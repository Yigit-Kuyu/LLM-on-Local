import streamlit as st
import logging
import tempfile
import os
import shutil
from typing import Dict, List, Any, Tuple, Optional
import ollama
import pdfplumber
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOllama
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.retrievers import MultiQueryRetriever
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
import os
from pathlib import Path
import logging
from langchain.document_loaders import PyPDFLoader

st.set_page_config(page_title="YcK UI", page_icon="🎈", layout="wide")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Define a persistent directory for the vector database
PERSIST_DIRECTORY = Path("./persistent_db_yck")


def create_or_update_vector_db(uploaded_files) -> Chroma:
    PERSIST_DIRECTORY.mkdir(exist_ok=True)
    
    embedding_function = OllamaEmbeddings(model="nomic-embed-text")
    
    # Load existing database if it exists, otherwise create a new one
    if PERSIST_DIRECTORY.exists() and any(PERSIST_DIRECTORY.iterdir()):
        vector_db = Chroma(persist_directory=str(PERSIST_DIRECTORY), embedding_function=embedding_function)
        logger.info(f"Loaded existing vector database with {vector_db._collection.count()} documents")
    else:
        vector_db = Chroma(persist_directory=str(PERSIST_DIRECTORY), embedding_function=embedding_function)
        logger.info("Created new vector database")
    
    all_chunks = []
    for file_upload in uploaded_files:
        try:
            temp_dir = tempfile.mkdtemp()
            path = os.path.join(temp_dir, file_upload.name)
            with open(path, "wb") as f:
                f.write(file_upload.getvalue())
            
            # Use PyPDFLoader instead of UnstructuredPDFLoader
            loader = PyPDFLoader(path)
            pages = loader.load_and_split()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(pages)
            
            all_chunks.extend(chunks)
            logger.info(f"Successfully processed {file_upload.name}")
        except Exception as e:
            logger.error(f"Error processing {file_upload.name}: {str(e)}")
            st.warning(f"Failed to process {file_upload.name}. This file will be skipped.")
        finally:
            shutil.rmtree(temp_dir)
    
    if all_chunks:
        # Add new documents to the existing database
        vector_db.add_documents(all_chunks)
        vector_db.persist()
        logger.info(f"Added {len(all_chunks)} chunks to the vector database")
    else:
        logger.warning("No documents were successfully processed")
    
    return vector_db


def clear_vector_db():
    if PERSIST_DIRECTORY.exists():
        shutil.rmtree(PERSIST_DIRECTORY)
    if "vector_db" in st.session_state:
        del st.session_state["vector_db"]


@st.cache_resource 
def extract_model_names(models_info: Dict[str, List[Dict[str, Any]]]) -> Tuple[str, ...]: #  extracts model names from the Ollama API response.
    return tuple(model["name"] for model in models_info["models"])


'''
def create_vector_db(uploaded_files) -> Chroma:
    all_chunks = []
    for file_upload in uploaded_files:
        temp_dir = tempfile.mkdtemp()
        path = os.path.join(temp_dir, file_upload.name)
        with open(path, "wb") as f:
            f.write(file_upload.getvalue())
        chunks = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200).split_documents(UnstructuredPDFLoader(path).load())
        all_chunks.extend(chunks)
        shutil.rmtree(temp_dir)
    return Chroma.from_documents(documents=all_chunks, embedding=OllamaEmbeddings(model="nomic-embed-text"), collection_name="myRAG")

'''

prompt_template="""
            Task: Answer the given question based solely on the provided context.

            Context:
            {context}

            Question: {question}

            Instructions:
            1. Carefully read the context and question.
            2. If the answer is explicitly stated in the context, provide it directly.
            3. If the answer can be inferred from the context, explain your reasoning.
            4. If the answer cannot be determined from the context, state "I don't have enough information to answer this question based on the given context."
            5. Do not use any external knowledge or make assumptions beyond the provided context.
            6. Include relevant quotes from the context to support your answer, using quotation marks.
            7. Keep your answer concise and to the point.

            Output Format:
            Answer: [Your answer here]
            Supporting Evidence: [Relevant quote(s) from the context]
            Confidence: [High/Medium/Low] - Based on how directly the context addresses the question

            Example:
            Context: The Eiffel Tower was completed in 1889. It stands 324 meters tall and was the tallest man-made structure in the world for 41 years.
            Question: When was the Eiffel Tower built?
            Answer: The Eiffel Tower was completed in 1889.
            Supporting Evidence: "The Eiffel Tower was completed in 1889."
            Confidence: High

            Now, please provide your answer to the given question:
            """

# This prompt is for generating multiple queries
multi_query_prompt = """
Given the following question, please generate 3 different versions of the question to help retrieve relevant information from a vector database:

Original question: {question}

1.
2.
3.
"""

def process_question(question: str, vector_db: Chroma, selected_model: str) -> str:
    llm = ChatOllama(model=selected_model, temperature=0.0)
    # Use the multi_query_prompt for the MultiQueryRetriever
    multi_query_prompt_template = PromptTemplate(
        input_variables=["question"],
        template=multi_query_prompt
    )
    
    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(),
        llm,
        prompt=multi_query_prompt_template
    )

    
    # Retrieve relevant context
    retrieved_docs = retriever.get_relevant_documents(question)
    if not retrieved_docs:
        return "Answer: I don't have enough information to answer this question based on the given context. Confidence: Low"
    
    
    # Combine retrieved documents into a single string for the context
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # Use the prompt_template in the final QA step
    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )
    
    rag_chain = (
        {"context": lambda x: context, "question": RunnablePassthrough()}
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    
    answer = rag_chain.invoke(question)
    return answer

@st.cache_data
def extract_all_pages_as_images(file_upload) -> List[Any]:
    with pdfplumber.open(file_upload) as pdf:
        return [page.to_image().original for page in pdf.pages]


def main() -> None:
    st.subheader("🧠 YcK-Model", divider="gray", anchor=False)
    models_info = ollama.list()
    available_models = extract_model_names(models_info)
    col1, col2 = st.columns([1.5, 2])

    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "vector_db" not in st.session_state:
        st.session_state["vector_db"] = None
        if "vector_db" not in st.session_state and PERSIST_DIRECTORY.exists() and any(PERSIST_DIRECTORY.iterdir()):
            st.session_state["vector_db"] = Chroma(persist_directory=str(PERSIST_DIRECTORY), embedding_function=OllamaEmbeddings(model="nomic-embed-text"))

    selected_model = col2.selectbox("Pick a model available locally on your system ↓", available_models) if available_models else None
    file_upload = col1.file_uploader("Upload PDF file(s) ↓", type="pdf", accept_multiple_files=True)

    if file_upload:
        with st.spinner("Uploading PDF files..."):
            st.session_state["vector_db"] =  create_or_update_vector_db(file_upload)
            st.session_state["pdf_pages"] = [page for file in file_upload for page in extract_all_pages_as_images(file)]

    if col1.button("Clear Uploads", type="secondary"):
        clear_vector_db()
        for key in ["pdf_pages", "file_uploads"]:
            st.session_state.pop(key, None)
        st.success("Uploads cleared successfully.")

    with col2:
        #message_container = st.container(height=500, border=True)
        message_container = st.container()
        for message in st.session_state["messages"]:
            with message_container.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else "😎"):
                st.markdown(message["content"])

        # Disable chat input while files are being processed
        # Disable chat input while files are being processed OR database is not ready
        input_disabled = (file_upload is not None and "vector_db" not in st.session_state) or (
            "vector_db" in st.session_state and st.session_state["vector_db"] is None
        )
        prompt = st.chat_input("Enter a prompt here...", disabled=input_disabled)
        
        if prompt:
            st.session_state["messages"].append({"role": "user", "content": prompt})
            message_container.chat_message("user", avatar="🤖").markdown(prompt)
            with message_container.chat_message("assistant", avatar="😎"):
                with st.spinner(":green[processing...]"):
                    if st.session_state["vector_db"] is not None:
                        response = process_question(prompt, st.session_state["vector_db"], selected_model)
                        st.markdown(response)
                        st.session_state["messages"].append({"role": "assistant", "content": response})
                    else:
                        st.warning("Please upload PDF file(s) first.")
        elif st.session_state["vector_db"] is None:
            st.warning("Upload PDF file(s) to begin chat...")




if __name__ == "__main__":
    main()