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
from langchain.chains import LLMChain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.prompts import PromptTemplate
import random


st.set_page_config(page_title="YcK UI", page_icon="🎈", layout="wide")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Define a persistent directory for the vector database
PERSIST_DIRECTORY = Path("./persistent_db_yck")


def create_or_update_vector_db(uploaded_files=None) -> Chroma:
    PERSIST_DIRECTORY.mkdir(exist_ok=True)
    
    embedding_function = OllamaEmbeddings(model="nomic-embed-text")
    # Always try to load existing database
    vector_db = Chroma(persist_directory=str(PERSIST_DIRECTORY), embedding_function=embedding_function)
    logger.info(f"Loaded vector database with {vector_db._collection.count()} documents")

    if uploaded_files:
    
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


prompt_template = """
Task: Provide a detailed and comprehensive answer to the given question based on the provided context.

Context:
{context}

Question: {question}

Previous Question (if any): {previous_question}

Instructions:
1. Analyze the context thoroughly and provide a detailed answer.
2. Structure your response using bullet points or numbered lists for clarity.
3. Include specific details, dates, and relevant information from the context.
4. If this is a follow-up question, focus on aspects not covered in the previous answer.
5. Use direct quotes from the context, citing them with quotation marks.
6. If the context doesn't provide enough information, clearly state what's missing.
7. Provide any relevant background information or context that helps answer the question.
8. If applicable, mention achievements, skills, or experiences highlighted in the context.
9. If applicable, you should write 2 paragraphs at least.

Output Format:
Answer: [Your detailed, structured answer here]
Supporting Evidence: [Relevant quote(s) from the context]
Confidence: [High/Medium/Low] - Explain your confidence level

Now, please provide your comprehensive answer to the given question:
"""

def generate_query_variations(question: str, llm: ChatOllama) -> List[str]:
    variation_prompt = PromptTemplate(
        input_variables=["question"],
        template="Generate 3 variations of the following question, maintaining its core meaning:\n\nOriginal: {question}\n\n1."
    )
    variation_chain = LLMChain(llm=llm, prompt=variation_prompt)
    variations = variation_chain.run(question).split("\n")
    return [question] + [v.split(". ", 1)[1] for v in variations if ". " in v]


def process_question(question: str, vector_db: Chroma, selected_model: str,previous_question: str = "") -> str:
    llm = ChatOllama(model=selected_model, temperature=0.2)
    
     # Generate query variations
    query_variations = generate_query_variations(question, llm)
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=LLMChainExtractor.from_llm(llm),
        base_retriever=retriever
    )
    all_docs = []
    for query in query_variations:
        docs = compression_retriever.get_relevant_documents(query)
        all_docs.extend(docs)
    
    # Remove duplicates and randomly sample if too many
    unique_docs = []
    seen_contents = set()
    for doc in all_docs:
        if doc.page_content not in seen_contents:
            unique_docs.append(doc)
            seen_contents.add(doc.page_content)
    # Randomly sample if too many
    if len(unique_docs) > 10:
        unique_docs = random.sample(unique_docs, 10)
    
    if not unique_docs:
        return "Answer: I don't have unique information to answer this question based on the given context. Confidence: Low"

    context = "\n\n".join([doc.page_content for doc in unique_docs])
    

    # Retrieve relevant context
    retrieved_docs = retriever.get_relevant_documents(question)
    if not retrieved_docs:
        return "Answer: I don't have enough information to answer this question based on the given context. Confidence: Low"
    
    
    # Combine retrieved documents, penalizing repetition
    #context = "\n\n".join(set([doc.page_content for doc in retrieved_docs]))  # Deduplicate
    
    # Use the prompt_template in the final QA step
    qa_prompt = PromptTemplate(
        input_variables=["context", "question", "previous_question"],
        template=prompt_template
    )
    
    rag_chain = (
        {"context": lambda x: context, "question": RunnablePassthrough(), "previous_question": lambda x: previous_question}
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
        #st.session_state["vector_db"] = None
        st.session_state["vector_db"] = create_or_update_vector_db()
        if "vector_db" not in st.session_state and PERSIST_DIRECTORY.exists() and any(PERSIST_DIRECTORY.iterdir()):
            st.session_state["vector_db"] = Chroma(persist_directory=str(PERSIST_DIRECTORY), embedding_function=OllamaEmbeddings(model="nomic-embed-text"))
    if "previous_question" not in st.session_state:
        st.session_state["previous_question"] = ""

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

        
        prompt = st.chat_input("Enter a prompt here...")
        
        if prompt:
            # Add user message
            st.session_state["messages"].append({"role": "user", "content": prompt})
            with message_container.chat_message("user", avatar="😎"):
                st.markdown(prompt)

            # Generate and display response
            if st.session_state["vector_db"] is not None:
                with st.spinner(":green[processing...]"):
                    response = process_question(prompt, st.session_state["vector_db"], selected_model, st.session_state["previous_question"])
                
                st.session_state["messages"].append({"role": "assistant", "content": response})
                with message_container.chat_message("assistant", avatar="🤖"):
                    st.markdown(response)
                
                st.session_state["previous_question"] = prompt
            else:
                with message_container.chat_message("assistant", avatar="🤖"):
                    st.warning("No documents have been uploaded yet. The response may not be accurate or relevant.")


if __name__ == "__main__":
    main()