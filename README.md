## Info
This repository contains a Streamlit-based web application for document question-answering using large language models. The app allows users to upload PDF files, which are processed and stored in a vector database using Chroma and Ollama embeddings. Users can then ask questions about the uploaded documents, and the application uses a retrieval-augmented generation (RAG) approach to provide answers. The system leverages locally available Ollama models for text generation and supports multiple query generation for improved retrieval. Key features include PDF processing, persistent vector storage, multi-query retrieval, and a user-friendly chat interface. The application is designed to be easily deployable and customizable for various document QA scenarios.

### Specifications
- The framework is able to work on local or on internet via *ngrok*.
- The framework has a memory so you don't need to upload the same pdf files again for each run.
- By changing the database name in the code, you can create different frameworks for different tasks (best base models according to experiments *mistral* and *qwen2*).
- To add a new base model, use `ollama pull model_name` (for example, ollama pull aya:8b). The alternative base models can be found [here](https://ollama.com/library).
- If you push "Clear Uploads" button, you will delete the whole database.
- To decrease processing time:
  - Reduce the number of query variations
  - Optimize chunk size
  - Reduce number of documents retrieved

The UI is here:              

![AV_Module](https://github.com/Yigit-Kuyu/LLM-on-Local/blob/main/Interface.jpg)

### Running on local
`streamlit run LLM_V4.py`
This will create a database in your local and network to the local host with port number 8501. Thats all.

### Running on the internet
- Login [ngrok](https://ngrok.com/) and follow the instructions for the installation including the addition of authtoken.
- Run *ngrok* with the same port number `ngrok http 8501` on different terminal, this will give you a web-link as seen in the figure below.

![AV_Module](https://github.com/Yigit-Kuyu/LLM-on-Local/blob/main/ngrok_api.jpg)

- Use this link on any computer with internet access. As seen in the below figure, the framework can recognize me without any upload due to the database in the main PC.  


![AV_Module](https://github.com/Yigit-Kuyu/LLM-on-Local/blob/main/ngrok_web.jpg)






