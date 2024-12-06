# Libraries 
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from milvus import default_server
from pymilvus import connections
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import tempfile

# Set layout wide
st.set_page_config(layout='wide')

# Create title in the center
st.markdown(
    """
    <h1 style="text-align: center; color: white;">
        PDF Summarizer
    </h1>
    """,
    unsafe_allow_html=True
)

# File uploader for PDF
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file_path = temp_file.name
        temp_file.write(uploaded_file.read())

    try:
        # Use PyPDFLoader to read the content of the uploaded PDF
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()

        # Create splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1500,
            chunk_overlap = 150
        )

        # Split texts into chunks
        splits = text_splitter.split_documents(docs)

    finally:
        # Delete the temporary file after processing
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

if st.button("Analyze PDF"):
    with st.spinner("Analyzing PDF...Please wait!"):
    # Start local connection to Milvus
        URI = "http://127.0.0.1:19530"
        default_server.start()
        connections.connect(host='127.0.0.1', port=default_server.listen_port)

        # Create Vector Database
        vectordb = Milvus.from_documents(
            documents=splits,
            embedding=HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
            ),
            collection_name="dtsense_streamlit",
            connection_args={"uri": URI},
        )

        # Stop connection to Milvus
        default_server.stop()

        # Get API KEY from .env
        load_dotenv()

        # Use model with Groq
        llm = ChatGroq(model_name="llama3-8b-8192", temperature=0, groq_api_key=os.environ["GROQ_API_KEY"])

        # Define the prompt template for generating AI responses
        PROMPT_TEMPLATE = """
        Human: You are an AI assistant, and provides answers to questions by using fact based and statistical information when possible.
        Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        <context>
        {context}
        </context>

        <question>
        {question}
        </question>

        The response should be specific and use statistics or numbers when possible.

        Assistant:"""

        # Create a PromptTemplate instance with the defined template and input variables
        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE, input_variables=["context", "question"]
        )

        # Convert the vector store to a retriever
        retriever = vectordb.as_retriever()

        # Define a function to format the retrieved documents
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Define the RAG (Retrieval-Augmented Generation) chain for AI response generation
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    # finally:
    #     # Delete the temporary file after processing
    #     if os.path.exists(temp_file_path):
    #         os.remove(temp_file_path)

# Create text box for user
text = st.text_area("Text to analyze: ")

# Invoke LLM
if st.button("Generate Responses"):
    st.markdown(rag_chain.invoke(text))