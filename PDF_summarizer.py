# Libraries 
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
import faiss
import numpy as np
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import tempfile

# Set layout wide
st.set_page_config(layout='wide')

# Detect theme (dark or light mode)
current_theme = st.get_option("theme.base")
text_color = "white" if current_theme == "dark" else "black"

# Create title in the center
st.markdown(
    """
    <h1 style="text-align: center; color: {text_color}; font-size: 32px">
        PDF Summarizer
    </h1>
    """,
    unsafe_allow_html=True
)

# Add description
st.write(
    """
    <p style="text-align: justify; font-size: 18px; line-height: 1.6;">
        This application allows users to upload a PDF document, process it, and generate AI-assisted summaries or responses based on the content of the document.
        It utilizes state-of-the-art embeddings, document retrieval, and generative AI technologies to provide fact-based and concise answers to user queries.
        <br><br>
        How to use the application:
        <ol>
            <li>Upload your PDF</li>
            <li>Ask the content of the PDF however you like</li>
            <li>Click [Generate Response] button</li>
        </ol>
    </p>
    <br>
    """,
    unsafe_allow_html=True
)



# File uploader for PDF
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Create text box for user
text = st.text_area("Text to analyze: ")

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file_path = temp_file.name
        temp_file.write(uploaded_file.read())

        # Use PyPDFLoader to read the content of the uploaded PDF
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()

        # Create splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=150
        )

        # Split texts into chunks
        splits = text_splitter.split_documents(docs)

        if st.button("Generate Response"):
            # Create embedding using HuggingFace
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
            )

            # Generate embeddings for the document chunks using embed_documents()
            vectors = np.array(embeddings.embed_documents([doc.page_content for doc in splits]), dtype=np.float32)

            # Create a FAISS index
            dimension = vectors.shape[1]  # Vector dimension
            index = faiss.IndexFlatL2(dimension)  # L2 distance (Euclidean)
            index.add(vectors)  # Add vectors to the index

            # Save the index to use for later retrieval
            faiss.write_index(index, "faiss_index.idx")

            # Load API KEY from .env
            load_dotenv()

            # Use model with Groq
            llm = ChatGroq(model_name="llama3-8b-8192", temperature=0, groq_api_key=os.environ["GROQ_API_KEY"])

            # Define the prompt template for generating AI responses
            PROMPT_TEMPLATE = """
            Human: You are an AI assistant, and provides answers to questions by using fact based and statistical information when possible.
            Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.

            <question>
            {question}
            </question>

            The response should be specific and use statistics or numbers when possible.

            Don't re-write the questions when providing a generation responses and don't write the <question> tag.

            Assistant:"""

            # Create a PromptTemplate instance with the defined template and input variables
            prompt = PromptTemplate(
                template=PROMPT_TEMPLATE, input_variables=["context", "question"]
            )

            # Define a function to retrieve the most similar documents using FAISS
            def retrieve_docs(query):
                # Ensure the output is converted to a NumPy array
                query_vector = np.array(embeddings.embed_documents([query])).reshape(1, -1)
                _, indices = index.search(query_vector, k=5)  # Change `k` for more/less results
                # Return the document objects based on the indices
                return [splits[i] for i in indices[0] if i < len(splits)]

            # Convert the FAISS index to a retriever
            retriever = RunnablePassthrough(lambda query: retrieve_docs(query))

            def format_docs(docs):
                # Ensure docs are document objects and not strings
                return "\n\n".join(doc.page_content for doc in docs if hasattr(doc, 'page_content'))


            # Define the RAG (Retrieval-Augmented Generation) chain for AI response generation
            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            # Invoke LLM
            st.markdown(rag_chain.invoke(text))

            # Delete the temporary file after processing
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

# Add sidebar title
st.sidebar.markdown(
    """
    <h1 style="text-align: center; color: {text_color};">
        About the Developer
    </h1>
    """,
    unsafe_allow_html=True
)

# Add a profile picture to the sidebar & biography
st.sidebar.markdown(
    """
    <div style="text-align: center;">
        <img src="https://raw.githubusercontent.com/NicNicNicNicNic5/PDF_summarizer/main/myPic.jpeg" 
             style="border-radius: 50%; width: 150px; height: 150px; margin-bottom: 20px;">
    </div>
    <p style="text-align: justify; font-size: 14px; line-height: 1.6;">
        Hi! I'm Nicholas Suardiman, a passionate developer with expertise in Python, AI/ML, and Streamlit applications.
        <br><br>
        I specialize in building AI-driven solutions that simplify complex workflows and enhance user experience.
        <br><br>
        This project I created is my final project in Generative AI Engineer (LLM & RAG) class from DTSense.
    </p>
    <hr style="border: 1px solid #ccc; margin: 20px 0;">
    <div style="text-align: center;">
        <img src="https://raw.githubusercontent.com/NicNicNicNicNic5/PDF_summarizer/main/DTSense.jpeg" 
             style="border-radius: 50%; width: 150px; height: 150px; margin-bottom: 20px;">
    </div>
    <p style="text-align: justify; font-size: 14px; line-height: 1.6;">
        DTSense Academy is an institution that provides consulting and training services in the field of Information Technology, focusing on Data Science and Artificial Intelligence.
        <br><br>
        DTSense Academy also offers various classes, including the Generative AI Engineer (LLM & RAG) class, which I attended.
        <br><br>
        In this class, I learned Python from scratch and gained skills to fine-tune LLM models.
    </p>
    """,
    unsafe_allow_html=True
)