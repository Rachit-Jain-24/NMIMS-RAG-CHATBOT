import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="NMIMS Hyderabad FAQ Chatbot",
    page_icon="🤖",
    layout="centered",
)

# --- Caching the RAG Chain ---
@st.cache_resource
def create_rag_chain(pdf_bytes, api_key):
    """
    Creates and returns a conversational retrieval chain from PDF bytes.
    This function is cached to avoid recreating the chain on every interaction.
    """
    if not api_key:
        st.error("OpenRouter API key is missing.")
        st.stop()
        
    os.environ["OPENROUTER_API_KEY"] = api_key

    # Save bytes to a temporary file to be loaded by PyPDFLoader
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_bytes)
        tmp_file_path = tmp_file.name

    # 1. Load and Split the PDF
    try:
        with st.spinner("Reading and preparing the PDF..."):
            loader = PyPDFLoader(tmp_file_path)
            pages = loader.load_and_split()
            pdf_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            docs = pdf_splitter.split_documents(pages)
    finally:
        # Clean up the temporary file
        os.remove(tmp_file_path)

    # 2. Create Embeddings and Vector Store
    with st.spinner("Creating document embeddings..."):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vector_db = Chroma.from_documents(documents=docs, embedding=embeddings)

    # 3. Set up Conversational Memory
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True
    )

    # 4. Create the Language Model (LLM)
    llm = ChatOpenAI(
        model="openai/gpt-3.5-turbo",
        temperature=0.2,
        openai_api_base="https://openrouter.ai/api/v1",
        max_tokens=500,
        openai_api_key=api_key
    )

    # 5. Create the Conversational Retrieval Chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_db.as_retriever(search_type="mmr"),
        memory=memory,
        return_source_documents=False
    )
    st.success("PDF processed successfully. You can now ask questions.")
    return qa_chain

# --- Streamlit App UI ---
st.image("nmimslogo.webp", width=200)
st.title("NMIMS Hyderabad FAQ Chatbot")

st.info("Ask any question about NMIMS Hyderabad based on the provided FAQ document.")

# --- API Key ---
# The API key is hardcoded for faster setup as requested.
# Avoid committing this directly to a public repository in a real-world scenario.
openrouter_api_key = "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"


# --- Sidebar Information ---
with st.sidebar:
    st.header("About")
    st.info("This chatbot uses the 'NMIMS_FAQ.pdf' as its knowledge base.")

# --- Session State Management ---
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Load Default PDF ---
pdf_bytes = None
try:
    with open("NMIMS_FAQ.pdf", "rb") as f:
        pdf_bytes = f.read()
except FileNotFoundError:
    st.error("Default file 'NMIMS_FAQ.pdf' not found. Please make sure it's in the same directory as the app.")
    st.stop()


# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Main Chat Logic ---
if pdf_bytes and openrouter_api_key:
    # Create the RAG chain (will be cached)
    qa_chain = create_rag_chain(pdf_bytes, openrouter_api_key)
    
    if prompt := st.chat_input("Ask your question here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = qa_chain({"question": prompt})
                bot_message = response["answer"]
                st.markdown(bot_message)
        
        st.session_state.messages.append({"role": "assistant", "content": bot_message})
else:
    # This part should ideally not be reached if the PDF is missing due to st.stop()
    st.warning("There was an issue loading the PDF or API key.")

