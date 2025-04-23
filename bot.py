import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
import os

# Set up Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY", "AIzaSyAccMg6J9cA1BgyAEOvrGtLQ9RH7YbGQhc"))  # Replace with your actual API key
gemini = genai.GenerativeModel("gemini-1.5-flash")  # Use a valid model name

# Load and split documents (cached globally)
@st.cache_resource
def load_documents():
    loader = TextLoader("rec_azamgarh_info.txt")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_documents(docs)

# Embed documents (cached globally)
@st.cache_resource
def embed_documents(_docs):  # Use underscore to bypass hashing
    texts = [doc.page_content for doc in _docs]  # Extract text from documents
    # Use HuggingFaceEmbeddings to wrap the SentenceTransformer model
    embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    # Create FAISS index using the embedding model
    db = FAISS.from_texts(texts, embedding=embedding_model, metadatas=[{"text": text} for text in texts])
    return db

# Initialize documents and vector store once
docs = load_documents()
db = embed_documents(docs)

# Streamlit UI
st.title("📚 REC Azamgarh Chatbot")
st.write("Ask me anything about REC Azamgarh!")

user_input = st.text_input("Your Question:")

if user_input:
    # Get context from FAISS
    search_results = db.similarity_search_with_score(user_input, k=3)
    context = "\n".join([doc.page_content for doc, _ in search_results])

    prompt = f"Use the following context to answer the question:\n\n{context}\n\nQuestion: {user_input}"

    try:
        response = gemini.generate_content(prompt)
        st.markdown(f"### 🧠 Answer:\n{response.text}")
    except Exception as e:
        st.error(f"❌ Gemini Error: {e}")