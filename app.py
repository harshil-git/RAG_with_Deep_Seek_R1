import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain.storage import InMemoryStore
from langchain.retrievers import BM25Retriever,ParentDocumentRetriever

st.markdown("""
    <style>
    .stApp {
        background-color: #1A1F36;
        color: #E3E7F1;
    }
    
    /* Chat Input Styling */
    .stChatInput input {
        background-color: #2B2F4A !important;
        color: #E3E7F1 !important;
        border: 1px solid #4A506D !important;
    }
    
    /* User Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #3E2C57 !important;
        border: 1px solid #5A3E7E !important;
        color: #F3E5F5 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Assistant Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #214F4B !important;
        border: 1px solid #3B6B68 !important;
        color: #D1F7F5 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Avatar Styling */
    .stChatMessage .avatar {
        background-color: #FF9E64 !important;
        color: #1A1F36 !important;
    }
    
    /* Text Color Fix */
    .stChatMessage p, .stChatMessage div {
        color: #E3E7F1 !important;
    }
    
    .stFileUploader {
        background-color: #2B2F4A;
        border: 1px solid #4A506D;
        border-radius: 5px;
        padding: 15px;
    }
    
    h1, h2, h3 {
        color: #FF9E64 !important;
    }
    </style>
""", unsafe_allow_html=True)

PROMPT_TEMPLATE = """
You are helpful, respectful and honest assistant. Always answer as helpfully as possible using the context text provided. Your answers should only answer the question once and not have any other text after the answer is done.
if a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.

Query: {user_query} 
Context: {document_context} 
Answer:
"""

#using HuggingFaceEmbeddings to generate embedding vectors of the data.

model_name = "BAAI/bge-small-en-v1.5"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


PDF_STORAGE_PATH = 'document_store/pdfs/'
EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")
DOCUMENT_VECTOR_DB = Chroma(persist_directory="document_store/chroma",collection_name="full_documents", embedding_function=hf)
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")


def save_uploaded_file(uploaded_file):
    file_path = PDF_STORAGE_PATH + uploaded_file.name
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

def load_pdf_documents(file_path):
    document_loader = PyPDFLoader(file_path)
    return document_loader.load()

def index_documents(raw_docs):
  parent_splitter = RecursiveCharacterTextSplitter(separators=['\n\n','\n',','], chunk_size=2000,chunk_overlap=75)
  child_splitter = RecursiveCharacterTextSplitter(separators=['\n\n','\n',','], chunk_size=400,chunk_overlap=75)
  store = InMemoryStore()
  full_doc_retriever= ParentDocumentRetriever(
    vectorstore=DOCUMENT_VECTOR_DB,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    docstore=store,)

  full_doc_retriever.add_documents(raw_docs)
  return full_doc_retriever


def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    return response_chain.invoke({"user_query": user_query, "document_context": context_text})


# UI Configuration


st.title("📘 PDF Researcher Agent")
st.markdown("### Your Intelligent Document Assistant")
st.markdown("---")

# File Upload Section
uploaded_pdf = st.file_uploader(
    "Upload Research Document (PDF)",
    type="pdf",
    help="Select a PDF document for analysis",
    accept_multiple_files=False

)

if uploaded_pdf:
    saved_path = save_uploaded_file(uploaded_pdf)
    raw_docs = load_pdf_documents(saved_path)
    docs_retriever = index_documents(raw_docs)
    
    st.success("✅ Document processed successfully! Ask your questions below.")
    
    user_input = st.chat_input("Enter your question about the document...")
    
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.spinner("Analyzing document..."):
            relevant_docs = docs_retriever.get_relevant_documents(user_input)
            ai_response = generate_answer(user_input, relevant_docs)

        with st.chat_message("assistant", avatar="🤖"):
            st.write(ai_response)