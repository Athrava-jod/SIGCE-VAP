from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# ==========================================
# 1. Load local PDF documents
# ==========================================

pdf_files = [
    "hack.pdf",
    "yd.pdf"
]

documents = []

for pdf_file in pdf_files:
    documents.extend(PyPDFLoader(pdf_file).load())

print(f"Loaded {len(documents)} pages")


# ==========================================
# 2. Split documents into chunks
# ==========================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")


# ==========================================
# 3. Create embeddings
# ==========================================

embeddings = OllamaEmbeddings(
    model="qwen3-embedding:0.6b"
)


# ==========================================
# 4. Create Chroma vector database
# ==========================================

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("Vector database created and persisted locally")