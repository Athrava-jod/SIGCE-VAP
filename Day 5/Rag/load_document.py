import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from langchain_community.document_loaders import PyPDFLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

pdf_files = ["sample_bank_statement.pdf"]

document =[]
for pdf_file in pdf_files:
    document.extend(PyPDFLoader("sample_bank_statement.pdf").load())
