# the main work of this file is indexing => mean convert file into chunks
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

# for creating chunks using text splitter
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

#print("API KEY =", os.getenv("GEMINI_API_KEY"))

# convert chunks into embeddings

from langchain_openai import OpenAIEmbeddings
# openai is paid api key so i use genai

from langchain_google_genai import GoogleGenerativeAIEmbeddings

# now we need to save these embeddings and chunks into qdrant db

from langchain_qdrant import QdrantVectorStore



pdf_path = Path(__file__).parent / "Software.pdf"

# now we need to load this file
# to load this will return pages
loader = PyPDFLoader(file_path =pdf_path)
docx = loader.load()
#print(len(pages))

# now we need to make chunks from this files or pdf or pgae

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000 ,
    chunk_overlap=300 
)

chunks = text_splitter.split_documents(documents=docx)
#print(chunks)


# now converts these chunks into vector embeddings

# embeddings = GoogleGenerativeAIEmbeddings(
#     model="models/embedding-001"   # Gemini embedding model
# )
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

# now we need to move these embeddings and chuks into qdrant db

vectors = QdrantVectorStore.from_documents(
    documents = chunks ,
    embedding = embeddings ,
    collection_name="Rag Learn" ,
    url = "http://localhost:6333"

)

print('Vector done..........')

