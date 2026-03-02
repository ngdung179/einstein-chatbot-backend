import os
import glob
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from typing import List
from pydantic import Field
from langchain_core.documents import Document
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# ==============================
# LOAD ENV
# ==============================
import os

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found")
# ==============================
# CONFIG
# ==============================
MODEL = "models/gemini-2.5-flash"
EMBED_MODEL = "models/gemini-embedding-001"
DB_NAME = "vector_db"
DATA_DIR = "data_EH"

def build_rag():
    print("Building RAG system...")

    documents = []
    for path in glob.glob(f"{DATA_DIR}/**/*.md", recursive=True):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        documents.append(
            Document(
                page_content=content,
                metadata={"source": os.path.basename(path)},
            )
        )

    print(f"Loaded {len(documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)

    if os.path.exists(DB_NAME):
        vectorstore = Chroma(
            persist_directory=DB_NAME,
            embedding_function=embeddings,
        )
        print("Loaded existing vector DB")
    else:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=DB_NAME,
        )
        print("Created new vector DB")

    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    class HybridRetriever(BaseRetriever):
        def _get_relevant_documents(self, query):
            vec_docs = vector_retriever.invoke(query)
            return vec_docs

    llm = ChatGoogleGenerativeAI(
        model=MODEL,
        temperature=0.4,
        convert_system_message_to_human=True,
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(HybridRetriever(), document_chain)

    return rag_chain

# ==============================
# MEMORY
# ==============================
store = {}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

conversational_rag = None

def get_rag():
    global conversational_rag
    if conversational_rag is None:
        conversational_rag = build_rag()  # hàm tạo RAG của bạn
    return conversational_rag

# ==============================
# CHAT FUNCTION
# ==============================
def chat(message, history):
    result = conversational_rag.invoke(
        {"input": message},
        config={"configurable": {"session_id": "default"}}
    )
    return result["answer"]