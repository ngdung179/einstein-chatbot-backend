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

# ==============================
# LOAD DOCUMENTS
# ==============================
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

# ==============================
# SPLIT CHUNKS
# ==============================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
)

chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# ==============================
# EMBEDDINGS
# ==============================
embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)

# ==============================
# VECTORSTORE (NO REBUILD IF EXISTS)
# ==============================
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

# ==============================
# KEYWORD RETRIEVER
# ==============================
class KeywordRetriever(BaseRetriever):
    docs: List[Document] = Field(default_factory=list)

    def _get_relevant_documents(self, query: str):
        results = []
        for doc in self.docs:
            if any(word in doc.page_content.lower() for word in query.lower().split()):
                results.append(doc)
        return results[:5]

keyword_retriever = KeywordRetriever(docs=chunks)

# ==============================
# HYBRID RETRIEVER
# ==============================
class HybridRetriever(BaseRetriever):
    def _get_relevant_documents(self, query):
        vec_docs = vector_retriever.invoke(query)
        key_docs = keyword_retriever.invoke(query)

        combined = {doc.page_content: doc for doc in vec_docs}
        for doc in key_docs:
            combined[doc.page_content] = doc

        return list(combined.values())[:10]

hybrid_retriever = HybridRetriever()

# ==============================
# LLM
# ==============================
llm = ChatGoogleGenerativeAI(
    model=MODEL,
    temperature=0.4,
    convert_system_message_to_human=True,
)

# ==============================
# SYSTEM PROMPT
# ==============================
system_message = """
"Bạn là chatbot hỗ trợ chính thức cho trung tâm Einstein House (website: https://einsteinhouse.vn/). "
    "Einstein House là trung tâm trải nghiệm khoa học và STEAM dành cho trẻ em, cung cấp không gian đọc sách, hoạt động trải nghiệm khoa học, lập trình, và các sản phẩm giáo dục. Einstein House có thư viện sách phong phú, trải nghiệm thí nghiệm, công nghệ và nhiều khu vực học tập hấp dẫn cho trẻ em từ 5-12 tuổi. "  # thông tin chính xác từ website
    "Nhiệm vụ của bạn là tư vấn và trả lời các câu hỏi liên quan đến Einstein House cho cả khách hàng và nhân viên trung tâm một cách ngắn gọn, chính xác và chuyên nghiệp. "
    
    "Đối với khách hàng: "
    "- Trả lời các câu hỏi về sản phẩm, sách và dịch vụ tại Einstein House. "
    "- Hướng dẫn khách khi đến trung tâm, các trải nghiệm có thể tham gia, và các thông tin liên quan đến sự kiện hoặc mua hàng. "
    "- Giải đáp thắc mắc về thời gian hoạt động, địa chỉ, liên hệ hỗ trợ. "

    "Đối với nhân viên: "
    "- Trả lời các câu hỏi về nội quy, quy trình nội bộ, công việc, quy định cho nhân viên mới, nghỉ phép, báo cáo hay liên hệ phòng ban phù hợp. "
    
    "Bạn chỉ sử dụng thông tin đã có trong ngữ cảnh được cung cấp. "
    "Nếu không biết câu trả lời, hãy nói rõ rằng bạn không biết thay vì bịa ra thông tin mới. "
    "Trả lời thân thiện, dễ hiểu và phù hợp với vai trò người hỏi."
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_message),
    ("human", 
     """Dựa trên thông tin sau:
     
{context}

Hãy trả lời câu hỏi:
{input}
""")
])

# ==============================
# CREATE RAG CHAIN
# ==============================
document_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(hybrid_retriever, document_chain)

# ==============================
# MEMORY
# ==============================
store = {}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

conversational_rag = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer", 
)

# ==============================
# CHAT FUNCTION
# ==============================
def chat(message, history):
    result = conversational_rag.invoke(
        {"input": message},
        config={"configurable": {"session_id": "default"}}
    )
    return result["answer"]
