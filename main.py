from fastapi import FastAPI
from pydantic import BaseModel
from app import conversational_rag  # import từ file app.py của bạn

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat_api(request: ChatRequest):
    result = conversational_rag.invoke(
        {"input": request.message},
        config={"configurable": {"session_id": "default"}}
    )
    return {"answer": result["answer"]}
@app.get("/")
def root():
    return {"status": "Backend is running 🚀"}