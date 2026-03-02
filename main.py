from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from app import get_rag
import os

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat_api(request: ChatRequest):
    rag = get_rag()
    result = rag.invoke(
        {"input": request.message},
        config={"configurable": {"session_id": "default"}}
    )
    return {"answer": result["answer"]}
@app.get("/")
def root():
    return {"status": "Backend is running 🚀"}
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)