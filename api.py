from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from workflow import build_workflow
from langchain.schema import HumanMessage, AIMessage
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
workflow = build_workflow().compile()

@app.get("/")
def home():
    return {"message": "Welcome to the Banking Chatbot API!"}

class UserMessage(BaseModel):
    customer_id: int
    message: str

# CORS settings to allow requests from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://oguzhantasci.github.io"],  # GitHub Pages URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat/")
async def chat(user_msg: UserMessage):
    state = {"messages": [HumanMessage(content=user_msg.message)], "customer_id": user_msg.customer_id}
    state = workflow.invoke(state)
    bot_response = next((msg.content for msg in reversed(state["messages"]) if isinstance(msg, AIMessage)), "Üzgünüm, isteğinizi anlayamadım.")
    return {"response": bot_response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
