from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from workflow import build_workflow
from langchain.schema import HumanMessage, AIMessage
import uvicorn

app = FastAPI()
workflow = build_workflow().compile()

class UserMessage(BaseModel):
    customer_id: int
    message: str

@app.post("/chat/")
async def chat(user_msg: UserMessage):
    state = {"messages": [HumanMessage(content=user_msg.message)], "customer_id": user_msg.customer_id}
    state = workflow.invoke(state)
    bot_response = next((msg.content for msg in reversed(state["messages"]) if isinstance(msg, AIMessage)), "Üzgünüm, isteğinizi anlayamadım.")
    return {"response": bot_response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
