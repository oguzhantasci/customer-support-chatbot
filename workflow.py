from langgraph.graph import StateGraph, START, END
from agents import financial_agent, transaction_agent, complaint_agent, query_agent, transaction_history_agent
from tools import classify_intent
from langchain.schema import BaseMessage
from typing import Sequence, TypedDict


class AgentState(TypedDict):
    """Defines the structure of state data passed between agents."""
    messages: Sequence[BaseMessage]
    next: str
    customer_id: int


def supervisor_agent(state):
    """Routes user messages to the correct agent based on intent."""
    last_message = state["messages"][-1].content
    intent = classify_intent(last_message)

    return {"messages": state["messages"], "next": intent}


def build_workflow() -> StateGraph:
    workflow = StateGraph(AgentState)

    # Add nodes (agents)
    workflow.add_node("Supervisor_Agent", supervisor_agent)
    workflow.add_node("Financial_Agent", financial_agent)
    workflow.add_node("Transaction_Agent", transaction_agent)
    workflow.add_node("Transaction_History_Agent", transaction_history_agent)
    workflow.add_node("Complaint_Agent", complaint_agent)
    workflow.add_node("Query_Agent", query_agent)
    workflow.add_node("END", lambda state: state)  # ✅ Explicitly define END as a valid state

    # Define possible routes
    workflow.add_edge(START, "Supervisor_Agent")

    # Use conditional edges for dynamic routing
    workflow.add_conditional_edges("Supervisor_Agent", lambda x: x["next"], {
        "BALANCE_INQUIRY": "Financial_Agent",
        "TRANSACTION_REQUEST": "Transaction_Agent",
        "TRANSACTION_HISTORY": "Transaction_History_Agent",
        "COMPLAINT": "Complaint_Agent",
        "GENERAL_QUERY": "Query_Agent"
    })

    # ✅ Ensure END is explicitly recognized in the workflow
    workflow.add_edge("Financial_Agent", END)
    workflow.add_edge("Transaction_Agent", END)
    workflow.add_edge("Transaction_History_Agent", END)
    workflow.add_edge("Complaint_Agent", END)
    workflow.add_edge("Query_Agent", END)

    return workflow
