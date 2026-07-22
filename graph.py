import os
import json
from typing import TypedDict, Literal, List
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from Agent_Nodes import (
    llm, router_node, sql_retriever_node, 
    vector_retriever_node, context_grader_node, query_rewriter_node
)

# 1. Define the State that flows through the graph
class AgentState(TypedDict):
    user_query: str
    current_search_query: str
    routing_decision: str
    retrieved_context: str
    is_sufficient: str
    loop_count: int
    final_output: dict

# 2. Define the Pydantic Schema for the Final Audit Report
class AuditReport(BaseModel):
    executive_summary: str = Field(description="Direct, factual answer to the user's question.")
    cited_metrics_or_quotes: List[str] = Field(
        description="A JSON array/list of individual strings containing exact numbers from SQL or quotes from transcripts. Example format: ['Quote 1', '143756000000.0']. NEVER output a single stringified list."
    )
    risk_assessment: Literal["LOW", "MEDIUM", "HIGH", "UNCLEAR"] = Field(description="Risk level based on executive tone and financial exposure.")
    data_source_used: str = Field(description="State whether DuckDB (SQL), ChromaDB (Vector), or BOTH were used.")

# ==========================================
# 3. DEFINE THE GRAPH NODES
# ==========================================
def node_router(state: AgentState):
    decision = router_node(state["user_query"])
    return {"routing_decision": decision, "current_search_query": state["user_query"], "loop_count": 0}

def node_retrieve(state: AgentState):
    decision = state["routing_decision"]
    query = state["current_search_query"]
    context = ""
    
    if decision in ["SQL", "BOTH"]:
        context += f"\n[SQL RELATIONAL DATA]:\n{sql_retriever_node(query)}\n"
    if decision in ["VECTOR", "BOTH"]:
        context += f"\n[VECTOR TRANSCRIPT DATA]:\n{vector_retriever_node(query)}\n"
        
    return {"retrieved_context": context}

def node_grade(state: AgentState):
    # Only grade vector retrieval; SQL math is assumed deterministic for this scope
    if state["routing_decision"] == "SQL":
        return {"is_sufficient": "YES"}
        
    grade = context_grader_node(state["user_query"], state["retrieved_context"])
    return {"is_sufficient": grade}

def node_rewrite(state: AgentState):
    new_query = query_rewriter_node(state["current_search_query"])
    return {"current_search_query": new_query, "loop_count": state["loop_count"] + 1}

def node_synthesize(state: AgentState):
    print("✨ [Synthesis Node] Generating final deterministic JSON audit report...")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert financial auditor. Output a structured JSON audit report strictly based on the retrieved context. Do NOT hallucinate numbers or facts not present in the text. Ensure that `cited_metrics_or_quotes` is output as a valid JSON list/array of strings, not a single text string."),
        ("human", "User Query: {query}\n\nRetrieved Evidence:\n{context}\n\nData Sources Used: {source}")
    ])
    
    structured_llm = llm.with_structured_output(AuditReport)
    chain = prompt | structured_llm
    report = chain.invoke({
        "query": state["user_query"], 
        "context": state["retrieved_context"],
        "source": state["routing_decision"]
    })
    return {"final_output": report.model_dump()}

# ==========================================
# 4. CONDITIONAL EDGE LOGIC
# ==========================================
def decide_to_rewrite(state: AgentState):
    if state["is_sufficient"] == "YES":
        return "synthesize"
    #Stop looping after 2 rewrites to prevent infinite loops
    if state["loop_count"] >= 2:
        print("⚠️ [Guardrail] Max loops reached. Proceeding to synthesis with available data.")
        return "synthesize"
    return "rewrite"

# ==========================================
# 5. BUILD AND COMPILE THE WORKFLOW
# ==========================================
workflow = StateGraph(AgentState)

workflow.add_node("router", node_router)
workflow.add_node("retrieve", node_retrieve)
workflow.add_node("grade", node_grade)
workflow.add_node("rewrite", node_rewrite)
workflow.add_node("synthesize", node_synthesize)

workflow.set_entry_point("router")
workflow.add_edge("router", "retrieve")
workflow.add_edge("retrieve", "grade")
workflow.add_conditional_edges("grade", decide_to_rewrite, {"synthesize": "synthesize", "rewrite": "rewrite"})
workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("synthesize", END)

# Compile the runnable app
agent_app = workflow.compile()