import os
import duckdb
import chromadb
from chromadb.utils import embedding_functions
from pydantic import BaseModel, Field
from typing import Literal
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Initialize LLM
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.0,
    api_key=os.environ.get("GROQ_API_KEY")
)

# ==========================================
# 1. THE ROUTER NODE (Generalized & Transcript Aware)
# ==========================================
class RouteDecision(BaseModel):
    destination: Literal["SQL", "VECTOR", "BOTH"] = Field(
        description="Choose SQL only for structured historical database queries. Choose VECTOR for transcripts, audio webcasts, executive sentiment, commentary, Q&A, or custom uploaded documents. Choose BOTH if comparing structured table numbers against transcript commentary."
    )
    reasoning: str = Field(description="Brief explanation for why this routing decision was made.")

def router_node(user_query: str):
    """Decides which database to query based on user intent."""
    print(f"\n🧭 [Router Node] Analyzing query: '{user_query}'")
    
    system_prompt = """You are an expert financial routing agent. 
    Analyze the user's question and decide where to route it:
    - 'VECTOR': Use this if the question is about an uploaded transcript, audio webcast, executive commentary, Q&A session, guidance, or custom document notes (even if numbers/metrics are requested from the transcript!).
    - 'SQL': Use this ONLY if the question specifically asks for structured database table queries, historical annual/quarterly financial statements, or database aggregation across tables.
    - 'BOTH': Use this if comparing structured database tables against transcript/audio commentary."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])
    
    structured_llm = llm.with_structured_output(RouteDecision)
    router = prompt | structured_llm
    
    decision = router.invoke({"question": user_query})
    print(f"👉 [Decision]: Route to {decision.destination} (Reason: {decision.reasoning})")
    return decision.destination

# ==========================================
# 2. THE RETRIEVAL NODES (Dynamic Schema Detection)
# ==========================================
def get_duckdb_schema_info(db_path="financial_data.duckdb"):
    """Dynamically fetches table names and column headers from DuckDB."""
    try:
        conn = duckdb.connect(db_path)
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = [t[0] for t in tables]
        
        schema_summary = ""
        for t_name in table_names:
            cols = conn.execute(f"PRAGMA table_info('{t_name}')").fetchall()
            col_names = [c[1] for c in cols]
            schema_summary += f"\nTable `{t_name}` has columns: {col_names}"
            
        conn.close()
        return schema_summary
    except Exception as e:
        return f"Error reading schema: {str(e)}"

def sql_retriever_node(user_query: str, db_path="financial_data.duckdb"):
    """Translates natural language to SQL using DYNAMIC schema detection."""
    print("📊 [SQL Retriever] Inspecting DuckDB schema and generating query...")
    
    # 1. Inspect live DuckDB schema dynamically
    schema_info = get_duckdb_schema_info(db_path)
    
    # 2. Instruct LLM with the live dynamic schema
    sql_prompt = ChatPromptTemplate.from_template("""
    You are a SQL expert. Write a clean, valid DuckDB SQL query to answer the user's question based strictly on the available database schema.
    
    CURRENT LIVE DATABASE SCHEMA:
    {schema}
    
    CRITICAL RULES:
    1. ONLY use table and column names that exist in the schema provided above.
    2. Wrap column names that contain spaces in double quotes (e.g. `"Total Revenue"`).
    3. Do NOT assume or hardcode any company names unless present in the user query or database filters.
    4. Output ONLY the raw SQL string—no markdown formatting, no backticks, no explanations.
    
    User Question: {question}
    SQL Query:
    """)
    
    sql_chain = sql_prompt | llm
    sql_query = sql_chain.invoke({"schema": schema_info, "question": user_query}).content.strip()
    sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
    
    print(f"   Generated SQL: {sql_query}")
    
    try:
        conn = duckdb.connect(db_path)
        results = conn.execute(sql_query).fetchall()
        columns = [desc[0] for desc in conn.description]
        conn.close()
        
        return f"Columns: {columns}\nData Rows: {results}"
    except Exception as e:
        print(f"   ❌ SQL Error: {str(e)}")
        return f"SQL Execution Error: {str(e)}. Attempted Query: {sql_query}"

def vector_retriever_node(search_query: str, db_path="./chroma_db"):
    """Queries ChromaDB for transcript/audio text chunks."""
    print(f"🧠 [Vector Retriever] Searching ChromaDB transcripts for: '{search_query}'...")
    
    local_ef = embedding_functions.DefaultEmbeddingFunction()
    client = chromadb.PersistentClient(path=db_path)
    
    try:
        collection = client.get_collection(name="earnings_calls", embedding_function=local_ef)
    except Exception:
        return "No transcript collection found in ChromaDB. Please upload a document or audio file first."
    
    results = collection.query(query_texts=[search_query], n_results=3)
    
    if not results or not results["documents"] or len(results["documents"][0]) == 0:
        return "No relevant transcript documents found matching the query."
        
    retrieved_text = ""
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i] if results["metadatas"] else {}
        source = meta.get("speaker", meta.get("source", "Uploaded Document"))
        retrieved_text += f"\n--- [Chunk {i+1} | Source: {source}] ---\n{doc}\n"
        
    return retrieved_text

# ==========================================
# 3. THE REFLEXION GRADER NODE
# ==========================================
class GradeDecision(BaseModel):
    is_sufficient: Literal["YES", "NO"] = Field(
        description="YES if the context contains clear factual evidence to answer the user question. NO if the context is missing key facts."
    )
    explanation: str = Field(description="Why the data passed or failed evaluation.")

def context_grader_node(user_query: str, retrieved_context: str):
    """Evaluates if the retrieved data actually answers the user's prompt."""
    print("⚖️ [Reflexion Grader] Grading retrieved context...")
    
    system_prompt = """You are a strict financial audit grader.
    Look at the user's question and the retrieved database documents/transcripts.
    Does the context contain clear, factual evidence to answer the question?
    Do NOT assume or guess. If the answer is not in the text, grade 'NO'."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "User Question: {question}\n\nRetrieved Context:\n{context}")
    ])
    
    structured_llm = llm.with_structured_output(GradeDecision)
    grader = prompt | structured_llm
    
    grade = grader.invoke({"question": user_query, "context": retrieved_context})
    print(f"👉 [Grade]: {grade.is_sufficient} (Reason: {grade.explanation})")
    return grade.is_sufficient

# ==========================================
# 4. THE QUERY REWRITER NODE
# ==========================================
def query_rewriter_node(original_query: str):
    """Rewrites a failed query for semantic search."""
    print("✍️ [Query Rewriter] Rewriting search terms...")
    
    prompt = ChatPromptTemplate.from_template("""
    You are an expert search optimizer. The original query failed to retrieve relevant documents.
    Rewrite the query to use alternative business/financial synonyms or key speaker terms.
    Only output the rewritten query string.
    
    Original Query: {question}
    Optimized Search Query:
    """)
    
    chain = prompt | llm
    return chain.invoke({"question": original_query}).content.strip()