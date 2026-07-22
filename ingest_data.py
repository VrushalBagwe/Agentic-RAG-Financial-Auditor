import os
import duckdb
import yfinance as yf
import chromadb
from chromadb.utils import embedding_functions

def setup_structured_database(ticker_symbol="AAPL", db_path="financial_data.duckdb"):
    """
    Fetches quarterly financial tables via yfinance and stores them in DuckDB.
    """
    print(f"📊 [DuckDB] Fetching financial tables for {ticker_symbol}...")
    ticker = yf.Ticker(ticker_symbol)
    
    # Extract Quarterly Income Statement and Balance Sheet
    income_stmt = ticker.quarterly_income_stmt.T.reset_index()
    income_stmt.rename(columns={"index": "Date"}, inplace=True)
    
    # Convert Date column to string for cleaner SQL querying
    income_stmt["Date"] = income_stmt["Date"].astype(str)
    
    # Connect to in-process DuckDB
    conn = duckdb.connect(db_path)
    
    # Create clean SQL table
    conn.execute("DROP TABLE IF EXISTS quarterly_income_statement")
    conn.execute("CREATE TABLE quarterly_income_statement AS SELECT * FROM income_stmt")
    
    # Verify insertion
    row_count = conn.execute("SELECT COUNT(*) FROM quarterly_income_statement").fetchone()[0]
    print(f"[DuckDB] Successfully inserted {row_count} quarters of financial data into '{db_path}'.\n")
    conn.close()

def setup_vector_database(db_path="./chroma_db"):
    """
    Simulates semantic chunking of an earnings call transcript and embeds it into ChromaDB
    using local open-source embeddings (zero API cost).
    """
    print("🧠 [ChromaDB] Initializing local vector database and embedding model...")
    
    # Use Chroma's built-in local embedding function (uses lightweight ONNX MiniLM on your CPU)
    local_ef = embedding_functions.DefaultEmbeddingFunction()
    
    client = chromadb.PersistentClient(path=db_path)
    
    # Recreate collection cleanly for demo reproducibility
    try:
        client.delete_collection("earnings_calls")
    except Exception:
        pass
        
    collection = client.create_collection(
        name="earnings_calls",
        embedding_function=local_ef
    )
    
    # High-fidelity sample transcript chunks with metadata (CEO Script vs CFO Q&A)
    transcript_chunks = [
        {
            "id": "aapl_q3_2025_ceo_1",
            "text": "CEO Tim Cook (Prepared Remarks): We are thrilled to report outstanding Q3 results. Our iPhone revenue remained resilient despite macro headwinds, and we expect gross margins for Q4 to expand to a range of 46.0% to 46.5%, driven by supply chain efficiencies and a favorable product mix.",
            "meta": {"speaker": "Tim Cook", "role": "CEO", "quarter": "Q3-2025", "section": "Prepared Remarks", "sentiment": "Bullish"}
        },
        {
            "id": "aapl_q3_2025_cfo_1",
            "text": "CFO Luca Maestri (Prepared Remarks): Services revenue hit an all-time record. Operating expenses for next quarter are projected to be between $14.2 billion and $14.4 billion. We continue to return capital to shareholders aggressively through our share repurchase program.",
            "meta": {"speaker": "Luca Maestri", "role": "CFO", "quarter": "Q3-2025", "section": "Prepared Remarks", "sentiment": "Bullish"}
        },
        {
            "id": "aapl_q3_2025_qa_1",
            "text": "Analyst Question (Goldman Sachs): Can you comment on potential margin compression from rising memory chip costs and regulatory scrutiny in Europe? Are you seeing any demand softness?",
            "meta": {"speaker": "Analyst", "role": "Analyst", "quarter": "Q3-2025", "section": "Q&A", "sentiment": "Neutral"}
        },
        {
            "id": "aapl_q3_2025_cfo_qa_response",
            "text": "CFO Luca Maestri (Q&A Response): To be candid, while our headline margin guidance is strong at 46% to 46.5%, we are closely monitoring commodity pricing. If NAND and DRAM chip prices spike faster than forecasted in late Q4, it could create a 50 to 75 basis point headwind against our gross margins. We are hedging where possible.",
            "meta": {"speaker": "Luca Maestri", "role": "CFO", "quarter": "Q3-2025", "section": "Q&A", "sentiment": "Cautious/Evasive"}
        }
    ]
    
    # Add chunks to Vector Store
    collection.add(
        documents=[chunk["text"] for chunk in transcript_chunks],
        metadatas=[chunk["meta"] for chunk in transcript_chunks],
        ids=[chunk["id"] for chunk in transcript_chunks]
    )
    
    print(f"[ChromaDB] Successfully embedded {len(transcript_chunks)} transcript chunks into '{db_path}'.")

if __name__ == "__main__":
    setup_structured_database(ticker_symbol="AAPL")
    setup_vector_database()
    print("\nBoth databases are ready for querying.")