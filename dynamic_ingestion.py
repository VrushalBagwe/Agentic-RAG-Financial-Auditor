import os
import io
import duckdb
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq

# Initialize Groq client
groq_client = Groq()

def transcribe_audio_with_groq(file_bytes, file_name):
    """
    Takes audio file bytes, sends them to Groq's Whisper-Large-V3, 
    and returns the verbatim text transcript.
    """
    print(f"🎙️ [Whisper API] Transcribing audio file: {file_name}...")
    
    transcription = groq_client.audio.transcriptions.create(
        file=(file_name, file_bytes),
        model="whisper-large-v3",
        response_format="json",
        temperature=0.0
    )
    
    print("✅ [Whisper API] Audio transcription complete!")
    return transcription.text

def ingest_custom_table(file_bytes, file_name, db_path="financial_data.duckdb"):
    """
    Reads an uploaded CSV or Excel file and overwrites the DuckDB table for SQL querying.
    """
    print(f"📊 [DuckDB] Loading custom table: {file_name}...")
    
    if file_name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes))
    else:
        df = pd.read_excel(io.BytesIO(file_bytes))
        
    # Standardize date column if present as string for SQL LIKE filters
    if "Date" in df.columns:
        df["Date"] = df["Date"].astype(str)
        
    conn = duckdb.connect(db_path)
    # Overwrite quarterly table with the custom uploaded data
    conn.execute("DROP TABLE IF EXISTS quarterly_income_statement")
    conn.execute("CREATE TABLE quarterly_income_statement AS SELECT * FROM df")
    
    row_count = conn.execute("SELECT COUNT(*) FROM quarterly_income_statement").fetchone()[0]
    conn.close()
    
    return f"Successfully loaded {row_count} rows from `{file_name}` into DuckDB SQL engine."

def ingest_custom_text_or_transcript(text_content, source_name="Uploaded File", db_path="./chroma_db"):
    """
    Chunks a raw string (or transcribed audio) and embeds it into ChromaDB.
    """
    print(f"🧠 [ChromaDB] Embedding custom text from: {source_name}...")
    
    local_ef = embedding_functions.DefaultEmbeddingFunction()
    client = chromadb.PersistentClient(path=db_path)
    
    # Reset collection cleanly for custom demo analysis
    try:
        client.delete_collection("earnings_calls")
    except Exception:
        pass
        
    collection = client.create_collection(name="earnings_calls", embedding_function=local_ef)
    
    # Simple semantic/paragraph chunking (splitting by double line breaks or periods)
    paragraphs = [p.strip() for p in text_content.split("\n\n") if len(p.strip()) > 40]
    
    # Fallback if text doesn't have double line breaks
    if not paragraphs:
        paragraphs = [p.strip() + "." for p in text_content.split(".") if len(p.strip()) > 40]
        
    ids = [f"custom_chunk_{i}" for i in range(len(paragraphs))]
    metadatas = [{"source": source_name, "chunk_id": i, "section": "Custom Ingestion"} for i in range(len(paragraphs))]
    
    collection.add(
        documents=paragraphs,
        metadatas=metadatas,
        ids=ids
    )
    
    return f"Successfully embedded {len(paragraphs)} searchable chunks into ChromaDB Vector store."