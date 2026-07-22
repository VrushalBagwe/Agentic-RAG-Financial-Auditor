import streamlit as st
import json
import phoenix as px
from openinference.instrumentation.langchain import LangChainInstrumentor
from graph import agent_app
from dynamic_ingestion import (
    transcribe_audio_with_groq, 
    ingest_custom_table, 
    ingest_custom_text_or_transcript
)

# 1. Initialize Open-Source Tracing
@st.cache_resource
def setup_tracing():
    try:
        session = px.launch_app()
        LangChainInstrumentor().instrument()
        return session.url
    except Exception as e:
        return None

trace_url = setup_tracing()

# 2. Page Config
st.set_page_config(page_title="Autonomous Earnings & Guidance Auditor", page_icon="📈", layout="wide")
st.title("📈 Autonomous Financial & Earnings Auditor")
st.markdown("An **Agentic RAG Engine** with self-correcting retrieval loops (Reflexion), dynamic tool routing, and deterministic JSON guardrails.")

# 3. Initialize Session State for Persisting Transcripts and Files across button clicks
if "audio_transcript" not in st.session_state:
    st.session_state.audio_transcript = None
if "custom_data_status" not in st.session_state:
    st.session_state.custom_data_status = None

# Sidebar
with st.sidebar:
    st.header("⚙️ Architecture Specs")
    st.markdown("- **Orchestration:** LangGraph")
    st.markdown("- **LLM:** Meta Llama-3.3-70B (Groq)")
    st.markdown("- **Audio Transcription:** Groq Whisper-Large-V3")
    st.markdown("- **Structured DB:** DuckDB (Dynamic SQL)")
    st.markdown("- **Vector DB:** ChromaDB (Local CPU)")
    st.markdown("- **Guardrails:** Pydantic Schema")
    if trace_url:
        st.success(f"🟢 **Arize Phoenix Live!**\n\nOpen [localhost:6006]({trace_url}) to view agent execution trees.")

# ==========================================
# DUAL WORKFLOW SELECTION (Default vs Custom)
# ==========================================
mode_tab1, mode_tab2 = st.tabs(["🍎 Mode 1: Default AAPL Benchmark Data", "📂 Mode 2: Custom Data & Audio Transcripts"])

query_to_run = None
execution_button_clicked = False
data_mode_label = "AAPL Sample Data"

# --- MODE 1: DEFAULT AAPL DATA ---
with mode_tab1:
    st.subheader("Audit Pre-Loaded Apple (AAPL) Financials & Transcripts")
    sample_queries = [
        "What was Apple's total revenue in 2025?",
        "Did the CFO mention any risks about memory chip prices during the Q&A?",
        "Compare actual revenue against what Tim Cook said about margins."
    ]
    
    selected_sample = st.selectbox("Select a benchmark sample query:", ["-- Select Sample --"] + sample_queries)
    custom_aapl_query = st.text_input("Or type a custom query for Apple dataset:")
    
    aapl_query = custom_aapl_query if custom_aapl_query else (selected_sample if selected_sample != "-- Select Sample --" else None)
    
    if st.button("🚀 Run AAPL Audit", type="primary", key="btn_aapl"):
        query_to_run = aapl_query
        execution_button_clicked = True
        data_mode_label = "AAPL Sample Data"

# --- MODE 2: CUSTOM DATA & AUDIO ---
with mode_tab2:
    st.subheader("Upload Custom Financial Tables, Notes, or Webcast Audio")
    
    tab_csv, tab_txt, tab_audio = st.tabs(["📊 Upload Table (SQL)", "📝 Upload Text (Vector)", "🎙️ Upload Audio (Whisper AI)"])
    
    with tab_csv:
        table_file = st.file_uploader("Upload CSV or Excel Statement", type=["csv", "xlsx"])
        if table_file and st.button("Index Table into DuckDB", key="btn_idx_table"):
            with st.spinner("Writing custom table to DuckDB..."):
                msg = ingest_custom_table(table_file.getvalue(), table_file.name)
                st.session_state.custom_data_status = msg
                
    with tab_txt:
        text_file = st.file_uploader("Upload Text Document", type=["txt", "md"])
        if text_file and st.button("Embed Text into ChromaDB", key="btn_idx_text"):
            with st.spinner("Embedding document into ChromaDB..."):
                raw_text = text_file.getvalue().decode("utf-8")
                msg = ingest_custom_text_or_transcript(raw_text, source_name=text_file.name)
                st.session_state.custom_data_status = msg
                
    with tab_audio:
        audio_file = st.file_uploader("Upload Earnings Webcast Audio", type=["mp3", "wav", "m4a"])
        if audio_file and st.button("Transcribe Audio & Embed", key="btn_idx_audio"):
            with st.status("🎧 Transcribing Audio with Whisper AI...", expanded=True) as status:
                st.write("🎙️ Processing speech with Groq Whisper-Large-V3...")
                transcript_text = transcribe_audio_with_groq(audio_file.getvalue(), audio_file.name)
                
                # Store in session state so it NEVER disappears on reruns!
                st.session_state.audio_transcript = transcript_text
                
                st.write("🧠 Chunking and vectorizing transcript into ChromaDB...")
                msg = ingest_custom_text_or_transcript(transcript_text, source_name=audio_file.name)
                st.session_state.custom_data_status = msg
                
                status.update(label="✅ Audio Transcribed & Vectorized!", state="complete", expanded=False)

    # Show active index status if available
    if st.session_state.custom_data_status:
        st.success(st.session_state.custom_data_status)
        
    # PERSISTENT AUDIO TRANSCRIPTION DISPLAY
    if st.session_state.audio_transcript:
        with st.expander("📜 View Active Audio Transcription (Persisted)", expanded=True):
            st.text_area("Verbatim Whisper Output:", st.session_state.audio_transcript, height=200)

    st.markdown("---")
    custom_query_input = st.text_input("Enter your query for the uploaded custom data/transcript:")
    
    if st.button("🚀 Run Custom Data Audit", type="primary", key="btn_custom"):
        query_to_run = custom_query_input
        execution_button_clicked = True
        data_mode_label = "Custom Uploaded Data"

# ==========================================
# AGENTIC EXECUTION & RESULTS DISPLAY
# ==========================================
if execution_button_clicked:
    if not query_to_run:
        st.warning("⚠️ Please select or enter a valid query before running the audit.")
    else:
        st.markdown(f"### 🎯 Audit Results ({data_mode_label})")
        with st.status("🤖 Running Agentic RAG Workflow...", expanded=True) as status:
            st.write(f"🧭 Routing query: *'{query_to_run}'*")
            
            initial_state = {
                "user_query": query_to_run, 
                "current_search_query": "", 
                "routing_decision": "", 
                "retrieved_context": "", 
                "is_sufficient": "", 
                "loop_count": 0, 
                "final_output": {}
            }
            
            result = agent_app.invoke(initial_state)
            
            st.write(f"👉 Target Engine Selected: **{result['routing_decision']}**")
            if result['loop_count'] > 0:
                st.write(f"🔄 Self-Correction Triggered! Agent rewrote search **{result['loop_count']} time(s)**.")
            st.write("✨ Validating response against Pydantic JSON schema...")
            status.update(label="✅ Audit Complete!", state="complete", expanded=False)

        # Display Final Structured Report
        report = result["final_output"]
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("📝 Executive Summary")
            st.info(report.get("executive_summary", "No summary generated."))
            
            st.subheader("🔍 Cited Metrics & Verbatim Evidence")
            for quote in report.get("cited_metrics_or_quotes", []):
                st.markdown(f"> 📌 *{quote}*")
                
        with col2:
            st.subheader("🛡️ Audit Metadata")
            risk = report.get("risk_assessment", "UNCLEAR")
            risk_color = "red" if risk in ["HIGH", "CRITICAL"] else ("orange" if risk == "MEDIUM" else "green")
            st.markdown(f"**Risk Severity:** :{risk_color}[**{risk}**]")
            st.markdown(f"**Data Sources Routed:** `{report.get('data_source_used')}`")
            st.markdown(f"**Reflexion Loops:** `{result['loop_count']}`")
            
        with st.expander("📦 View Raw Deterministic JSON Output"):
            st.json(report)