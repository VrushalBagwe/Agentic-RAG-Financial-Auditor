# 📈 Autonomous Financial & Regulatory Audit Engine (Agentic RAG)

An autonomous, multi-agent financial auditing engine built with **Python**, **LangGraph**, and **Llama-3-70B**. The system dynamically routes complex investor queries across structured relational tables and unstructured transcript embeddings including raw earnings call **audio**, transcribed on the fly, utilizing cyclic self-correction loops to eliminate LLM hallucinations.

## 🏛️ Architecture & Tech Stack
* **Orchestration:** LangGraph (Stateful, cyclic multi-agent workflow)
* **LLM Engine:** Meta Llama-3.3-70B via Groq API (Zero-latency inference)
* **Speech-to-Text:** Groq Whisper-Large-V3 API (Native transcription of earnings call webcast audio — zero local GPU/PyTorch overhead)
* **Relational Database (SQL):** DuckDB (In-process execution of historical quantitative statements)
* **Vector Database (Semantic Search):** ChromaDB + Local ONNX CPU Embeddings (Unstructured earnings call transcripts)
* **Guardrails & Schema Enforcement:** Pydantic (Deterministic JSON outputs for downstream enterprise APIs)
* **Observability & Tracing:** Arize Phoenix (Open-source OpenInference telemetry and visual graph debugging)
* **Frontend UI:** Streamlit

## ⚡ Key Agentic Behaviors
1. **Adaptive Tool Routing:** A Supervisor Agent evaluates user intent and dynamically directs queries to `DuckDB` (for hard math/dates), `ChromaDB` (for executive sentiment/guidance), or both simultaneously.
2. **The Reflexion Loop (Self-Correction):** Includes a specialized Grader Node that evaluates retrieved context against the prompt. If retrieved data lacks factual evidence, it triggers a Query Rewriter Node to optimize search keywords and re-query the database up to a maximum loop threshold.
3. **Zero-Hallucination Guardrails:** Enforces strict adherence to retrieved context; the agent explicitly flags missing periods rather than guessing or approximating financial figures.
4. **Multimodal Audio Ingestion:** Raw earnings call webcast audio (`.mp3` / `.wav`) is offloaded to Groq's hosted Whisper-Large-V3 endpoint for transcription, converting spoken executive commentary into text that flows directly into the ChromaDB retrieval pipeline in seconds — enabling audits on webcasts that have no official written transcript yet, with no local Whisper/PyTorch install required.

## 🚀 Quickstart
```bash
# 1. Clone repo and install dependencies
pip install -r requirements.txt

# 2. Set Groq API Key (powers both the Llama-3.3-70B agent and Whisper-Large-V3 transcription)
export GROQ_API_KEY="your_api_key_here"

# 3. Ingest synthetic data into DuckDB and ChromaDB
python ingest_data.py

# 4. Launch the interactive audit dashboard and background telemetry
streamlit run app.py

# 5. (Optional) Audit a live webcast: upload an .mp3/.wav file in the
#    "Upload Earnings Webcast Audio" tab — Whisper-Large-V3 transcribes it automatically
#    and the transcript is indexed into ChromaDB for querying.
```