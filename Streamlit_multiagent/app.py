# app.py

from __future__ import annotations

import os
import concurrent.futures
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st
import requests

from agents import (
    SupervisorAgent,
    SupervisorConfig,
    WorkerAgent,
    WorkerAgentConfig,
)
from vectorstores import SimpleVectorStore
from embeddings import DEVICE


# ---------- Paths & filenames ----------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

EVENTS_FILE = DATA_DIR / "AllServerEvents_2025-11-14.csv"
PERFMON_FILE = DATA_DIR / "perfmon_metrics_2025-04-29.csv"
AGG_METRICS_FILE = DATA_DIR / "ServerPerformanceMetrics_2025-11-18.csv"


# ---------- Utility: load CSV/Excel & build text docs ----------


def df_to_docs(df: pd.DataFrame, source_name: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Convert each row into a single text "document":
    'col1=value1, col2=value2, ...'
    """
    texts: List[str] = []
    metas: List[Dict[str, Any]] = []

    df = df.reset_index(drop=True)

    for idx, row in df.iterrows():
        parts = []
        for col in df.columns:
            val = row[col]
            parts.append(f"{col}={val}")
        text = f"Row index={idx}; " + ", ".join(parts)
        texts.append(text)
        metas.append({"source": source_name, "row_id": int(idx)})

    return texts, metas


def load_vector_store_for_file(path: Path) -> SimpleVectorStore:
    """
    Load CSV/Excel and build a SimpleVectorStore.
    NO caching: embeddings are rebuilt every time the app starts.
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in [".xls", ".xlsx"]:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type for: {path}")

    texts, metas = df_to_docs(df, source_name=path.name)
    print(f"[BUILD] Embedding {len(texts)} rows for {path.name} ...")
    store = SimpleVectorStore.from_texts(texts, metas)
    return store


# ---------- Small Ollama status helper ----------


def get_llm_status_and_host() -> Tuple[str, str]:
    """
    Ping Ollama to show a simple 'Connected' / 'Offline' status.
    Uses /api/tags which is cheap.
    """
    host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    url = host.rstrip("/") + "/api/tags"
    try:
        r = requests.get(url, timeout=0.5)
        if r.ok:
            return "Connected", host
        return f"HTTP {r.status_code}", host
    except Exception:
        return "Offline", host


# ---------- Streamlit state init ----------


def init_state():
    if "initialized" in st.session_state:
        return

    st.session_state["messages"] = []
    st.session_state["agent_memories"] = {}
    st.session_state["agents"] = {}
    st.session_state["supervisor"] = None

    # Ensure data files exist
    if not EVENTS_FILE.exists() or not PERFMON_FILE.exists() or not AGG_METRICS_FILE.exists():
        st.error(
            "One or more data files not found in the ./data directory.\n"
            f"- {EVENTS_FILE.name}\n"
            f"- {PERFMON_FILE.name}\n"
            f"- {AGG_METRICS_FILE.name}"
        )
        st.stop()

    # Load vector stores (no cache)
    events_store = load_vector_store_for_file(EVENTS_FILE)
    perfmon_store = load_vector_store_for_file(PERFMON_FILE)
    agg_store = load_vector_store_for_file(AGG_METRICS_FILE)

    # Empty memory stores (per agent)
    events_mem = SimpleVectorStore.from_texts([], [])
    perfmon_mem = SimpleVectorStore.from_texts([], [])
    agg_mem = SimpleVectorStore.from_texts([], [])

    # ---------- Agent configs ----------

    events_cfg = WorkerAgentConfig(
        name="EventLogsAgent",
        description=f"Windows / application event logs from {EVENTS_FILE.name}.",
        model_name="llama3:latest",
        temperature=0.1,
        schema_hint="""
Columns include: Message, ID, ProviderName, LogName, MachineName,
TimeCreated, LevelDisplayName, CreatedDate
""",
        analysis_focus="""
Find event log anomalies around servers/timeframes:
- identify errors, warnings, repeated event IDs
- detect service failures, timeouts, authentication problems
- correlate with reported slow/failed operations
"""
    )

    perfmon_cfg = WorkerAgentConfig(
        name="PerfMonAgent",
        description=f"Windows PerfMon metrics from {PERFMON_FILE.name}.",
        model_name="llama3.1:8b",
        temperature=0.1,
        schema_hint="""
PerfMon counters: Timestamp, MachineName, CounterObject, CounterName,
InstanceName, CookedValue
""",
        analysis_focus="""
Detect CPU/disk/memory/network stress:
- CPU saturation
- disk latency spikes
- memory pressure / paging
- network bottlenecks
"""
    )

    agg_cfg = WorkerAgentConfig(
        name="ServerPerfSummaryAgent",
        description=f"SQL Server wait statistics from {AGG_METRICS_FILE.name}.",
        model_name="gpt-oss:20b",
        temperature=0.2,
        schema_hint="""
Columns: Date, Servername, WAITTYPE, WAIT_S, RESOURCE_S,
SIGNAL_S, WAIT_Count, Percentage, AvgWait_s
""",
        analysis_focus="""
Explain SQL wait-type patterns:
- dominant waits
- CPU-bound vs IO-bound waits
- high SIGNAL_S → CPU pressure
- high RESOURCE_S → IO/lock contention
"""
    )

    # Instantiate worker agents
    st.session_state["agents"] = {
        "events": WorkerAgent(events_cfg, events_store, events_mem),
        "perfmon": WorkerAgent(perfmon_cfg, perfmon_store, perfmon_mem),
        "agg": WorkerAgent(agg_cfg, agg_store, agg_mem),
    }

    # Supervisor agent:
    # IMPORTANT: use a *general chat model*, not a SQL-specialist model.
    sup_cfg = SupervisorConfig(
        model_name="llama3:latest",  # was "duckdb-nsql:7b"
        temperature=0.2,
    )
    st.session_state["supervisor"] = SupervisorAgent(config=sup_cfg)

    st.session_state["initialized"] = True


# ---------- UI rendering ----------


def render_chat_history():
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("agents"):
                with st.expander("Advanced: what each agent said"):
                    for agent_res in msg["agents"]:
                        st.markdown(f"### {agent_res.get('agent_name','Unknown')}")
                        st.markdown(f"**Summary:** {agent_res.get('short_summary')}")
                        st.markdown(agent_res.get("detailed_reasoning", ""))

                        # SAFE evidence snippet rendering
                        for snip in agent_res.get("evidence_snippets", []):
                            if not isinstance(snip, dict):
                                continue

                            src = snip.get("source", "")
                            row_id = snip.get("row_id", "?")
                            text = snip.get("text", "")
                            st.markdown(f"- `{src}` (row {row_id}): {text}")

                        # Optional debug: show retrieved rows used as context
                        rows = agent_res.get("retrieved_rows", [])
                        if rows:
                            with st.expander("Debug: retrieved rows (agent context)"):
                                for r in rows:
                                    if not isinstance(r, dict):
                                        continue
                                    src = r.get("source", "")
                                    row_id = r.get("row_id", "?")
                                    text = r.get("text", "")
                                    st.markdown(f"- **{src}** (row {row_id}): {text}")


# ---------- Main ----------


def main():
    st.set_page_config(page_title="Multi-Agent Log Analyzer", page_icon="🖥️")

    # Initialize state
    init_state()

    # --- Layout CSS: centered content, FIXED chat input at bottom ---
    st.markdown(
        """
        <style>
            .block-container {
                max-width: 1100px;
                margin: 0 auto;
                padding-top: 30px !important;
                padding-bottom: 200px !important;  /* space so last message not hidden */
            }

            /* Fixed chat input at bottom, centered */
            div[data-testid="stChatInput"] {
                position: fixed !important;
                bottom: 16px !important;
                left: 50% !important;
                transform: translateX(-50%) !important;
                width: min(900px, 90%) !important;
                z-index: 1000 !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --- Title ---
    st.title("🖥️ Multi-Agent Log Analyzer")

    agents: Dict[str, WorkerAgent] = st.session_state["agents"]
    supervisor: SupervisorAgent = st.session_state["supervisor"]

    # --- Two-column layout: left = chat, right = Config card ---
    left_col, right_col = st.columns([4, 1.5])

    # Right column: Config card
    with right_col:
        st.markdown(
            """
            <style>
                .config-code-box pre {
                    white-space: pre;
                    overflow-x: auto;
                    font-size: 14px !important;
                    padding: 12px !important;
                    width: 100% !important;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("Config")

        status, host = get_llm_status_and_host()
        st.markdown(
            f"**LLM status:** {'✅' if status == 'Connected' else '⚠️'} {status}"
        )

        bge_model_name = os.getenv("BGE_MODEL_NAME", "BAAI/bge-base-en")

        cfg_text = f"""
OLLAMA_HOST        = {host}
EVENT_LOGS_MODEL   = {agents['events'].config.model_name}
PERFMON_MODEL      = {agents['perfmon'].config.model_name}
SQL_WAITS_MODEL    = {agents['agg'].config.model_name}
EMBEDDING_MODEL    = {bge_model_name}
EMBEDDING_DEVICE   = {DEVICE}
"""

        st.markdown("<div class='config-code-box'>", unsafe_allow_html=True)
        st.code(cfg_text.strip(), language="text")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("Start Ollama to enable real calls:")
        st.code(
            "ollama serve\n"
            "ollama run llama3:latest\n"
            "ollama run llama3.1:8b\n"
            "ollama run gpt-oss:20b",
            language="bash",
        )

    # Left column: chat history + assistant answers
    with left_col:
        render_chat_history()

    # Chat input (fixed at bottom via CSS)
    user_input = st.chat_input("Ask a question about server behavior...")
    if not user_input:
        return

    # Add user message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    chat_window = st.session_state["messages"]

    # Run agents in parallel
    with st.spinner("Agents are analyzing logs and metrics..."):

        def run_agent(agent: WorkerAgent):
            return agent.answer(user_input, chat_history_window=chat_window)

        worker_outputs: List[Dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as exe:
            futures = {name: exe.submit(run_agent, ag) for name, ag in agents.items()}

            for name, fut in futures.items():
                try:
                    worker_outputs.append(fut.result())
                except Exception as e:
                    worker_outputs.append(
                        {
                            "agent_name": name,
                            "short_summary": f"Agent {name} failed: {e}",
                            "detailed_reasoning": (
                                f"Agent {name} encountered an error: {e}"
                            ),
                            "evidence_snippets": [],
                        }
                    )

        # Supervisor synthesizes
        final_answer = supervisor.answer(
            question=user_input,
            worker_outputs=worker_outputs,
            chat_history_window=chat_window,
        )

    # Save supervisor answer
    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": final_answer,
            "agents": worker_outputs,
        }
    )

    # Render latest assistant message in left column
    with left_col:
        with st.chat_message("assistant"):
            st.markdown(final_answer)
            with st.expander("Advanced: what each agent said"):
                for agent_res in worker_outputs:
                    st.markdown(f"### {agent_res.get('agent_name', 'Unknown')}")
                    st.markdown(agent_res.get("detailed_reasoning", ""))

                    # SAFE evidence snippet rendering here too
                    for snip in agent_res.get("evidence_snippets", []):
                        if not isinstance(snip, dict):
                            continue
                        src = snip.get("source", "")
                        row_id = snip.get("row_id", "?")
                        text = snip.get("text", "")
                        st.markdown(f"- `{src}` (row {row_id}): {text}")

                    # Optional debug: show retrieved rows used as context
                    rows = agent_res.get("retrieved_rows", [])
                    if rows:
                        with st.expander("Debug: retrieved rows (agent context)"):
                            for r in rows:
                                if not isinstance(r, dict):
                                    continue
                                src = r.get("source", "")
                                row_id = r.get("row_id", "?")
                                text = r.get("text", "")
                                st.markdown(f"- **{src}** (row {row_id}): {text}")


if __name__ == "__main__":
    main()
