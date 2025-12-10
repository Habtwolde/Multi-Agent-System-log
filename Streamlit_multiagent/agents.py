# agents.py

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from vectorstores import SimpleVectorStore


# ---------- LLM client (Ollama via OpenAI-compatible API) ----------

OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434/v1")


def call_ollama_chat(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 1024,
    timeout: int = 180,
) -> str:
    """
    Call Ollama's OpenAI-compatible /v1/chat/completions endpoint.
    """
    url = f"{OLLAMA_API_BASE}/chat/completions"

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


# ---------- Worker agent ----------


@dataclass
class WorkerAgentConfig:
    name: str
    description: str
    model_name: str
    temperature: float = 0.2
    top_k_data: int = 5
    top_k_memory: int = 5

    # Help each agent understand its data domain & job
    schema_hint: str = ""
    analysis_focus: str = ""


@dataclass
class WorkerAgent:
    config: WorkerAgentConfig
    data_store: SimpleVectorStore
    memory_store: SimpleVectorStore

    # ---- helper: extract possible server tokens from a question ----
    @staticmethod
    def _extract_server_tokens(question: str) -> List[str]:
        """
        Crude heuristic to pull out server names like HC1DBSQ36PV, HC1DBSQ27PV, etc.
        Adjust regex if your naming pattern is different.
        """
        # Example: words that start with HC and contain letters/digits
        return re.findall(r"\bHC[0-9A-Z]+(?:PV|VP)?\b", question.upper())

    # ---- prompts ----

    def _build_system_prompt(self) -> str:
        return f"""
You are a specialized server analysis assistant named {self.config.name}.

Data domain:
{self.config.description}

Schema / fields (approximate):
{self.config.schema_hint}

Your primary analysis goals:
{self.config.analysis_focus}

CRITICAL RULES ABOUT CONTEXT AND HALLUCINATIONS
------------------------------------------------
- You ONLY have access to:
  1) The retrieved log / performance rows provided in the CONTEXT.
  2) The retrieved memory snippets (prior questions / answers).
- You MUST NOT invent servers, metrics, timestamps, or log events that are
  not clearly supported by the provided CONTEXT.
- You CANNOT assume that the retrieved rows represent the whole dataset.
  They are only a subset.
- Therefore, you MUST NOT say things like:
    "there are no events in the logs"
    "there is no evidence anywhere in the data"
  Instead, you may say:
    "in the retrieved context, I do / do not see evidence of X".

If the retrieved CONTEXT does not contain enough information to answer the
user's question (e.g. no matching server or timestamp), you MUST:
1) Use the token: INSUFFICIENT_EVIDENCE_IN_CONTEXT
2) Explain briefly what is missing (e.g. "no rows for server HC1DBSQ36PV").

Output format (VERY IMPORTANT):
You MUST respond in compact JSON with the following keys ONLY:
- "agent_name": string
- "short_summary": string    (1–3 sentences, non-technical if possible)
- "detailed_reasoning": string
- "evidence_snippets": array of objects with keys:
    - "source": string
    - "row_id": integer or null
    - "text": string (short quotation or paraphrase from the CONTEXT)

Every important claim in your reasoning should be backed by at least one
evidence_snippet.

Do NOT include markdown code fences, backticks, or any keys other than those listed above.
"""

    def _build_user_prompt(
        self,
        question: str,
        data_results: List[Dict[str, Any]],
        memory_results: List[Dict[str, Any]],
        chat_history_window: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Build the user prompt that includes context + prior memory.
        """
        history_txt = ""
        if chat_history_window:
            for msg in chat_history_window[-6:]:
                history_txt += f"{msg['role'].upper()}: {msg['content']}\n"

        def fmt_results(title: str, results: List[Dict[str, Any]]) -> str:
            lines = [f"=== {title} ==="]
            for r in results:
                meta = r.get("metadata", {})
                src = meta.get("source", "unknown_source")
                row_id = meta.get("row_id", "n/a")
                text = r.get("text", "").strip().replace("\n", " ")
                lines.append(f"[{src} | row_id={row_id}] {text}")
            if not results:
                lines.append("(no results)")
            return "\n".join(lines)

        data_block = fmt_results("Retrieved log/performance rows", data_results)
        mem_block = fmt_results("Retrieved memory snippets", memory_results)

        prompt = f"""
Conversation history (recent turns):
{history_txt}

User question:
{question}

You are {self.config.name}. CONTEXT below.

{data_block}

{mem_block}

IMPORTANT:
- You may ONLY use information from the CONTEXT above.
- If the CONTEXT is insufficient, you MUST output the token
  INSUFFICIENT_EVIDENCE_IN_CONTEXT in your reasoning and clearly explain
  what is missing.
- Do NOT claim that something does or does not exist in the entire dataset,
  only what is or is not visible in this retrieved subset.

Return your answer STRICTLY as valid JSON with keys:
"agent_name", "short_summary", "detailed_reasoning", "evidence_snippets".
Do NOT include any markdown code fences, backticks, or extra commentary.
"""
        return prompt

    def _parse_agent_json(self, raw: str) -> Dict[str, Any]:
        """
        Try to parse JSON. If it fails, wrap the raw text into a fallback structure.
        """
        cleaned = raw.strip()
        # Remove code fences if present
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            # possible "json\n{...}"
            if cleaned.lstrip().lower().startswith("json"):
                cleaned = cleaned.split("\n", 1)[1]

        try:
            data = json.loads(cleaned)
            if not isinstance(data, dict):
                raise ValueError("JSON is not an object")

            # basic sanity check / defaults
            data.setdefault("agent_name", self.config.name)
            data.setdefault("short_summary", "")
            data.setdefault("detailed_reasoning", "")
            data.setdefault("evidence_snippets", [])

            if not isinstance(data.get("evidence_snippets"), list):
                data["evidence_snippets"] = []

            return data
        except Exception:
            # Fallback
            return {
                "agent_name": self.config.name,
                "short_summary": cleaned[:300],
                "detailed_reasoning": cleaned,
                "evidence_snippets": [],
            }

    def answer(
        self,
        question: str,
        chat_history_window: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Main entry for the worker agent.
        - Retrieves from data_store + memory_store
        - Calls LLM
        - Updates memory_store with this turn
        - Returns structured dict (plus debug retrieved_rows)
        """
        # ---- Retrieval from CSV/Excel vector store (with server bias) ----
        server_tokens = self._extract_server_tokens(question)
        retrieval_query = question
        if server_tokens:
            # Prepend server names to strongly bias similarity search
            retrieval_query = " ".join(server_tokens) + " " + question

        data_results = self.data_store.similarity_search(
            query=retrieval_query,
            k=self.config.top_k_data,
        )

        # Build debug view of retrieved rows so UI can show them
        retrieved_rows = []
        for r in data_results:
            meta = r.get("metadata", {})
            retrieved_rows.append(
                {
                    "source": meta.get("source", "unknown_source"),
                    "row_id": meta.get("row_id", None),
                    "text": r.get("text", "").strip().replace("\n", " "),
                }
            )

        # ---- Retrieval from memory store ----
        memory_results = self.memory_store.similarity_search(
            query=question,
            k=self.config.top_k_memory,
        )

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(
            question=question,
            data_results=data_results,
            memory_results=memory_results,
            chat_history_window=chat_history_window,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        raw = call_ollama_chat(
            model=self.config.model_name,
            messages=messages,
            temperature=self.config.temperature,
        )
        parsed = self._parse_agent_json(raw)

        # Add this Q&A to memory
        memory_text = (
            f"Q: {question}\n"
            f"A: {parsed.get('short_summary', '')}\n"
            f"Details: {parsed.get('detailed_reasoning', '')[:1000]}"
        )
        self.memory_store.add_document(
            text=memory_text,
            metadata={"source": "agent_memory", "agent_name": self.config.name},
        )

        # Attach retrieval debug info so the UI can show it
        parsed["retrieved_rows"] = retrieved_rows
        return parsed


# ---------- Supervisor agent ----------


@dataclass
class SupervisorConfig:
    model_name: str
    temperature: float = 0.2


@dataclass
class SupervisorAgent:
    config: SupervisorConfig

    def answer(
        self,
        question: str,
        worker_outputs: List[Dict[str, Any]],
        chat_history_window: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Combine worker outputs into a single final answer.
        """
        history_txt = ""
        if chat_history_window:
            for msg in chat_history_window[-6:]:
                history_txt += f"{msg['role'].upper()}: {msg['content']}\n"

        workers_json = json.dumps(worker_outputs, indent=2)

        system_prompt = """
You are a Supervisor LLM coordinating several specialized agents that
analyze server logs and performance metrics.

Goal:
- Read their structured JSON outputs.
- Compare and reconcile their findings.
- Explain clearly to a human what is going on, including likely root causes.
- Highlight cross-signals (events + CPU spikes + waits, etc.).

Rules about uncertainty and context:
- Each worker only sees a subset (retrieved CONTEXT) of the full dataset.
- If a worker's "detailed_reasoning" contains the token
  INSUFFICIENT_EVIDENCE_IN_CONTEXT, you MUST treat that as:
    "this agent's retrieved context was insufficient to answer",
  NOT as proof that the entire dataset has no such evidence.
- Never claim that something does or does not exist in the whole dataset.
  You can only say what the agents did or did not see in their contexts.
- When evidence is weak or conflicting, be explicit about that.

Your response MUST be:
1) A concise summary in simple language.
2) A short bullet list of key points.
3) A section "What each agent found" summarizing each agent.

Be honest about uncertainty and conflicts.
"""

        user_prompt = f"""
Conversation history (recent turns):
{history_txt}

User question:
{question}

Worker agent outputs (JSON):
{workers_json}

Now produce a single, coherent answer for the user, as described in the system instructions.
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        final_answer = call_ollama_chat(
            model=self.config.model_name,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=1536,
        )
        return final_answer
