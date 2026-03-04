from __future__ import annotations

from dataclasses import dataclass
from typing import Type

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel
from omegaconf import DictConfig


def _get_llm(cfg: DictConfig, model: str) -> BaseChatModel:
    """Return ChatOpenAI or ChatOllama based on config."""
    provider = getattr(cfg, "llm", None) and getattr(cfg.llm, "provider", None)
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model,
            base_url=getattr(cfg.llm, "base_url", "http://localhost:11434"),
            max_tokens=getattr(cfg.llm, "max_tokens", 200),
        )
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=model,
        api_key=cfg.openai.openai_api_key,
        max_tokens=getattr(cfg.openai, "max_tokens", 200),
    )


@dataclass
class LLMCallRecord:
    t: int | None
    agent_name: str | None
    call_type: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class SimulationLLM:
    """LangChain-based LLM wrapper for the evacuation simulation.

    Supports OpenAI (ChatOpenAI) or Ollama (ChatOllama) via config.llm.provider.
    Uses with_structured_output() for Pydantic-validated responses.
    """

    def __init__(self, cfg: DictConfig, system_prompt: str, model: str | None = None):
        self.cfg = cfg
        self.system_prompt = system_prompt

        if model is None:
            model = (
                getattr(cfg.llm, "model", None)
                if getattr(cfg, "llm", None)
                else None
            ) or "gpt-4.1-mini"

        self.llm = _get_llm(cfg, model)

        self.history: list = []

        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0
        self.call_log: list[LLMCallRecord] = []

    def _build_messages(self, user_prompt: str) -> list:
        return [
            SystemMessage(content=self.system_prompt),
            *self.history,
            HumanMessage(content=user_prompt),
        ]

    def decide(
        self,
        user_prompt: str,
        output_schema: Type[BaseModel],
        meta: dict | None = None,
    ) -> BaseModel:
        """Make a structured decision via LangChain's structured-output API.

        The Pydantic *output_schema* is enforced by the model (tool/function
        calling under the hood).  ``include_raw=True`` lets us extract token
        counts from the raw ``AIMessage``.
        """
        messages = self._build_messages(user_prompt)
        structured_llm = self.llm.with_structured_output(
            output_schema, include_raw=True,
        )

        result = structured_llm.invoke(messages)
        parsed: BaseModel = result["parsed"]
        raw_msg: AIMessage = result["raw"]

        self.history.append(HumanMessage(content=user_prompt))
        self.history.append(AIMessage(content=parsed.model_dump_json()))

        token_usage = _extract_token_usage(raw_msg)
        self._record_usage(token_usage, meta)

        return parsed

    def ask(self, user_prompt: str, meta: dict | None = None) -> str:
        """Plain text response (connectivity check, free-form queries, etc.)."""
        messages = self._build_messages(user_prompt)
        response = self.llm.invoke(messages)

        self.history.append(HumanMessage(content=user_prompt))
        self.history.append(response)

        token_usage = _extract_token_usage(response)
        self._record_usage(token_usage, meta)

        return response.content.strip()

    def _record_usage(self, token_usage: dict, meta: dict | None) -> None:
        pt = token_usage.get("prompt_tokens", 0)
        ct = token_usage.get("completion_tokens", 0)
        tt = token_usage.get("total_tokens", pt + ct)

        self.total_prompt_tokens += pt
        self.total_completion_tokens += ct
        self.total_calls += 1

        m = meta or {}
        self.call_log.append(
            LLMCallRecord(
                t=m.get("t"),
                agent_name=m.get("agent_name"),
                call_type=m.get("call_type", "unknown"),
                prompt_tokens=pt,
                completion_tokens=ct,
                total_tokens=tt,
            )
        )


def _extract_token_usage(msg) -> dict:
    """Safely pull token counts from AIMessage response_metadata.

    OpenAI: token_usage.prompt_tokens, completion_tokens, total_tokens
    Ollama: prompt_eval_count, eval_count (no total_tokens)
    """
    meta = getattr(msg, "response_metadata", None)
    if not isinstance(meta, dict):
        return {}

    # OpenAI format
    usage = meta.get("token_usage", {})
    if usage:
        return usage

    # Ollama format
    pt = meta.get("prompt_eval_count", 0)
    ct = meta.get("eval_count", 0)
    if pt or ct:
        return {
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": pt + ct,
        }
    return {}


def test_connection(cfg: DictConfig, user_text: str = "Say: 'OpenAI comms successful.'") -> str:
    """Quick connectivity check. Uses Ollama or OpenAI based on config."""
    provider = getattr(cfg, "llm", None) and getattr(cfg.llm, "provider", None)
    model = (
        getattr(cfg.llm, "model", "llama3.2:latest")
        if provider == "ollama"
        else "gpt-4.1-mini"
    )
    llm = _get_llm(cfg, model)
    sys_content = (
        f"You are a helpful assistant. The key owner is {cfg.openai.key_owner}."
        if provider != "ollama"
        else "You are a helpful assistant."
    )
    response = llm.invoke([
        SystemMessage(content=sys_content),
        HumanMessage(content=user_text),
    ])
    return response.content.strip()
