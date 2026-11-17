"""
LiteLLM Gateway Engine for Multimodal Agentic RAG.
Manages:
- Primary and Fallback model failover chains (e.g. Groq Qwen 27B -> Groq GPT-OSS 20B / OpenAI / Anthropic)
- Configurable Max Token limits
- Max Tool Hit / Iteration guardrails
- Request timeout and retry orchestration
- Gateway utilization & health metrics
"""
import logging
import threading
from typing import Any, Dict, List, Optional

from langchain_community.chat_models import ChatLiteLLM
from langchain_core.language_models import BaseChatModel
from src.config import settings

logger = logging.getLogger(__name__)


class GatewayMetrics:
    """Thread-safe telemetry tracker for LiteLLM Gateway."""

    def __init__(self):
        self._lock = threading.Lock()
        self.total_calls = 0
        self.primary_calls = 0
        self.fallback_calls = 0
        self.total_tokens_estimated = 0

    def record_call(self, is_fallback: bool = False, tokens: int = 0):
        with self._lock:
            self.total_calls += 1
            if is_fallback:
                self.fallback_calls += 1
            else:
                self.primary_calls += 1
            self.total_tokens_estimated += tokens

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            fallback_list = [
                m.strip()
                for m in settings.LITELLM_FALLBACK_MODELS.split(",")
                if m.strip()
            ]
            return {
                "gateway": "LiteLLM Embedded Gateway",
                "status": "active",
                "primary_model": settings.LITELLM_PRIMARY_MODEL,
                "fallback_models": fallback_list,
                "max_tokens_limit": settings.LITELLM_MAX_TOKENS,
                "max_tool_iterations": settings.MAX_TOOL_ITERATIONS,
                "request_timeout_seconds": settings.LITELLM_TIMEOUT,
                "temperature": settings.LITELLM_TEMPERATURE,
                "telemetry": {
                    "total_gateway_calls": self.total_calls,
                    "primary_model_calls": self.primary_calls,
                    "fallback_model_calls": self.fallback_calls,
                    "estimated_tokens_consumed": self.total_tokens_estimated,
                },
            }


_metrics = GatewayMetrics()


def get_gateway_llm(
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> BaseChatModel:
    """
    Constructs the LiteLLM Gateway model chain.
    If the primary model fails due to rate limits (HTTP 429), context limit,
    or service errors (500/503), LiteLLM automatically fails over to the
    configured fallback model(s).
    """
    settings.validate()

    primary_model_str = settings.LITELLM_PRIMARY_MODEL
    api_key = settings.GROQ_API_KEY
    effective_temp = temperature if temperature is not None else settings.LITELLM_TEMPERATURE
    effective_tokens = max_tokens if max_tokens is not None else settings.LITELLM_MAX_TOKENS

    # 1. Primary Model
    primary_llm = ChatLiteLLM(
        model=primary_model_str,
        api_key=api_key,
        temperature=effective_temp,
        max_tokens=effective_tokens,
        request_timeout=settings.LITELLM_TIMEOUT,
    )

    # 2. Parse Fallback Models
    fallback_model_names = [
        m.strip() for m in settings.LITELLM_FALLBACK_MODELS.split(",") if m.strip()
    ]
    fallback_llms: List[BaseChatModel] = []

    for fb_name in fallback_model_names:
        fallback_llm = ChatLiteLLM(
            model=fb_name,
            api_key=api_key,
            temperature=effective_temp,
            max_tokens=effective_tokens,
            request_timeout=settings.LITELLM_TIMEOUT,
        )
        fallback_llms.append(fallback_llm)

    # 3. Build Fallback Chain
    if fallback_llms:
        logger.info(
            f"[LiteLLM Gateway] Primary: '{primary_model_str}' -> Fallbacks: {fallback_model_names}"
        )
        gateway_chain = primary_llm.with_fallbacks(fallback_llms)
    else:
        logger.info(f"[LiteLLM Gateway] Single Model: '{primary_model_str}'")
        gateway_chain = primary_llm

    _metrics.record_call(is_fallback=False)
    return gateway_chain


def get_gateway_stats() -> Dict[str, Any]:
    """Retrieve runtime configuration and metrics of the LiteLLM gateway."""
    return _metrics.get_stats()
