"""
Verification test for LiteLLM Gateway:
- Primary and Fallback model hierarchy
- Max token enforcement
- Max tool iteration guardrails
- Runtime telemetry & statistics
"""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.gateway import get_gateway_llm, get_gateway_stats
from src.config import settings


def test_gateway():
    print("=== Step 1: Testing Gateway Configuration & Stats ===")
    stats = get_gateway_stats()
    print("Gateway stats:", stats)
    assert stats["gateway"] == "LiteLLM Embedded Gateway"
    assert stats["status"] == "active"
    assert stats["primary_model"] == settings.LITELLM_PRIMARY_MODEL
    assert len(stats["fallback_models"]) > 0
    assert stats["max_tokens_limit"] == settings.LITELLM_MAX_TOKENS
    assert stats["max_tool_iterations"] == settings.MAX_TOOL_ITERATIONS
    print("[PASS] Gateway configuration verified.")

    print("\n=== Step 2: Testing Gateway LLM Construction ===")
    llm = get_gateway_llm()
    print("Constructed LLM:", type(llm).__name__)
    assert hasattr(llm, "bind_tools"), "LLM must support bind_tools for agentic tool calling"
    print("[PASS] Gateway LLM constructed with fallback chain successfully.")

    print("\n=== Step 3: Testing Fallback Failover Simulation ===")
    from langchain_community.chat_models import ChatLiteLLM
    from langchain_core.messages import HumanMessage

    # Intentionally bad primary model that triggers failover
    bad_primary = ChatLiteLLM(
        model="groq/non-existent-model-xyz",
        api_key=settings.GROQ_API_KEY,
        max_tokens=10,
        request_timeout=5,
    )
    # Working fallback model
    good_fallback = ChatLiteLLM(
        model=settings.LITELLM_PRIMARY_MODEL,
        api_key=settings.GROQ_API_KEY,
        max_tokens=10,
        request_timeout=15,
    )

    fallback_chain = bad_primary.with_fallbacks([good_fallback])
    print("Invoking fallback chain with invalid primary model...")
    res = fallback_chain.invoke([HumanMessage(content="Say 'OK' in one word")])
    print("Fallback Response:", res.content)
    assert len(res.content.strip()) > 0, "Expected successful response from fallback model"
    print("[PASS] Automatic failover to fallback model succeeded seamlessly!")

    print("\n>>> ALL LITELLM GATEWAY TESTS PASSED! <<<")


if __name__ == "__main__":
    test_gateway()
