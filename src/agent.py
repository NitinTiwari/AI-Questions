"""
Builds the tool-calling agent using ChatGroq as the reasoning LLM
and PostgreSQL (pgvector) as the multimodal vector store.
"""
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
try:
    from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
except ImportError:
    from langchain.agents import AgentExecutor, create_tool_calling_agent

from src.config import settings
from src.tools import get_tools, init_tools
from src.vector_store import get_or_build_vector_store
from src.gateway import get_gateway_llm

SYSTEM_PROMPT = """You are an expert AI assistant that answers questions using ONLY \
the contents of the indexed Word document(s), accessed through the \
`search_document` tool.

The document contains standard text explanations, structured comparison tables (formatted in Markdown), \
and technical architecture diagrams/figures with detailed descriptions.

Rules:
- Always call `search_document` at least once before answering any factual or comparison question.
- If the first search result doesn't fully answer the question, refine the query and search once more (maximum 2 searches total).
- Do NOT repeatedly search in a loop if the concept is missing. If the answer or concept is truly not in the document, state clearly: "Based on the indexed document (AI-Notes.docx), information about this topic was not found."
- When answering from a table, present the retrieved data clearly (using concise bullet points or markdown tables).
- When answering about a diagram or architecture flow, explain the components and mention the image file path if present in the retrieved passage metadata.
- Cite the source file name(s) and relevant section(s) at the end of your answer.
- Keep answers directly grounded in retrieved passages.
"""


def build_agent(force_rebuild_index: bool = False) -> AgentExecutor:
    settings.validate()

    vector_store = get_or_build_vector_store(force_rebuild=force_rebuild_index)
    init_tools(vector_store)
    tools = get_tools()

    # Obtain reasoning model with LiteLLM gateway fallback and token limits
    llm = get_gateway_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=settings.MAX_TOOL_ITERATIONS,
        early_stopping_method="generate",
        handle_parsing_errors=True,
    )


def ask(agent_executor: AgentExecutor, question: str) -> str:
    try:
        result = agent_executor.invoke({"input": question})
        return result["output"]
    except Exception as e:
        err_msg = str(e)
        if "rate_limit" in err_msg.lower() or "429" in err_msg:
            return "The AI reasoning provider is currently rate-limited (token limit reached on Groq free tier). Please retry in 10-15 seconds."
        raise e
