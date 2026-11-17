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

SYSTEM_PROMPT = """You are an expert AI assistant that answers questions using ONLY \
the contents of the indexed Word document(s), accessed through the \
`search_document` tool.

The document contains standard text explanations, structured comparison tables (formatted in Markdown), \
and technical architecture diagrams/figures with detailed descriptions.

Rules:
- Always call `search_document` at least once before answering any factual or comparison question.
- If the first search result doesn't fully answer the question, refine the query and search again (up to 2-3 times).
- When answering from a table, present the retrieved data clearly (using concise bullet points or markdown tables).
- When answering about a diagram or architecture flow, explain the components and mention the image file path if present in the retrieved passage metadata.
- If the answer truly isn't in the document, say so clearly instead of making something up.
- Cite the source file name(s) and relevant section(s) at the end of your answer.
- Keep answers directly grounded in retrieved passages.
"""


def build_agent(force_rebuild_index: bool = False) -> AgentExecutor:
    settings.validate()

    vector_store = get_or_build_vector_store(force_rebuild=force_rebuild_index)
    init_tools(vector_store)
    tools = get_tools()

    llm = ChatGroq(
        model=settings.GROQ_MODEL,
        api_key=settings.GROQ_API_KEY,
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=6)


def ask(agent_executor: AgentExecutor, question: str) -> str:
    result = agent_executor.invoke({"input": question})
    return result["output"]
