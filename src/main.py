"""
CLI & Service entry point.

Usage:
    python -m src.main                 # interactive CLI Q&A loop
    python -m src.main --serve         # start the FastAPI server for end users
    python -m src.main --rebuild-index # force re-index the documents into PostgreSQL
    python -m src.main -q "question"   # ask a single question and exit
"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Multimodal Agentic RAG over Word documents with PostgreSQL (pgvector)")
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start the FastAPI web server for end users",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the FastAPI server on (default: 8000)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host interface to bind the FastAPI server (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Force rebuild of the PostgreSQL (pgvector) index from the source .docx files",
    )
    parser.add_argument(
        "-q", "--question",
        type=str,
        default=None,
        help="Ask a single question non-interactively and exit",
    )
    args = parser.parse_args()

    # Launch FastAPI server if --serve is requested
    if args.serve:
        import uvicorn
        print(f"Starting FastAPI server on http://{args.host}:{args.port} (Interactive docs: http://localhost:{args.port}/docs)")
        uvicorn.run("src.api:app", host=args.host, port=args.port, reload=True)
        return

    # Lazy import to keep --help instantaneous
    from src.agent import build_agent, ask
    from src.cache import get_cached_response, set_cached_response, clear_all_caches

    if args.rebuild_index:
        clear_all_caches()

    print("Connecting to PostgreSQL vector store and initializing agent...")
    try:
        agent_executor = build_agent(force_rebuild_index=args.rebuild_index)
    except Exception as e:
        print(f"Startup failed: {e}", file=sys.stderr)
        sys.exit(1)

    if args.question:
        cached = get_cached_response(args.question)
        if cached:
            tier = cached.get("tier", "cache")
            print(f"[⚡ Cached ({tier})]\n{cached['answer']}")
        else:
            answer = ask(agent_executor, args.question)
            set_cached_response(args.question, answer, images=[])
            print(answer)
        return

    print("Multimodal Agentic RAG ready (PostgreSQL pgvector). Ask a question about your document (type 'exit' to quit).\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        cached = get_cached_response(question)
        if cached:
            tier = cached.get("tier", "cache")
            sim_str = f", similarity: {cached['similarity']}" if cached.get("similarity") else ""
            print(f"\nAssistant [⚡ Cached ({tier}{sim_str})]: {cached['answer']}\n")
            continue

        try:
            answer = ask(agent_executor, question)
            set_cached_response(question, answer, images=[])
        except Exception as e:
            print(f"[Error] {e}")
            continue

        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    main()
