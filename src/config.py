"""
Centralized configuration loaded from environment variables (.env file).
Every other module imports its settings from here instead of
reading os.environ directly.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root regardless of current working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class Settings:
    # --- PostgreSQL Database (pgvector) ---
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    PG_COLLECTION_NAME: str = os.getenv("PG_COLLECTION_NAME", "ai_notes_docs")

    # --- LLM Provider (Groq) ---
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

    # --- Models ---
    VISION_MODEL: str = os.getenv("VISION_MODEL", "llama-3.2-11b-vision-preview")
    HF_EMBEDDING_MODEL: str = os.getenv(
        "HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # --- Paths ---
    DOCS_PATH: Path = Path(os.getenv("DOCS_PATH", "./data")).resolve()
    EXTRACTED_IMAGES_PATH: Path = Path(os.getenv("EXTRACTED_IMAGES_PATH", "./data/extracted_images")).resolve()

    # --- Chunking ---
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))

    # --- Retrieval ---
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "4"))

    # --- Caching (L1 In-Memory + L2 PostgreSQL pgvector) ---
    CACHE_ENABLED: bool = os.getenv("CACHE_ENABLED", "true").lower() in {"true", "1", "yes"}
    CACHE_L1_MAXSIZE: int = int(os.getenv("CACHE_L1_MAXSIZE", "500"))
    CACHE_L1_TTL: int = int(os.getenv("CACHE_L1_TTL", "7200"))
    CACHE_SEMANTIC_THRESHOLD: float = float(os.getenv("CACHE_SEMANTIC_THRESHOLD", "0.92"))

    def validate(self):
        """Fail fast with a clear message if required configurations are missing."""
        missing = []
        if not self.DATABASE_URL:
            missing.append("DATABASE_URL")
        if not self.GROQ_API_KEY:
            missing.append("GROQ_API_KEY")
        if missing:
            raise EnvironmentError(
                f"Missing required environment variable(s): {', '.join(missing)} in .env."
            )


settings = Settings()
