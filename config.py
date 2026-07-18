"""
config.py
---------
Central configuration for the AI Knowledge Assistant.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# --- Paths ---
KNOWLEDGE_BASE_DIR = os.environ.get("KNOWLEDGE_BASE_DIR", "sample_kb")
VECTOR_STORE_DIR = os.environ.get("VECTOR_STORE_DIR", ".vector_store")
EXPORTS_DIR = os.environ.get("EXPORTS_DIR", "exports")

# --- Models ---
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "gemini-2.5-flash")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# --- Chunking ---
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "120"))

# --- Retrieval ---
RETRIEVAL_TOP_K = int(os.environ.get("RETRIEVAL_TOP_K", "6"))
RETRIEVAL_FETCH_MULTIPLIER = 3  # over-fetch factor before de-duplication

# --- Validation ---
MIN_CONFIDENCE_TO_ACCEPT = float(os.environ.get("MIN_CONFIDENCE_TO_ACCEPT", "0.55"))
MAX_REGENERATION_ATTEMPTS = int(os.environ.get("MAX_REGENERATION_ATTEMPTS", "2"))

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv"}


def validate_config():
    """Fail fast if required configuration is missing."""
    if not GOOGLE_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY is not set. Create a .env file (see .env.example) "
            "and add your Google AI Studio API key there."
        )
