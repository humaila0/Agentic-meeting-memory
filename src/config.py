from pathlib import Path
import os
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
DATA_DIR = ROOT / os.getenv("DATA_DIR", "data/source/transcripts")
QUESTIONS_FILE = ROOT / os.getenv("QUESTIONS_FILE", "data/source/questions.json")
CHROMA_DIR = ROOT / os.getenv("CHROMA_DIR", "chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
