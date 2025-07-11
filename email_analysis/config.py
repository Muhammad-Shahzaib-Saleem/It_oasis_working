import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama3-8b-8192"
QWEN_MODEL = "qwen/qwq-32b:free"
DEEP_SEEK_MODEL="deepseek/deepseek-chat-v3-0324:free"
LLAMA_MODEL="meta-llama/llama-4-maverick:free"

OPEN_ROUTER_API="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"

# Vector Database Configuration
VECTOR_DB_PATH = "./vector_db"
COLLECTION_NAME = "email_collection"

# Embedding Model Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# File Upload Configuration
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
SUPPORTED_FORMATS = ['.csv', '.json', '.txt','.eml']

# Streamlit Configuration
PAGE_TITLE = "Email Analysis System"
PAGE_ICON = "📧"