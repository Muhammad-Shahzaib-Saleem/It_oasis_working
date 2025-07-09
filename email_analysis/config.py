import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama3-8b-8192"

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