import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base URL and port settings
BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1")  # Default to localhost
PORT = int(os.getenv("PORT", 10000))
