import os
from dotenv import load_dotenv

load_dotenv()

# BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1")  # Default to localhost
HOST = os.getenv("HOST", "127.0.0.1")  # Host for Flask to bind to
PORT = int(os.getenv("PORT", 10000))


