from supabase import create_client
import os
from datetime import datetime

class Database:
    _instance = None
    _client = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Database, cls).__new__(cls)
            cls._client = create_client(
                os.getenv('SUPABASE_URL'),
                os.getenv('SUPABASE_KEY')
            )
        return cls._instance

    @classmethod
    def get_client(cls):
        if cls._client is None:
            cls._instance = cls()
        return cls._client 