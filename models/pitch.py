import os
from supabase import create_client, Client
from dotenv import load_dotenv
from typing import List, Dict
# import json
# import re
from datetime import datetime
# import spacy
# from fuzzywuzzy import fuzz
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
from models.matcher import OutletMatcher


# nlp = spacy.load("en_core_web_md")

load_dotenv()

supabase_url: str = os.environ.get("SUPABASE_URL")
supabase_key: str = os.environ.get("SUPABASE_SERVICE_KEY")

if not supabase_url or not supabase_key:
    raise ValueError("Supabase URL and Key must be set in environment variables.")

supabase: Client = create_client(supabase_url, supabase_key)

class Pitch:
    def __init__(self, abstract: str, industry: str):
        self.abstract = abstract
        self.industry = industry
        # self.abstract_doc = nlp(self.abstract)
        self.matcher = OutletMatcher(supabase)

    def find_matching_outlets(self) -> List[Dict]:
        """Find matching outlets for the pitch."""
        query = f"{self.abstract} {self.industry}"
        return self.matcher.find_matches(query)

    def insert_pitch(self):
        try:
            matched_outlets = self.find_matching_outlets()
            match_count = len(matched_outlets)

            data = {
                "abstract": self.abstract,
                "industry": self.industry,
                "status": "Submitted",
                "matches_found": match_count,
                "created_at": datetime.utcnow().isoformat()
            }
            
            response = supabase.table("pitches").insert(data).execute()
                        
            if response.data:
                print("Insert successful:", response.data)
                return True
            return False
        
        except Exception as e:
            print(f"Detailed error inserting pitch: {str(e)}")
            return False
        
    
    @staticmethod
    def get_dashboard_data():
        try:
            # pitches = supabase.table("pitches").select("*").eq("user_id", user_id).execute().data
            pitches = supabase.table("pitches").select("*").execute().data
            # print("pitches: ", pitches)            
            # activity = supabase.table("activity_log").select("*").eq("user_id", user_id).order("timestamp", desc=True).execute().data
            total_pitches = len(pitches)
            total_matches = sum(p["matches_found"] if p["matches_found"] is not None else 0 for p in pitches)

        
            print("total pitches: ", total_pitches)
            print("total matches: ", total_matches)

            return {
                "pitches_sent": total_pitches,
                "matches_found": total_matches,
                "my_pitches": pitches,
                # "activity": activity
            }
        except Exception as e:
            print(f"Error fetching dashboard data: {str(e)}")
            return None
