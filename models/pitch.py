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
from collections import defaultdict

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
                # print("Insert successful:", response.data)
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

    @staticmethod
    def save_selected_outlets(pitch_id: str, outlet_ids: List[str]) -> bool:
        """Save selected outlets for a pitch in the `saved_outlets` table."""
        print("Pitch_id, Outlet_ids: ", pitch_id, outlet_ids)
        
        try:
            if not pitch_id or not outlet_ids:
                return False

            data = [{"pitch_id": pitch_id, "outlet_id": outlet_id} for outlet_id in outlet_ids]
            response = supabase.table("selected_outlets").insert(data).execute()
            
            if response.data:
                # print("reponse:", response.data) 
                return True
            
        except Exception as e:
            print(f"Error saving selected outlets: {str(e)}")
            return False
        
    # def get_all_selected_outlets() -> List[dict]:
    #     """Fetch all saved outlets from the selected_outlets table."""
    #     try:
    #         response = supabase.table("selected_outlets").select("*").execute()
    #         # print("Response: ", response)
    #         if response.data:
    #             grouped_outlets = defaultdict(list)

    #             # Group outlet_id values under their respective pitch_id
    #             for record in response.data:
    #                 pitch_id = record["pitch_id"]
    #                 outlet_id = record["outlet_id"]
    #                 grouped_outlets[pitch_id].append(outlet_id)

    #             # Convert grouped dictionary into the required list format
    #             formatted_data = [
    #                 {"description": pitch, "outlets": outlets}
    #                 for pitch, outlets in grouped_outlets.items()
    #             ]

    #             return formatted_data
                
    #         return []

    #     except Exception as e:
    #         print(f"Error fetching saved outlets: {str(e)}")
    #         return []



    def get_all_selected_outlets() -> List[dict]:
        """Fetch all saved outlets from the selected_outlets table, ensuring unique pitch groups based on created_at order."""
        try:
            # Fetch the selected outlets with pitch_id, outlet_id, and created_at
            response = supabase.table("selected_outlets").select("pitch_id, outlet_id, created_at").order("created_at", desc=False).execute()
            
            # Check if data exists in response
            if response.data:
                grouped_outlets = []
                last_pitch_id = None
                last_created_at = None
                current_group = None

                for record in response.data:
                    pitch_id = record["pitch_id"]
                    outlet_id = record["outlet_id"]
                    created_at = record["created_at"]

                    # Convert created_at to a comparable format
                    created_at = datetime.fromisoformat(created_at) if isinstance(created_at, str) else created_at

                    # If it's a new pitch_id or a new created_at, start a new group
                    if last_pitch_id != pitch_id or (last_created_at and (created_at - last_created_at).total_seconds() > 1):
                        if current_group:  # Save the previous group before starting a new one
                            grouped_outlets.append(current_group)
                        
                        # Start a new group
                        current_group = {"description": pitch_id, "outlets": []}

                    # Append the outlet to the current group
                    current_group["outlets"].append(outlet_id)

                    # Update last seen values
                    last_pitch_id = pitch_id
                    last_created_at = created_at

                # Append the last group if not empty
                if current_group:
                    grouped_outlets.append(current_group)

                return grouped_outlets

            return []

        except Exception as e:
            print(f"Error fetching saved outlets: {str(e)}")
            return []
