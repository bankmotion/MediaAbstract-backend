import os
from dotenv import load_dotenv
from typing import List, Dict
from datetime import datetime
from collections import defaultdict

from models.matcher import OutletMatcher
from services.supabase_service import supabase

load_dotenv()
class Pitch:
    def __init__(self, abstract: str, industry: str):
        self.abstract = abstract
        self.industry = industry
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
                return True
            return False
        
        except Exception as e:
            print(f"Detailed error inserting pitch: {str(e)}")
            return False
        
    
    @staticmethod
    def get_dashboard_data():
        try:
            pitches = supabase.table("pitches").select("*").execute().data
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
                return True
            
        except Exception as e:
            print(f"Error saving selected outlets: {str(e)}")
            return False
        
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
