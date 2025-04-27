import os
from dotenv import load_dotenv
from typing import List, Dict
from datetime import datetime, timedelta
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
        return self.matcher.find_matches(self.abstract, self.industry)

    def insert_pitch(self):
        try:
            matched_outlets = self.find_matching_outlets()
            match_count = len(matched_outlets)

            # Insert pitch data
            pitch_data = {
                "abstract": self.abstract,
                "industry": self.industry,
                "status": "Matched",
                "matches_found": match_count,
                "matched_outlets": matched_outlets,  # Store the matched outlets data
                "created_at": datetime.utcnow().isoformat()
            }
            
            # Insert pitch and get the ID
            response = supabase.table("pitches").insert(pitch_data).execute()
            
            if response.data:
                return response.data[0]["id"]  # Return the ID of the inserted pitch
            return None
        
        except Exception as e:
            print(f"Detailed error inserting pitch: {str(e)}")
            return None
        
    
    @staticmethod
    def get_dashboard_data():
        try:
            pitches = supabase.table("pitches").select("*").order("created_at", desc=True).execute().data
            total_pitches = len(pitches)
            total_matches = sum(p["matches_found"] if p["matches_found"] is not None else 0 for p in pitches)

            # Format pitch data for frontend
            formatted_pitches = []
            for pitch in pitches:
                # Get first few words of abstract as title (or use full abstract if short)
                title_words = pitch["abstract"].split()[:8]  # First 8 words
                title = " ".join(title_words) + ("..." if len(pitch["abstract"].split()) > 8 else "")

                # Format matched outlets data
                matched_outlets = []
                if pitch.get("matched_outlets"):
                    for outlet_match in pitch["matched_outlets"]:
                        outlet = outlet_match.get("outlet", {})
                        outlet_name = outlet.get("Outlet Name", "")
                        outlet_url = outlet.get("URL", "")
                        outlet_email = outlet.get("Editor Contact", "")
                        outlet_ai_partnered = outlet.get("AI Partnered", "")
                        match_score = outlet_match.get("score", 0)
                        match_percentage = f"{int(match_score * 100)}%"
                        match_explanation = outlet_match.get("match_explanation", "")

                        matched_outlets.append({
                            "name": outlet_name,
                            "match_percentage": match_percentage,
                            "url": outlet_url,
                            "email": outlet_email,
                            "ai_partnered": outlet_ai_partnered,
                            "match_explanation": match_explanation
                        })

                formatted_pitch = {
                    "id": pitch["id"],
                    "title": title,
                    "abstract": pitch["abstract"],
                    "industry": pitch["industry"],
                    "status": pitch["status"],
                    "matched_outlets": matched_outlets,
                    "created_at": pitch["created_at"]
                }
                formatted_pitches.append(formatted_pitch)

            return {
                "pitches_sent": total_pitches,
                "matches_found": total_matches,
                "my_pitches": formatted_pitches
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
                        
                        # Start a new group with selected date
                        current_group = {
                            "description": pitch_id, 
                            "outlets": [],
                            "selected_date": created_at.strftime("%Y-%m-%d %H:%M:%S")  # Format date for frontend
                            # "selected_date": created_at
                        }

                    # Append the outlet to the current group
                    current_group["outlets"].append(outlet_id)

                    # Update last seen values
                    last_pitch_id = pitch_id
                    last_created_at = created_at

                # Append the last group if not empty
                if current_group:
                    grouped_outlets.append(current_group)

                # print("grouped_outlets: ", grouped_outlets)

                return grouped_outlets

            return []

        except Exception as e:
            print(f"Error fetching saved outlets: {str(e)}")
            return []

    def get_all_outlets() -> List[dict]:
        """Fetch all outlets from the outlets table."""
        try:
            response = supabase.table("outlets").select("*").execute()
            
            if not response.data:
                return []
                
            outlets = []
            for outlet in response.data:
                formatted_outlet = {
                    "name": outlet.get("Outlet Name"),
                    "audience": outlet.get("Audience"),
                    "section_name": outlet.get("Section Name"),
                    "contact_email": outlet.get("Editor Contact"),
                    "ai_partnered": outlet.get("AI Partnered"),
                    "url": outlet.get("URL"),
                    "guidelines": outlet.get("Guidelines"),
                    "pitch_tips": outlet.get("Pitch Tips"),
                    "keywords": outlet.get("Keywords"),
                    "last_updated": outlet.get("Last Updated"),
                    "prestige": outlet.get("Prestige"),
                }
                
                outlets.append(formatted_outlet)
            
            return outlets
            
        except Exception as e:
            print(f"Error fetching all outlets: {str(e)}")
            return []

    @staticmethod
    def update_pitch_status(pitch_id: str) -> bool:
        """Update the status of a pitch to Submitted."""
        try:
            # Update the pitch status in the database
            update_response = supabase.table("pitches").update(
                {"status": "Submitted"}
            ).eq("id", pitch_id).execute()
            
            return bool(update_response.data)
            
        except Exception as e:
            print(f"Error updating pitch status: {str(e)}")
            return False

    @staticmethod
    def update_pitch_status_and_notes(pitch_id: str, status: str, notes: str) -> bool:
        """Update both the status and notes of a pitch."""
        try:
            # Update the pitch status and notes in the database
            response = supabase.table("pitches").update(
                {
                    "status": status,
                    "notes": notes
                }
            ).eq("id", pitch_id).execute()
            
            print("response: ", response)
            # Check if the update was successful by verifying the response data
            if response and response.data:
                return True
            return False
            
        except Exception as e:
            print(f"Error updating pitch status and notes: {str(e)}")
            return False

    @staticmethod
    def delete_saved_pitch(description: str, selected_date: str) -> bool:
        """
        Delete saved outlets where pitch_id matches and created_at matches the given second (ignoring fractional seconds and timezone).
        Args:
            description (str): The pitch_id to match
            selected_date (str): The created_at timestamp to match (format: YYYY-MM-DD HH:MM:SS)
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            if not description or not selected_date:
                print("Error: Both description and selected_date are required")
                return False

            # Parse the input date string (no timezone/fractional)
            try:
                dt = datetime.strptime(selected_date, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                print(f"Error: Invalid date format for selected_date: {selected_date}. Expected format: YYYY-MM-DD HH:MM:SS")
                return False

            # Build range for the second
            start = dt.isoformat()
            end = (dt + timedelta(seconds=1)).isoformat()

            delete_response = (
                supabase
                .table("selected_outlets")
                .delete()
                .eq("pitch_id", description)
                .gte("created_at", start)
                .lt("created_at", end)
                .execute()
            )

            if not delete_response.data:
                print(f"No records found to delete for pitch_id: {description} and date: {selected_date}")
                return False

            print(f"Successfully deleted saved pitch with pitch_id: {description} and date: {selected_date}")
            return True

        except Exception as e:
            print(f"Error deleting saved pitch: {str(e)}")
            return False
