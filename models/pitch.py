import os
from supabase import create_client, Client
from dotenv import load_dotenv
import json
import re

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

    def insert_pitch(self):
        try:
            data = {
                "abstract": self.abstract,
                "industry": self.industry
            }
            
            response = supabase.table("pitches").insert(data).execute()
                        
            if response.data:
                print("Insert successful:", response.data)
                return True
            return False
        
        except Exception as e:
            print(f"Detailed error inserting pitch: {str(e)}")
            return False
        
    def match_outlets(self):
        try:
            response = supabase.table("outlets").select("*").execute()
            outlets = response.data

            if not outlets:
                return []
            
            abstract_words = set(re.findall(r'\b\w+\b', self.abstract))
            print("abstract word", abstract_words)
            matches = []
            for outlet in outlets:
                outlet_keywords = set(re.findall(r'\b\w+\b', outlet.get("Keywords", "").lower()))
                outlet_name = outlet.get("Outlet Name", "Unknown Outlet")
                
                # Count matching score
                common_words = abstract_words.intersection(outlet_keywords)
                match_score = len(common_words)
                print("Common word, Match score:", common_words, match_score)
                # If there's a match, calculate confidence percentage
                if match_score > 0:
                    match_confidence = round((match_score / len(outlet_keywords)) * 100, 2) if outlet_keywords else 0
                    matches.append({
                        "name": outlet_name,
                        "url": outlet.get("URL", "N/A"),
                        "contact_email": outlet.get("Editor Contact", "N/A"),
                        "match_confidence": match_confidence,
                        "ai_partnered": outlet.get("AI Partnered", False),
                        "matched_keywords": list(common_words),
                        "pitch_tips": outlet.get("Pitch Tips", "No pitch tips available."),
                        "guidelines": outlet.get("Guidelines", "No submission guidelines available."),
                    })
            
            #Sort outlets by match confidence
            matches.sort(key=lambda x: x["match_confidence"], reverse=True)
            # print("Matches Result:", matches)
            return matches
        
        except Exception as e:
            print(f"Error matching outlets: {str(e)}")
            return []