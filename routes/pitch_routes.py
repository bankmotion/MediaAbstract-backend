from flask import Blueprint, request, jsonify
from models.pitch import Pitch
from supabase import create_client
from datetime import datetime
import os

# Initialize Supabase client
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))

pitch_routes = Blueprint("pitch_routes", __name__)

@pitch_routes.route("/submit_pitch", methods=["POST"])
def submit_pitch():
    try:
        data = request.json
        if not data:
            return jsonify({"matched_outlets": []}), 400
        if "abstract" not in data or "industry" not in data:
            return jsonify({"matched_outlets": []}), 400
        
        pitch = Pitch(
            data["abstract"], 
            data["industry"],
            user_id=data["userId"],  
            plan_type=data.get("planType")  
        )
        
        # Get enhanced analysis of user input
        user_analysis = pitch.analyze_user_input()
        print(f"User analysis: {user_analysis}")
        # Find matches using enhanced matcher
        matches = pitch.find_matching_outlets()
        pitch_id = pitch.insert_pitch()
        
        if pitch_id is None:
            return jsonify({"matched_outlets": []}), 500
        
        abstract = data["abstract"]
        title_words = abstract.split()[:8]
        pitch_title = " ".join(title_words) + ("..." if len(abstract.split()) > 8 else "")
        action = f"Matched '{pitch_title}'"
        user_id = data["userId"]
        created_at = datetime.utcnow().isoformat()
        
        supabase.table("activity_log").insert({
            "user_id": user_id,
            "action": action,
            "created_at": created_at
        }).execute()
        
        serializable_matches = []
        for match in matches:
            if data.get("planType", "").lower() == "basic":
                serializable_match = {
                    "pitch_id": pitch_id,
                    "outlet": {
                        "name": match["outlet"].get("Outlet Name", ""),
                        "contact_email": match["outlet"].get("Editor Contact", ""),
                        "url": match["outlet"].get("URL", "")
                    }
                }
            else:
                # Enhanced response with detailed analysis
                outlet = match["outlet"]
                serializable_match = {
                    "pitch_id": pitch_id,
                    "outlet": {
                        "name": outlet.get("Outlet Name", ""),
                        "audience": outlet.get("Audience", ""),
                        "section_name": outlet.get("Section Name", ""),
                        "contact_email": outlet.get("Editor Contact", ""),
                        "ai_partnered": outlet.get("AI Partnered", ""),
                        "url": outlet.get("URL", ""),
                        "guidelines": outlet.get("Guidelines", ""),
                        "pitch_tips": outlet.get("Pitch Tips", ""),
                        "keywords": outlet.get("Keywords", ""),
                        "prestige": outlet.get("Prestige", ""),
                        "last_updated": outlet.get("Last Updated", "")
                    },
                    "score": float(match["score"]),
                    "match_confidence": match["match_confidence"],
                    "match_explanation": match["match_explanation"]
                }
            serializable_matches.append(serializable_match)
        
        # Include user analysis in response for premium plans
        response_data = {
            "matched_outlets": serializable_matches,
            "analysis": user_analysis if data.get("planType", "").lower() != "basic" else None
        }
        
        return jsonify(response_data), 200
    except Exception as e:
        print(f"Error in submit_pitch: {str(e)}")
        return jsonify({"matched_outlets": []}), 500

@pitch_routes.route("/update_pitch_status", methods=["PUT"])
def update_pitch_status():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        # Validate required fields
        pitch_id = data.get("pitchId")
        outlet_name = data.get("outletName")
        status = data.get("status")
        user_id = data.get("userId")

        print(f"Received data: {data}")
        print(f"Pitch ID: {pitch_id}")
        print(f"Outlet Name: {outlet_name}")
        print(f"Status: {status}")
        print(f"User ID: {user_id}")

        if not all([pitch_id, outlet_name, status, user_id]):
            return jsonify({"error": "Missing required fields: pitchId, outletName, status, or userId"}), 400
        
        # Update the pitch status
        success = Pitch.update_pitch_status(pitch_id)

        print(f"Success: {success}")

        if success:
            # Log activity: Submitted 'outletname'
            action = f"Submitted '{outlet_name}'"
            created_at = datetime.utcnow().isoformat()
            supabase.table("activity_log").insert({
                "user_id": user_id,
                "action": action,
                "created_at": created_at
            }).execute()
            return jsonify({
                "success": True,
                "message": f"Successfully updated status to {status} for outlet {outlet_name}"
            }), 200
        else:
            return jsonify({
                "success": False,
                "error": "Failed to update pitch status"
            }), 500
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Internal server error: {str(e)}"
        }), 500

@pitch_routes.route("/update_pitch_status_and_notes", methods=["PUT"])
def update_pitch_status_and_notes():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        # Validate required fields
        pitch_id = data.get("pitchId")
        status = data.get("status")
        notes = data.get("notes")
        print(f"Received data: {data}")
        
        if not pitch_id:
            return jsonify({"error": "Missing required field: pitchId"}), 400
            
        if not status and not notes:
            return jsonify({"error": "At least one of status or notes must be provided"}), 400
        
        # Update the pitch status and notes
        success = Pitch.update_pitch_status_and_notes(pitch_id, status, notes)

        if success:
            return jsonify({
                "success": True,
                "message": "Successfully updated pitch status and notes"
            }), 200
        else:
            return jsonify({
                "success": False,
                "error": "Failed to update pitch status and notes"
            }), 500
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Internal server error: {str(e)}"
        }), 500

@pitch_routes.route("/get_dashboard_data", methods=["GET"])
def get_dashboard_data():
    user_id = request.args.get("userId")
    dashboard_data = Pitch.get_dashboard_data(user_id=user_id)
    
    if dashboard_data:
        return jsonify(dashboard_data), 200
    else:
        return jsonify({"error": "Failed to fetch dashboard data"}), 500

@pitch_routes.route("/save_selected_outlets", methods=["POST"])
def save_selected_outlets():
    try:
        data = request.json
        pitch_id = data.get("description")
        outlet_ids = data.get("outlets")
        user_id = data.get("userId")  # Get user_id from request data

        if not pitch_id or not outlet_ids or not user_id:
            return jsonify({"error": "Missing required fields"}), 400

        success = Pitch.save_selected_outlets(pitch_id, outlet_ids, user_id)
        print(f"Success: {success}")
        if success:
            # Use pitch_id (description/abstract) directly as the title
            abstract = pitch_id
            title_words = abstract.split()[:8]
            pitch_title = " ".join(title_words) + ("..." if len(abstract.split()) > 8 else "")


            # Use outlet_ids directly as names
            outlet_names_str = ", ".join(outlet_ids)
            selected_count = len(outlet_ids) if outlet_ids else 0

            action = f"Saved outlets ({selected_count}): {outlet_names_str} for '{pitch_title}'"
            created_at = datetime.utcnow().isoformat()
            
            supabase.table("activity_log").insert({
                "user_id": user_id,
                "action": action,
                "created_at": created_at
            }).execute()
            return jsonify({"success": True, "message": "Outlets saved successfully"}), 200
        else:
            return jsonify({"success": False, "error": "Failed to save outlets"}), 500

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@pitch_routes.route("/get_saved_outlets", methods=["GET"])
def get_saved_outlets():
    """Fetch all saved outlets for pitches."""
    user_id = request.args.get("userId")
    
    if not user_id:
        return jsonify({"error": "Missing required field: userId"}), 400
    
    saved_outlets = Pitch.get_all_selected_outlets(user_id)

    if saved_outlets:
        return jsonify(saved_outlets), 200
    else:
        return jsonify({"error": "Failed to fetch saved outlets data"}), 500

@pitch_routes.route("/get_all_outlets", methods=["GET"])
def get_all_outlets():
    outlets = Pitch.get_all_outlets()
    if outlets:
        return jsonify(outlets), 200
    else:
        return jsonify({"error": "Failed to fetch all outlets"}), 500

@pitch_routes.route("/delete_saved_pitch", methods=["DELETE"])
def delete_saved_pitch():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        print(f"Received data: {data}")
        description = data.get("description")
        selected_date = data.get("selected_date")
        user_id = data.get("userId")
        
        if not all([description, selected_date, user_id]):
            return jsonify({"error": "Missing required fields: description, selected_date, or userId"}), 400

        success = Pitch.delete_saved_pitch(description, selected_date, user_id)

        if success:
            return jsonify({"success": True, "message": "Saved pitch deleted successfully."}), 200
        else:
            return jsonify({"success": False, "error": "Failed to delete saved pitch."}), 500

    except Exception as e:
        print(f"Error deleting saved pitch: {str(e)}")
        return jsonify({"success": False, "error": f"Internal server error: {str(e)}"}), 500

@pitch_routes.route("/analyze_input", methods=["POST"])
def analyze_input():
    """Analyze user input to extract topics, themes, and characteristics."""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        if "abstract" not in data or "industry" not in data:
            return jsonify({"error": "Missing abstract or industry"}), 400
        
        pitch = Pitch(
            data["abstract"], 
            data["industry"],
            user_id=data.get("userId"),  
            plan_type=data.get("planType")  
        )
        
        # Get comprehensive analysis
        analysis = pitch.analyze_user_input()
        
        return jsonify({
            "analysis": analysis,
            "message": "Input analyzed successfully"
        }), 200
        
    except Exception as e:
        print(f"Error in analyze_input: {str(e)}")
        return jsonify({"error": "Failed to analyze input"}), 500

        
