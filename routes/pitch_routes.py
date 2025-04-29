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
            return jsonify({"error": "No data provided"}), 400
            
        # Validate required fields
        if "abstract" not in data or "industry" not in data:
            return jsonify({"error": "Missing required fields: abstract and industry"}), 400
    
        # Create pitch object
        pitch = Pitch(data["abstract"], data["industry"])
        
        # Find matching outlets
        matches = pitch.find_matching_outlets()
        
        # Insert pitch and matches into database
        pitch_id = pitch.insert_pitch()

        if pitch_id is None:
            return jsonify({"error": "Failed to submit pitch"}), 500
        
        # Format the matches for response
        serializable_matches = []
        for match in matches:
            serializable_match = {
                "pitch_id": pitch_id,
                "outlet": {
                      # Add pitch_id to each match
                    "name": match["outlet"].get("Outlet Name", ""),
                    "audience": match["outlet"].get("Audience", ""),
                    "section_name": match["outlet"].get("Section Name", ""),
                    "contact_email": match["outlet"].get("Editor Contact", ""),
                    "ai_partnered": match["outlet"].get("AI Partnered", ""),
                    "url": match["outlet"].get("URL", ""),
                    "guidelines": match["outlet"].get("Guidelines", ""),
                    "pitch_tips": match["outlet"].get("Pitch Tips", ""),
                    # Add other relevant outlet fields
                },
                "score": float(match["score"]),  # Convert to float to ensure serializability
                "match_explanation": match["match_explanation"],
                "match_confidence": match["match_confidence"]
            }
            serializable_matches.append(serializable_match)

        return jsonify({
            "success": True,
            "message": "Pitch submitted successfully",
            "pitch_id": pitch_id,
            "matched_outlets": serializable_matches
        }), 200
            
    except Exception as e:
        print(f"Error in submit_pitch: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Internal server error: {str(e)}"
        }), 500

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
        
        print(f"Received data: {data}")
        print(f"Pitch ID: {pitch_id}")
        print(f"Outlet Name: {outlet_name}")
        print(f"Status: {status}")
        
        if not all([pitch_id, outlet_name, status]):
            return jsonify({"error": "Missing required fields: pitchId, outletName, or status"}), 400
        
        # Update the pitch status
        success = Pitch.update_pitch_status(pitch_id)

        print(f"Success: {success}")

        if success:
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
    dashboard_data = Pitch.get_dashboard_data()
    
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

        if not pitch_id or not outlet_ids:
            return jsonify({"error": "Missing required fields"}), 400

        success = Pitch.save_selected_outlets(pitch_id, outlet_ids)

        if success:
            return jsonify({"success": True, "message": "Outlets saved successfully"}), 200
        else:
            return jsonify({"success": False, "error": "Failed to save outlets"}), 500

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@pitch_routes.route("/get_saved_outlets", methods=["GET"])
def get_saved_outlets():
    """Fetch all saved outlets for pitches."""
    
    saved_outlets = Pitch.get_all_selected_outlets()

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
        
        if not all([description, selected_date]):
            return jsonify({"error": "Missing required fields: description or selected_date"}), 400

        success = Pitch.delete_saved_pitch(description, selected_date)

        if success:
            return jsonify({"success": True, "message": "Saved pitch deleted successfully."}), 200
        else:
            return jsonify({"success": False, "error": "Failed to delete saved pitch."}), 500

    except Exception as e:
        print(f"Error deleting saved pitch: {str(e)}")
        return jsonify({"success": False, "error": f"Internal server error: {str(e)}"}), 500

        
