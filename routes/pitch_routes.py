from flask import Blueprint, request, jsonify
from models.pitch import Pitch
from supabase import create_client

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
        
        # Insert pitch into database
        success = pitch.insert_pitch()

        if not success:
            return jsonify({"error": "Failed to submit pitch"}), 500
        

        if success:
            # Convert matches to serializable format
            serializable_matches = []
            for match in matches:
                serializable_match = {
                    "outlet": {
                        "name": match["outlet"].get("Outlet Name", ""),
                        "audience": match["outlet"].get("Audience", ""),
                        "section_name": match["outlet"].get("Section Name", ""),
                        "contact_email": match["outlet"].get("Editor Contact", ""),
                        "ai_partnered": match["outlet"].get("AI Partnered", ""),
                        "url": match["outlet"].get("URL", ""),
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
                "matched_outlets": serializable_matches
            }), 200
        else:
            return jsonify({
                "success": False,
                "error": "Failed to insert pitch"
            }), 500
            
    except Exception as e:
        print(f"Error in submit_pitch: {str(e)}")
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
    print("Saved_Outlets: ", saved_outlets)

    if saved_outlets:
        return jsonify(saved_outlets), 200
    else:
        return jsonify({"error": "Failed to fetch saved outlets data"}), 500

