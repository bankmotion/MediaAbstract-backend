from flask import Blueprint, request, jsonify
from models.pitch import Pitch

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
                print("=Matches=:", serializable_matches)

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

        # return jsonify({
        #     "message": "Pitch submitted successfully",
        #     "matched_outlets": matches
        # }), 200
            
    except Exception as e:
        print(f"Error in submit_pitch: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Internal server error: {str(e)}"
        }), 500

@pitch_routes.route("/get_dashboard_data", methods=["GET"])
def get_dashboard_data():
    # user_id = request.args.get("user_id")
    # if not user_id:
    #     return jsonify({"error": "User ID required"}), 400
    
    # dashboard_data = Pitch.get_dashboard_data(user_id)
    dashboard_data = Pitch.get_dashboard_data()
    print("Dashboard Data: ", dashboard_data)
    if dashboard_data:
        return jsonify(dashboard_data), 200
    else:
        return jsonify({"error": "Failed to fetch dashboard data"}), 500
