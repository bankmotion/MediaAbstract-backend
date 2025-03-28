from flask import Blueprint, request, jsonify
from models.pitch import Pitch

pitch_routes = Blueprint("pitch_routes", __name__)

@pitch_routes.route("/submit_pitch", methods=["POST"])
def submit_pitch():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        if "abstract" not in data or "industry" not in data:
            return jsonify({"error": "Missing required fields"}), 400
            
        pitch = Pitch(data["abstract"], data["industry"])
        pitch_data = pitch.insert_pitch()
        print("Pitched data", pitch_data)
        # if result:
        #     return jsonify({"message": "Pitch submitted successfully"}), 200
        # else:
        #     return jsonify({"error": "Failed to submit pitch"}), 500

        if not pitch_data:
            return jsonify({"error": "Failed to submit pitch"}), 500
        
        #Run keyword-matching algorithm
        matched_outlets = pitch.match_outlets()
        print("Mathced outlets",matched_outlets)

        return jsonify({
            "message": "Pitch submitted successfully",
            "matched_outlets": matched_outlets
        }), 200
            
    except Exception as e:
        print(f"Error in submit_pitch: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500
    

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
