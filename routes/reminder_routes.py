from flask import Blueprint, request, jsonify
import os
import requests
from datetime import datetime
from services.supabase_service import supabase

reminder_routes = Blueprint("reminder_routes", __name__)

@reminder_routes.route("/create_reminder", methods=["POST"])
def create_reminder():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Validate required fields
        required_fields = ["user_id", "pitch_id", "reminder_date", "email", "status"]
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing required fields"}), 400

        # Insert reminder into Supabase
        reminder_data = {
            "user_id": data["user_id"],
            "pitch_id": data["pitch_id"],
            "reminder_date": data["reminder_date"],
            "email": data["email"],
            "status": data["status"],
            "created_at": datetime.utcnow().isoformat()
        }

        result = supabase.table("reminders").insert(reminder_data).execute()
        
        if not result.data:
            return jsonify({"error": "Failed to create reminder"}), 500

        # Fetch pitch details
        pitch_id = data["pitch_id"]
        pitch_result = supabase.table("pitches").select("abstract, industry, created_at").eq("id", pitch_id).single().execute()
        pitch_data = pitch_result.data if pitch_result and pitch_result.data else {}

        # Generate pitch title from first 8 words of abstract
        title = ""
        if pitch_data.get("abstract"):
            title_words = pitch_data["abstract"].split()[:8]
            title = " ".join(title_words) + ("..." if len(pitch_data["abstract"].split()) > 8 else "")

        # Truncate abstract to first sentence or 250 characters
        truncated_abstract = ""
        if pitch_data.get("abstract"):
            abs_text = pitch_data["abstract"]
            first_sentence = abs_text.split(". ")[0] + "." if "." in abs_text else abs_text
            truncated_abstract = first_sentence if len(first_sentence) <= 250 else abs_text[:250] + "..."

        # Get submission date
        submission_date = pitch_data.get("created_at", "")

        # Fetch user's full name from user_profiles
        user_id = data["user_id"]
        user_profile_result = supabase.table("user_profiles").select("full_name").eq("user_id", user_id).single().execute()
        user_profile = user_profile_result.data if user_profile_result and user_profile_result.data else {}
        full_name = user_profile.get("full_name", "")
        user_first_name = full_name.split()[0] if full_name else "[User]"

        # Prepare payload for Zapier
        zapier_payload = {
            **reminder_data,
            "id": pitch_id,
            "title": title,
            "submission_date": submission_date,
            "abstract": truncated_abstract,
            "industry": pitch_data.get("industry", ""),
            "first_name": user_first_name
        }
        
        # Send to Zapier webhook
        zapier_webhook_url = os.getenv("ZAPIER_WEBHOOK_URL")
        if zapier_webhook_url:
            try:
                zapier_response = requests.post(zapier_webhook_url, json=zapier_payload)
                if zapier_response.status_code != 200:
                    print(f"Warning: Zapier webhook returned status code {zapier_response.status_code}")
            except Exception as e:
                print(f"Error sending to Zapier: {str(e)}")

        return jsonify({
            "success": True,
            "message": "Reminder created successfully",
            "data": reminder_data
        }), 200

    except Exception as e:
        print(f"Error in create_reminder: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Internal server error: {str(e)}"
        }), 500

@reminder_routes.route("/get_reminders", methods=["GET"])
def get_reminders():
    try:
        result = supabase.table("reminders").select("*").execute()
        
        if not result.data:
            return jsonify({"error": "Failed to fetch reminders"}), 500
        return jsonify({
            "success": True,
            "data": result.data
        }), 200

    except Exception as e:
        print(f"Error in get_reminders: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Internal server error: {str(e)}"
        }), 500

@reminder_routes.route("/update_reminder_status", methods=["PUT"])
def update_reminder_status():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        reminder_id = data.get("reminderId")
        status = data.get("status")

        if not all([reminder_id, status]):
            return jsonify({"error": "Missing required fields: reminderId or status"}), 400

        result = supabase.table("reminders").update({"status": status}).eq("id", reminder_id).execute()

        if not result.data:
            return jsonify({"error": "Failed to update reminder status"}), 500

        return jsonify({
            "success": True,
            "message": "Reminder status updated successfully",
            "data": result.data[0]
        }), 200

    except Exception as e:
        print(f"Error in update_reminder_status: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Internal server error: {str(e)}"
        }), 500

@reminder_routes.route("/receive-reminder-status", methods=["POST"])
def receive_reminder_status():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        reminder_id = data.get("id")
        print("reminder_id (raw)", reminder_id)

        if not reminder_id:
            return jsonify({"error": "Missing required field: id"}), 400

        # Convert to int (if id is stored as integer in Supabase)
        try:
            reminder_id = int(reminder_id)
        except ValueError:
            return jsonify({"error": "Invalid reminder id format"}), 400

        # Optional: Check existence first
        existing = supabase.table("reminders").select("*").eq("pitch_id", reminder_id).execute()
        print("existing reminder:", existing.data)

        if not existing.data:
            return jsonify({"error": f"Reminder ID {reminder_id} not found"}), 404

        # Update reminder status
        result = supabase.table("reminders").update({
            "status": "sent",
            
        }).eq("pitch_id", reminder_id).execute()

        return jsonify({
            "success": True,
            "message": "Reminder status updated successfully",
            "data": result.data[0] if result.data else {}
        }), 200

    except Exception as e:
        print(f"Error in receive_reminder_status: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Internal server error: {str(e)}"
        }), 500

    