from flask import Blueprint, request, jsonify
import stripe
import os
from services.supabase_service import supabase


webhook_bp = Blueprint("webhook", __name__)

# STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
# stripe.api_key = STRIPE_SECRET_KEY


WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")

@webhook_bp.route("/webhook", methods=["POST"])
def stripe_webhook():
    payload = request.get_data()
    sig_header = request.headers.get("Stripe-Signature")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, WEBHOOK_SECRET)

        if event["type"] == "checkout.session.completed":
            session = event["data"]["object"]
            email = session["customer_email"]

            # Update user payment status in Supabase
            supabase.table("users").update({"is_paid": True}).eq("email", email).execute()

        return jsonify(success=True)

    except stripe.error.SignatureVerificationError:
        return jsonify({"error": "Webhook signature verification failed"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400
