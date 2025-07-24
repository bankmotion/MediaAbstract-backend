from flask import Blueprint, request, jsonify, current_app
from services.auth_service import create_user_and_return_token
from services.payment_service import create_checkout_session

auth_bp = Blueprint("auth", __name__)

@auth_bp.route("/signup", methods=["POST"])
def singup():
    data = request.json
    email = data.get("email")
    password = data.get("password")
    plan_id = data.get("plan_id")

    if not email or not password or not plan_id:
        return jsonify({"error": "Missing required fields"}), 400

    try:
        # Step 1: Create user in Supabase and get authentication token
        token = create_user_and_return_token(email, password, plan_id)
        
         # Step 2: Create a Stripe checkout session
        session = create_checkout_session(email, plan_id)

        return jsonify({
            "token": token,
            "sessionId": session.id,
            "stripePublicKey": current_app.config["STRIPE_PUBLIC_KEY"]
        }), 200
    
    
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    
@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.json
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"error": "Invalid credentials"}), 401

    try:
        token = create_user_and_return_token(email, password, "default_plan")
        return jsonify({"token": token}), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 401
