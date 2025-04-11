from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash
from models.user import User
import stripe
import os
from datetime import datetime

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    try:    
        # Check if user already exists
        existing_user = User.get_by_email(email)
        if existing_user:
            return jsonify({'error': 'Email already registered'}), 409

        # Create temporary user record (not activated)
        hashed_password = generate_password_hash(password)
        new_user = User.create(email, hashed_password)
        
        if not new_user:
            return jsonify({'error': 'Failed to create user'}), 500

        return jsonify({
            'id': new_user.id,
            'email': new_user.email,
            'message': 'User pre-registered successfully'
        }), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@auth_bp.route('/verify-email', methods=['POST'])
def verify_email():
    try:
        data = request.get_json()
        email = data.get('email')

        # Check if email exists
        user = User.get_by_email(email)
        return jsonify({'exists': bool(user)}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500 