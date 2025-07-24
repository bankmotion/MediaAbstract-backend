import jwt
from datetime import datetime, timedelta
from flask import current_app

def create_jwt_token(email):
    expiration = datetime.utcnow() + timedelta(days=1)  # Token valid for 1 day
    payload = {
        'email': email,
        'exp': expiration
    }
    token = jwt.encode(payload, current_app.config['JWT_SECRET_KEY'], algorithm='HS256')
    return token