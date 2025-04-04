import stripe
from flask import current_app
from models.user import User, users_db
from utils.jwt_utils import create_jwt_token

from services.supabase_service import supabase


def create_user_and_return_token(email: str, password: str, plan_id: str):
    try:
        #Try creating user in supabase
        response = supabase.auth.sign_up(email=email, password=password)
        
        if response.error:
            raise Exception(response.error.message)
        
        #User successfully created
        user_id = response["user"]["id"]

        #Insert user details into users table
        supabase.table("users").insert({
            "id": user_id,
            "email": email,
            "plan": plan_id,
            "is_paid": False #Payment is pending
        }).execute()
        
        #Authentication and get JWT token
        login_response = supabase.auth.sign_in_with_password(email=email, password=password)
        
        if login_response.get("error"):
            raise ValueError(login_response["error"]["message"])
        
        return login_response["session"]["access_token"]
        
    except Exception as e:
        raise ValueError(f"Error creating user: {str(e)}")
