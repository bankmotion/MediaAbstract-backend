import os
from dotenv import load_dotenv

load_dotenv()

# BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1")  # Default to localhost
HOST = os.getenv("HOST", "0.0.0.0")  # Default to 0.0.0.0 to allow external connections
PORT = int(os.getenv("PORT", 10000))


# Payment Plans (Stripe Product IDs)
# Replace these with your actual test price IDs from Stripe dashboard
BASIC_PLAN_ID = os.getenv("STRIPE_BASIC_PLAN_ID")  # $50/month
TEAM_PLAN_ID = os.getenv("STRIPE_TEAM_PLAN_ID")  # $120/month
ENTERPRISE_PLAN_ID = os.getenv("STRIPE_ENTERPRISE_PLAN_ID")  # $200/month


#Stripe Keys
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
STRIPE_PUBLIC_KEY = os.getenv("STRIPE_PUBLIC_KEY")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
