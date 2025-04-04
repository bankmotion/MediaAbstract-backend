import os
import stripe

# Stripe Configuration
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
stripe.api_key = STRIPE_SECRET_KEY

# Define available pricing plans (match with frontend)
PLAN_PRICING = {
    "agency75": {"price": 7500, "name": "Agency & Team - $75"},
    "agency150": {"price": 15000, "name": "Agency & Team - $150"},
    "agency250": {"price": 25000, "name": "Agency & Team - $250"}
}

def create_checkout_session(email: str, plan_id: str):
    """
    Creates a Stripe Checkout session for the selected plan.
    """
    if plan_id not in PLAN_PRICING:
        raise ValueError("Invalid plan selection")

    plan_details = PLAN_PRICING[plan_id]

    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            customer_email=email,
            line_items=[{
                "price_data": {
                    "currency": "usd",
                    "unit_amount": plan_details["price"],
                    "product_data": {"name": plan_details["name"]},
                },
                "quantity": 1,
            }],
            mode="payment",
            success_url="http://localhost:3000/payment-success?session_id={CHECKOUT_SESSION_ID}",
            cancel_url="http://localhost:3000/payment-failed",
        )

        return session

    except Exception as e:
        raise ValueError(f"Stripe Checkout Error: {str(e)}")
