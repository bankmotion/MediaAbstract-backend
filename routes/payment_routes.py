from flask import Blueprint, request, jsonify
from models.user import User
from models.subscription import Subscription
import stripe
from config import STRIPE_SECRET_KEY, BASIC_PLAN_ID, TEAM_PLAN_ID, ENTERPRISE_PLAN_ID
from datetime import datetime
import os

payment_bp = Blueprint('payment', __name__)

# Initialize Stripe
stripe.api_key = STRIPE_SECRET_KEY

# Define subscription plans with actual Stripe price IDs
SUBSCRIPTION_PLANS = {
    BASIC_PLAN_ID: {
        'name': 'Basic Plan',
        'price': 75.00,
        'interval': 'month',
        'value': 'basic',
        'pitch_limit': 5,
        'features': ['1 user', '5 matches/day', 'Basic media guidelines']
    },
    TEAM_PLAN_ID: {
        'name': 'Team Plan',
        'price': 150.00,
        'interval': 'month',
        'value': 'team',
        'pitch_limit': 15,
        'features': ['3 users', '15 matches/day', 'CRM export', 'Enhanced outreach tools']
    },
    ENTERPRISE_PLAN_ID: {
        'name': 'Enterprise Plan',
        'price': 250.00,
        'interval': 'month',
        'value': 'enterprise',
        'pitch_limit': float('inf'),
        'features': ['Unlimited users', 'Unlimited matches', 'Priority support', 'Premium insights']
    }
}

@payment_bp.route('/create-checkout-session', methods=['POST', 'OPTIONS'])
def create_checkout_session():
    # Handle OPTIONS request
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        print("Starting checkout session creation...")
        data = request.get_json()
        print("Received request data:", data)
        
        email = data.get('email')
        plan_type = data.get('planType')
        password = data.get('password')
        
        print("Extracted data - Email:", email, "Plan Type:", plan_type, "Password:", password)

        if not email or not plan_type or not password:
            print("Missing required fields")
            return jsonify({'error': 'Email, password and plan type are required'}), 400

        # Map plan type to price ID
        price_id = {
            'basic': BASIC_PLAN_ID,
            'team': TEAM_PLAN_ID,
            'enterprise': ENTERPRISE_PLAN_ID
        }.get(plan_type)

        # Validate price ID
        if not price_id or price_id not in SUBSCRIPTION_PLANS:
            print(f"Invalid plan type: {plan_type}")
            return jsonify({'error': 'Invalid plan type'}), 400

        # Create Stripe customer
        print("Creating Stripe customer...")
        customer = stripe.Customer.create(
            email=email,
            metadata={
                "email": email,
                "password": password,
                "price_id": price_id
            }
        )
        print("Stripe customer created:", customer)

        # Create Stripe checkout session
        print("Creating Stripe checkout session...")
        session = stripe.checkout.Session.create(   
            customer=customer.id,
            payment_method_types=['card'],
            line_items=[{
                'price': price_id,
                'quantity': 1,
            }],
            mode='subscription',
            success_url="http://localhost:3002/payment-success?session_id={CHECKOUT_SESSION_ID}",
            cancel_url="http://localhost:3002/signup/agencies",
            metadata={
                "email": email,
                "password": password,
                "price_id": price_id
            }
        )
        print("Stripe checkout session created:", session)
        
        # Return the session URL for redirection
        return jsonify({
            'sessionId': session.id,
            'url': session.url
        }), 200

    except Exception as e:
        print("Error in create_checkout_session:", str(e))
        return jsonify({'error': str(e)}), 500

@payment_bp.route('/verify', methods=['POST', 'OPTIONS'])
def verify_payment():
    # Handle OPTIONS request
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        data = request.get_json()
        session_id = data.get('sessionId')
        
        if not session_id:
            return jsonify({'error': 'Session ID is required'}), 400

        # Retrieve the checkout session
        checkout_session = stripe.checkout.Session.retrieve(session_id)
        
        if checkout_session.payment_status != 'paid':
            return jsonify({'error': 'Payment not completed'}), 400

        # Get user details from session metadata
        email = checkout_session.metadata.get('email')
        password = checkout_session.metadata.get('password')
        price_id = checkout_session.metadata.get('price_id')

        # Check if user already exists
        existing_user = User.get_by_email(email)
        if existing_user:
            return jsonify({'error': 'User already exists'}), 400

        # Create new user
        user = User.create(email, password)
        if not user:
            return jsonify({'error': 'Failed to create user'}), 500

        # Activate user and set Stripe customer ID
        updated_user = user.update({
            'is_active': True,
            'stripe_customer_id': checkout_session.customer
        })

        if not updated_user:
            return jsonify({'error': 'Failed to update user'}), 500

        # Get plan details
        plan_details = SUBSCRIPTION_PLANS[price_id]

        # Create subscription record
        subscription_data = {
            'user_id': user.id,
            'plan_id': price_id,
            'stripe_subscription_id': checkout_session.subscription,
            'status': 'active',
            'current_period_start': datetime.fromtimestamp(checkout_session.created).isoformat(),
            'current_period_end': datetime.fromtimestamp(checkout_session.created + 30*24*60*60).isoformat(),  # 30 days
            'pitch_limit': plan_details['pitch_limit'],
            'features': plan_details['features'],
            'user_limit': 1 if plan_details['value'] == 'basic' else (3 if plan_details['value'] == 'team' else float('inf')),
            'has_crm_export': plan_details['value'] in ['team', 'enterprise'],
            'has_enhanced_outreach': plan_details['value'] in ['team', 'enterprise'],
            'has_priority_support': plan_details['value'] == 'enterprise',
            'has_premium_insights': plan_details['value'] == 'enterprise'
        }

        subscription = Subscription.create(subscription_data)
        if not subscription:
            return jsonify({'error': 'Failed to create subscription'}), 500

        return jsonify({
            'success': True,
            'user': updated_user.to_dict(),
            'subscription': subscription.to_dict()
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@payment_bp.route('/webhook', methods=['POST'])
def stripe_webhook():
    try:
        event = None
        payload = request.data
        sig_header = request.headers.get('Stripe-Signature')

        try:
            event = stripe.Webhook.construct_event(
                payload, sig_header, os.getenv('STRIPE_WEBHOOK_SECRET')
            )
        except ValueError as e:
            return jsonify({'error': 'Invalid payload'}), 400
        except stripe.error.SignatureVerificationError as e:
            return jsonify({'error': 'Invalid signature'}), 400

        # Handle specific events
        if event['type'] == 'customer.subscription.updated':
            subscription = event['data']['object']
            handle_subscription_updated(subscription)
        elif event['type'] == 'customer.subscription.deleted':
            subscription = event['data']['object']
            handle_subscription_deleted(subscription)

        return jsonify({'success': True}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def handle_subscription_updated(stripe_subscription):
    try:
        # Find subscription in database
        subscription = Subscription.get_by_stripe_id(stripe_subscription.id)
        if subscription:
            subscription.update({
                'status': stripe_subscription.status,
                'current_period_end': datetime.fromtimestamp(stripe_subscription.current_period_end).isoformat()
            })

    except Exception as e:
        print(f"Error handling subscription update: {str(e)}")

def handle_subscription_deleted(stripe_subscription):
    try:
        # Find subscription in database
        subscription = Subscription.get_by_stripe_id(stripe_subscription.id)
        if subscription:
            # Update subscription status
            subscription.update({
                'status': 'cancelled'
            })

            # Deactivate user
            user = User.get_by_email(subscription.user_id)
            if user:
                user.update({
                    'is_active': False
                })

    except Exception as e:
        print(f"Error handling subscription deletion: {str(e)}") 