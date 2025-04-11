from datetime import datetime
from .base import Database

class User:
    def __init__(self, data: dict = None):
        self.id = data.get('id') if data else None
        self.email = data.get('email') if data else None
        self.password = data.get('password') if data else None
        self.is_active = data.get('is_active', False)
        self.stripe_customer_id = data.get('stripe_customer_id')
        self.created_at = data.get('created_at')
        self.updated_at = data.get('updated_at')

    @classmethod
    def get_by_email(cls, email: str):
        """Get user by email."""
        try:
            response = Database.get_client().table('users').select('*').eq('email', email).execute()
            if response.data:
                return cls(response.data[0])
            return None
        except Exception as e:
            print(f"Error getting user by email: {str(e)}")
            return None

    @classmethod
    def create(cls, email: str, hashed_password: str):
        """Create a new user."""
        try:
            user_data = {
                'email': email,
                'password': hashed_password,
                'is_active': False,
                'created_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat()
            }
            response = Database.get_client().table('users').insert(user_data).execute()
            if response.data:
                return cls(response.data[0])
            return None
        except Exception as e:
            print(f"Error creating user: {str(e)}")
            return None

    def update(self, data: dict):
        """Update user data."""
        try:
            data['updated_at'] = datetime.utcnow().isoformat()
            response = Database.get_client().table('users').update(data).eq('id', self.id).execute()
            if response.data:
                updated_data = response.data[0]
                self.is_active = updated_data.get('is_active', self.is_active)
                self.stripe_customer_id = updated_data.get('stripe_customer_id', self.stripe_customer_id)
                self.updated_at = updated_data.get('updated_at', self.updated_at)
                return self
            return None
        except Exception as e:
            print(f"Error updating user: {str(e)}")
            return None

    def to_dict(self):
        """Convert user object to dictionary."""
        return {
            'id': self.id,
            'email': self.email,
            'is_active': self.is_active,
            'stripe_customer_id': self.stripe_customer_id,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }

    def check_subscription_active(self):
        """Check if user has an active subscription."""
        from .subscription import Subscription
        subscription = Subscription.get_by_user_id(self.id)
        if not subscription:
            return False
        return subscription.is_active()

    def get_subscription(self):
        """Get user's current subscription."""
        from .subscription import Subscription
        return Subscription.get_by_user_id(self.id)

    def get_pitch_limit(self):
        """Get user's pitch limit based on subscription."""
        from .subscription import Subscription
        subscription = Subscription.get_by_user_id(self.id)
        if not subscription:
            return 0
        return subscription.pitch_limit

    def can_create_pitch(self):
        """Check if user can create a new pitch."""
        if not self.is_active or not self.check_subscription_active():
            return False
            
        # For unlimited pitches
        if self.get_pitch_limit() == float('inf'):
            return True
            
        # Check against monthly limit
        from .pitch import Pitch
        current_month_pitches = len([p for p in Pitch.get_by_user_id(self.id) if p.is_current_month()])
        return current_month_pitches < self.get_pitch_limit()

    def save_to_db(self):
        pass

users_db = []