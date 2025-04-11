from datetime import datetime
from .base import Database

class Subscription:
    def __init__(self, data: dict = None):
        self.id = data.get('id') if data else None
        self.user_id = data.get('user_id') if data else None
        self.plan_id = data.get('plan_id') if data else None
        self.stripe_subscription_id = data.get('stripe_subscription_id')
        self.status = data.get('status', 'active')
        self.current_period_start = data.get('current_period_start')
        self.current_period_end = data.get('current_period_end')
        self.pitch_limit = data.get('pitch_limit')
        self.features = data.get('features', [])
        self.user_limit = data.get('user_limit', 1)  # Default to 1 user
        self.has_crm_export = data.get('has_crm_export', False)
        self.has_enhanced_outreach = data.get('has_enhanced_outreach', False)
        self.has_priority_support = data.get('has_priority_support', False)
        self.has_premium_insights = data.get('has_premium_insights', False)
        self.created_at = data.get('created_at')
        self.updated_at = data.get('updated_at')

    @classmethod
    def create(cls, data: dict):
        """Create a new subscription."""
        try:
            data.update({
                'created_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat()
            })
            response = Database.get_client().table('subscriptions').insert(data).execute()
            if response.data:
                return cls(response.data[0])
            return None
        except Exception as e:
            print(f"Error creating subscription: {str(e)}")
            return None

    @classmethod
    def get_by_user_id(cls, user_id: str):
        """Get subscription by user ID."""
        try:
            response = Database.get_client().table('subscriptions').select('*').eq('user_id', user_id).eq('status', 'active').execute()
            if response.data:
                return cls(response.data[0])
            return None
        except Exception as e:
            print(f"Error getting subscription: {str(e)}")
            return None

    @classmethod
    def get_by_stripe_id(cls, stripe_subscription_id: str):
        """Get subscription by Stripe subscription ID."""
        try:
            response = Database.get_client().table('subscriptions').select('*').eq('stripe_subscription_id', stripe_subscription_id).execute()
            if response.data:
                return cls(response.data[0])
            return None
        except Exception as e:
            print(f"Error getting subscription: {str(e)}")
            return None

    def update(self, data: dict):
        """Update subscription data."""
        try:
            data['updated_at'] = datetime.utcnow().isoformat()
            response = Database.get_client().table('subscriptions').update(data).eq('id', self.id).execute()
            if response.data:
                updated_data = response.data[0]
                self.status = updated_data.get('status', self.status)
                self.current_period_end = updated_data.get('current_period_end', self.current_period_end)
                self.updated_at = updated_data.get('updated_at', self.updated_at)
                return self
            return None
        except Exception as e:
            print(f"Error updating subscription: {str(e)}")
            return None

    def to_dict(self):
        """Convert subscription object to dictionary."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'plan_id': self.plan_id,
            'stripe_subscription_id': self.stripe_subscription_id,
            'status': self.status,
            'current_period_start': self.current_period_start,
            'current_period_end': self.current_period_end,
            'pitch_limit': self.pitch_limit,
            'features': self.features,
            'user_limit': self.user_limit,
            'has_crm_export': self.has_crm_export,
            'has_enhanced_outreach': self.has_enhanced_outreach,
            'has_priority_support': self.has_priority_support,
            'has_premium_insights': self.has_premium_insights,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }

    def is_active(self):
        """Check if subscription is active."""
        if self.status != 'active':
            return False
        current_time = datetime.utcnow().isoformat()
        return current_time <= self.current_period_end

    def days_remaining(self):
        """Calculate days remaining in current period."""
        if not self.current_period_end:
            return 0
        end_date = datetime.fromisoformat(self.current_period_end.replace('Z', '+00:00'))
        delta = end_date - datetime.utcnow()
        return max(0, delta.days)

    def can_create_pitch(self, current_pitch_count: int):
        """Check if user can create more pitches under this subscription."""
        if not self.is_active():
            return False
        if self.pitch_limit == float('inf'):
            return True
        return current_pitch_count < self.pitch_limit 