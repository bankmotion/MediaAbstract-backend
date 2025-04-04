class User:
    def __init__(self, email, password, plan):
        self.email = email
        self.password = password
        self.plan = plan
        self.id = None

    def save_to_db(self):
        pass

users_db = []