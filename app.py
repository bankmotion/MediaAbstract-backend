import os
import sys
from dotenv import load_dotenv
from werkzeug.urls import url_quote

from logging.handlers import RotatingFileHandler

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables first
load_dotenv()

from flask import Flask, jsonify
from flask_cors import CORS
from routes.pitch_routes import pitch_routes
from routes.auth_routes import auth_bp
from routes.payment_routes import payment_bp
from routes.reminder_routes import reminder_routes
from config import HOST, PORT

app = Flask(__name__)
CORS(app)

app.register_blueprint(pitch_routes)
app.register_blueprint(auth_bp, url_prefix="/api")
app.register_blueprint(payment_bp)
app.register_blueprint(reminder_routes)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Flask is connected to Supabase!"})

if __name__ == "__main__":
    # host = "146.190.131.130"  # Use the server's IP address directly
    host = "127.0.0.1"
    print(f"Starting server on host: {host}")
    app.run(host=host, port=PORT, debug=True, use_reloader=False)

    # app.run(host=HOST, debug=True, use_reloader=False)
