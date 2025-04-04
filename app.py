import os
from dotenv import load_dotenv
from werkzeug.urls import url_quote
# from werkzeug.urls import unquote

# Load environment variables first
load_dotenv()

from flask import Flask, jsonify
from flask_cors import CORS
from routes.pitch_routes import pitch_routes
from routes.auth import auth_bp
from config import HOST, PORT

app = Flask(__name__)
CORS(app)

app.register_blueprint(pitch_routes)
app.register_blueprint(auth_bp, url_prefix="/api")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Flask is connected to Supabase!"})


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=True, use_reloader=False)