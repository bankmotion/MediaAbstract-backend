import os
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

from flask import Flask, jsonify
from flask_cors import CORS
from routes.pitch_routes import pitch_routes

app = Flask(__name__)
CORS(app)

app.register_blueprint(pitch_routes)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Flask is connected to Supabase!"})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000, debug=True, use_reloader=False)