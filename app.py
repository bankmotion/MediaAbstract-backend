import os
from dotenv import load_dotenv
# from werkzeug.urls import url_quote
from werkzeug.urls import unquote

# Load environment variables first
load_dotenv()

from flask import Flask, jsonify
from flask_cors import CORS
from routes.pitch_routes import pitch_routes
from config import BASE_URL, PORT

app = Flask(__name__)
CORS(app)

app.register_blueprint(pitch_routes)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Flask is connected to Supabase!"})


if __name__ == "__main__":
    app.run(host=BASE_URL, port=PORT, debug=True, use_reloader=False)