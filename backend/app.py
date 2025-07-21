from flask import Flask, request, jsonify
from flask_cors import CORS 
import os
import json

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def root():
    return "✅ Shared Space backend is live!"

@app.route("/generate", methods=["GET"])
def generate():
    name = request.args.get("name")
    if not name:
        return jsonify({"error": "No name provided"}), 400

    filename = f"{name.lower()}.json"

    # Get absolute path to the JSON file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(base_dir, "aesthetic_profiles", filename)

    # Log filepath for debugging
    print(f"Looking for file at: {filepath}")

    if not os.path.exists(filepath):
        return jsonify({"error": f"Profile not found at {filepath}"}), 404

    with open(filepath, "r") as f:
        data = json.load(f)

    return jsonify(data)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use Render-assigned port if available
    app.run(host="0.0.0.0", port=port, debug=False)


