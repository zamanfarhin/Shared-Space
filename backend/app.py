from flask import Flask, request, jsonify
from flask_cors import CORS 
import os
import json

app = Flask(__name__)
CORS(app)

@app.route("/generate", methods=["GET"])
def generate():
    name = request.args.get("name")
    if not name:
        return jsonify({"error": "No name provided"}), 400

    filename = f"{name.lower()}.json"
    filepath = os.path.join("aesthetic_profiles", filename)

    if not os.path.exists(filepath):
        return jsonify({"error": "Profile not found"}), 404

    with open(filepath, "r") as f:
        data = json.load(f)

    return jsonify(data)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use Render-assigned port if available
    app.run(host="0.0.0.0", port=port, debug=False)


