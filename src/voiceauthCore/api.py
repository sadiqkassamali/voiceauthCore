import base64

from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
from core import analyze_audio, analyze_image

app = Flask(__name__)

# Configurable API keys (can be moved to a database later)
VALID_API_KEYS = {"your-secure-api-key"}
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(get_remote_address, app=app, default_limits=["100 per minute"])
def require_api_key(f):
    """Decorator to check for a valid Base64-encoded API key with 128 raw text length."""
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get("x-api-key")

        if not api_key:
            return jsonify({"error": "Missing API key"}), 403

        try:
            decoded_key = base64.b64decode(api_key).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            return jsonify({"error": "Invalid API key format"}), 403

        if len(decoded_key) != 128:
            return jsonify({"error": "Invalid API key length"}), 403

        return f(*args, **kwargs)

    return decorated_function


@app.route("/analyze-audio", methods=["POST"])
@require_api_key
def analyze_audio_api():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty file name"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    try:
        audio_results = analyze_audio(file_path)
    except Exception as e:
        os.remove(file_path)
        return jsonify({"error": str(e)}), 500

    os.remove(file_path)

    # Process combined decision
    fake_count = sum(1 for result in audio_results.values() if "Fake" in result[0])
    real_count = sum(1 for result in audio_results.values() if "Real" in result[0])

    confidence = max(result[1] for result in audio_results.values()) if audio_results else 0.0
    final_label = "Fake" if fake_count > real_count else "Real" if real_count > fake_count else "Uncertain"

    combined_result = {
        "label": final_label,
        "confidence": confidence,
        "reasoning": f"{fake_count} out of {len(audio_results)} models classified as {final_label}"
    }

    return jsonify({
        "individual_results": audio_results,
        "combined_result": combined_result
    })


@app.route("/analyze-image", methods=["POST"])
@require_api_key
def analyze_image_api():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty file name"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    try:
        image_results = analyze_image(file_path)
    except Exception as e:
        os.remove(file_path)
        return jsonify({"error": str(e)}), 500

    os.remove(file_path)

    # Process combined decision
    fake_count = sum(1 for result in image_results.values() if "Fake" in result[0])
    real_count = sum(1 for result in image_results.values() if "Real" in result[0])

    confidence = max(result[1] for result in image_results.values()) if image_results else 0.0
    final_label = "Fake" if fake_count > real_count else "Real" if real_count > fake_count else "Uncertain"

    combined_result = {
        "label": final_label,
        "confidence": confidence,
        "reasoning": f"{fake_count} out of {len(image_results)} models classified as {final_label}"
    }

    return jsonify({
        "individual_results": image_results,
        "combined_result": combined_result
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
