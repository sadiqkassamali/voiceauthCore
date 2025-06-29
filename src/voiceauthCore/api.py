import base64
import logging
import os
import platform
import shutil
import tempfile
from functools import wraps
from typing import Dict, Any

import psutil
from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

from core import analyze_audio, analyze_image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
app.config.update(
    MAX_CONTENT_LENGTH=50 * 1024 * 1024,  # 50MB max file size
    SECRET_KEY=os.environ.get('SECRET_KEY', 'dev-key-change-in-production'),
    UPLOAD_FOLDER=os.environ.get('UPLOAD_FOLDER', 'uploads'),
    DEBUG=os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
)

# Create upload directory
UPLOAD_FOLDER = app.config['UPLOAD_FOLDER']
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allowed file extensions
ALLOWED_AUDIO_EXTENSIONS = {
    'mp3', 'wav', 'ogg', 'flac', 'aac', 'm4a', 'mp4', 'mov', 'avi', 'mkv', 'webm'
}
ALLOWED_IMAGE_EXTENSIONS = {
    'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp'
}

# API Keys - In production, use environment variables or database
VALID_API_KEYS = {
    os.environ.get('API_KEY', 'your-secure-api-key-change-in-production')
}

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per minute", "500 per hour"]
)

def allowed_file(filename: str, allowed_extensions: set) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in allowed_extensions

def validate_file_content(file_path: str, expected_type: str) -> bool:
    """Basic file content validation."""
    try:
        if expected_type == 'audio':
            # Try to read with librosa to validate audio file
            import librosa
            librosa.load(file_path, duration=1.0)  # Just check first second
            return True
        elif expected_type == 'image':
            # Try to open with PIL to validate image file
            from PIL import Image
            with Image.open(file_path) as img:
                img.verify()
            return True
    except Exception as e:
        logger.warning(f"File validation failed for {file_path}: {e}")
        return False
    return False

def require_api_key(f):
    """Decorator to check for a valid API key."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get("x-api-key") or request.headers.get("X-API-Key")

        if not api_key:
            logger.warning(f"Missing API key from {request.remote_addr}")
            return jsonify({"error": "Missing API key"}), 401

        # Check if it's base64 encoded
        try:
            decoded_key = base64.b64decode(api_key).decode("utf-8")
            if len(decoded_key) != 128:
                return jsonify({"error": "Invalid API key format"}), 401
            # In production, validate against database
            if decoded_key not in VALID_API_KEYS:
                return jsonify({"error": "Invalid API key"}), 401
        except (ValueError, UnicodeDecodeError):
            # Try direct key comparison for non-base64 keys
            if api_key not in VALID_API_KEYS:
                logger.warning(f"Invalid API key from {request.remote_addr}")
                return jsonify({"error": "Invalid API key"}), 401

        return f(*args, **kwargs)
    return decorated_function

def add_security_headers(response):
    """Add security headers to response."""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    return response

@app.after_request
def after_request(response):
    """Add security headers to all responses."""
    return add_security_headers(response)

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({"error": "File too large. Maximum size is 50MB."}), 413

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    """Handle file size limit exceeded."""
    return jsonify({"error": "File size exceeds maximum allowed limit of 50MB"}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {e}")
    return jsonify({"error": "Internal server error"}), 500

def process_audio_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Process audio analysis results into consistent format."""
    if "error" in results:
        return {"error": results["error"]}

    # Count predictions
    fake_count = 0
    real_count = 0
    confidences = []

    for model_name, result in results.items():
        if isinstance(result, dict) and "label" in result:
            label = result["label"]
            confidence = result.get("confidence", 0.0)

            if "fake" in label.lower():
                fake_count += 1
            elif "real" in label.lower() or "authentic" in label.lower():
                real_count += 1

            if isinstance(confidence, (int, float)) and confidence > 0:
                confidences.append(confidence)

    # Determine final result
    total_models = len(results)
    if fake_count > real_count:
        final_label = "Fake"
    elif real_count > fake_count:
        final_label = "Real"
    else:
        final_label = "Uncertain"

    # Calculate confidence
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    return {
        "label": final_label,
        "confidence": round(avg_confidence, 3),
        "reasoning": f"{fake_count} models detected fake, {real_count} models detected real out of {total_models} total models"
    }

def process_image_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Process image analysis results into consistent format."""
    if "error" in results:
        return {"error": results["error"]}

    # Handle single result format
    if "Image Detection" in results:
        result = results["Image Detection"]
        label = result.get("label", "Unknown")
        confidence = result.get("confidence", 0.0)

        return {
            "label": label,
            "confidence": round(confidence, 3),
            "reasoning": f"Image detection model classified as {label}"
        }

    return {"error": "No valid results found"}

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "version": "1.0.0"}), 200

@app.route("/analyze-audio", methods=["POST"])
@limiter.limit("10 per minute")
@require_api_key
def analyze_audio_api():
    """Analyze audio file for deepfake detection."""
    temp_file_path = None

    try:
        # Validate request
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if not file.filename:
            return jsonify({"error": "No file selected"}), 400

        # Validate file extension
        if not allowed_file(file.filename, ALLOWED_AUDIO_EXTENSIONS):
            return jsonify({
                "error": f"Invalid file type. Allowed types: {', '.join(ALLOWED_AUDIO_EXTENSIONS)}"
            }), 400

        # Save file securely
        filename = secure_filename(file.filename)
        temp_file_path = os.path.join(UPLOAD_FOLDER, f"{os.urandom(16).hex()}_{filename}")
        file.save(temp_file_path)

        # Validate file content
        if not validate_file_content(temp_file_path, 'audio'):
            return jsonify({"error": "Invalid audio file content"}), 400

        logger.info(f"Processing audio file: {filename}")

        # Analyze audio
        audio_results = analyze_audio(temp_file_path)

        if not audio_results:
            return jsonify({"error": "Analysis failed - no results returned"}), 500

        # Process results
        combined_result = process_audio_results(audio_results)

        response_data = {
            "individual_results": audio_results,
            "combined_result": combined_result,
            "file_info": {
                "filename": filename,
                "file_size": os.path.getsize(temp_file_path)
            }
        }

        logger.info(f"Audio analysis completed for: {filename}")
        return jsonify(response_data), 200

    except Exception as e:
        logger.error(f"Error in audio analysis: {str(e)}")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {temp_file_path}: {e}")

@app.route("/analyze-image", methods=["POST"])
@limiter.limit("10 per minute")
@require_api_key
def analyze_image_api():
    """Analyze image file for deepfake detection."""
    temp_file_path = None

    try:
        # Validate request
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if not file.filename:
            return jsonify({"error": "No file selected"}), 400

        # Validate file extension
        if not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
            return jsonify({
                "error": f"Invalid file type. Allowed types: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}"
            }), 400

        # Save file securely
        filename = secure_filename(file.filename)
        temp_file_path = os.path.join(UPLOAD_FOLDER, f"{os.urandom(16).hex()}_{filename}")
        file.save(temp_file_path)

        # Validate file content
        if not validate_file_content(temp_file_path, 'image'):
            return jsonify({"error": "Invalid image file content"}), 400

        logger.info(f"Processing image file: {filename}")

        # Analyze image
        image_results = analyze_image(temp_file_path)

        if not image_results:
            return jsonify({"error": "Analysis failed - no results returned"}), 500

        # Process results
        combined_result = process_image_results(image_results)

        response_data = {
            "individual_results": image_results,
            "combined_result": combined_result,
            "file_info": {
                "filename": filename,
                "file_size": os.path.getsize(temp_file_path)
            }
        }

        logger.info(f"Image analysis completed for: {filename}")
        return jsonify(response_data), 200

    except Exception as e:
        logger.error(f"Error in image analysis: {str(e)}")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {temp_file_path}: {e}")

def display_system_info():
    """Display system information on startup."""
    try:
        system_info = {
            "Machine Name": platform.node(),
            "OS": platform.system() + " " + platform.release(),
            "Processor": platform.processor(),
            "CPU Cores": psutil.cpu_count(logical=True),
            "Total RAM (GB)": round(psutil.virtual_memory().total / (1024 ** 3), 2),
            "Free RAM (GB)": round(psutil.virtual_memory().available / (1024 ** 3), 2),
            "Disk Space (GB)": round(shutil.disk_usage("/").free / (1024 ** 3), 2),
            "Home Directory": os.path.expanduser("~"),
            "Current Working Directory": os.getcwd(),
        }
    except Exception:
        # Fallback for systems where some info might not be available
        system_info = {
            "Machine Name": platform.node(),
            "OS": platform.system(),
            "Python Version": platform.python_version(),
        }

    ascii_banner = """
██╗   ██╗ ██████╗ ██╗███████╗███████╗ █████╗ ████╗   ██╗████████╗██╗  ██╗
██║   ██║██╔═══██╗██║██╔════╝██╔════╝██╔══██╗██║   ██║╚══██╔══╝██║  ██║
██║   ██║██║   ██║██║██║     █████╗  ███████║██║   ██║   ██║   ███████║
╚██╗ ██╔╝██║   ██║██║██║     ██╔══╝  ██╔══██║██║   ██║   ██║   ██╔══██║
 ╚████╔╝ ╚██████╔╝██║╚██████╗███████╗██║  ██║╚██████╔╝   ██║   ██║  ██║
  ╚═══╝   ╚═════╝ ╚═╝ ╚═════╝╚══════╝╚═╝  ╚═╝ ╚═════╝    ╚═╝   ╚═╝  ╚═╝
                    
            🔊 VoiceAuth Core API Server 🔊
            
            Available Endpoints:
            📊 GET  /health         - Health check
            🎵 POST /analyze-audio  - Audio deepfake detection  
            🖼️  POST /analyze-image  - Image deepfake detection
            
            Security: API Key Required (x-api-key header)
            Rate Limit: 10 requests/minute per endpoint
    """

    print(ascii_banner)
    print("=" * 80)
    for key, value in system_info.items():
        print(f"{key}: {value}")
    print("=" * 80)
    print(f"🚀 Server starting on http://0.0.0.0:5000")
    print("=" * 80)

if __name__ == "__main__":
    display_system_info()

    # Production deployment should use a proper WSGI server like Gunicorn
    if app.config['DEBUG']:
        app.run(host="0.0.0.0", port=5000, debug=True)
    else:
        app.run(host="0.0.0.0", port=5000, debug=False)
