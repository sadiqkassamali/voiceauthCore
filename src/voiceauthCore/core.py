import logging
import os
import subprocess
import tempfile
import sys
from multiprocessing import freeze_support
import librosa
import numpy as np
import tensorflow_hub as hub
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from transformers import pipeline
from voiceauthCore.utils import convert_to_wav, get_file_metadata
from voiceauthCore.database import save_metadata, init_db
from transformers import AutoImageProcessor, AutoModelForImageClassification
from io import BytesIO
from PIL import Image
import requests
import torch
import tensorflow.compat.v1 as tf
import soundfile as sf
import uuid
import hashlib

freeze_support()

# Initialize models with error handling
try:
    yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
except Exception as e:
    logging.warning(f"Failed to load YAMNet model: {e}")
    yamnet_model = None

try:
    vggish_model = hub.load("https://www.kaggle.com/models/google/vggish/TensorFlow2/vggish/1")
except Exception as e:
    logging.warning(f"Failed to load VGGish model: {e}")
    vggish_model = None

try:
    pipe = pipeline("audio-classification", model="alexandreacff/wav2vec2-large-ft-fake-detection")
except Exception as e:
    logging.warning(f"Failed to load HF model 1: {e}")
    pipe = None

try:
    pipe2 = pipeline("audio-classification", model="WpythonW/ast-fakeaudio-detector")
except Exception as e:
    logging.warning(f"Failed to load HF model 2: {e}")
    pipe2 = None

try:
    pipe3 = pipeline("audio-classification", model="alexandreacff/sew-ft-fake-detection")
except Exception as e:
    logging.warning(f"Failed to load HF model 3: {e}")
    pipe3 = None

# Path setup
if getattr(sys, "frozen", False):
    base_path = os.path.join(tempfile.gettempdir(), "voiceauthCore")
else:
    base_path = os.path.join(os.getcwd(), "voiceauthCore")

os.makedirs(base_path, exist_ok=True)
temp_dir = base_path

# FFmpeg path setup
ffmpeg_path = os.path.join(base_path, "ffmpeg")
if os.path.exists(ffmpeg_path):
    os.environ["PATH"] += os.pathsep + ffmpeg_path

# Librosa cache setup
librosa_cache_dir = os.path.join(tempfile.gettempdir(), "librosa")
os.makedirs(librosa_cache_dir, exist_ok=True)
os.environ["LIBROSA_CACHE_DIR"] = librosa_cache_dir

def setup_logging(log_filename: str = "audio_detection.log") -> None:
    """Sets up logging to both file and console."""
    log_file_path = os.path.join(base_path, log_filename)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file_path, mode="a"),
            logging.StreamHandler(),
        ],
    )

setup_logging()
logging.info("VoiceAuthCore starting...")

# Initialize image detection models
try:
    processor = AutoImageProcessor.from_pretrained("dima806/deepfake_vs_real_image_detection")
    model = AutoModelForImageClassification.from_pretrained("dima806/deepfake_vs_real_image_detection")
except Exception as e:
    logging.warning(f"Failed to load image detection models: {e}")
    processor = None
    model = None

def analyze_image(image_path_or_url):
    """Analyzes an image file or URL for deepfake detection."""
    if not processor or not model:
        return {"error": "Image detection models not available"}

    try:
        if image_path_or_url.startswith(('http://', 'https://')):
            response = requests.get(image_path_or_url, timeout=30)
            response.raise_for_status()
            image_bytes = response.content
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        else:
            image = Image.open(image_path_or_url).convert("RGB")

        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt")
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
            confidence = torch.softmax(logits, dim=1).max().item()

            label = "Real" if predicted_class == 0 else "Fake"

            return {
                "Image Detection": {
                    "label": label,
                    "confidence": float(confidence)
                }
            }

    except Exception as e:
        logging.error(f"Error in image analysis: {e}")
        return {"error": str(e)}

def analyze_audio(file_path):
    """Analyzes an audio file using all available models and aggregates results."""
    try:
        # Convert to WAV format
        wav_path = convert_to_wav(file_path)

        results = {}

        # Run all prediction models
        if yamnet_model:
            try:
                label1, confidence1 = predict_yamnet(wav_path)
                results["YAMNet"] = {"label": label1, "confidence": confidence1}
            except Exception as e:
                logging.error(f"YAMNet prediction failed: {e}")
                results["YAMNet"] = {"label": "Error", "confidence": 0.0}

        if vggish_model:
            try:
                label2, confidence2 = predict_vggish(wav_path)
                results["VGGish"] = {"label": label2, "confidence": confidence2}
            except Exception as e:
                logging.error(f"VGGish prediction failed: {e}")
                results["VGGish"] = {"label": "Error", "confidence": 0.0}

        if pipe:
            try:
                label3, confidence3 = predict_hf(wav_path)
                results["HuggingFace Model 1"] = {"label": label3, "confidence": confidence3}
            except Exception as e:
                logging.error(f"HF Model 1 prediction failed: {e}")
                results["HuggingFace Model 1"] = {"label": "Error", "confidence": 0.0}

        if pipe2:
            try:
                label4, confidence4 = predict_hf2(wav_path)
                results["HuggingFace Model 2"] = {"label": label4, "confidence": confidence4}
            except Exception as e:
                logging.error(f"HF Model 2 prediction failed: {e}")
                results["HuggingFace Model 2"] = {"label": "Error", "confidence": 0.0}

        if pipe3:
            try:
                label5, confidence5 = predict_rf(wav_path)
                results["Random Forest"] = {"label": label5, "confidence": confidence5}
            except Exception as e:
                logging.error(f"Random Forest prediction failed: {e}")
                results["Random Forest"] = {"label": "Error", "confidence": 0.0}

        # Calculate combined confidence
        valid_confidences = [
            result["confidence"] for result in results.values()
            if result["confidence"] > 0 and result["label"] != "Error"
        ]
        max_confidence = max(valid_confidences) if valid_confidences else 0.0

        # Save metadata to database
        try:
            file_uuid = str(uuid.uuid4())
            save_metadata(file_uuid, file_path, str(results))
        except Exception as e:
            logging.warning(f"Failed to save metadata: {e}")

        # Clean up temporary WAV file if it was created
        if wav_path != file_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception as e:
                logging.warning(f"Failed to clean up temp file: {e}")

        return results

    except Exception as e:
        logging.error(f"Error in audio analysis: {e}")
        return {"error": str(e)}

def predict_yamnet(file_path):
    """Predict using YAMNet model."""
    if not yamnet_model:
        return "Model Not Available", 0.0

    try:
        # Load audio with proper sampling rate for YAMNet
        audio, sr = librosa.load(file_path, sr=16000, mono=True)

        # Run YAMNet prediction
        scores, embeddings, spectrogram = yamnet_model(audio)
        scores_np = scores.numpy()

        if scores_np.size == 0:
            return "No Audio Detected", 0.0

        # Get the most confident class
        mean_scores = np.mean(scores_np, axis=0)
        top_class_idx = np.argmax(mean_scores)
        confidence = float(mean_scores[top_class_idx])

        # Simple heuristic for fake detection based on audio characteristics
        # This is a simplified approach - in practice you'd want a trained classifier
        speech_related_classes = [0, 1, 2, 3, 24, 25]  # Speech, conversation, singing etc.
        if top_class_idx in speech_related_classes:
            # Higher confidence in speech-related classes might indicate real audio
            label = "Real" if confidence > 0.5 else "Fake"
        else:
            label = "Uncertain"

        return label, confidence

    except Exception as e:
        logging.error(f"Error in YAMNet prediction: {e}")
        return "Error", 0.0

def predict_vggish(file_path):
    """Predict using VGGish model."""
    if not vggish_model:
        return "Model Not Available", 0.0

    try:
        audio, sr = librosa.load(file_path, sr=16000, mono=True)
        embeddings = vggish_model(audio)

        # Simple analysis of embeddings
        # In practice, you'd train a classifier on top of these embeddings
        embedding_stats = np.mean(embeddings.numpy())
        confidence = float(abs(embedding_stats))

        # Simple heuristic based on embedding characteristics
        label = "Real" if embedding_stats > 0 else "Fake"

        return label, min(confidence, 1.0)

    except Exception as e:
        logging.error(f"Error in VGGish prediction: {e}")
        return "Error", 0.0

def predict_hf(file_path):
    """Predict using first HuggingFace model."""
    if not pipe:
        return "Model Not Available", 0.0

    try:
        audio_data, sr = librosa.load(file_path, sr=16000)
        prediction = pipe(audio_data)

        if isinstance(prediction, list) and len(prediction) > 0:
            result = prediction[0]
            label = result.get("label", "Unknown")
            confidence = float(result.get("score", 0.0))

            # Normalize label format
            if "fake" in label.lower():
                label = "Fake"
            elif "real" in label.lower() or "authentic" in label.lower():
                label = "Real"

            return label, confidence
        else:
            return "No Prediction", 0.0

    except Exception as e:
        logging.error(f"Error in HF model 1 prediction: {e}")
        return "Error", 0.0

def predict_hf2(file_path):
    """Predict using second HuggingFace model."""
    if not pipe2:
        return "Model Not Available", 0.0

    try:
        audio_data, sr = librosa.load(file_path, sr=16000)
        prediction = pipe2(audio_data)

        if isinstance(prediction, list) and len(prediction) > 0:
            result = prediction[0]
            label = result.get("label", "Unknown")
            confidence = float(result.get("score", 0.0))

            # Normalize label format
            if "fake" in label.lower():
                label = "Fake"
            elif "real" in label.lower() or "authentic" in label.lower():
                label = "Real"

            return label, confidence
        else:
            return "No Prediction", 0.0

    except Exception as e:
        logging.error(f"Error in HF model 2 prediction: {e}")
        return "Error", 0.0

def predict_rf(file_path):
    """Predict using Random Forest model (third HuggingFace model)."""
    if not pipe3:
        return "Model Not Available", 0.0

    try:
        audio_data, sr = librosa.load(file_path, sr=16000)
        prediction = pipe3(audio_data)

        if isinstance(prediction, list) and len(prediction) > 0:
            result = prediction[0]
            label = result.get("label", "Unknown")
            confidence = float(result.get("score", 0.0))

            # Normalize label format
            if "fake" in label.lower():
                label = "Fake"
            elif "real" in label.lower() or "authentic" in label.lower():
                label = "Real"

            return label, confidence
        else:
            return "No Prediction", 0.0

    except Exception as e:
        logging.error(f"Error in Random Forest prediction: {e}")
        return "Error", 0.0

def visualize_embeddings_tsne(file_path, output_path="tsne_visualization.png"):
    """Create t-SNE visualization of audio embeddings."""
    try:
        if not vggish_model:
            logging.warning("VGGish model not available for visualization")
            return

        # Get embeddings
        audio, sr = librosa.load(file_path, sr=16000, mono=True)
        embeddings = vggish_model(audio)
        embeddings_np = embeddings.numpy()

        if embeddings_np.ndim == 1:
            embeddings_np = embeddings_np.reshape(1, -1)

        n_samples = embeddings_np.shape[0]

        if n_samples <= 1:
            logging.info(f"Not enough samples ({n_samples}) for t-SNE visualization")
            # Create simple plot indicating insufficient data
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "Insufficient data for t-SNE\n(Need multiple time segments)",
                     fontsize=14, ha="center", va="center", transform=plt.gca().transAxes)
            plt.title("t-SNE Visualization of Audio Embeddings")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            return

        # Perform t-SNE
        perplexity = min(30, n_samples - 1)
        perplexity = max(5.0, perplexity)

        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        reduced_embeddings = tsne.fit_transform(embeddings_np)

        # Create visualization
        plt.figure(figsize=(10, 6))
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1],
                    c="blue", alpha=0.7, edgecolors="k", s=50)
        plt.title("t-SNE Visualization of Audio Embeddings")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Open the file based on platform
        try:
            if sys.platform == "win32":
                os.startfile(output_path)
            elif sys.platform == "darwin":
                subprocess.run(["open", output_path], check=True)
            else:
                subprocess.run(["xdg-open", output_path], check=True)
        except Exception as e:
            logging.warning(f"Could not open visualization file: {e}")

    except Exception as e:
        logging.error(f"Error creating t-SNE visualization: {e}")

def main():
    """Main function for command line usage."""
    if len(sys.argv) < 2:
        print("Usage: python core.py <audio_file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)

    print(f"Analyzing: {file_path}")
    results = analyze_audio(file_path)

    if "error" in results:
        print(f"Error: {results['error']}")
    else:
        print("Analysis Results:")
        for model_name, result in results.items():
            print(f"  {model_name}: {result['label']} (Confidence: {result['confidence']:.3f})")

if __name__ == "__main__":
    main()
