import logging
import os
import tempfile
import sys
from multiprocessing import freeze_support

import librosa
import tensorflow_hub as hub
from transformers import pipeline
from voiceauthCore.utils import convert_to_wav, get_file_metadata
from voiceauthCore.database import save_metadata, init_db
from transformers import AutoImageProcessor, AutoModelForImageClassification
from io import BytesIO
from PIL import Image
import requests
import torch

freeze_support()
# Load ML models
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
vggish_model = hub.load("https://www.kaggle.com/models/google/vggish/TensorFlow2/vggish/1")
pipe = pipeline("audio-classification", model="alexandreacff/wav2vec2-large-ft-fake-detection")
pipe2 = pipeline("audio-classification", model="WpythonW/ast-fakeaudio-detector")
pipe3 = pipeline("audio-classification", model="alexandreacff/sew-ft-fake-detection")

# Determine base path (handles both PyInstaller frozen & normal script execution)
if getattr(sys, "frozen", False):
    base_path = os.path.join(tempfile.gettempdir(), "voiceauthCore")  # Temp directory for frozen app
else:
    base_path = os.path.join(os.getcwd(), "voiceauthCore")  # Local directory for regular execution

# Create temp directory once (if it doesn't exist)
os.makedirs(base_path, exist_ok=True)

# Set temp_dir and ensure it's not repeatedly deleted
temp_dir = base_path
ffmpeg_path = os.path.join(base_path, "ffmpeg")

if os.path.exists(ffmpeg_path):
    os.environ["PATH"] += os.pathsep + ffmpeg_path + "ffmpeg.exe"

# Set Librosa cache directory
librosa_cache_dir = os.path.join(tempfile.gettempdir(), "librosa")
os.makedirs(librosa_cache_dir, exist_ok=True)  # Ensure it exists
os.environ["LIBROSA_CACHE_DIR"] = librosa_cache_dir


def setup_logging(log_filename: str = "audio_detection.log") -> None:
    """Sets up logging to both file and console."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename, mode="a"),
            logging.StreamHandler(),
        ],
    )


setup_logging()  # Call it early in the script
logging.info("App starting...")

# Load the pre-trained model and processor
processor = AutoImageProcessor.from_pretrained("dima806/deepfake_vs_real_image_detection")
model = AutoModelForImageClassification.from_pretrained("dima806/deepfake_vs_real_image_detection")


def analyze_image(image_url):
    response = requests.get(image_url)
    image_bytes = response.content

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        return predicted_class == 0  # Return True if predicted as real, False otherwise


def analyze_audio(file_path):
    """Analyzes an audio file using all three models and aggregates results."""
    try:
        wav_path = convert_to_wav(file_path)
        label1, confidence1 = predict_yamnet(wav_path)
        label2, confidence2 = predict_vggish(wav_path)
        label3, confidence3 = predict_hf(wav_path)
        label4, confidence4 = predict_hf2(wav_path)
        label5, confidence5 = predict_rf(wav_path)

        results = {
            "YAMNet": {"label": label1, "confidence": confidence1},
            "VGGish": {"label": label2, "confidence": confidence2},
            "HuggingFace Model 1": {"label": label3, "confidence": confidence3},
            "HuggingFace Model 2": {"label": label4, "confidence": confidence4},
            "R-Forest": {"label": label5, "confidence": confidence5},
        }

        save_metadata(file_path, results, max(confidence1, confidence2, confidence3, confidence4, confidence5))
        return results
    except Exception as e:
        return {"error": str(e)}


def predict_yamnet(file_path):
    audio, sr = librosa.load(file_path, sr=16000, mono=True)
    scores, embeddings, spectrogram = yamnet_model(audio)
    inferred_class = scores.numpy().mean(axis=0).argmax()
    return inferred_class, scores.numpy().mean()


def predict_vggish(file_path):
    audio, sr = librosa.load(file_path, sr=16000, mono=True)
    embeddings = vggish_model(audio)
    return "VGGish_Features", embeddings.numpy().mean()


def predict_hf(file_path):
    audio_data, sr = librosa.load(file_path, sr=16000)
    prediction = pipe(audio_data)
    return prediction[0]["label"], prediction[0]["score"]


def predict_hf2(file_path):
    audio_data, sr = librosa.load(file_path, sr=16000)
    prediction = pipe2(audio_data)
    return prediction[0]["label"], prediction[0]["score"]


def predict_rf(file_path):
    audio_data, sr = librosa.load(file_path, sr=16000)
    prediction = pipe3(audio_data)
    return prediction[0]["label"], prediction[0]["score"]


if __name__ == "__main__":
    import sys

    file_path = sys.argv[1]
    print(analyze_audio(file_path))
    print(analyze_image(file_path))

