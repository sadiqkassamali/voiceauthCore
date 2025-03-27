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
import params as yamnet_params
freeze_support()

yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
vggish_model = hub.load("https://www.kaggle.com/models/google/vggish/TensorFlow2/vggish/1")
pipe = pipeline("audio-classification", model="alexandreacff/wav2vec2-large-ft-fake-detection")
pipe2 = pipeline("audio-classification", model="WpythonW/ast-fakeaudio-detector")
pipe3 = pipeline("audio-classification", model="alexandreacff/sew-ft-fake-detection")


if getattr(sys, "frozen", False):
    base_path = os.path.join(tempfile.gettempdir(), "voiceauthCore")
else:
    base_path = os.path.join(os.getcwd(), "voiceauthCore")


os.makedirs(base_path, exist_ok=True)


temp_dir = base_path
ffmpeg_path = os.path.join(base_path, "ffmpeg")

if os.path.exists(ffmpeg_path):
    os.environ["PATH"] += os.pathsep + ffmpeg_path + "ffmpeg.exe"


librosa_cache_dir = os.path.join(tempfile.gettempdir(), "librosa")
os.makedirs(librosa_cache_dir, exist_ok=True)  
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


setup_logging()
logging.info("App starting...")

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
        return predicted_class == 0

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


def class_names_from_csv(class_map_csv_text):
    """Parses class names from YAMNet class map CSV."""
    return [line.split(',')[0] for line in class_map_csv_text.splitlines() if line]

def predict_yamnet(file_path):
    try:

        audio, sr = librosa.load(file_path, sr=16000, mono=True)

        outputs = yamnet_model(audio)

        scores, embeddings, spectrogram = outputs

        scores_np = scores.numpy()

        if scores_np.size == 0:
            raise ValueError("YAMNet model returned empty scores.")

        inferred_class_idx = np.mean(scores_np, axis=0).argmax()

        class_map_csv_bytes = tf.io.read_file(yamnet_model.class_map_path('yamnet_class_map.csv'))
        class_map_text = class_map_csv_bytes.numpy().decode('utf-8')
        class_names = class_names_from_csv(class_map_text)
        if inferred_class_idx >= len(class_names):
            raise IndexError(f"Inferred class index {inferred_class_idx} is out of range. Total classes: {len(class_names)}")

        inferred_class_name = class_names[inferred_class_idx]
        confidence = np.mean(scores_np)

        wav_data, sr = sf.read(outputs, dtype=np.int16)
        waveform = wav_data / 32768.0

        # The graph is designed for a sampling rate of 16 kHz, but higher rates should work too.
        # We also generate scores at a 10 Hz frame rate.
        params = yamnet_params.Params(sample_rate=sr, patch_hop_seconds=0.1)
        print("Sample rate =", params.sample_rate)

        yamnet = yamnet_model.yamnet_frames_model(params)
        yamnet.load_weights('yamnet.h5')

        # Run the model.
        scores, embeddings, spectrogram = yamnet(waveform)
        scores = scores.numpy()
        spectrogram = spectrogram.numpy()

        # Visualize the results.
        plt.figure(figsize=(10, 8))

        # Plot the waveform.
        plt.subplot(3, 1, 1)
        plt.plot(waveform)
        plt.xlim([0, len(waveform)])
        # Plot the log-mel spectrogram (returned by the model).
        plt.subplot(3, 1, 2)
        plt.imshow(spectrogram.T, aspect='auto', interpolation='nearest', origin='bottom')

        # Plot and label the model output scores for the top-scoring classes.
        mean_scores = np.mean(scores, axis=0)
        top_N = 10
        top_class_indices = np.argsort(mean_scores)[::-1][:top_N]
        plt.subplot(3, 1, 3)
        plt.imshow(scores[:, top_class_indices].T, aspect='auto', interpolation='nearest', cmap='gray_r')
        # Compensate for the patch_window_seconds (0.96s) context window to align with spectrogram.
        patch_padding = (params.patch_window_seconds / 2) / params.patch_hop_seconds
        plt.xlim([-patch_padding, scores.shape[0] + patch_padding])
        # Label the top_N classes.
        yticks = range(0, top_N, 1)
        plt.yticks(yticks, [class_names[top_class_indices[x]] for x in yticks])
        _ = plt.ylim(-0.5 + np.array([top_N, 0]))
        return inferred_class_idx, inferred_class_name, confidence

    except Exception as e:
        print(f"Error in predict_yamnet: {e}")
        return None, "Unknown", 0.0


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


def visualize_embeddings_tsne(file_path, output_path="tsne_visualization.png"):

    embeddings, _ = predict_vggish(file_path)


    if isinstance(embeddings, np.ndarray):
        n_samples = embeddings.shape[0]
    else:
        print("Error: embeddings is not a valid NumPy array.")
        return

    if n_samples <= 1:
        print(
            f"t-SNE cannot be performed with only {n_samples} sample(s). Skipping visualization."
        )

        plt.figure(figsize=(10, 6))
        plt.text(
            0.5,
            0.5,
            "Not enough samples for t-SNE",
            fontsize=12,
            ha="center")
        plt.title("t-SNE Visualization of Audio Embeddings")
        plt.savefig(output_path)
        plt.close()
        os.startfile(output_path)
        return

    perplexity = min(30, n_samples - 1)
    perplexity = max(5.0, perplexity)


    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    reduced_embeddings = tsne.fit_transform(embeddings)


    plt.figure(figsize=(10, 6))
    plt.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        c="blue",
        alpha=0.7,
        edgecolors="k",
    )
    plt.title("t-SNE Visualization of Audio Embeddings")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()


    plt.savefig(output_path)
    plt.close()


    if sys.platform.system() == "Windows":
        os.startfile(output_path)
    elif sys.platform.system() == "Darwin":
        subprocess.run(["open", output_path], check=True)
    else:
        subprocess.run(["xdg-open", output_path], check=True)

if __name__ == "__main__":
    import sys
    file_path = sys.argv[1]
    print(analyze_audio(file_path))

