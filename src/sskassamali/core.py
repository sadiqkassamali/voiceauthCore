import os
import librosa
import tensorflow_hub as hub
from transformers import pipeline
from src.sskassamali.utils import convert_to_wav, get_file_metadata
from src.sskassamali.database import save_metadata, init_db

# Load ML models
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
vggish_model = hub.load("https://www.kaggle.com/models/google/vggish/TensorFlow2/vggish/1")
pipe = pipeline("audio-classification", model="alexandreacff/wav2vec2-large-ft-fake-detection")
pipe2 = pipeline("audio-classification", model="WpythonW/ast-fakeaudio-detector")
pipe3 = pipeline("audio-classification", model="alexandreacff/sew-ft-fake-detection")

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