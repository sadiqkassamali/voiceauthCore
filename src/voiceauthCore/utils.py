import os
import platform
import subprocess
import threading
import soundfile as sf
import librosa
import numpy as np
from matplotlib import pyplot as plt
from pydub import AudioSegment

def convert_to_wav(file_path):
    """Converts audio files to WAV format."""
    temp_wav_path = file_path.replace(os.path.splitext(file_path)[-1], ".wav")
    audio = AudioSegment.from_file(file_path)
    audio.export(temp_wav_path, format="wav")
    return temp_wav_path

def typewriter_effect(text_widget, text, typing_speed=0.009):
    if hasattr(text_widget, "delete") and hasattr(text_widget, "insert"):

        for i in range(len(text) + 1):
            text_widget.delete("1.0", "end")

            text_widget.insert("end", text[:i])
            text_widget.yview("end")
            text_widget.update()
            threading.Event().wait(
                typing_speed
            )
    else:
        pass


def get_score_label(confidence):
    if confidence is None or not isinstance(confidence, (int, float)):
        return "Invalid confidence value"

    if confidence > 0.90:
        return "Almost certainly real"
    elif confidence > 0.80:
        return "Probably real but with slight doubt"
    elif confidence > 0.65:
        return "High likelihood of being fake, use caution"
    else:
        return "Considered fake: quality of audio does matter, do check for false positive just in case.."


def visualize_mfcc(temp_file_path):
    """Function to visualize MFCC features."""

    audio_data, sr = librosa.load(temp_file_path, sr=None)

    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)

    plt.figure(figsize=(10, 4))
    plt.imshow(mfccs, aspect="auto", origin="lower", cmap="coolwarm")
    plt.title("MFCC Features")
    plt.ylabel("MFCC Coefficients")
    plt.xlabel("Time Frames")
    plt.colorbar(format="%+2.0f dB")

    plt.tight_layout()
    plt_file_path = os.path.join(
        os.path.dirname(temp_file_path),
        "mfccfeatures.png")
    plt.savefig(plt_file_path)

    if platform.system() == "Windows":
        os.startfile(plt_file_path)
    elif platform.system() == "Darwin":  # macOS
        subprocess.run(["open", plt_file_path], check=True)
    else:  # Linux/Unix
        subprocess.run(["xdg-open", plt_file_path], check=True)


def create_mel_spectrogram(temp_file_path):
    audio_file = os.path.join(temp_file_path)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)

    y, sr = librosa.load(audio_file)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    librosa.display.specshow(
        log_mel_spectrogram, sr=sr, x_axis="time", y_axis="mel", cmap="inferno"
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel Spectrogram")
    plt.tight_layout()
    plt.savefig("melspectrogram.png")
    mel_file_path = os.path.join(
        os.path.dirname(temp_file_path),
        "melspectrogram.png")
    plt.savefig(mel_file_path)
    if platform.system() == "Windows":
        os.startfile(mel_file_path)
    elif platform.system() == "Darwin":  # macOS
        subprocess.run(["open", mel_file_path], check=True)
    else:  # Linux/Unix
        subprocess.run(["xdg-open", mel_file_path], check=True)



def get_file_metadata(file_path):
    """Returns detailed metadata about an audio file."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Get file size
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert bytes to MB

        # Load audio
        y, sr = librosa.load(file_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)

        # Get file format
        file_format = os.path.splitext(file_path)[-1].replace(".", "").upper()  # Extract extension (e.g., WAV, MP3)

        # Calculate bitrate (approximate for non-lossy formats)
        bitrate = (file_size * 8) / duration if duration > 0 else 0  # Convert MB to megabits

        # Extract additional metadata if needed
        with sf.SoundFile(file_path) as audio_file:
            additional_metadata = {
                "channels": audio_file.channels,
                "samplerate": audio_file.samplerate,
                "subtype": audio_file.subtype
            }

        return file_format, file_size, duration, bitrate, additional_metadata

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None, None, None, None, None  # Ensure five values are always returned

