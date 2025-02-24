import os
import librosa
from pydub import AudioSegment

def convert_to_wav(file_path):
    """Converts audio files to WAV format."""
    temp_wav_path = file_path.replace(os.path.splitext(file_path)[-1], ".wav")
    audio = AudioSegment.from_file(file_path)
    audio.export(temp_wav_path, format="wav")
    return temp_wav_path

def get_file_metadata(file_path):
    """Returns metadata about an audio file."""
    y, sr = librosa.load(file_path, sr=None)
    return {"sample_rate": sr, "duration": librosa.get_duration(y=y, sr=sr)}