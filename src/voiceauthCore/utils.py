import os
import platform
import subprocess
import threading
import time
import tempfile
import sys
import logging
from typing import Optional, Tuple, Dict, Any, Union

import soundfile as sf
import librosa
import numpy as np
from matplotlib import pyplot as plt
from pydub import AudioSegment

# Configure logging
logger = logging.getLogger(__name__)

def get_safe_temp_dir():
    """Get a safe temporary directory for the application."""
    if getattr(sys, "frozen", False):
        temp_base = os.path.join(tempfile.gettempdir(), "voiceauth_temp")
    else:
        temp_base = os.path.join(os.getcwd(), "temp")

    os.makedirs(temp_base, exist_ok=True)
    return temp_base

def safe_open_file(file_path: str) -> bool:
    """Safely open a file using the system's default application."""
    try:
        if not os.path.exists(file_path):
            logger.warning(f"File does not exist: {file_path}")
            return False

        system = platform.system()
        if system == "Windows":
            os.startfile(file_path)
        elif system == "Darwin":  # macOS
            subprocess.run(["open", file_path], check=True)
        else:  # Linux and other Unix-like systems
            subprocess.run(["xdg-open", file_path], check=True)

        return True

    except Exception as e:
        logger.error(f"Error opening file {file_path}: {e}")
        return False

def convert_to_wav(file_path: str, target_sr: int = 16000) -> str:
    """
    Converts audio files to WAV format with specified sample rate.

    Args:
        file_path: Path to the input audio file
        target_sr: Target sample rate (default: 16000 Hz for models)

    Returns:
        str: Path to the converted WAV file
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")

        # Generate a unique temporary WAV file path
        temp_dir = get_safe_temp_dir()
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        temp_wav_path = os.path.join(temp_dir, f"{base_name}_{os.urandom(8).hex()}.wav")

        # Check if the file is already WAV with correct sample rate
        if file_path.lower().endswith('.wav'):
            try:
                with sf.SoundFile(file_path) as audio_file:
                    if audio_file.samplerate == target_sr:
                        logger.info(f"File already in correct WAV format: {file_path}")
                        return file_path
            except Exception:
                pass  # Continue with conversion if reading fails

        # Convert using pydub
        try:
            audio = AudioSegment.from_file(file_path)
            # Set sample rate and mono
            audio = audio.set_frame_rate(target_sr).set_channels(1)
            audio.export(temp_wav_path, format="wav")

            logger.info(f"Successfully converted {file_path} to {temp_wav_path}")
            return temp_wav_path

        except Exception as e:
            # Fallback to librosa + soundfile
            logger.warning(f"Pydub conversion failed, trying librosa: {e}")
            audio_data, sr = librosa.load(file_path, sr=target_sr, mono=True)
            sf.write(temp_wav_path, audio_data, target_sr)

            logger.info(f"Successfully converted using librosa: {file_path} to {temp_wav_path}")
            return temp_wav_path

    except Exception as e:
        logger.error(f"Error converting file {file_path} to WAV: {e}")
        raise

def typewriter_effect(text_widget, text: str, typing_speed: float = 0.009) -> None:
    """
    Create a typewriter effect in a text widget.

    Args:
        text_widget: Text widget that supports delete and insert methods
        text: Text to display with typewriter effect
        typing_speed: Delay between characters in seconds
    """
    try:
        if not hasattr(text_widget, "delete") or not hasattr(text_widget, "insert"):
            logger.warning("Text widget doesn't support required methods for typewriter effect")
            # Fallback: just insert the text normally
            if hasattr(text_widget, "insert"):
                text_widget.insert("end", text)
            return

        def type_char(index: int = 0):
            """Recursively type each character with delay."""
            if index <= len(text):
                try:
                    text_widget.delete("1.0", "end")
                    text_widget.insert("end", text[:index])
                    if hasattr(text_widget, "yview"):
                        text_widget.yview("end")
                    if hasattr(text_widget, "update"):
                        text_widget.update()

                    if index < len(text):
                        # Schedule next character
                        threading.Timer(typing_speed, lambda: type_char(index + 1)).start()

                except Exception as e:
                    logger.error(f"Error in typewriter effect: {e}")
                    # Fallback: insert remaining text
                    try:
                        text_widget.delete("1.0", "end")
                        text_widget.insert("end", text)
                    except Exception:
                        pass

        # Start the typing effect
        type_char()

    except Exception as e:
        logger.error(f"Error setting up typewriter effect: {e}")
        # Fallback: insert text normally
        try:
            if hasattr(text_widget, "insert"):
                text_widget.insert("end", text)
        except Exception:
            pass

def get_score_label(prediction_result: Union[bool, str, float]) -> str:
    """
    Convert prediction result to human-readable label.

    Args:
        prediction_result: Can be boolean (True=fake, False=real),
                         string ("fake"/"real"), or confidence score

    Returns:
        str: Human-readable confidence label
    """
    try:
        # Handle boolean values (True = fake, False = real)
        if isinstance(prediction_result, bool):
            return "Likely Fake - Use Caution" if prediction_result else "Likely Real - High Confidence"

        # Handle string values
        if isinstance(prediction_result, str):
            label_lower = prediction_result.lower()
            if "fake" in label_lower:
                return "Likely Fake - Use Caution"
            elif "real" in label_lower or "authentic" in label_lower:
                return "Likely Real - High Confidence"
            else:
                return f"Uncertain Result: {prediction_result}"

        # Handle numeric confidence scores (0.0 to 1.0)
        if isinstance(prediction_result, (int, float)):
            confidence = float(prediction_result)

            if confidence > 0.95:
                return "Very High Confidence - Almost Certainly Real"
            elif confidence > 0.85:
                return "High Confidence - Likely Real"
            elif confidence > 0.70:
                return "Moderate Confidence - Probably Real"
            elif confidence > 0.55:
                return "Low Confidence - Uncertain Result"
            elif confidence > 0.30:
                return "Moderate Confidence - Probably Fake"
            elif confidence > 0.15:
                return "High Confidence - Likely Fake"
            else:
                return "Very High Confidence - Almost Certainly Fake"

        return f"Unknown result format: {prediction_result}"

    except Exception as e:
        logger.error(f"Error processing score label: {e}")
        return "Error processing result"

def visualize_mfcc(file_path: str, output_dir: Optional[str] = None) -> Optional[str]:
    """
    Generate and save MFCC features visualization.

    Args:
        file_path: Path to the audio file
        output_dir: Directory to save the visualization (optional)

    Returns:
        str: Path to the saved visualization file, or None if failed
    """
    try:
        if not os.path.exists(file_path):
            logger.error(f"Audio file not found: {file_path}")
            return None

        # Load audio data
        audio_data, sr = librosa.load(file_path, sr=None)

        if len(audio_data) == 0:
            logger.error("Empty audio data")
            return None

        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)

        # Create visualization
        plt.figure(figsize=(12, 6))
        plt.imshow(mfccs, aspect="auto", origin="lower", cmap="coolwarm")
        plt.title(f"MFCC Features - {os.path.basename(file_path)}")
        plt.ylabel("MFCC Coefficients")
        plt.xlabel("Time Frames")
        plt.colorbar(format="%+2.0f dB", label="Amplitude (dB)")
        plt.tight_layout()

        # Determine output path
        if output_dir is None:
            output_dir = get_safe_temp_dir()
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, f"mfcc_features_{os.urandom(4).hex()}.png")

        # Save the plot
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()  # Important: close the figure to free memory

        logger.info(f"MFCC visualization saved: {output_file}")

        # Attempt to open the file
        safe_open_file(output_file)

        return output_file

    except Exception as e:
        logger.error(f"Error creating MFCC visualization: {e}")
        return None

def create_mel_spectrogram(file_path: str, output_dir: Optional[str] = None) -> Optional[str]:
    """
    Generate and save mel spectrogram visualization.

    Args:
        file_path: Path to the audio file
        output_dir: Directory to save the visualization (optional)

    Returns:
        str: Path to the saved visualization file, or None if failed
    """
    try:
        if not os.path.exists(file_path):
            logger.error(f"Audio file not found: {file_path}")
            return None

        # Load audio data
        audio_data, sr = librosa.load(file_path, sr=None)

        if len(audio_data) == 0:
            logger.error("Empty audio data")
            return None

        # Generate mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # Create visualization
        plt.figure(figsize=(12, 8))
        librosa.display.specshow(
            log_mel_spectrogram,
            sr=sr,
            x_axis="time",
            y_axis="mel",
            cmap="inferno"
        )
        plt.colorbar(format="%+2.0f dB", label="Power (dB)")
        plt.title(f"Mel Spectrogram - {os.path.basename(file_path)}")
        plt.xlabel("Time (s)")
        plt.ylabel("Mel Frequency")
        plt.tight_layout()

        # Determine output path
        if output_dir is None:
            output_dir = get_safe_temp_dir()
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, f"mel_spectrogram_{os.urandom(4).hex()}.png")

        # Save the plot
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()  # Important: close the figure to free memory

        logger.info(f"Mel spectrogram saved: {output_file}")

        # Attempt to open the file
        safe_open_file(output_file)

        return output_file

    except Exception as e:
        logger.error(f"Error creating mel spectrogram: {e}")
        return None

def get_file_metadata(file_path: str) -> Tuple[Optional[str], Optional[float], Optional[float], Optional[float], Optional[Dict[str, Any]]]:
    """
    Extract comprehensive metadata from an audio file.

    Args:
        file_path: Path to the audio file

    Returns:
        Tuple containing:
        - file_format: File format (e.g., "MP3", "WAV")
        - file_size: File size in MB
        - duration: Audio duration in seconds
        - bitrate: Estimated bitrate in kbps
        - additional_metadata: Dictionary with channels, sample rate, etc.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Get file size
        file_size_bytes = os.path.getsize(file_path)
        file_size_mb = file_size_bytes / (1024 * 1024)

        # Get audio properties using librosa
        try:
            audio_data, sr = librosa.load(file_path, sr=None)
            duration = librosa.get_duration(y=audio_data, sr=sr)
        except Exception as e:
            logger.warning(f"Librosa failed to load {file_path}: {e}")
            # Fallback: try to get duration from soundfile
            try:
                with sf.SoundFile(file_path) as f:
                    duration = len(f) / f.samplerate
                    sr = f.samplerate
            except Exception as e2:
                logger.error(f"Could not determine duration: {e2}")
                return None, None, None, None, None

        # Get file format
        file_format = os.path.splitext(file_path)[-1].replace(".", "").upper()

        # Calculate estimated bitrate
        bitrate_kbps = (file_size_bytes * 8) / (duration * 1000) if duration > 0 else 0

        # Get additional metadata using soundfile
        additional_metadata = {}
        try:
            with sf.SoundFile(file_path) as audio_file:
                additional_metadata = {
                    "channels": audio_file.channels,
                    "samplerate": audio_file.samplerate,
                    "subtype": audio_file.subtype,
                    "format": audio_file.format,
                    "frames": len(audio_file)
                }
        except Exception as e:
            logger.warning(f"Could not extract detailed metadata: {e}")
            additional_metadata = {
                "channels": "Unknown",
                "samplerate": sr if 'sr' in locals() else "Unknown",
                "subtype": "Unknown",
                "format": file_format,
                "frames": "Unknown"
            }

        logger.info(f"Successfully extracted metadata for {file_path}")
        return file_format, file_size_mb, duration, bitrate_kbps, additional_metadata

    except FileNotFoundError as e:
        logger.error(str(e))
        return None, None, None, None, None
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return None, None, None, None, None

def cleanup_temp_files(temp_dir: Optional[str] = None, max_age_hours: int = 24) -> int:
    """
    Clean up old temporary files.

    Args:
        temp_dir: Directory to clean (defaults to app temp dir)
        max_age_hours: Remove files older than this many hours

    Returns:
        int: Number of files removed
    """
    try:
        if temp_dir is None:
            temp_dir = get_safe_temp_dir()

        if not os.path.exists(temp_dir):
            return 0

        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        removed_count = 0

        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)

            try:
                if os.path.isfile(file_path):
                    file_age = current_time - os.path.getmtime(file_path)
                    if file_age > max_age_seconds:
                        os.remove(file_path)
                        removed_count += 1
                        logger.debug(f"Removed old temp file: {file_path}")
            except Exception as e:
                logger.warning(f"Could not remove temp file {file_path}: {e}")

        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} temporary files")

        return removed_count

    except Exception as e:
        logger.error(f"Error during temp file cleanup: {e}")
        return 0

# Initialize cleanup on module import
try:
    cleanup_temp_files()
except Exception:
    pass  # Don't fail module loading if cleanup fails
