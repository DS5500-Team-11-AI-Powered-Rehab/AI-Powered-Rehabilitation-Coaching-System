"""
tts.py — Simple Text-to-Speech Utility

Usage:
    python tts.py

Or import:
    from tts import speak, save_audio
"""

from TTS.api import TTS
import os

# Optional playback
try:
    import sounddevice as sd

    PLAY_AVAILABLE = True
except:
    PLAY_AVAILABLE = False


# ── Config ────────────────────────────────────────────────────────────────

MODEL_NAME = "tts_models/en/ljspeech/tacotron2-DDC"
USE_CUDA = False  # set True if you want GPU


# ── Initialize model (singleton) ──────────────────────────────────────────

_tts_model = None


def get_tts_model():
    global _tts_model

    if _tts_model is None:
        print("Loading TTS model...")
        _tts_model = TTS(model_name=MODEL_NAME, progress_bar=False)

        if USE_CUDA:
            _tts_model = _tts_model.to("cuda")

    return _tts_model


# ── Core functions ────────────────────────────────────────────────────────


def speak(text: str):
    """
    Generate speech and play it immediately.
    """
    tts = get_tts_model()
    audio = tts.tts(text)

    if PLAY_AVAILABLE:
        sd.play(audio, samplerate=tts.synthesizer.output_sample_rate)
        sd.wait()
    else:
        print("Playback not available (install sounddevice).")


def save_audio(text: str, file_path: str = "output.wav", to_mp3: bool = False):
    """
    Save speech to file.
    """
    tts = get_tts_model()

    # Save WAV
    tts.tts_to_file(text=text, file_path=file_path)
    print(f"Saved WAV to {file_path}")

    # Optional MP3 conversion
    if to_mp3:
        mp3_path = file_path.replace(".wav", ".mp3")
        os.system(f"ffmpeg -y -i {file_path} {mp3_path}")
        print(f"Converted to MP3: {mp3_path}")


# ── Demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    demo_text = """
    Great job on your exercise!
    Try to keep your knees aligned with your toes.
    You're making good progress, keep it up!
    """

    print("▶ Playing demo speech...")
    speak(demo_text)

    print("▶ Saving audio...")
    save_audio(demo_text, "demo_output.wav", to_mp3=True)
