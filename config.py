import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - only used on incomplete local runtimes.
    load_dotenv = None


if os.getenv("MEDIA_ART_LOAD_DOTENV", "1") != "0" and load_dotenv is not None:
    load_dotenv()


ROOT = Path(__file__).resolve().parent

DEVICE = int(os.getenv("AUDIO_DEVICE", "0"))
CHANNELS = int(os.getenv("AUDIO_CHANNELS", "1"))
RATE = int(os.getenv("AUDIO_RATE", "16000"))
CHUNK = int(os.getenv("AUDIO_CHUNK", "1024"))
THRESHOLD = float(os.getenv("AUDIO_THRESHOLD", "400"))
SILENCE_LIMIT = float(os.getenv("AUDIO_SILENCE_LIMIT", "0.5"))

ARCHIVE_DIR = os.getenv("ARCHIVE_DIR", "wav_archive")

OSC_IP = os.getenv("OSC_IP", "127.0.0.1")
OSC_PORT = int(os.getenv("OSC_PORT", "5000"))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
