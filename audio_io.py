import os
import time
import wave
from datetime import datetime

import numpy as np

from audio_processing import analyze_audio_volume
from config import ARCHIVE_DIR, CHANNELS, CHUNK, DEVICE, RATE, SILENCE_LIMIT, THRESHOLD


class SoundDeviceFacade:
    def __init__(self):
        self._module = None

    def _load(self):
        if self._module is None:
            import sounddevice

            self._module = sounddevice
        return self._module

    def query_devices(self):
        return self._load().query_devices()

    def InputStream(self, *args, **kwargs):
        return self._load().InputStream(*args, **kwargs)


sd = SoundDeviceFacade()


def record_audio():
    print("\nWaiting for microphone input...")

    frames = []
    recording = False
    silence_start_time = None

    stream = sd.InputStream(device=DEVICE, samplerate=RATE, channels=CHANNELS, dtype="int16", blocksize=CHUNK)
    stream.start()

    try:
        while True:
            data, _overflowed = stream.read(CHUNK)
            volume = analyze_audio_volume(data)

            if volume > THRESHOLD:
                if not recording:
                    print(f"Sound detected. Recording started. volume={volume:.2f}")
                    recording = True
                frames.append(data.copy())
                silence_start_time = None
            elif recording:
                frames.append(data.copy())
                if silence_start_time is None:
                    silence_start_time = time.time()
                elif time.time() - silence_start_time > SILENCE_LIMIT:
                    print("Recording stopped.")
                    break
    finally:
        stream.stop()
        stream.close()

    if not frames:
        return None

    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(ARCHIVE_DIR, f"audio_{timestamp}.wav")

    with wave.open(filepath, "wb") as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(2)
        wav_file.setframerate(RATE)
        wav_file.writeframes(np.concatenate(frames, axis=0).tobytes())

    return filepath


def list_audio_input_devices(selected_device=DEVICE):
    devices = []
    for index, device in enumerate(sd.query_devices()):
        max_input_channels = int(device.get("max_input_channels", 0))
        if max_input_channels <= 0:
            continue
        devices.append(
            {
                "index": index,
                "name": device.get("name", f"device {index}"),
                "maxInputChannels": max_input_channels,
                "defaultSamplerate": device.get("default_samplerate"),
                "selected": index == selected_device,
            }
        )
    return {"selected": selected_device, "devices": devices}


def probe_audio_input_device(device=None, duration=0.5):
    selected = DEVICE if device is None else int(device)
    safe_duration = min(max(float(duration), 0.1), 2.0)
    frames_needed = max(1, int(RATE * safe_duration))
    chunks = []
    frames_read = 0

    stream = sd.InputStream(
        device=selected,
        samplerate=RATE,
        channels=CHANNELS,
        dtype="int16",
        blocksize=CHUNK,
    )
    try:
        stream.start()
        while frames_read < frames_needed:
            data, _overflowed = stream.read(min(CHUNK, frames_needed - frames_read))
            chunks.append(data)
            frames_read += len(data)
    finally:
        stream.stop()
        stream.close()

    if not chunks:
        return {"device": selected, "durationSeconds": safe_duration, "rms": 0.0, "peak": 0.0}

    audio = np.concatenate(chunks, axis=0)
    samples = audio.astype(np.float32)
    rms = float(np.sqrt(np.mean(samples * samples)))
    peak = float(np.max(np.abs(samples)))
    return {
        "device": selected,
        "durationSeconds": safe_duration,
        "frames": int(frames_read),
        "rms": rms,
        "peak": peak,
    }


def manage_archive_limit(archive_dir, max_files=20):
    try:
        files = [os.path.join(archive_dir, filename) for filename in os.listdir(archive_dir) if filename.endswith(".wav")]
        if len(files) <= max_files:
            return

        files.sort(key=os.path.getmtime)
        for filepath in files[: len(files) - max_files]:
            os.remove(filepath)
            print(f"[archive] removed old wav file: {filepath}")
    except Exception as exc:
        print(f"Archive cleanup failed: {exc}")
