import os
import sys

import numpy as np

import ai_clients
import emotion_analysis as _emotion_analysis
from ai_clients import GEMINI_MODEL, genai
from audio_io import manage_archive_limit, record_audio, sd
from audio_processing import (
    analyze_arousal,
    analyze_audio_volume,
    compute_dual_live_audio_features,
    compute_live_audio_features,
    should_collect_live_segment,
)
from config import ARCHIVE_DIR, CHANNELS, CHUNK, DEVICE, LEFT_DEVICE, LOCAL_SER_BACKEND, LOCAL_SER_MODEL_ID, NOISE_GATE_DB, OSC_IP, OSC_PORT, RATE, RIGHT_DEVICE, SILENCE_LIMIT, THRESHOLD
from mood_meter import MOOD_METER_GRID, clamp, find_word_in_grid, map_touchdesigner_values
from osc_sender import osc_client, send_emotion_osc, send_live_osc


gemini_client = None
openai_client = None


if sys.platform == "win32":
    import io

    if (sys.stdout.encoding or "").lower().replace("-", "") != "utf8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
    if (sys.stderr.encoding or "").lower().replace("-", "") != "utf8":
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", line_buffering=True)


def _sync_ai_cache_to_module():
    ai_clients.gemini_client = gemini_client
    ai_clients.openai_client = openai_client


def _sync_ai_cache_from_module():
    global gemini_client, openai_client
    gemini_client = ai_clients.gemini_client
    openai_client = ai_clients.openai_client


def get_gemini_client():
    _sync_ai_cache_to_module()
    client = ai_clients.get_gemini_client()
    _sync_ai_cache_from_module()
    return client


def get_openai_client():
    _sync_ai_cache_to_module()
    client = ai_clients.get_openai_client()
    _sync_ai_cache_from_module()
    return client


def extract_json(text):
    return _emotion_analysis.extract_json(text)


def estimate_valence_confidence(transcript, valence):
    return _emotion_analysis.estimate_valence_confidence(transcript, valence)


def get_ai_emotion(transcript, arousal=0.0):
    _sync_ai_cache_to_module()
    result = _emotion_analysis.get_ai_emotion(transcript, arousal=arousal)
    _sync_ai_cache_from_module()
    return result


def analyze_text_result(text, audio_arousal=0.0, send_osc=True):
    _sync_ai_cache_to_module()
    result = _emotion_analysis.analyze_text_result(text, audio_arousal=audio_arousal, send_osc=send_osc)
    _sync_ai_cache_from_module()
    return result


def process_audio_result(filepath, send_osc=True):
    _sync_ai_cache_to_module()
    result = _emotion_analysis.process_audio_result(filepath, send_osc=send_osc)
    _sync_ai_cache_from_module()
    return result


def process_audio(filepath):
    return bool(process_audio_result(filepath).get("ok"))


def main():
    print("=== Media Art Emotion Backend ===")
    test_mode_input = input("Use text mode instead of microphone? (y/n): ").strip().lower()
    is_text_mode = test_mode_input == "y"

    try:
        while True:
            if is_text_mode:
                text = input("\n[text mode] Enter a sentence, or q to quit: ")
                if text.lower() == "q":
                    break

                result = analyze_text_result(text, audio_arousal=0.0, send_osc=True)
                print("\n==================================================")
                print(f"Text: {result['transcript']}")
                print(f"Emotion word: {result['emotion_word']} -> color: {result['color_name']}")
                print(f"TouchDesigner values: valence={result['td_valence']}, arousal={result['td_arousal']}")
                print("==================================================")
                continue

            filepath = record_audio()
            if not filepath:
                continue

            success = process_audio(filepath)
            if success:
                manage_archive_limit(ARCHIVE_DIR, max_files=20)
                continue

            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    print(f"Removed unhelpful recording: {filepath}")
            except OSError:
                pass
    except KeyboardInterrupt:
        print("\nExiting.")

if __name__ == "__main__":
    main()
