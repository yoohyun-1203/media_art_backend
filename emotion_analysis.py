import concurrent.futures
import json
import os
import time

from ai_clients import GEMINI_MODEL, gemini_valence_config, get_gemini_client, get_openai_client
from audio_processing import analyze_arousal
from mood_meter import clamp, map_touchdesigner_values, word_from_values
from osc_sender import OSC_IP, OSC_PORT, send_emotion_osc


def estimate_valence_confidence(transcript, valence):
    clean_text = (transcript or "").strip()
    if not clean_text:
        return 0.0
    length_score = clamp(len(clean_text) / 16.0, 0.0, 1.0)
    polarity_score = clamp(abs(valence), 0.0, 1.0)
    return clamp(0.15 + (length_score * 0.55) + (polarity_score * 0.30), 0.0, 1.0)


def extract_json(text):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("JSON object not found in Gemini response")
    return json.loads(text[start : end + 1])


def get_ai_emotion(transcript, arousal=0.0):
    start_time = time.time()
    prompt = f"""
Analyze the emotional valence of this Korean utterance.
Return only JSON. Valence must be a number from -1.0 to 1.0.

Utterance: {transcript}
Voice-based arousal hint: {arousal:.3f}

Format:
{{"valence": 0.0}}
"""

    try:
        response = get_gemini_client().models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=gemini_valence_config(),
        )
        result = extract_json(response.text or "{}")
        valence = clamp(result.get("valence", 0.0))
    except RuntimeError as exc:
        if str(exc) == "GEMINI_API_KEY is required":
            raise
        print(f"Gemini emotion analysis failed: {exc}")
        valence = 0.0
    except Exception as exc:
        print(f"Gemini emotion analysis failed: {exc}")
        valence = 0.0

    emotion_word = word_from_values(valence, arousal)
    elapsed_time = time.time() - start_time
    print(f"Gemini emotion analysis took {elapsed_time:.3f}s (valence={valence:.2f})")
    return emotion_word, valence


def analyze_text_result(text, audio_arousal=0.0, send_osc=True):
    emotion_word, valence = get_ai_emotion(text, arousal=audio_arousal)
    color_name, td_valence, td_arousal = map_touchdesigner_values(emotion_word, audio_arousal)

    result = {
        "ok": True,
        "transcript": text,
        "emotion_word": emotion_word,
        "valence": valence,
        "audio_arousal": audio_arousal,
        "color_name": color_name,
        "td_valence": td_valence,
        "td_arousal": td_arousal,
        "osc_ip": OSC_IP,
        "osc_port": OSC_PORT,
    }

    if send_osc:
        send_emotion_osc(emotion_word, color_name, td_valence, td_arousal, text)
        result["osc_sent"] = True
    else:
        result["osc_sent"] = False

    return result


def process_audio_result(filepath, send_osc=True):
    absolute_path = os.path.abspath(filepath)
    print(f"\n[1/3] Running speech-to-text for: {absolute_path}")

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as arousal_executor:
            arousal_future = arousal_executor.submit(analyze_arousal, absolute_path)

            prompt_hint = "안녕? 난 지금 기분이 아주 좋아. 넌 어때? 우울하거나 슬프진 않아? 정말 짜증나고 화가 나. 너무 신기하고 재미있다!"
            with open(absolute_path, "rb") as audio_file:
                transcript = get_openai_client().audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="ko",
                    prompt=prompt_hint,
                ).text.strip()

            print(f"Recognized text: {transcript}")

            if not transcript:
                return {"ok": False, "error": "No recognized text.", "filepath": absolute_path}

            print("[2/3 & 3/3] Running text emotion and audio arousal analysis...")
            audio_arousal = arousal_future.result()

        result = analyze_text_result(transcript, audio_arousal=audio_arousal, send_osc=False)
        print("\n==================================================")
        print(f"Text: {transcript}")
        print(f"Emotion word: {result['emotion_word']} -> color: {result['color_name']}")
        print(f"TouchDesigner values: valence={result['td_valence']}, arousal={result['td_arousal']}")
        print("==================================================")

        if send_osc:
            send_emotion_osc(
                result["emotion_word"],
                result["color_name"],
                result["td_valence"],
                result["td_arousal"],
                transcript,
            )
            result["osc_sent"] = True

        result["filepath"] = absolute_path
        return result

    except Exception as exc:
        print(f"Audio processing failed: {exc}")
        return {"ok": False, "error": str(exc), "filepath": absolute_path}


def process_audio(filepath):
    return bool(process_audio_result(filepath).get("ok"))
