import sounddevice as sd
import wave
import numpy as np
import time
import os
from dotenv import load_dotenv
from datetime import datetime
import json
import warnings


from pythonosc.udp_client import SimpleUDPClient
from google import genai

# -------- API 키 보안 설정 --------
# .env 파일에 숨겨둔 변수들을 불러옵니다.
load_dotenv()

# Gemini API 설정 (발급받은 키를 여기에 입력하세요)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("오류: .env 파일에 GEMINI_API_KEY가 없습니다!")
    exit(1)

# Suppress warnings
warnings.filterwarnings("ignore")

import sys
# Windows 터미널 한글 깨짐 방지
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)

# 1. 로컬 AI 모델 로드 (초기 실행 시 다운로드 필요)
print("로컬 AI 모델을 로드하는 중입니다. (약간의 시간이 소요될 수 있습니다)")
try:
    import whisper
    from transformers import pipeline
    import librosa
    import torch # 얘는 맥북 GPU 쓰려고 import 했어용가리
    import ssl
    
    # macOS SSL 인증서 우회 (모델 다운로드 에러 방지)
    ssl._create_default_https_context = ssl._create_unverified_context

    # Mac의 강력한 GPU(Metal Performance Shaders)를 사용하도록 설정
    device_type = "cpu" 
    print(f"✅ 사용 중인 하드웨어 가속: {device_type} (안정성 최우선)")

    print("정확도가 높은 medium 모델을 로드합니다...")
    # medium 모델을 M3 GPU에 로드 (가속 적용)
    whisper_model = whisper.load_model("medium", device=device_type)
    
    # 생성형 모델 초기화 (속도가 빠른 flash 모델 사용)
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    print("AI 모델 로드 완료!")
except Exception as e:
    print(f"모델 로드 중 오류 발생: {e}")
    exit(1)

# 오디오 설정
DEVICE = 0  # 마이크 디바이스 번호 (0번)
CHANNELS = 1
RATE = 16000
CHUNK = 1024
THRESHOLD = 400  # RMS 임계값 (사람 목소리가 보통 100~300 내외이므로 하향 조정)
SILENCE_LIMIT = 0.7  # 초 (이 시간 동안 조용하면 녹음 종료)

# 폴더 설정
ARCHIVE_DIR = "wav_archive"
os.makedirs(ARCHIVE_DIR, exist_ok=True)

# OSC 설정
OSC_IP = "127.0.0.1"
OSC_PORT = 5000
osc_client = SimpleUDPClient(OSC_IP, OSC_PORT)

def analyze_audio_volume(data):
    """오디오 데이터의 볼륨(RMS)을 계산합니다."""
    rms = np.sqrt(np.mean(np.square(data.astype(np.float32))))
    return rms

def record_audio():
    """VAD를 사용하여 일정 볼륨 이상의 유의미한 소리를 녹음합니다."""
    print("\n마이크 대기 중... (소리를 내면 녹음이 시작됩니다)")
    
    frames = []
    recording = False
    silence_start_time = None

    stream = sd.InputStream(device=DEVICE, samplerate=RATE, channels=CHANNELS, dtype='int16', blocksize=CHUNK)
    stream.start()

    last_debug_time = time.time()

    try:
        while True:
            data, overflowed = stream.read(CHUNK)
            volume = analyze_audio_volume(data)
            
            # 마이크가 고장났는지 파악하기 위해 미녹음 중일 땐 3초마다 현재 볼륨을 보여줌
            # if not recording and time.time() - last_debug_time > 3:
            #     print(f"[디버그] 현재 마이크 입력 볼륨: {volume:.2f} (녹음 시작 임계값: {THRESHOLD})")
            #     last_debug_time = time.time()

            if volume > THRESHOLD:
                if not recording:
                    print(f"소리 감지! 녹음 시작 (현재 볼륨: {volume:.2f})")
                    recording = True
                frames.append(data.copy())
                silence_start_time = None
            elif recording:
                frames.append(data.copy())
                if silence_start_time is None:
                    silence_start_time = time.time()
                elif time.time() - silence_start_time > SILENCE_LIMIT:
                    print("녹음 종료.")
                    break
    finally:
        stream.stop()
        stream.close()

    if frames:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(ARCHIVE_DIR, f"audio_{timestamp}.wav")
        
        wf = wave.open(filepath, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2) # 16-bit
        wf.setframerate(RATE)
        audio_data = np.concatenate(frames, axis=0)
        wf.writeframes(audio_data.tobytes())
        wf.close()
        
        return filepath
    return None

def analyze_arousal(filepath):
    """librosa를 사용하여 어투(Pitch 변화량, RMS 에너지) 기반 Arousal 분석"""
    try:
        y, sr = librosa.load(filepath, sr=16000)
        
        # 1. RMS 에너지 계산
        rms = librosa.feature.rms(y=y)[0]
        mean_rms = np.mean(rms)
        
        # 2. 피치(Pitch) 변화량 계산
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, 
            fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7')
        )
        f0 = f0[voiced_flag] # 음성이 있는 부분만 추출
        
        if len(f0) > 0:
            std_pitch = np.std(f0)
        else:
            std_pitch = 0
            
        # 3. 추가 특징: Zero Crossing Rate (발화의 선명도/속도)
        zcr = librosa.feature.zero_crossing_rate(y=y)[0]
        mean_zcr = np.mean(zcr)
        
        # 4. 추가 특징: Spectral Centroid (목소리의 밝기)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        mean_centroid = np.mean(centroid)
            
        # 5. Arousal 계산 (멀티모달 휴리스틱 매핑)
        # 에너지, 피치 변화량, 발화 속도(ZCR), 밝기(Centroid) 종합
        norm_rms = min(mean_rms / 0.1, 1.0)
        norm_pitch_std = min(std_pitch / 65.0, 1.0)
        norm_zcr = min(mean_zcr / 0.15, 1.0)
        norm_centroid = min(mean_centroid / 3000.0, 1.0)
        
        # 가중치: RMS(40%), Pitch(30%), ZCR(15%), Centroid(15%)
        arousal_raw = (norm_rms * 0.4) + (norm_pitch_std * 0.3) + (norm_zcr * 0.15) + (norm_centroid * 0.15)
        
        # -1.0 ~ 1.0 범위로 매핑
        arousal = (arousal_raw * 2.0) - 1.0
        arousal = max(min(arousal, 1.0), -1.0)
        
        print(f"[어투 분석] 에너지: {mean_rms:.3f}, 피치변화: {std_pitch:.1f}Hz, ZCR: {mean_zcr:.3f}, 밝기: {mean_centroid:.0f}Hz -> Arousal: {arousal:.2f}")
        return arousal
    except Exception as e:
        print(f"어투 분석 중 오류 발생: {e}")
        return 0.0

MOOD_METER_GRID = [
    ["격분한", "공황에 빠진", "스트레스 받는", "초조한", "충격받은", "놀란", "긍정적인", "흥겨운", "아주 신나는", "황홀한"],
    ["격노한", "몹시 화가 난", "좌절한", "신경이 날카로운", "망연자실한", "들뜬", "쾌활한", "동기 부여된", "영감을 받은", "의기양양한"],
    ["화가 치밀어 오른", "겁먹은", "화난", "초조한", "안절부절못하는", "기운이 넘치는", "활발한", "흥분한", "낙관적인", "열광하는"],
    ["불안한", "우려하는", "근심하는", "짜증나는", "거슬리는", "만족스러운", "집중하는", "행복한", "자랑스러운", "짜릿한"],
    ["불쾌한", "골치 아픈", "염려하는", "마음이 불편한", "언짢은", "유쾌한", "기쁜", "희망찬", "재미있는", "더없이 행복한"],
    ["역겨운", "침울한", "실망스러운", "의욕 없는", "냉담한", "속 편한", "태평한", "자족하는", "다정한", "충만한"],
    ["비관적인", "시무룩한", "낙담한", "슬픈", "지루한", "평온한", "안전한", "만족스러운", "감사하는", "감동적인"],
    ["소외된", "비참한", "쓸쓸한", "기죽은", "피곤한", "여유로운", "차분한", "편안한", "축복받은", "안정적인"],
    ["의기소침한", "우울한", "뚱한", "기진맥진한", "지친", "한가로운", "생각에 잠긴", "평화로운", "편한", "근심 걱정 없는"],
    ["절망한", "가망 없는", "고독한", "소모된", "진이 빠진", "나른한", "흐뭇한", "고요한", "안락한", "안온한"]
]

def find_word_in_grid(word):
    for r in range(10):
        for c in range(10):
            if MOOD_METER_GRID[r][c] == word:
                return r, c
    return -1, -1

def process_audio(filepath):
    """로컬 모델을 사용하여 STT, Valence, Arousal 분석 후 OSC 전송"""
    print(f"\n[1/3] 음성 인식(STT) 진행 중... (파일: {filepath})")
    
    try:
        # --- 👇 여기에 경로를 절대 경로로 바꿔주는 한 줄을 추가하세요! 👇 ---
        absolute_path = os.path.abspath(filepath)
        # -------------------------------------------------------------

        # 1. Whisper STT (한국어 지정하여 정확도 향상)
        #result = whisper_model.transcribe(filepath, language="ko")
        # 변경: 감정적인 대화나 구어체 위주로 힌트 제공
        prompt_hint = "안녕? 난 지금 기분이 아주 좋아. 넌 어때? 우울하거나 슬프진 않아? 정말 짜증나고 화가 나. 너무 신기하고 재미있다!"
        
        result = whisper_model.transcribe(
            absolute_path, 
            language="ko",
            initial_prompt=prompt_hint
        )
        transcript = result["text"].strip()
        print(f" -> 인식된 텍스트: {transcript}")

        if not transcript:
            print("인식된 텍스트가 없어 분석을 건너뜁니다.")
            return False

        # 2. 텍스트 기반 감정 단어 분류 및 Valence 분석 (Gemini API 사용)
        print("[2/3] Gemini API 기반 무드 미터(100개 감정) 분류 및 Valence 분석 중...")
        
        prompt = f"""
        다음 문장의 텍스트를 분석하여 사용자의 감정 상태를 파악해줘.
        1. 아래 100개의 감정 단어 중 텍스트의 문맥(비꼬기, 이모티콘 등 포함)과 가장 잘 어울리는 단어 1개를 선택해.
        2. 가장 부정적인 상태를 -1.0, 가장 긍정적인 상태를 1.0으로 하는 긍정/부정(Valence) 수치를 평가해.
        
        응답은 반드시 '감정단어|수치' 형식으로만 출력해 (부연 설명 절대 금지).
        예시: 슬픈|-0.6
        
        [100개의 감정 단어 목록]
        격분한, 공황에 빠진, 스트레스 받는, 초조한, 충격받은, 격노한, 몹시 화가 난, 좌절한, 신경이 날카로운, 망연자실한, 
        화가 치밀어 오른, 겁먹은, 화난, 안절부절못하는, 불안한, 우려하는, 근심하는, 짜증나는, 거슬리는, 불쾌한, 
        골치 아픈, 염려하는, 마음이 불편한, 언짢은, 놀란, 긍정적인, 흥겨운, 아주 신나는, 황홀한, 들뜬, 
        쾌활한, 동기 부여된, 영감을 받은, 의기양양한, 기운이 넘치는, 활발한, 흥분한, 낙관적인, 열광하는, 만족스러운, 
        집중하는, 행복한, 자랑스러운, 짜릿한, 유쾌한, 기쁜, 희망찬, 재미있는, 더없이 행복한, 역겨운, 
        침울한, 실망스러운, 의욕 없는, 냉담한, 비관적인, 시무룩한, 낙담한, 슬픈, 지루한, 소외된, 
        비참한, 쓸쓸한, 기죽은, 피곤한, 의기소침한, 우울한, 뚱한, 기진맥진한, 지친, 절망한, 
        가망 없는, 고독한, 소모된, 진이 빠진, 속 편한, 태평한, 자족하는, 다정한, 충만한, 평온한, 
        안전한, 감사하는, 감동적인, 여유로운, 차분한, 편안한, 축복받은, 안정적인, 한가로운, 생각에 잠긴, 
        평화로운, 편한, 근심 걱정 없는, 나른한, 흐뭇한, 고요한, 안락한, 안온한
        
        텍스트: "{transcript}"
        """
        
        emotion_word = "분석 불가"
        valence = 0.0
        
        try:
            response = gemini_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            response_text = response.text.strip()
            print(f" -> 제미나이 원본 응답: {response_text}")
            
            if "|" in response_text:
                parts = response_text.split("|")
                emotion_word = parts[0].strip()
                valence_str = parts[1].strip()
                valence = float(''.join(c for c in valence_str if c.isdigit() or c in '.-'))
            else:
                valence = float(''.join(c for c in response_text if c.isdigit() or c in '.-'))
                
            valence = max(min(valence, 1.0), -1.0)
        except Exception as e:
            print(f" -> Gemini API 오류 또는 파싱 실패 (기본값 적용): {e}")

        # 3. 음향 기반 Arousal 분석 (참고용)
        print("[3/3] 오디오 기반 어투/흥분도(Arousal) 분석 중...")
        audio_arousal = analyze_arousal(filepath)

        # 4. 무드 미터 그리드에서 정확한 사분면 및 색상 찾기
        row, col = find_word_in_grid(emotion_word)
        
        if row != -1 and col != -1:
            # 사분면에 따라 터치디자이너가 보라색 등 섞인 색을 내지 않도록 극단값(스냅) 부여
            if col < 5 and row < 5:
                color_name = "빨강 (Red)"
                td_valence, td_arousal = -0.7, 0.7
            elif col >= 5 and row < 5:
                color_name = "노랑 (Yellow)"
                td_valence, td_arousal = 0.7, 0.7
            elif col < 5 and row >= 5:
                color_name = "파랑 (Blue)"
                td_valence, td_arousal = -0.7, -0.7
            else:
                color_name = "초록 (Green)"
                td_valence, td_arousal = 0.7, -0.7
        else:
            color_name = "알 수 없음"
            td_valence, td_arousal = valence, audio_arousal

        # 결과 요약 출력
        print(f"\n==================================================")
        print(f"🗣️ 인식된 말: {transcript}")
        print(f"🤖 최종 감정단어: {emotion_word} -> 🎨 매핑 색상: {color_name}")
        print(f"📊 터치디자이너 전송 수치 - Valence: {td_valence}, Arousal: {td_arousal}")
        print(f"==================================================")

        # 5. OSC 전송
        print(f"\n>> OSC 데이터 전송 (포트 5000) - 단어: {emotion_word}, 색상: {color_name}")
        osc_client.send_message("/emotion/word", emotion_word)
        osc_client.send_message("/emotion/color_name", color_name)
        osc_client.send_message("/emotion/valence", float(td_valence))
        osc_client.send_message("/emotion/arousal", float(td_arousal))
        osc_client.send_message("/emotion/text", transcript)
        print("전송 완료!")
        
        return True

    except Exception as e:
        print(f"데이터 처리 중 오류 발생: {e}")
        return False

def manage_archive_limit(archive_dir, max_files=20):
    """폴더 내의 wav 파일 개수를 제한하고, 초과 시 오래된 파일부터 삭제합니다."""
    try:
        files = [os.path.join(archive_dir, f) for f in os.listdir(archive_dir) if f.endswith('.wav')]
        if len(files) > max_files:
            # 오래된 파일이 앞에 오도록 수정 시간 기준 정렬
            files.sort(key=os.path.getmtime)
            
            # 초과하는 개수만큼 삭제
            files_to_delete = len(files) - max_files
            for i in range(files_to_delete):
                os.remove(files[i])
                print(f"[아카이브 정리] 최대 개수({max_files}개) 초과로 오래된 파일을 삭제했습니다: {files[i]}")
    except Exception as e:
        print(f"아카이브 관리 중 오류 발생: {e}")

def main():
    print("=== 미디어아트 100% 로컬 백엔드 시스템 시작 ===")
    try:
        while True:
            # 1. 소리 감지 및 녹음
            filepath = record_audio()
            
            # 2. 녹음 파일이 있으면 분석 및 OSC 전송
            if filepath:
                success = process_audio(filepath)
                if success:
                    # 분석이 완료된 경우에만 최대 20개 유지 규칙 적용
                    manage_archive_limit(ARCHIVE_DIR, max_files=20)
                else:
                    # 텍스트가 없거나 에러가 난 경우 해당 파일은 삭제
                    try:
                        if os.path.exists(filepath):
                            os.remove(filepath)
                            print(f"[알림] 의미 없는 소리이므로 저장하지 않고 삭제했습니다: {filepath}\n")
                    except OSError:
                        pass
    except KeyboardInterrupt:
        print("\n프로그램을 종료합니다.")

if __name__ == "__main__":
    main()
