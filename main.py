import sounddevice as sd
import wave
import numpy as np
import time
import os
from datetime import datetime
import json
import warnings

from pythonosc.udp_client import SimpleUDPClient

# Suppress warnings
warnings.filterwarnings("ignore")

import sys
# Windows 터미널 한글 깨짐 방지
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 1. 로컬 AI 모델 로드 (초기 실행 시 다운로드 필요)
print("로컬 AI 모델을 로드하는 중입니다. (약간의 시간이 소요될 수 있습니다)")
try:
    import whisper
    from transformers import pipeline
    import librosa
    
    # Whisper STT 모델 로드 ('small' 모델로 업그레이드하여 정확도 향상)
    whisper_model = whisper.load_model("small")
    
    # 긍정/부정(Valence) 분석을 위한 파이프라인 (다국어 지원 모델)
    # 더 정확한 다국어 감정 분석 모델 (positive, neutral, negative 지원)
    sentiment_model = pipeline(
        "sentiment-analysis", 
        model="lxyuan/distilbert-base-multilingual-cased-sentiments-student"
    )
    # 감정 키워드 보정 (한국어 전용 - 각 50개 이상)
    POSITIVE_WORDS = [
        "행복", "좋아", "감사", "최고", "사랑", "기뻐", "즐거워", "웃음", "안녕", "환상",
        "보람", "감동", "흐뭇", "희망", "열정", "완벽", "훌륭", "소중", "든든", "설렘",
        "평화", "행운", "활기", "만족", "칭찬", "축복", "멋져", "예뻐", "아름다워", "대단",
        "기적", "은혜", "친절", "배려", "용기", "자신감", "보물", "달콤", "상쾌", "유쾌",
        "쾌적", "뿌듯", "신나", "짜릿", "반가워", "건강", "성공", "승리", "평온", "낭만",
        "따뜻", "포근", "화사", "밝아", "빛나", "넉넉", "알차", "기특", "기운", "활짝"
    ]
    NEGATIVE_WORDS = [
        "슬퍼", "화나", "짜증", "미워", "죽고", "힘들어", "나빠", "최악", "안돼", "우울",
        "고통", "절망", "분노", "억울", "무서워", "공포", "걱정", "근심", "불안", "초조",
        "지겨워", "끔찍", "혐오", "불쾌", "비참", "창피", "부끄러워", "귀찮아", "싫어", "답답",
        "허탈", "공허", "외로워", "고독", "좌절", "피곤", "괴로워", "아파", "속상", "원망",
        "후회", "부실", "부족", "실패", "엉망", "지옥", "어두워", "차가워", "무거워", "딱해",
        "가여워", "얄미워", "야속", "비난", "모욕", "차별", "편견", "의심", "거짓", "꼴불견",
        "빡치", "킹받", "열받", "개같", "망했", "울고"
    ]
    # 감정 부정 단어 (Valence 반전용)
    NEGATION_WORDS = ["안 ", "못 ", "아니", "않아", "않은", "않고", "않게", "없어"]
    print("AI 모델 로드 완료!")
except Exception as e:
    print(f"모델 로드 중 오류 발생: {e}")
    exit(1)

# 오디오 설정
CHANNELS = 1
RATE = 16000
CHUNK = 1024
THRESHOLD = 500  # RMS 임계값
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

    stream = sd.InputStream(samplerate=RATE, channels=CHANNELS, dtype='int16', blocksize=CHUNK)
    stream.start()

    try:
        while True:
            data, overflowed = stream.read(CHUNK)
            volume = analyze_audio_volume(data)

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

def process_audio(filepath):
    """로컬 모델을 사용하여 STT, Valence, Arousal 분석 후 OSC 전송"""
    print(f"\n[1/3] 음성 인식(STT) 진행 중... (파일: {filepath})")
    
    try:
        # 1. Whisper STT (한국어 지정하여 정확도 향상)
        result = whisper_model.transcribe(filepath, language="ko")
        transcript = result["text"].strip()
        print(f" -> 인식된 텍스트: {transcript}")

        if not transcript:
            print("인식된 텍스트가 없어 분석을 건너뜁니다.")
            return False

        # 2. 텍스트 기반 Valence 분석
        print("[2/3] 텍스트 기반 긍정/부정(Valence) 분석 중...")
        sentiment_result = sentiment_model(transcript)[0]
        label = sentiment_result['label'].lower() # 'positive', 'neutral', 'negative'
        score = sentiment_result['score']
        
        if label == 'positive':
            valence = score
        elif label == 'negative':
            valence = -score
        else:
            valence = 0.0
            
        # 키워드 기반 강력 보정 (부정어 문맥 파악 포함)
        pos_found = any(word in transcript for word in POSITIVE_WORDS)
        neg_found = any(word in transcript for word in NEGATIVE_WORDS)
        negation_found = any(word in transcript for word in NEGATION_WORDS)
        
        if pos_found and not neg_found:
            if negation_found:
                # "안 행복해" 처럼 긍정 키워드 + 부정어 조합
                valence = min(valence, -0.5)
            else:
                valence = max(valence, 0.6)
        elif neg_found and not pos_found:
            if negation_found:
                # "나쁘지 않아" 처럼 부정 키워드 + 부정어 조합
                valence = max(valence, 0.2)
            else:
                valence = min(valence, -0.6)
        elif pos_found and neg_found:
            # 혼합된 경우 중립으로 수렴
            valence = valence * 0.3
        
        print(f" -> 감정 분석 결과: {label} (신뢰도: {score:.2f}) -> 최종 문맥보정 Valence: {valence:.2f}")
        
        # 3. 음향 기반 Arousal 분석
        print("[3/3] 오디오 기반 어투/흥분도(Arousal) 분석 중...")
        arousal = analyze_arousal(filepath)

        # 4. OSC 전송
        print(f"\n>> OSC 데이터 전송 (포트 5000) - Valence: {valence:.2f}, Arousal: {arousal:.2f}, Text: {transcript}")
        osc_client.send_message("/emotion/valence", float(valence))
        osc_client.send_message("/emotion/arousal", float(arousal))
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
