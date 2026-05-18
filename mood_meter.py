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
    ["절망한", "가망 없는", "고독한", "소모된", "진이 빠진", "나른한", "흐뭇한", "고요한", "안락한", "안온한"],
]


def clamp(value, min_value=-1.0, max_value=1.0):
    return max(min(float(value), max_value), min_value)


def find_word_in_grid(word):
    for row_index, row in enumerate(MOOD_METER_GRID):
        for col_index, item in enumerate(row):
            if item == word:
                return row_index, col_index
    return -1, -1


def word_from_values(valence, arousal):
    row = int(max(min((1.0 - float(arousal)) / 2.0 * 9.99, 9.0), 0.0))
    col = int(max(min((float(valence) + 1.0) / 2.0 * 9.99, 9.0), 0.0))
    return MOOD_METER_GRID[row][col]


def map_touchdesigner_values(emotion_word, audio_arousal):
    row, col = find_word_in_grid(emotion_word)

    if row != -1 and col != -1:
        if col < 5 and row < 5:
            return "빨강 (Red)", -0.7, 0.7
        if col >= 5 and row < 5:
            return "노랑 (Yellow)", 0.7, 0.7
        if col < 5 and row >= 5:
            return "파랑 (Blue)", -0.7, -0.7
        return "초록 (Green)", 0.7, -0.7

    if audio_arousal > 0:
        return "노랑 (Yellow) - 로컬대체", 0.5, audio_arousal
    return "파랑 (Blue) - 로컬대체", -0.5, audio_arousal


def clamp_mood_value(value, name):
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    return clamp(numeric)


def mood_payload(valence, arousal):
    safe_valence = clamp_mood_value(valence, "valence")
    safe_arousal = clamp_mood_value(arousal, "arousal")
    return safe_valence, safe_arousal, f"v,{safe_valence:.3f},{safe_arousal:.3f}"
