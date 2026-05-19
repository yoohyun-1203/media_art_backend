def _clamp(value, low=-1.0, high=1.0):
    return max(low, min(high, float(value)))


class NoiseFloorTracker:
    def __init__(self, initial_floor=0.01, learning_rate=0.05):
        self.floor = max(float(initial_floor), 1e-9)
        self.learning_rate = float(learning_rate)

    def update(self, rms, is_speech):
        if not is_speech:
            rms = max(float(rms), 0.0)
            self.floor = (
                (1.0 - self.learning_rate) * self.floor
                + self.learning_rate * rms
            )
            self.floor = max(self.floor, 1e-9)
        return self.floor

    def relative_level(self, rms):
        return max(float(rms), 0.0) / self.floor


class EnvelopeSmoother:
    def __init__(self, value=0.0, attack=0.65, release=0.12):
        self.value = float(value)
        self.attack = float(attack)
        self.release = float(release)

    def update(self, target):
        target = float(target)
        coefficient = self.attack if target > self.value else self.release
        self.value = self.value + (target - self.value) * coefficient
        return self.value


class RollingVoiceBaseline:
    """Continuously adapt to the current speaker/environment without identity."""

    def __init__(
        self,
        rms_baseline=0.02,
        arousal_baseline=0.0,
        speech_learning_rate=0.015,
        quiet_learning_rate=0.06,
    ):
        self.rms_baseline = max(float(rms_baseline), 1e-6)
        self.arousal_baseline = float(arousal_baseline)
        self.speech_learning_rate = float(speech_learning_rate)
        self.quiet_learning_rate = float(quiet_learning_rate)

    def reset(self):
        self.rms_baseline = 0.02
        self.arousal_baseline = 0.0

    def update(self, rms, arousal_live, has_signal):
        rms = max(float(rms), 0.0)
        arousal_live = _clamp(arousal_live)
        learning_rate = self.speech_learning_rate if has_signal else self.quiet_learning_rate
        self.rms_baseline = max(
            1e-6,
            self.rms_baseline + (rms - self.rms_baseline) * learning_rate,
        )
        self.arousal_baseline = _clamp(
            self.arousal_baseline + (arousal_live - self.arousal_baseline) * learning_rate
        )
        relative_level = rms / self.rms_baseline
        relative_arousal = _clamp(arousal_live - self.arousal_baseline)
        return {
            "rms_baseline": self.rms_baseline,
            "arousal_baseline": self.arousal_baseline,
            "relative_level": relative_level,
            "relative_arousal": relative_arousal,
        }


class SegmentEndpoint:
    def __init__(self, silence_seconds=1.0):
        self.silence_seconds = float(silence_seconds)
        self.active = False
        self.last_signal_at = None

    def update(self, has_signal, now):
        now = float(now)

        if has_signal:
            self.last_signal_at = now
            if not self.active:
                self.active = True
                return "start"
            return "continue"

        if not self.active:
            return "idle"

        if self.last_signal_at is not None and now - self.last_signal_at >= self.silence_seconds:
            self.active = False
            self.last_signal_at = None
            return "end"

        return "continue"


class UtteranceValenceTracker:
    """Hold valence steady during speech, then commit one value per utterance."""

    def __init__(
        self,
        silence_seconds=0.5,
        max_utterance_seconds=4.0,
        early_commit_min_candidates=3,
        early_commit_min_confidence=0.6,
        min_hold_seconds=3.0,
        switch_min_candidates=5,
        switch_min_confidence=0.75,
        mood_attack=0.22,
        mood_release=0.08,
    ):
        self.endpoint = SegmentEndpoint(silence_seconds=silence_seconds)
        self.max_utterance_seconds = float(max_utterance_seconds)
        self.early_commit_min_candidates = int(early_commit_min_candidates)
        self.early_commit_min_confidence = float(early_commit_min_confidence)
        self.min_hold_seconds = float(min_hold_seconds)
        self.switch_min_candidates = int(switch_min_candidates)
        self.switch_min_confidence = float(switch_min_confidence)
        self.mood_attack = float(mood_attack)
        self.mood_release = float(mood_release)
        self.committed_valence = 0.0
        self.committed_confidence = 0.0
        self.mood_score = 0.0
        self.last_committed_at = None
        self._candidates = []
        self._utterance_started_at = None

    def reset(self):
        self.endpoint = SegmentEndpoint(silence_seconds=self.endpoint.silence_seconds)
        self.committed_valence = 0.0
        self.committed_confidence = 0.0
        self.mood_score = 0.0
        self.last_committed_at = None
        self._candidates = []
        self._utterance_started_at = None

    def _candidate_summary(self):
        if not self._candidates:
            return None
        total_weight = sum(confidence for _valence, confidence in self._candidates)
        if total_weight <= 0.0:
            return None
        valence = sum(
            valence * confidence for valence, confidence in self._candidates
        ) / total_weight
        confidence = max(confidence for _valence, confidence in self._candidates)
        return valence, confidence

    def _commit(self, now):
        summary = self._candidate_summary()
        if summary is None:
            return False
        candidate_valence, candidate_confidence = summary
        if not self._can_replace_committed(candidate_valence, candidate_confidence, now):
            self._candidates = []
            return False
        coefficient = self.mood_attack if abs(candidate_valence) > abs(self.mood_score) else self.mood_release
        self.mood_score = self.mood_score + (candidate_valence - self.mood_score) * coefficient
        self.committed_valence = self.mood_score
        self.committed_confidence = candidate_confidence
        self.last_committed_at = float(now)
        self._candidates = []
        return True

    def _can_early_commit(self):
        if len(self._candidates) < self.early_commit_min_candidates:
            return False
        recent = self._candidates[-self.early_commit_min_candidates :]
        signs = {1 if valence >= 0.0 else -1 for valence, _confidence in recent}
        confidences = [confidence for _valence, confidence in recent]
        return len(signs) == 1 and min(confidences) >= self.early_commit_min_confidence

    def _can_replace_committed(self, candidate_valence, candidate_confidence, now):
        if self.committed_confidence <= 0.0 or self.last_committed_at is None:
            return True

        same_direction = (candidate_valence >= 0.0) == (self.committed_valence >= 0.0)
        if same_direction:
            return True

        if float(now) - self.last_committed_at < self.min_hold_seconds:
            return False

        recent = self._candidates[-self.switch_min_candidates :]
        if len(recent) < self.switch_min_candidates:
            return False
        signs = {1 if valence >= 0.0 else -1 for valence, _confidence in recent}
        confidences = [confidence for _valence, confidence in recent]
        return (
            len(signs) == 1
            and min(confidences) >= self.switch_min_confidence
            and float(candidate_confidence) >= self.switch_min_confidence
        )

    def update(self, candidate_valence, candidate_confidence, has_signal, now):
        event = self.endpoint.update(has_signal=has_signal, now=now)
        if event == "start":
            self._utterance_started_at = float(now)

        confidence = max(0.0, min(1.0, float(candidate_confidence)))
        if has_signal and confidence > 0.0:
            self._candidates.append((_clamp(candidate_valence), confidence))

        committed = False
        if event == "end":
            committed = self._commit(now)
            self._utterance_started_at = None
        elif self.committed_confidence <= 0.0 and self._can_early_commit():
            committed = self._commit(now)
        elif (
            self.endpoint.active
            and self._utterance_started_at is not None
            and float(now) - self._utterance_started_at >= self.max_utterance_seconds
        ):
            # Long uninterrupted speech: refresh slowly without word-level flicker.
            committed = self._commit(now)
            self._utterance_started_at = float(now)

        return {
            "valence": self.committed_valence,
            "confidence": self.committed_confidence,
            "mood_score": self.mood_score,
            "event": event,
            "committed": committed,
            "candidate_count": len(self._candidates),
        }


class SpeakerBleedGate:
    """Two-mic speaker-bleed attenuation heuristic, not AEC or standalone VAD."""

    def __init__(self, reference_weight=0.8):
        self.reference_weight = float(reference_weight)

    def score_residual_voice_confidence(self, voice_rms, reference_rms):
        voice = max(float(voice_rms), 0.0)
        reference = max(float(reference_rms), 0.0)
        adjusted = voice - (reference * self.reference_weight)
        if voice <= 1e-6:
            return 0.0
        return max(0.0, min(1.0, adjusted / voice))

    def score_voice_likelihood(self, voice_rms, reference_rms):
        return self.score_residual_voice_confidence(voice_rms, reference_rms)


def compose_led_mood_signal(
    arousal_live,
    arousal_confidence,
    latest_valence,
    latest_valence_confidence,
    ambient_valence,
    ambient_arousal,
    has_mic_activity,
):
    if has_mic_activity and float(arousal_confidence) > 0.0:
        arousal = arousal_live
    else:
        arousal = ambient_arousal

    if float(latest_valence_confidence) > 0.0:
        valence = latest_valence
    elif has_mic_activity:
        valence = 0.0
    else:
        valence = ambient_valence

    return {
        "serial_prefix": "v",
        "valence": _clamp(valence),
        "arousal": _clamp(arousal),
    }
