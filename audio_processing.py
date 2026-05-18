import numpy as np

from config import RATE, THRESHOLD
from mood_meter import clamp


def analyze_audio_volume(data):
    samples = np.asarray(data).astype(np.float32)
    return float(np.sqrt(np.mean(np.square(samples))))


def _as_mono_float_audio(data):
    samples = np.asarray(data)
    if samples.size == 0:
        return np.zeros(0, dtype=np.float32)

    input_dtype = samples.dtype
    if samples.ndim > 1:
        samples = samples[:, 0]
    samples = samples.astype(np.float32, copy=False)

    if np.issubdtype(input_dtype, np.integer):
        info = np.iinfo(input_dtype)
        samples = samples / float(max(abs(info.min), info.max))
    elif np.max(np.abs(samples)) > 1.5:
        samples = samples / 32768.0

    return samples


def compute_live_audio_features(data, rate=RATE):
    samples = _as_mono_float_audio(data)
    if samples.size == 0:
        samples = np.zeros(1, dtype=np.float32)

    rms = float(np.sqrt(np.mean(np.square(samples))))
    if samples.size > 1:
        zcr = float(np.mean(np.diff(np.signbit(samples)) != 0))
    else:
        zcr = 0.0

    magnitude = np.abs(np.fft.rfft(samples))
    if float(np.sum(magnitude)) > 0.0:
        freqs = np.fft.rfftfreq(samples.size, d=1.0 / float(rate))
        spectral_centroid = float(np.sum(freqs * magnitude) / np.sum(magnitude))
    else:
        spectral_centroid = 0.0

    norm_energy = clamp(rms / 0.08, 0.0, 1.0)
    norm_zcr = clamp(zcr / 0.15, 0.0, 1.0)
    norm_centroid = clamp(spectral_centroid / 3000.0, 0.0, 1.0)
    arousal_raw = (norm_energy * 0.6) + (norm_zcr * 0.2) + (norm_centroid * 0.2)
    arousal_live = clamp((arousal_raw * 2.0) - 1.0)
    arousal_confidence = clamp((rms - 0.01) / 0.08, 0.0, 1.0)

    return {
        "arousal_live": arousal_live,
        "arousal_confidence": arousal_confidence,
        "rms": rms,
        "zcr": zcr,
        "spectral_centroid": spectral_centroid,
    }


def analyze_arousal(filepath):
    try:
        import librosa

        y, sr = librosa.load(filepath, sr=RATE)
        rms = librosa.feature.rms(y=y)[0]
        mean_rms = float(np.mean(rms))

        f0, voiced_flag, _voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
        )
        f0 = f0[voiced_flag]
        std_pitch = float(np.std(f0)) if len(f0) > 0 else 0.0

        zcr = librosa.feature.zero_crossing_rate(y=y)[0]
        mean_zcr = float(np.mean(zcr))
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        mean_centroid = float(np.mean(centroid))

        norm_rms = min(mean_rms / 0.1, 1.0)
        norm_pitch_std = min(std_pitch / 65.0, 1.0)
        norm_zcr = min(mean_zcr / 0.15, 1.0)
        norm_centroid = min(mean_centroid / 3000.0, 1.0)

        arousal_raw = (
            (norm_rms * 0.4)
            + (norm_pitch_std * 0.3)
            + (norm_zcr * 0.15)
            + (norm_centroid * 0.15)
        )
        arousal = clamp((arousal_raw * 2.0) - 1.0)
        print(
            "[audio] rms={:.3f}, pitch_std={:.1f}Hz, zcr={:.3f}, centroid={:.0f}Hz -> arousal={:.2f}".format(
                mean_rms,
                std_pitch,
                mean_zcr,
                mean_centroid,
                arousal,
            )
        )
        return arousal
    except Exception as exc:
        print(f"Audio arousal analysis failed: {exc}")
        return 0.0


def should_collect_live_segment(volume, features, threshold=THRESHOLD, min_confidence=0.45):
    return float(volume) > float(threshold) and float(features.get("arousal_confidence", 0.0)) >= min_confidence
