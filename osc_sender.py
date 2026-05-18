from config import OSC_IP, OSC_PORT


class LazyOscClient:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self._client = None

    def _load(self):
        if self._client is None:
            from pythonosc.udp_client import SimpleUDPClient

            self._client = SimpleUDPClient(self.ip, self.port)
        return self._client

    def send_message(self, address, value):
        return self._load().send_message(address, value)


osc_client = LazyOscClient(OSC_IP, OSC_PORT)


def send_live_osc(
    arousal_live=None,
    arousal_confidence=None,
    left_arousal_live=None,
    right_arousal_live=None,
    left_arousal_confidence=None,
    right_arousal_confidence=None,
    valence_target=None,
    valence_confidence=None,
    text_partial=None,
    text_final=None,
):
    if arousal_live is None:
        live_values = [value for value in (left_arousal_live, right_arousal_live) if value is not None]
        if live_values:
            arousal_live = max(live_values)
    if arousal_confidence is None:
        confidence_values = [
            value for value in (left_arousal_confidence, right_arousal_confidence)
            if value is not None
        ]
        if confidence_values:
            arousal_confidence = max(confidence_values)

    values = {
        "/emotion/arousal_live": arousal_live,
        "/emotion/arousal_confidence": arousal_confidence,
        "/emotion/left_arousal_live": left_arousal_live,
        "/emotion/right_arousal_live": right_arousal_live,
        "/emotion/left_arousal_confidence": left_arousal_confidence,
        "/emotion/right_arousal_confidence": right_arousal_confidence,
        "/emotion/valence_target": valence_target,
        "/emotion/valence_confidence": valence_confidence,
        "/emotion/arousal": arousal_live,
        "/emotion/valence": valence_target,
        "/emotion/text_partial": text_partial,
        "/emotion/text_final": text_final,
    }
    for address, value in values.items():
        if value is not None:
            osc_client.send_message(address, value)


def send_emotion_osc(emotion_word, color_name, td_valence, td_arousal, text):
    print(f">> Sending OSC data to port {OSC_PORT}")
    osc_client.send_message("/emotion/word", emotion_word)
    osc_client.send_message("/emotion/color_name", color_name)
    osc_client.send_message("/emotion/valence", float(td_valence))
    osc_client.send_message("/emotion/arousal", float(td_arousal))
    osc_client.send_message("/emotion/text", text)
    print("OSC send complete.")
