import json
import os
from urllib import request as urlrequest


TD_BRIDGE_URL = os.getenv("TD_BRIDGE_URL", "http://127.0.0.1:9988/td")
TD_BRIDGE_TIMEOUT = float(os.getenv("TD_BRIDGE_TIMEOUT", "0.8"))
SERIAL_DAT_PATH = "/project1/serial1"
OSC_IN_PATH = "/project1/oscin2"


def td_bridge_action(action, **fields):
    payload = json.dumps({"action": action, **fields}).encode("utf-8")
    req = urlrequest.Request(
        TD_BRIDGE_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlrequest.urlopen(req, timeout=TD_BRIDGE_TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


def td_channels(path):
    return td_bridge_action("channels", path=path)


def read_touchdesigner_state():
    paths = [
        "/project1/select2",
        "/project1/select3",
        "/project1/joy",
        "/project1/sad",
        "/project1/angry",
        "/project1/relaxed",
        "/project1/RGBs",
        "/project1/oscin2",
    ]
    state = {}
    for path in paths:
        try:
            state[path] = td_channels(path).get("channels", {})
        except Exception as exc:
            state[path] = {"error": str(exc)}
    return state
