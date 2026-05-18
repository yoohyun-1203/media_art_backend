# TouchDesigner Bridge Recovery - 2026-05-18

## Failure

- TouchDesigner owned TCP 9988 and UDP 5000, but HTTP requests to http://127.0.0.1:9988/td timed out.
- The backend web controller was running at http://127.0.0.1:8765.
- Backend /api/debug/td-readback failed before recovery because the TouchDesigner bridge did not answer.

## Open Project

- Before restart, TouchDesigner was running:
  C:\Users\kksu1\Documents\New project\innerworld\Innerworld\Innerworld.1.toe
- After restart, process command line confirmed it opened:
  C:\Users\kksu1\Documents\New project\innerworld\Innerworld\Innerworld.1.bridge-fixed.toe
- TouchDesigner window title may still display Innerworld.1.toe because of internal project metadata.

## Root Cause

Found by expanding a copy of the .toe with toeexpand.

- /project1/webserver1.parm had: callbacks 0 webserver1_callbacks1
- /project1/webserver1_callbacks1.parm had: language 0 text
- WebServer DAT callbacks must execute as Python, so the bridge callback was not running correctly.

## Fix

- Changed /project1/webserver1_callbacks1.parm from language 0 text to language 0 python.
- Collapsed the edited copy with toecollapse.
- Wrote fixed file:
  C:\Users\kksu1\Documents\New project\innerworld\Innerworld\Innerworld.1.bridge-fixed.toe
- Restarted TouchDesigner with the fixed file.

## Verification

Passed after restart:

- Direct bridge ping returned: {"ok": true, "pong": true}
- Backend /api/debug/td-readback returned:
  - oscin2.ok=True
  - serialParams.ok=True
  - serialRows.ok=True
- /project1/oscin2 exposes:
  - emotion/arousal_live
  - emotion/arousal_confidence
  - emotion/valence_target
  - emotion/valence_confidence
  - legacy emotion/valence and emotion/arousal
- /project1/serial1 params:
  - port=COM6
  - baudrate=115200
  - callbacks=/project1/serial1_callbacks

## Remaining Check

The software bridge is restored. Physical Arduino/LED verification still requires sending a controlled serial payload, for example v,0.700,0.700, only when hardware output is expected and allowed.
