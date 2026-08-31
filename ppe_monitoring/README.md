# PPE Monitoring

A complete local Edge AI safety system: Ultralytics YOLO PPE detection, pose-assisted
wearing validation, ByteTrack IDs, temporal smoothing, a FastAPI/WebSocket dashboard,
SQLite analytics, and fail-safe Arduino LED control.

The system never treats PPE elsewhere in the image as worn. A helmet must pass
center and overlap tests against one person's head; a vest must pass equivalent
tests against that person's torso. Greedy strongest-match assignment prevents one
item from being shared by two people. Pose keypoints define regions when reliable;
configurable person-box proportions are the fallback.

## First command

After creating the environment and installing requirements, run this first from
the directory containing `ppe_monitoring`:

```powershell
python -m ppe_monitoring.scripts.download_models
```

It inspects the mandatory Hugging Face repository file list, considers actual
`.pt` files, downloads a candidate, loads it through Ultralytics, prints
`model.names`, and accepts it only if helmet and vest aliases exist. It then obtains
`yolo11n-pose.pt`, with `yolov8n-pose.pt` as a compatibility fallback. Success
means both of these files really exist and load:

```text
ppe_monitoring/models/ppe_model.pt
ppe_monitoring/models/pose_model.pt
```

The primary PPE repository is
`Tanishjain9/yolov8n-ppe-detection-6classes`. Its documented YOLOv8 Nano classes
are `Gloves`, `Vest`, `goggles`, `helmet`, `mask`, and `safety_shoe`; monitoring
uses only `Vest` and `helmet`. The script confirms the actual case-insensitive
names from `model.names` instead of assuming IDs. Review the repository's MIT
license, PyTorch checkpoint security implications, and accuracy for your site.

## 1. Windows PowerShell environment

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r .\ppe_monitoring\requirements.txt
```

Download and validate models:

```powershell
python -m ppe_monitoring.scripts.download_models
python -m ppe_monitoring.scripts.download_models --verify-only
```

The verification command prints the architecture and actual detected PPE class
names. To replace existing files from the mandatory repository:

```powershell
python -m ppe_monitoring.scripts.download_models --force
```

To use a fine-tuned model, copy it to `ppe_monitoring/models/ppe_model.pt` and run
`--verify-only`. No detector code changes are required.

## 2. Arduino sketch and safe wiring

Open and upload:

```text
ppe_monitoring/arduino/ppe_led_controller/ppe_led_controller.ino
```

The initial example pins are constants at the top of the sketch. Change them to
match your board before uploading; the final pin numbers are deliberately not
assumed.

| Arduino connection | Default example | Required wiring |
|---|---:|---|
| Red LED anode (+) | Digital 8 | Pin → 220–330 Ω resistor → LED anode |
| Blue LED anode (+) | Digital 9 | Pin → 220–330 Ω resistor → LED anode |
| Green LED anode (+) | Digital 10 | Pin → 220–330 Ω resistor → LED anode |
| Optional buzzer signal | Digital 11 | Use a suitable driver/transistor if current exceeds pin rating |
| All LED cathodes (−) | GND | Shared Arduino ground |

Never connect an LED directly to a GPIO pin without its own 220–330 ohm resistor.
Set `BUZZER_ENABLED = true` only after checking the buzzer's voltage/current needs.
The buzzer is silent for green and sounds only for red/blue.

Upload with Arduino IDE, or with `arduino-cli` after replacing the board FQBN and
port for your hardware:

```powershell
arduino-cli board list
arduino-cli compile --fqbn arduino:avr:uno .\ppe_monitoring\arduino\ppe_led_controller
arduino-cli upload -p COM3 --fqbn arduino:avr:uno .\ppe_monitoring\arduino\ppe_led_controller
```

The sketch accepts `RED`, `BLUE`, `GREEN`, `OFF`, and `PING`, replies with the
specified acknowledgement, permits only one LED at a time, starts off, and turns
everything off after six seconds without a valid command. There are no blocking
delays in its loop.

## 3. Find and test the serial port

Windows PowerShell:

```powershell
[System.IO.Ports.SerialPort]::GetPortNames()
```

Jetson/Linux:

```bash
python -m serial.tools.list_ports
ls -l /dev/ttyACM* /dev/ttyUSB* 2>/dev/null
```

Automatic discovery prefers Arduino/CH340/CP210/USB-serial descriptions. To set a
port manually, edit `config.yaml`:

```yaml
arduino:
  enabled: true
  port: COM3              # Windows
  # port: /dev/ttyACM0    # Jetson/Linux
```

Close Arduino Serial Monitor before testing because only one program can own the
port. Safely cycle OFF → RED → BLUE → GREEN → OFF:

```powershell
python .\ppe_monitoring\tools\test_arduino_leds.py --port COM3 --cycle
```

Linux example:

```bash
python ppe_monitoring/tools/test_arduino_leds.py --port /dev/ttyACM0 --cycle
```

## 4. Start the system and dashboard

Webcam 0:

```powershell
python -m ppe_monitoring.main --source 0
```

USB webcam:

```powershell
python -m ppe_monitoring.main --source 1
```

Video file:

```powershell
python -m ppe_monitoring.main --source "C:\videos\shift.mp4"
```

RTSP:

```powershell
python -m ppe_monitoring.main --source "rtsp://username:password@192.168.1.50:554/stream1"
```

Then open:

```text
http://127.0.0.1:8000
```

### Optional Supabase cloud telemetry

The read-only cloud dashboard is deployed at:

```text
https://ppe-monitoring.vercel.app
```

Inference, webcam video, model weights, and Arduino control stay on the Edge
computer. Only status summaries and compliance events are sent to the Supabase
project `Edge AI Projects` when cloud publishing is enabled.

For privacy, the deployed database starts locked down: anonymous and ordinary
authenticated browser users cannot read PPE telemetry. A specific viewer access
policy must be added before the dashboard can display records.

To let the Edge process publish, set the service-role key only in the local
process environment and enable the cloud block in `config.yaml`. Never commit
this key or place it in Vercel/browser environment variables:

```powershell
$env:SUPABASE_SERVICE_ROLE_KEY = "your-local-service-role-key"
```

```yaml
cloud:
  enabled: true
  url: https://cretvvkajdrqyvpzhujj.supabase.co
  service_key_env: SUPABASE_SERVICE_ROLE_KEY
```

The dashboard server starts even when a model, camera, or Arduino is unavailable;
the relevant status indicator and error explain the failure. Use the header button
to start or stop monitoring. Start without immediately opening a camera:

```powershell
python -m ppe_monitoring.main --no-auto-start
```

Test the entire web/runtime path without physical serial hardware:

```powershell
python -m ppe_monitoring.main --source 0 --mock-arduino
```

Inference, MJPEG video delivery, each WebSocket client, SQLite, and Arduino serial
run independently. A slow/disconnected browser or Arduino cannot stop inference.

## Dashboard and status rules

Each confirmed person has both a status and explanation:

- `COMPLIANT` — helmet and vest correctly worn — green.
- `PARTIAL PPE` — helmet missing or safety vest missing — blue.
- `NO PPE` — helmet and safety vest missing — red.

The Arduino uses the worst confirmed visible status: red beats blue, blue beats
green, and no people turns all LEDs off. Visible tracks still inside the configured
confirmation window show `ANALYZING` and do not trigger a compliance LED prematurely.
Shutdown and serial disconnection result in all LEDs off.

The dashboard provides current people, compliance breakdown/rate, FPS, today's
violations, overall state/reason, per-person states, live feed, Arduino status,
compliance timeline, violation types, hourly counts, and filterable events.
All CSS and JavaScript are local; runtime does not need internet access.

## Events and evidence

SQLite is stored at `ppe_monitoring/data/ppe_events.db`. A row includes UTC
timestamp, camera, tracking ID, helmet/vest booleans, status, reason, aggregate
and item confidence, and optional evidence path. Rows are written for meaningful
state changes and cooldown summaries, not every frame. Evidence images go to
`ppe_monitoring/evidence` only when an event is eligible to be stored.

Adjust:

```yaml
events:
  save_evidence: true
  cooldown_seconds: 10
```

## Detection tuning and debug mode

All initial defaults are in `config.yaml`. Important settings:

- `detection.ppe_confidence` and `person_confidence`
- `detection.keypoint_confidence`
- `helmet_head_overlap` and `vest_torso_overlap`
- fallback `head_region` and `torso_region`
- `tracking.history_frames`, `confirmation_frames`, and `lost_track_timeout`
- `camera.reconnect_seconds`

Enable debug overlays:

```yaml
dashboard:
  debug_regions: true
```

Debug mode displays person/head/torso boxes, confident pose keypoints, PPE match
lines, overlap values, and detection confidence. Raise overlap thresholds to reject
held/nearby items more aggressively; lower cautiously for occlusion. Validate changes
with representative clips containing overlapping workers and carried/floor PPE.

## Tests

```powershell
python -m pytest
```

Tests cover three-state PPE mapping, worst-status LEDs, spatial wearing checks,
wrong-person assignment, temporary loss, serial acknowledgements, missing hardware,
SQLite cooldown, dashboard APIs, and clear missing-model errors.

## Troubleshooting

### Model missing or incompatible

Run:

```powershell
python -m ppe_monitoring.scripts.download_models --verify-only
```

If missing, run without `--verify-only`. If classes lack helmet or vest aliases,
manually place a validated fine-tuned checkpoint at `models/ppe_model.pt`. A general COCO
`yolov8n.pt`/`yolo11n.pt` is not a PPE detector and will be rejected.

### Camera unavailable

Close other camera applications, test other indices, verify RTSP credentials and
network reachability, and increase `camera.reconnect_seconds`. For a video, use an
absolute quoted path.

### Arduino disconnected

The detector continues normally and the dashboard shows the serial error. Confirm
the uploaded baud rate is 115200, close Serial Monitor, verify the COM/device path,
try a data-capable USB cable, and run the LED tool. On Linux, add the user to the
appropriate serial group if needed, then sign out/in:

```bash
sudo usermod -aG dialout "$USER"
```

### Dashboard unavailable

Confirm the terminal says Uvicorn is listening and that port 8000 is free. Bind a
different port with `--port 8080`. For access from another machine, allow the port
through the host firewall and browse to the monitor machine's LAN address.

## Fine-tuning and Jetson Orin Nano

Fine-tune nano detection weights using images from the actual final camera. Include
hard negatives: held helmets/vests, floor or hanging PPE, overlapping workers,
partial people, poor lighting, blur, glare, and unusual angles. Do not label carried
or floor PPE as worn PPE. Evaluate false-compliance cases in full application replay,
not only detector mAP.

After validating `ppe_model.pt`, export ONNX:

```powershell
yolo export model=".\ppe_monitoring\models\ppe_model.pt" format=onnx imgsz=640 simplify=True
```

Build TensorRT on the target Jetson because engines depend on its TensorRT/CUDA/GPU
environment. Start with FP16:

```bash
yolo export model=ppe_monitoring/models/ppe_model.pt format=engine imgsz=640 half=True device=0
python -m ppe_monitoring.main --source 0
```

The association, smoothing, storage, dashboard, and Arduino layers are independent
of the inference backend, so only the detector adapters need to change for a custom
TensorRT execution path.

For an Ultralytics-exported local backend, update both the model path and backend:

```yaml
models:
  ppe_model: models/ppe_model.onnx   # or ppe_model.engine
  inference_backend: onnx           # or tensorrt
```

Monitoring never calls a Hugging Face hosted inference API. Hugging Face is used
only by the initial download script; `.pt`, ONNX, and TensorRT inference are local.

## Manual work still required

1. Review the mandatory PPE checkpoint and its license for your safety case.
2. Change sketch pin constants to your actual board wiring.
3. Wire one 220–330 Ω resistor per LED and a shared ground.
4. Upload the sketch and select/confirm the real COM or `/dev/tty*` port.
5. Validate thresholds and accuracy with footage from the installed camera.
