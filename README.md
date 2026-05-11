# Smart Pressure Ulcer Prevention System

A real-time, closed-loop IoT system that predicts and prevents pressure ulcers (bedsores) in bedridden hospital patients using a dense capacitive sensor mat, edge AI inference on an FPGA, and autonomous micro-pneumatic actuation — before tissue damage becomes visible.

---

## The Problem

Pressure ulcers affect over 2.5 million bedridden patients annually in India alone. They form when sustained body weight cuts off blood circulation to skin tissue, typically over bony prominences like the sacrum, heels, and hips. By the time a sore is visible, the damage has already reached deep tissue layers.

Current standard of care requires nurses to manually reposition patients every two hours. This is inconsistent, labor-intensive, missed at night, and completely reactive. Existing "smart mattress" products are expensive (USD 3,000+), low-resolution, and have no on-device intelligence — they stream raw data to cloud servers with no real-time response.

This project changes that.

---

## What This System Does

- Reads a 32x32 capacitive pressure map, a 32x24 thermopile skin temperature map, and reflective PPG tissue oxygenation simultaneously at 10 Hz
- Fuses all three sensor streams on an FPGA-based edge AI engine using a trained temporal CNN
- Predicts ischemia onset 20 to 40 minutes before irreversible tissue damage, per body zone
- Autonomously inflates micro-pneumatic bladders beneath high-risk zones to redistribute pressure silently, in under 3 seconds, without waking the patient
- Logs all events, predictions, and actuations to a hospital dashboard and cloud audit trail
- Improves its prediction model over time using federated learning across hospital beds, without sharing raw patient data

---

## System Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        SENSOR MAT                              │
│  [32x32 Capacitive Array]  [MLX90640 IR]  [MAX30102 PPG x8]   │
└────────────────────┬───────────────────────────────────────────┘
                     │ SPI / I2C @ 10 Hz
┌────────────────────▼───────────────────────────────────────────┐
│              EDGE AI CONTROLLER (FPGA + MCU)                   │
│  [CNN Inference Engine]  [Ischemia Predictor]  [PID Actuator]  │
│         Lattice CrossLink-NX  +  STM32H7                       │
└────────────────────┬───────────────────────────────────────────┘
                     │ MQTT over Wi-Fi (ESP32-S3)
┌────────────────────▼───────────────────────────────────────────┐
│                    CLOUD ANALYTICS                              │
│  [Patient Data Lake]  [Federated Learning]  [CDSCO Compliance] │
└────────────────────┬───────────────────────────────────────────┘
                     │ WebSocket
┌────────────────────▼───────────────────────────────────────────┐
│                  NURSE DASHBOARD                                │
│      [Live Heatmap]  [Risk Alerts]  [Actuation Audit Log]      │
└────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
smart-pressure-ulcer-prevention/
│
├── firmware/
│   ├── fpga/
│   │   ├── cnn_inference.v          # Verilog: CNN inference engine
│   │   ├── sensor_fusion.v          # Verilog: Multi-stream data fusion
│   │   └── actuator_ctrl.v          # Verilog: Pneumatic valve PWM driver
│   ├── stm32/
│   │   ├── main.c                   # STM32H7 main control loop
│   │   ├── pid_controller.c         # PID loop for bladder pressure
│   │   ├── sensor_read.c            # SPI/I2C sensor drivers
│   │   └── safety_interlock.c       # Hardware safety watchdog
│   └── esp32/
│       ├── mqtt_client.cpp          # MQTT publish to cloud
│       └── ble_gateway.cpp          # BLE 5.2 local gateway
│
├── ml/
│   ├── train_model.py               # Train temporal CNN locally
│   ├── quantize_model.py            # Post-training quantization for FPGA
│   ├── federated_client.py          # Federated learning client
│   ├── federated_server.py          # Hospital-level aggregation server
│   └── dataset/
│       └── README.md                # Dataset format and sources
│
├── src/
│   ├── dashboard/
│   │   ├── index.html               # Nurse station UI (this file)
│   │   ├── heatmap.js               # Real-time pressure heatmap render
│   │   └── websocket_client.js      # Live data from cloud backend
│   ├── backend/
│   │   ├── server.py                # FastAPI backend
│   │   ├── mqtt_bridge.py           # MQTT to WebSocket bridge
│   │   ├── db_models.py             # Patient and event data models
│   │   └── alert_engine.py          # Risk threshold and escalation logic
│   └── config/
│       └── thresholds.yaml          # Risk score thresholds, alert levels
│
├── hardware/
│   ├── sensor_mat_schematic.pdf     # PCB schematic for sensor mat
│   ├── actuator_board_schematic.pdf # Solenoid driver board schematic
│   └── bom.csv                      # Full bill of materials with pricing
│
├── docs/
│   ├── system_overview.md
│   ├── setup_guide.md
│   ├── api_reference.md
│   └── regulatory_notes.md
│
├── index.html                       # Project landing page (open in browser)
└── README.md
```

---

## Components Used and Why

### Sensing

| Component | Role | Why This Part |
|---|---|---|
| Custom 32x32 capacitive PCB | Pressure mapping | 1024-cell resolution gives body-zone granularity impossible with commercial FSR mats |
| MLX90640 thermopile array | Skin temperature IR map | Non-contact, 32x24 pixels, I2C, detects temperature differential caused by reduced blood flow before it is visible |
| MAX30102 (x8 nodes) | Reflectance PPG / SpO2 | Detects drop in tissue oxygenation at contact zones — the earliest biological signal of ischemia |

### Processing

| Component | Role | Why This Part |
|---|---|---|
| Lattice CrossLink-NX FPGA | CNN inference engine | Low-power, deterministic real-time inference, parallel sensor stream processing, no OS overhead |
| STM32H7 (480 MHz Cortex-M7) | System controller and PID | Dual-core, hardware FPU, 1 MB RAM, ideal for real-time PID and safety interlocks |
| ESP32-S3 | Wi-Fi + BLE 5.2 gateway | Dual-mode wireless, runs FreeRTOS, handles MQTT without burdening the main controller |

### Actuation

| Component | Role | Why This Part |
|---|---|---|
| Micro-pneumatic bladder grid (8x8) | Pressure redistribution | Silently redistributes load without mechanical movement or patient disturbance |
| Solenoid valve array (24V, normally closed) | Bladder control | Fast response (under 50 ms), fail-safe normally-closed design |
| Miniature air compressor (12V, oil-free) | Bladder inflation source | Quiet operation (under 35 dB), medical-grade oil-free output |

### Software and Cloud

| Component | Role | Why This Part |
|---|---|---|
| Flower (flwr) | Federated learning framework | Privacy-preserving model training across hospital nodes |
| FastAPI | Backend server | Async Python, auto-generates OpenAPI docs, handles WebSocket and REST |
| InfluxDB | Time-series data store | Optimized for sensor telemetry, native Grafana integration |
| MQTT (Mosquitto) | Device messaging protocol | Lightweight, QoS levels for reliable delivery even on hospital Wi-Fi |
| Chart.js | Dashboard heatmap | Lightweight canvas rendering, no heavy dependencies |

---

## How the Prediction Works

The model takes a rolling 5-minute window of all three sensor streams per body zone and outputs a risk score from 0 to 1.

```
Input (per zone, every 6 seconds):
  - pressure_kPa       [float, 10 Hz average]
  - duration_mins      [float, cumulative contact time]
  - skin_temp_delta_C  [float, deviation from baseline]
  - spo2_percent       [float, reflectance PPG reading]

Model: Temporal CNN (1D convolutions over 50-sample window)
  Conv1D(32, kernel=5) -> ReLU -> Conv1D(64, kernel=3) -> GlobalAvgPool -> Dense(1, sigmoid)

Output:
  risk_score [0.0 - 1.0]
    < 0.3   : Safe, no action
    0.3-0.6 : Caution, increase logging frequency
    0.6-0.8 : High risk, alert nurse
    > 0.8   : Critical, trigger autonomous actuation
```

The model is quantized to 8-bit integers after training for FPGA deployment using TensorFlow Lite quantization-aware training.

---

## Firmware: What Runs Where

### FPGA (Verilog)

- Reads all sensor data over SPI/I2C in parallel using custom state machines
- Runs the quantized CNN inference in hardware (fixed-point arithmetic, fully pipelined)
- Outputs risk scores per zone to STM32 over UART at 115200 baud
- Drives solenoid valve PWM at 1 kHz with hardware-enforced pressure limits

### STM32H7 (C, FreeRTOS)

- Reads risk scores from FPGA
- Runs PID controller to maintain target bladder pressure per zone
- Manages safety interlock: if any zone exceeds 200 mmHg, all valves close immediately
- Sends telemetry to ESP32 over UART2

### ESP32-S3 (C++, Arduino framework)

- Publishes sensor telemetry and risk scores to MQTT broker every 6 seconds
- Subscribes to actuation override commands from nurse dashboard
- Maintains BLE connection to bedside display tablet

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/smart-pressure-ulcer-prevention.git
cd smart-pressure-ulcer-prevention
```

### 2. Set up the Python environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Train the prediction model

```bash
cd ml
python train_model.py --epochs 50 --dataset dataset/sample_data.csv
python quantize_model.py --model outputs/model.h5 --output outputs/model_int8.tflite
```

### 4. Flash the firmware

For STM32H7 (using STM32CubeIDE or OpenOCD):
```bash
cd firmware/stm32
make flash BOARD=NUCLEO-H743ZI
```

For ESP32-S3 (using Arduino CLI):
```bash
cd firmware/esp32
arduino-cli compile --fqbn esp32:esp32:esp32s3 .
arduino-cli upload -p /dev/ttyUSB0 --fqbn esp32:esp32:esp32s3 .
```

For FPGA (using Lattice Radiant):
- Open `firmware/fpga/` as a Radiant project
- Synthesize and program via USB JTAG

### 5. Start the backend server

```bash
cd src/backend
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Open the nurse dashboard

Open `src/dashboard/index.html` in a browser, or serve it:
```bash
python3 -m http.server 3000 --directory src/dashboard
```
Navigate to `http://localhost:3000`

### 7. Open the project landing page

Simply open `index.html` in any browser. No server required.

---

## Federated Learning Setup

Each hospital runs a local aggregation server. Beds are clients.

```bash
# On the hospital server
cd ml
python federated_server.py --rounds 10 --min-clients 3

# On each bed node (Raspberry Pi or local compute)
python federated_client.py --server-address hospital-server:8080 --bed-id BED_042
```

Model weights are aggregated using FedAvg. Raw patient sensor data never leaves the bed node.

---

## Configuration

Edit `src/config/thresholds.yaml` to tune alert levels:

```yaml
risk_thresholds:
  safe: 0.30
  caution: 0.60
  high_risk: 0.80
  critical: 0.90

actuation:
  target_pressure_mmhg: 40
  max_pressure_mmhg: 200
  reposition_duration_sec: 180
  cooldown_sec: 600

alerts:
  nurse_pager: true
  sms_gateway: false
  dashboard_popup: true
```

---

## Regulatory and Safety Notes

- This is a research and prototype system. It is not certified for clinical use.
- Target certification path: CDSCO Class B medical device (India), IEC 62304 software lifecycle standard
- All actuation is bounded by hardware interlocks. The system cannot exceed 200 mmHg bladder pressure under any software condition.
- Patient data is encrypted at rest (AES-256) and in transit (TLS 1.3).
- See `docs/regulatory_notes.md` for full compliance discussion.

---

## Future Improvements

- Train on larger clinical datasets (currently using synthetic + limited real data)
- Add EMG muscle activity sensing to detect patient self-repositioning and pause actuation
- Integrate EHR systems (HL7 FHIR) for automatic risk factor adjustment based on diabetes, age, and BMI
- Replace FPGA with a custom ASIC for unit cost reduction below Rs. 8,000 per mat
- Add camera-based wound progression tracking using computer vision for post-ulcer monitoring
- Multi-patient ward dashboard with bed-level triage priority
- Offline-capable edge deployment with no cloud dependency for rural hospitals

---

## Target Users

- ICU and general ward nurses in hospitals with bedridden patients
- Hospital biomedical engineering teams deploying the hardware
- Medical device researchers and embedded systems engineers building on this codebase
- Healthcare IoT developers studying federated learning in clinical settings

---

## License

MIT License. See LICENSE for details. Note that clinical deployment requires separate regulatory approval.

---

## Contributing

Pull requests are welcome. For major changes, open an issue first to discuss the proposed change. Please ensure all firmware changes pass hardware-in-the-loop simulation before submitting.
