# Smart Pressure Ulcer Prevention System

> **Prototype / research concept only. Not a medical device.** This repository has not demonstrated clinical validity, safety, or reliable closed-loop actuation. Do not use its predictions for patient care or connect prototype actuation code to a person.

This repository contains several early-stage software artifacts for a pressure-risk monitoring concept: a simulated browser page, a Python model-training and quantization pipeline, a FastAPI/MQTT backend prototype, a Verilog inference module, and an STM32H7/FreeRTOS C source file. The checked-in files do not form a complete, buildable or validated end-to-end system.

## What is actually in this repository

| File | Purpose and current limitation |
|---|---|
| `index.html` | Standalone concept page with a simulated risk map/meters. Its JavaScript generates demonstration values; it does not connect to `server.py`, sensors, or an FPGA. |
| `train_model.py` | Trains a temporal 1D CNN from a CSV dataset and writes model/training artifacts. No dataset is included in the current tree. |
| `quantize_model.py` | Converts a trained Keras model to int8 TFLite and exports weight/LUT hex files. It expects the trained model and normalization arrays to exist. |
| `server.py` | FastAPI prototype with MQTT telemetry handling, WebSocket updates, and REST endpoints. Startup is currently blocked by the configuration path mismatch described below. |
| `thresholds.yaml` | Root-level risk and actuation threshold configuration. The backend currently looks for it at a different path. |
| `main.c` | STM32H7/FreeRTOS controller source skeleton. It includes `pid_controller.h`, `sensor_read.h`, and `safety_interlock.h`, which are not present in this repository; no STM32 project/build files are included. |
| `cnn_inference.v` | Verilog CNN inference module. It references `weights/*.hex` files that are not checked in, and no FPGA project or testbench is included. |
| `requirements (2).txt` | Python dependency list. The filename includes a suffix, so pass it exactly to pip. |
| `11. Bedsores.pdf` | Reference PDF included in the repository. |

## Architecture represented by the source

```text
CSV dataset -> train_model.py -> Keras model + normalization arrays
                                  |
                                  v
                           quantize_model.py -> TFLite + weight/LUT hex files

Sensor telemetry -> intended MQTT/FastAPI backend -> WebSocket/REST clients
FPGA risk score -> intended STM32H7/FreeRTOS controller -> intended telemetry/actuation
```

The arrows describe intended interfaces in the source and are not evidence that the components have been integrated or tested together.

## Open the browser concept

Open `index.html` directly in a modern browser. This is a visual simulation for demonstrating the concept only. The risk map and risk indicators are generated in the page; they are not measurements or model predictions from a patient or device.

## Python environment and model training

Use a virtual environment and install the dependencies from the exact tracked filename:

```bash
python -m venv .venv
python -m pip install -r "requirements (2).txt"
```

Activate `.venv` using the command appropriate for your shell before running project scripts. The dependency set includes TensorFlow, NumPy, pandas, scikit-learn, matplotlib, FastAPI, and MQTT/InfluxDB-related packages; installation may be large and was not tested during this review.

`train_model.py` defaults to `dataset/sample_data.csv`, but no `dataset/` directory or sample CSV is present in the checked-in tree. Supply a real, appropriately governed dataset yourself:

```bash
python train_model.py --dataset path/to/your_dataset.csv --epochs 50 --output outputs
```

The CSV must contain these columns: `patient_id`, `zone_id`, `timestamp_ms`, `pressure_kpa`, `duration_mins`, `skin_temp_delta_c`, `spo2_percent`, and `risk_label`. The script performs a random train/validation/test split of generated windows; it does not establish clinical performance or prevent data leakage between overlapping windows.

After training has produced `outputs/best_model.h5`, `outputs/norm_mean.npy`, and `outputs/norm_std.npy`, the quantization script can be invoked with:

```bash
python quantize_model.py --model outputs/best_model.h5 --output outputs
```

This command has not been run as part of the documentation review. The generated files still need format, numerical, synthesis, and hardware verification before they can be used by the Verilog module.

## Backend status and configuration blocker

`server.py` loads configuration at import time using `../config/thresholds.yaml` relative to the repository root. The tracked file is `thresholds.yaml` in the repository root, and the expected `config/` directory is absent. As committed, the backend therefore cannot load its configuration without a separate code or layout correction.

The backend also defaults to a local MQTT broker at port `1883` and InfluxDB at `http://localhost:8086`; it uses a development token fallback and permissive CORS. These are development defaults, not production deployment settings. Do not expose the backend publicly or send real patient information to it.

Once the configuration path is corrected and the required services/environment variables are configured, the intended ASGI entry point is `server:app`. No successful backend startup or live telemetry integration is claimed.

## Known gaps and safety limits

- There is no dataset, trained model, exported weight directory, FPGA project/testbench, STM32 project, or the header/source dependencies required by `main.c`.
- The browser page is simulated and is not wired to the FastAPI backend.
- The backend configuration path does not match the current repository layout.
- The Verilog module expects weight/LUT files that are not present.
- Model thresholds and training code are experimental; no clinical accuracy, early-warning lead time, or patient benefit has been established.
- Automated repositioning/pressure actuation is a safety-critical medical function. Do not connect this prototype to a patient or use it in clinical decisions.
- No model training, backend test, HDL simulation/synthesis, firmware build, or physical hardware test was performed for this documentation update.

## License

No conventional `LICENSE` file is present in the checked-in root tree. Confirm licensing before redistributing or deploying this repository.

## Maintenance & Audit

- **Audit date:** 18 November 2026
- **Documentation status:** Repository files and setup claims reviewed; major integration gaps recorded.
- **Verification status:** Documentation/source inspection only. No runtime, model, HDL, embedded, or clinical verification is claimed.
- **Scope:** Source, firmware, model, and hardware behavior were not changed by this documentation update.