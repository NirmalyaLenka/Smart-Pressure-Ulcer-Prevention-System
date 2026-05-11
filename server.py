"""
Smart Pressure Ulcer Prevention System
FastAPI Backend Server

Responsibilities:
  - Bridge MQTT telemetry from devices to WebSocket clients (nurse dashboard)
  - Store patient and event data in InfluxDB
  - Serve REST API for historical data and configuration
  - Apply alert thresholds and trigger nurse notifications
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import aiomqtt
from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync

app = FastAPI(
    title="Pressure Ulcer Prevention API",
    description="Real-time monitoring and alert system for hospital bed sensor mats",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config():
    config_path = os.path.join(os.path.dirname(__file__),
                               "../config/thresholds.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)

CONFIG = load_config()
THRESHOLDS = CONFIG["risk_thresholds"]

MQTT_BROKER   = os.getenv("MQTT_BROKER",   "localhost")
MQTT_PORT     = int(os.getenv("MQTT_PORT", "1883"))
MQTT_TOPIC    = "beds/+/telemetry"

INFLUX_URL    = os.getenv("INFLUX_URL",    "http://localhost:8086")
INFLUX_TOKEN  = os.getenv("INFLUX_TOKEN",  "dev-token")
INFLUX_ORG    = os.getenv("INFLUX_ORG",    "hospital")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET", "pressure_ulcer")

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class ZoneTelemetry(BaseModel):
    zone: int
    risk: float
    state: int
    pressure: float
    temp_delta: float
    spo2: float
    actuating: int

class BedTelemetry(BaseModel):
    bed_id: str
    timestamp: datetime
    zones: List[ZoneTelemetry]

class AlertEvent(BaseModel):
    bed_id: str
    zone: int
    risk_score: float
    alert_level: str
    timestamp: datetime
    auto_actuated: bool

# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------

class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        self.active.remove(ws)

    async def broadcast(self, message: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.active.remove(ws)

manager = ConnectionManager()

# ---------------------------------------------------------------------------
# In-memory state (replace with Redis in production)
# ---------------------------------------------------------------------------

bed_states: Dict[str, dict] = {}
alert_history: List[AlertEvent] = []

# ---------------------------------------------------------------------------
# Alert logic
# ---------------------------------------------------------------------------

def classify_alert(risk: float) -> Optional[str]:
    if risk >= THRESHOLDS["critical"]:
        return "CRITICAL"
    if risk >= THRESHOLDS["high_risk"]:
        return "HIGH"
    if risk >= THRESHOLDS["caution"]:
        return "CAUTION"
    return None

async def process_telemetry(bed_id: str, zone_data: dict):
    risk  = zone_data.get("risk", 0.0)
    zone  = zone_data.get("zone", -1)
    level = classify_alert(risk)

    if level in ("HIGH", "CRITICAL"):
        event = AlertEvent(
            bed_id=bed_id,
            zone=zone,
            risk_score=risk,
            alert_level=level,
            timestamp=datetime.utcnow(),
            auto_actuated=(zone_data.get("actuating", 0) == 1),
        )
        alert_history.append(event)
        if len(alert_history) > 1000:
            alert_history.pop(0)

        await manager.broadcast({
            "type": "alert",
            "bed_id": bed_id,
            "zone": zone,
            "risk": risk,
            "level": level,
            "auto_actuated": event.auto_actuated,
            "timestamp": event.timestamp.isoformat(),
        })

# ---------------------------------------------------------------------------
# MQTT bridge (runs as background task)
# ---------------------------------------------------------------------------

async def mqtt_bridge():
    async with aiomqtt.Client(MQTT_BROKER, MQTT_PORT) as client:
        await client.subscribe(MQTT_TOPIC)
        async for message in client.messages:
            topic_parts = str(message.topic).split("/")
            if len(topic_parts) < 3:
                continue
            bed_id = topic_parts[1]

            try:
                payload = json.loads(message.payload)
            except json.JSONDecodeError:
                continue

            zone_data = payload
            bed_states.setdefault(bed_id, {})[zone_data.get("zone", 0)] = zone_data

            await manager.broadcast({
                "type": "telemetry",
                "bed_id": bed_id,
                **zone_data,
            })

            await process_telemetry(bed_id, zone_data)

# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws/live")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        # Send current state snapshot on connect
        await ws.send_json({
            "type": "snapshot",
            "beds": bed_states,
        })
        while True:
            # Keep alive; actual data is pushed by MQTT bridge
            await asyncio.sleep(30)
            await ws.send_json({"type": "ping"})
    except WebSocketDisconnect:
        manager.disconnect(ws)

# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/api/beds")
async def get_beds():
    return {"beds": list(bed_states.keys()), "count": len(bed_states)}

@app.get("/api/beds/{bed_id}")
async def get_bed(bed_id: str):
    if bed_id not in bed_states:
        raise HTTPException(status_code=404, detail="Bed not found")
    return bed_states[bed_id]

@app.get("/api/alerts")
async def get_alerts(limit: int = 50, level: Optional[str] = None):
    alerts = alert_history[-limit:]
    if level:
        alerts = [a for a in alerts if a.alert_level == level]
    return [a.dict() for a in alerts]

@app.get("/api/config")
async def get_config():
    return CONFIG

@app.put("/api/config/thresholds")
async def update_thresholds(thresholds: dict):
    CONFIG["risk_thresholds"].update(thresholds)
    return {"status": "updated", "thresholds": CONFIG["risk_thresholds"]}

@app.get("/health")
async def health():
    return {"status": "ok", "ws_clients": len(manager.active)}

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    asyncio.create_task(mqtt_bridge())
