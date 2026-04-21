from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import sqlite3
from typing import List, Optional
from datetime import datetime

app = FastAPI(title="PMS IoT Backend", description="Backend for Predictive Maintenance System ESP32 integration")

DB_FILE = "sensor_data.db"

# Pydantic Model for incoming ESP32 Data
class SensorDataInput(BaseModel):
    machine_id: str = Field(..., example="ESP32-01")
    machine_type: str = Field(..., example="M", description="Quality type: L, M, or H")
    air_temp: float = Field(..., example=298.1)
    process_temp: float = Field(..., example=308.5)
    rotational_speed: int = Field(..., example=1500)
    torque: float = Field(..., example=40.5)
    tool_wear: int = Field(..., example=120)

class SensorDataOutput(SensorDataInput):
    id: int
    timestamp: str

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS telemetry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            machine_id TEXT,
            machine_type TEXT,
            air_temp REAL,
            process_temp REAL,
            rotational_speed INTEGER,
            torque REAL,
            tool_wear INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# Initialize DB on startup
init_db()

@app.post("/api/telemetry", status_code=201)
def add_telemetry(data: SensorDataInput):
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO telemetry 
            (machine_id, machine_type, air_temp, process_temp, rotational_speed, torque, tool_wear)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            data.machine_id, data.machine_type, data.air_temp, 
            data.process_temp, data.rotational_speed, data.torque, data.tool_wear
        ))
        conn.commit()
        conn.close()
        return {"status": "success", "message": "Telemetry data inserted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/telemetry/latest", response_model=List[SensorDataOutput])
def get_latest_telemetry(limit: int = 10):
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get the latest row for each machine
        cursor.execute('''
            SELECT * FROM telemetry 
            WHERE id IN (
                SELECT MAX(id) FROM telemetry GROUP BY machine_id
            )
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for r in rows:
            results.append(SensorDataOutput(
                id=r["id"],
                machine_id=r["machine_id"],
                machine_type=r["machine_type"],
                air_temp=r["air_temp"],
                process_temp=r["process_temp"],
                rotational_speed=r["rotational_speed"],
                torque=r["torque"],
                tool_wear=r["tool_wear"],
                timestamp=r["timestamp"]
            ))
            
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
