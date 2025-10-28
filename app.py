from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from datetime import datetime, timedelta
import json
import os
import psutil
import time
import threading
import random

app = Flask(__name__)
CORS(app)

# In-memory storage for incidents (replace with database in production)
incidents = [
    {
        "id": 1,
        "type": "Aggressive behavior detected",
        "timestamp": "1/9/2025, 6:05:51 pm",
        "confidence": 85,
        "camera": "Camera 01",
        "status": "Confirmed",
        "details": "High confidence detection of aggressive gestures"
    },
    {
        "id": 2,
        "type": "Physical altercation detected", 
        "timestamp": "1/9/2025, 9:05:51 am",
        "confidence": 94,
        "camera": "Camera 01", 
        "status": "Confirmed",
        "details": "Multiple persons involved in physical confrontation"
    }
]

# System status
system_status = {
    "status": "Normal",
    "last_updated": datetime.now().isoformat(),
    "camera_online": True,
    "ai_model_loaded": True,
    "alerts_enabled": True
}

# Settings storage
settings = {
    "night_vision": True,
    "incident_retention": "30 days",
    "auto_delete": False,
    "mobile_alarm": True,
    "buzzer": True,
    "led_alert": False,
    "alert_volume": 80,
    "alert_duration": "10 seconds",
    "camera_resolution": "High (1080p)",
    "user_profile": {
        "username": "Admin User",
        "email": "admin@security.com",
        "user_id": "USR001",
        "product_id": "PRD001"
    }
}

def simulate_system_monitoring():
    """Background thread to simulate system monitoring"""
    while True:
        # Randomly simulate harassment detection (very rarely)
        if random.random() < 0.02:  # 2% chance every 10 seconds
            new_incident = {
                "id": len(incidents) + 1,
                "type": random.choice(["Aggressive behavior detected", "Physical altercation detected", "Verbal harassment detected"]),
                "timestamp": datetime.now().strftime("%m/%d/%Y, %I:%M:%S %p"),
                "confidence": random.randint(75, 98),
                "camera": "Camera 01",
                "status": "Confirmed",
                "details": "AI detected potential harassment incident"
            }
            incidents.insert(0, new_incident)
            system_status["status"] = "Harassment Detected"
            system_status["last_updated"] = datetime.now().isoformat()
            
            # Reset to normal after 30 seconds
            threading.Timer(30.0, lambda: reset_status()).start()
        
        time.sleep(10)

def reset_status():
    system_status["status"] = "Normal"
    system_status["last_updated"] = datetime.now().isoformat()

# Start background monitoring
monitoring_thread = threading.Thread(target=simulate_system_monitoring, daemon=True)
monitoring_thread.start()

@app.route('/')
def home():
    return jsonify({
        "message": "AI Harassment Detector API",
        "status": "running",
        "endpoints": ["/api/status", "/api/incidents", "/api/system-health"]
    })

@app.route('/api/status')
def get_status():
    return jsonify(system_status)

@app.route('/api/incidents')
def get_incidents():
    # Filter by date if requested
    date_filter = request.args.get('date', 'today')
    status_filter = request.args.get('status', 'all')
    
    filtered_incidents = incidents.copy()
    
    if status_filter != 'all':
        filtered_incidents = [i for i in filtered_incidents if i['status'].lower() == status_filter.lower()]
    
    return jsonify(filtered_incidents)

@app.route('/api/system-health')
def get_system_health():
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get system uptime
        boot_time = psutil.boot_time()
        uptime_seconds = time.time() - boot_time
        uptime_hours = uptime_seconds // 3600
        
        # Simulate temperature (would be actual sensor reading on Pi)
        temperature = random.randint(35, 55)
        
        return jsonify({
            "cpu_usage": round(cpu_percent, 1),
            "memory_usage": round(memory.percent, 1),
            "disk_usage": round(disk.percent, 1),
            "uptime_hours": int(uptime_hours),
            "temperature": temperature,
            "camera_status": "Online",
            "ai_model_status": "Active",
            "network_status": "Connected"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/settings', methods=['GET', 'POST'])
def handle_settings():
    if request.method == 'GET':
        return jsonify(settings)
    
    if request.method == 'POST':
        try:
            updated_settings = request.json
            settings.update(updated_settings)
            return jsonify({"success": True, "message": "Settings updated successfully"})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/export/csv')
def export_csv():
    # Simulate CSV export
    return jsonify({"success": True, "message": "CSV export initiated", "download_url": "/downloads/incidents.csv"})

@app.route('/api/export/pdf') 
def export_pdf():
    # Simulate PDF export
    return jsonify({"success": True, "message": "PDF export initiated", "download_url": "/downloads/incidents.pdf"})

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    try:
        feedback_data = request.json
        # In production, save to database
        print(f"Feedback received: {feedback_data}")
        return jsonify({"success": True, "message": "Feedback submitted successfully"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/factory-reset', methods=['POST'])
def factory_reset():
    try:
        # Reset all settings to defaults
        global incidents, settings
        incidents.clear()
        settings.update({
            "night_vision": True,
            "incident_retention": "30 days", 
            "auto_delete": False,
            "mobile_alarm": True,
            "buzzer": True,
            "led_alert": False,
            "alert_volume": 80,
            "alert_duration": "10 seconds",
            "camera_resolution": "High (1080p)"
        })
        return jsonify({"success": True, "message": "System reset to factory defaults"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/login', methods=['POST'])
def login():
    try:
        credentials = request.json
        # Simple authentication (replace with proper auth in production)
        if (credentials.get('user_id') == 'USR001' and 
            credentials.get('product_id') == 'PRD001' and
            credentials.get('email') == 'admin@security.com'):
            return jsonify({
                "success": True, 
                "message": "Login successful",
                "user": settings["user_profile"]
            })
        else:
            return jsonify({"success": False, "error": "Invalid credentials"}), 401
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)