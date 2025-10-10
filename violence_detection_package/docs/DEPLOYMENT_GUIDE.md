# 📘 Violence Detection System - Complete Deployment Guide

**Comprehensive setup instructions for hardware team**

---

## 📋 Table of Contents

1. [Hardware Requirements](#hardware-requirements)
2. [Software Requirements](#software-requirements)
3. [Initial Setup](#initial-setup)
4. [Model Deployment](#model-deployment)
5. [Testing & Validation](#testing--validation)
6. [Running the System](#running-the-system)
7. [Monitoring & Maintenance](#monitoring--maintenance)

---

## 🔧 Hardware Requirements

### **Minimum Requirements:**

| Component | Specification | Notes |
|-----------|--------------|-------|
| **Raspberry Pi** | Pi 4 Model B, 4GB RAM | 8GB recommended for better performance |
| **Camera** | Pi Camera Module v2 | Or HQ Camera for better quality |
| **Microphone** | USB Microphone | Any USB mic compatible with ALSA |
| **Storage** | 32GB microSD, Class 10 | 64GB recommended for logs |
| **Power Supply** | Official 5V 3A adapter | Critical for stability |
| **Cooling** | Heatsinks + fan | Strongly recommended |

### **Optional Components:**

- **Case:** For protection and cooling
- **Display:** HDMI monitor for setup (can run headless after)
- **Network:** Ethernet cable (more stable than WiFi)

---

## 💻 Software Requirements

### **Operating System:**

- **Raspberry Pi OS** (32-bit or 64-bit)
- **Version:** Bullseye or later
- **Type:** Desktop or Lite (Lite recommended for performance)

### **Download OS:**

```bash
# Option 1: Use Raspberry Pi Imager (Recommended)
# Download from: https://www.raspberrypi.com/software/

# Option 2: Manual download
# https://www.raspberrypi.com/software/operating-systems/
```

---

## 🚀 Initial Setup

### **Step 1: Flash Raspberry Pi OS**

1. Insert microSD card into computer
2. Open Raspberry Pi Imager
3. Choose:
   - **OS:** Raspberry Pi OS (64-bit recommended)
   - **Storage:** Your microSD card
4. Click gear icon (⚙️) for advanced options:
   - Set hostname: `violence-detection`
   - Enable SSH
   - Set username and password
   - Configure WiFi (optional)
5. Click **Write** and wait

### **Step 2: First Boot**

1. Insert microSD into Raspberry Pi
2. Connect camera ribbon cable
3. Connect USB microphone
4. Connect monitor, keyboard, mouse
5. Power on

### **Step 3: Initial Configuration**

```bash
# Open configuration tool
sudo raspi-config
```

**Configure:**
1. **System Options** → **Password** → Change default password
2. **Interface Options** → **Camera** → Enable
3. **Performance Options** → **GPU Memory** → Set to 256
4. **Localisation Options** → Set timezone
5. **Finish** → Reboot when prompted

### **Step 4: Update System**

```bash
sudo apt-get update
sudo apt-get full-upgrade -y
sudo reboot
```

---

## 📦 Model Deployment

### **Step 1: Transfer Package**

**Option A: USB Drive**
```bash
# Insert USB drive
sudo mkdir /mnt/usb
sudo mount /dev/sda1 /mnt/usb
cp /mnt/usb/violence_detection_package.zip ~/
sudo umount /mnt/usb
```

**Option B: SCP (from another computer)**
```bash
# From your computer
scp violence_detection_package.zip pi@192.168.1.x:~/
```

**Option C: Download via wget (if hosted online)**
```bash
wget https://yourserver.com/violence_detection_package.zip
```

### **Step 2: Extract Package**

```bash
cd ~/
unzip violence_detection_package.zip
cd violence_detection_package
```

### **Step 3: Run Setup Script**

```bash
cd scripts
chmod +x rpi4_setup.sh
./rpi4_setup.sh
```

**What the script does:**
- ✅ Installs Python dependencies
- ✅ Installs OpenCV
- ✅ Installs audio libraries
- ✅ Installs TensorFlow Lite
- ✅ Enables camera
- ✅ Creates project directories
- ✅ Optimizes system settings

**Expected Duration:** 10-15 minutes

### **Step 4: Extract and Organize Models**

```bash
cd ~/violence_detection_package/models

# Extract vision model
unzip violence_detection_rpi4.zip
mkdir -p ~/violence_detection/models
mv violence_detection.tflite ~/violence_detection/models/vision_model.tflite
mv labels.txt ~/violence_detection/models/vision_labels.txt
mv config.json ~/violence_detection/models/vision_config.json

# Extract audio model
unzip audio_detection_rpi4.zip
mv audio_detection.tflite ~/violence_detection/models/audio_model.tflite
mv labels.txt ~/violence_detection/models/audio_labels.txt
# If there's another config.json, rename it
mv config.json ~/violence_detection/models/audio_config.json 2>/dev/null || true

# Clean up
cd ~/
rm -rf violence_detection_package/models/*.zip
```

### **Step 5: Copy Scripts and Config**

```bash
# Copy detection script
cp ~/violence_detection_package/scripts/integrated_detection.py ~/violence_detection/

# Copy configuration
cp ~/violence_detection_package/scripts/config.json ~/violence_detection/

# Make script executable
chmod +x ~/violence_detection/integrated_detection.py
```

### **Step 6: Verify File Structure**

```bash
cd ~/violence_detection
tree -L 2
```

**Expected output:**
```
~/violence_detection/
├── models/
│   ├── vision_model.tflite
│   ├── audio_model.tflite
│   ├── vision_labels.txt
│   ├── audio_labels.txt
│   └── *_config.json
├── logs/
├── alerts/
├── scripts/
├── integrated_detection.py
└── config.json
```

---

## 🧪 Testing & Validation

### **Test 1: Camera**

```bash
# Test camera capture
raspistill -o ~/test_camera.jpg

# View image
display ~/test_camera.jpg

# Or transfer to computer to view
scp pi@192.168.1.x:~/test_camera.jpg ./
```

**Troubleshooting:**
```bash
# If camera not detected
vcgencmd get_camera

# Should show:
# supported=1 detected=1

# If not, enable in raspi-config and reboot
```

### **Test 2: Microphone**

```bash
# List audio devices
arecord -l

# Should show your USB microphone

# Record 5 seconds of audio
arecord -D plughw:1,0 -d 5 -f cd ~/test_audio.wav

# Play back
aplay ~/test_audio.wav
```

**Troubleshooting:**
```bash
# If no microphone listed
lsusb  # Check if USB mic is detected

# Set default device
nano ~/.asoundrc

# Add:
pcm.!default {
    type hw
    card 1
}
ctl.!default {
    type hw
    card 1
}
```

### **Test 3: Python Dependencies**

```bash
python3 << EOF
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter
import pyaudio

print("✅ All dependencies imported successfully!")
print(f"OpenCV version: {cv2.__version__}")
print(f"NumPy version: {np.__version__}")
EOF
```

**Expected output:**
```
✅ All dependencies imported successfully!
OpenCV version: 4.x.x
NumPy version: 1.23.x
```

### **Test 4: Model Loading**

```bash
cd ~/violence_detection

python3 << EOF
from tflite_runtime.interpreter import Interpreter

# Test vision model
print("Loading vision model...")
v_interpreter = Interpreter('models/vision_model.tflite')
v_interpreter.allocate_tensors()
print("✅ Vision model loaded")

# Test audio model
print("Loading audio model...")
a_interpreter = Interpreter('models/audio_model.tflite')
a_interpreter.allocate_tensors()
print("✅ Audio model loaded")

print("\n✅ All models loaded successfully!")
EOF
```

### **Test 5: Quick System Test**

```bash
cd ~/violence_detection

# Run detection for 10 seconds
timeout 10 python3 integrated_detection.py || true

# Check if logs were created
ls -lh logs/
ls -lh alerts/
```

---

## 🎮 Running the System

### **Method 1: Interactive Mode (with display)**

```bash
cd ~/violence_detection
python3 integrated_detection.py
```

**Controls:**
- Press `q` to quit
- Press `s` to save screenshot
- Press `ESC` to exit

### **Method 2: Headless Mode (no display)**

```bash
# Edit config.json first
nano config.json

# Set:
"display": {
  "show_preview": false
}

# Run
python3 integrated_detection.py
```

### **Method 3: Background Service**

```bash
# Run in background
nohup python3 integrated_detection.py > system.log 2>&1 &

# Check if running
ps aux | grep integrated_detection

# View live logs
tail -f system.log

# Stop
pkill -f integrated_detection.py
```

### **Method 4: Auto-Start on Boot**

Create systemd service:

```bash
sudo nano /etc/systemd/system/violence-detection.service
```

**Add:**
```ini
[Unit]
Description=Violence Detection System
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/violence_detection
ExecStart=/usr/bin/python3 /home/pi/violence_detection/integrated_detection.py
Restart=always
RestartSec=10
StandardOutput=append:/home/pi/violence_detection/logs/service.log
StandardError=append:/home/pi/violence_detection/logs/service_error.log

[Install]
WantedBy=multi-user.target
```

**Enable and start:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable violence-detection.service
sudo systemctl start violence-detection.service

# Check status
sudo systemctl status violence-detection.service

# View logs
sudo journalctl -u violence-detection.service -f
```

**Manage service:**
```bash
sudo systemctl stop violence-detection.service    # Stop
sudo systemctl restart violence-detection.service # Restart
sudo systemctl disable violence-detection.service # Disable auto-start
```

---

## 📊 Monitoring & Maintenance

### **Real-Time Monitoring**

**Monitor logs:**
```bash
# Alert logs
tail -f ~/violence_detection/logs/alerts_$(date +%Y%m%d).log

# System logs
tail -f ~/violence_detection/logs/service.log
```

**Monitor system resources:**
```bash
# CPU and RAM usage
htop

# Temperature
vcgencmd measure_temp

# Disk space
df -h
```

### **Daily Checks**

```bash
#!/bin/bash
# Save as: ~/daily_check.sh

echo "=== Violence Detection System - Daily Check ==="
echo ""
echo "Date: $(date)"
echo ""

# Check if service is running
if systemctl is-active --quiet violence-detection.service; then
    echo "✅ Service: Running"
else
    echo "❌ Service: NOT running"
fi

# Check disk space
DISK_USAGE=$(df -h / | tail -1 | awk '{print $5}' | sed 's/%//')
if [ $DISK_USAGE -gt 80 ]; then
    echo "⚠️  Disk: ${DISK_USAGE}% used (HIGH)"
else
    echo "✅ Disk: ${DISK_USAGE}% used"
fi

# Check temperature
TEMP=$(vcgencmd measure_temp | cut -d'=' -f2 | cut -d"'" -f1)
if (( $(echo "$TEMP > 70" | bc -l) )); then
    echo "⚠️  Temperature: ${TEMP}°C (HIGH)"
else
    echo "✅ Temperature: ${TEMP}°C"
fi

# Count today's alerts
ALERT_COUNT=$(grep -c "ALERT" ~/violence_detection/logs/alerts_$(date +%Y%m%d).log 2>/dev/null || echo "0")
echo "📊 Alerts today: $ALERT_COUNT"

echo ""
echo "=== Check Complete ==="
```

**Make it executable and run:**
```bash
chmod +x ~/daily_check.sh
./daily_check.sh
```

### **Log Rotation**

**Automatic cleanup of old logs:**

```bash
# Create cleanup script
nano ~/violence_detection/cleanup_logs.sh
```

**Add:**
```bash
#!/bin/bash
# Delete logs older than 30 days

LOG_DIR=~/violence_detection/logs
ALERT_DIR=~/violence_detection/alerts

# Delete old logs
find $LOG_DIR -name "*.log" -mtime +30 -delete
echo "Deleted logs older than 30 days"

# Delete old alerts
find $ALERT_DIR -name "*.jpg" -mtime +30 -delete
echo "Deleted alerts older than 30 days"

# Optional: Compress logs older than 7 days
find $LOG_DIR -name "*.log" -mtime +7 ! -name "*.gz" -exec gzip {} \;
echo "Compressed logs older than 7 days"
```

**Schedule with cron:**
```bash
chmod +x ~/violence_detection/cleanup_logs.sh

# Edit crontab
crontab -e

# Add (runs daily at 2 AM):
0 2 * * * /home/pi/violence_detection/cleanup_logs.sh
```

### **Performance Monitoring**

**Create performance monitor:**
```bash
nano ~/performance_monitor.sh
```

**Add:**
```bash
#!/bin/bash

echo "=== System Performance ==="
echo ""

# CPU usage
CPU=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
echo "CPU Usage: ${CPU}%"

# RAM usage
RAM=$(free -m | awk 'NR==2{printf "%.0f", $3*100/$2 }')
echo "RAM Usage: ${RAM}%"

# Temperature
TEMP=$(vcgencmd measure_temp | cut -d'=' -f2)
echo "Temperature: $TEMP"

# Process info
PROCESS_CPU=$(ps aux | grep integrated_detection.py | grep -v grep | awk '{print $3}')
PROCESS_MEM=$(ps aux | grep integrated_detection.py | grep -v grep | awk '{print $4}')
echo ""
echo "Detection Process:"
echo "  CPU: ${PROCESS_CPU}%"
echo "  Memory: ${PROCESS_MEM}%"
```

### **Backup Configuration**

```bash
# Backup script
nano ~/backup_config.sh
```

**Add:**
```bash
#!/bin/bash

BACKUP_DIR=~/violence_detection_backups
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Backup config
cp ~/violence_detection/config.json $BACKUP_DIR/config_$DATE.json

# Backup logs summary
tail -100 ~/violence_detection/logs/alerts_$(date +%Y%m%d).log > $BACKUP_DIR/recent_alerts_$DATE.log

# Backup models info
ls -lh ~/violence_detection/models/ > $BACKUP_DIR/models_info_$DATE.txt

echo "✅ Backup created: $BACKUP_DIR"
```

---

## 🔄 Updates & Upgrades

### **Updating System Software**

```bash
sudo apt-get update
sudo apt-get upgrade -y
sudo reboot
```

### **Updating Detection Script**

```bash
# Stop service
sudo systemctl stop violence-detection.service

# Backup current version
cp ~/violence_detection/integrated_detection.py ~/integrated_detection.py.backup

# Copy new version
cp /path/to/new/integrated_detection.py ~/violence_detection/

# Restart service
sudo systemctl start violence-detection.service
```

### **Updating Models**

```bash
# Stop detection
sudo systemctl stop violence-detection.service

# Backup old models
mv ~/violence_detection/models ~/violence_detection/models.backup

# Copy new models
mkdir ~/violence_detection/models
# ... copy new model files ...

# Test new models
cd ~/violence_detection
python3 integrated_detection.py  # Test run

# If working, start service
sudo systemctl start violence-detection.service

# If not working, restore backup
rm -rf ~/violence_detection/models
mv ~/violence_detection/models.backup ~/violence_detection/models
```

---

## 🛡️ Security Best Practices

### **1. Change Default Credentials**

```bash
# Change password
passwd

# Change SSH keys
ssh-keygen -t rsa -b 4096
```

### **2. Enable Firewall**

```bash
sudo apt-get install ufw
sudo ufw allow ssh
sudo ufw enable
```

### **3. Secure Log Files**

```bash
chmod 700 ~/violence_detection/logs
chmod 700 ~/violence_detection/alerts
```

### **4. Regular Updates**

```bash
# Set up automatic security updates
sudo apt-get install unattended-upgrades
sudo dpkg-reconfigure --priority=low unattended-upgrades
```

### **5. Encrypted Storage (Optional)**

```bash
# Install encryption tools
sudo apt-get install cryptsetup

# Create encrypted container for sensitive logs
# (See full encryption guide online)
```

---

## 📈 Performance Tuning

### **Optimize for Speed**

**In config.json:**
```json
{
  "camera": {
    "width": 320,
    "height": 240,
    "fps": 20
  },
  "performance": {
    "frame_skip": 1,
    "optimize_for_speed": true
  }
}
```

### **Optimize for Accuracy**

**In config.json:**
```json
{
  "camera": {
    "width": 640,
    "height": 480,
    "fps": 30
  },
  "detection": {
    "alert_threshold": 0.75
  },
  "performance": {
    "frame_skip": 0,
    "optimize_for_speed": false
  }
}
```

### **Overclock (Use with caution)**

```bash
sudo raspi-config
# Performance Options → Overclock → Modest/Medium

# Monitor temperature after overclocking!
watch -n 1 vcgencmd measure_temp
```

---

## 🆘 Emergency Procedures

### **System Not Responding**

```bash
# Force stop detection
sudo pkill -9 -f integrated_detection.py

# Restart service
sudo systemctl restart violence-detection.service
```

### **Disk Full**

```bash
# Find large files
du -h ~/violence_detection | sort -rh | head -20

# Delete old alerts
rm ~/violence_detection/alerts/*.jpg

# Clean up logs
rm ~/violence_detection/logs/*.log
```

### **System Crash Recovery**

```bash
# Check system logs
sudo journalctl -xe

# Check detection logs
tail -100 ~/violence_detection/logs/service_error.log

# Restart from clean state
sudo systemctl stop violence-detection.service
rm ~/violence_detection/logs/*
sudo systemctl start violence-detection.service
```

### **Factory Reset**

```bash
# Backup important files first!
cp -r ~/violence_detection ~/violence_detection_backup

# Remove everything
rm -rf ~/violence_detection

# Re-run setup script
cd ~/violence_detection_package/scripts
./rpi4_setup.sh
```

---

## ✅ Deployment Checklist

### **Pre-Deployment:**
- [ ] Hardware assembled correctly
- [ ] Raspberry Pi OS installed
- [ ] Internet connection available (for setup)
- [ ] All package files transferred

### **During Deployment:**
- [ ] Setup script completed successfully
- [ ] Models extracted and verified
- [ ] Camera test passed
- [ ] Microphone test passed
- [ ] Detection system runs without errors

### **Post-Deployment:**
- [ ] Auto-start configured (if desired)
- [ ] Logs being created correctly
- [ ] Alerts trigger appropriately
- [ ] Performance acceptable
- [ ] Monitoring system in place
- [ ] Backup procedures documented
- [ ] Team trained on system operation

---

## 📞 Support Contacts

**For Technical Issues:**
- ML Team: [Your Email]
- Phone: [Your Phone]

**For Hardware Issues:**
- Hardware Team Lead: [Contact]

**Emergency Contact:**
- [24/7 Support Number]

---

## 📚 Additional Resources

**Official Documentation:**
- Raspberry Pi: https://www.raspberrypi.com/documentation/
- TensorFlow Lite: https://www.tensorflow.org/lite
- OpenCV: https://docs.opencv.org/

**Community Support:**
- Raspberry Pi Forums: https://forums.raspberrypi.com/
- Stack Overflow: https://stackoverflow.com/

---

**Document Version:** 1.0  
**Last Updated:** January 2025  
**Maintained By:** ML Team