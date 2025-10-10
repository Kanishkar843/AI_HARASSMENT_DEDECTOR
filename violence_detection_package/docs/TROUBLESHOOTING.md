# 🔧 Troubleshooting Guide

**Common issues and solutions for the Violence Detection System**

---

## 📋 Quick Diagnosis

Run this diagnostic script first:

```bash
cd ~/violence_detection

python3 << EOF
import sys
print("=== System Diagnostic ===\n")

# Check Python version
print(f"Python: {sys.version}")

# Check imports
try:
    import cv2
    print(f"✅ OpenCV: {cv2.__version__}")
except:
    print("❌ OpenCV: NOT installed")

try:
    import numpy as np
    print(f"✅ NumPy: {np.__version__}")
except:
    print("❌ NumPy: NOT installed")

try:
    from tflite_runtime.interpreter import Interpreter
    print("✅ TFLite Runtime: Installed")
except:
    print("❌ TFLite Runtime: NOT installed")

try:
    import pyaudio
    print("✅ PyAudio: Installed")
except:
    print("❌ PyAudio: NOT installed")

print("\n=== Hardware Checks ===\n")

# Check camera
import subprocess
try:
    result = subprocess.run(['vcgencmd', 'get_camera'], capture_output=True, text=True)
    print(f"Camera: {result.stdout.strip()}")
except:
    print("⚠️  Cannot check camera")

# Check temperature
try:
    result = subprocess.run(['vcgencmd', 'measure_temp'], capture_output=True, text=True)
    print(f"Temperature: {result.stdout.strip()}")
except:
    print("⚠️  Cannot check temperature")

EOF
```

---

## 🚨 Common Issues

### **Issue 1: "No module named 'tflite_runtime'"**

**Cause:** TensorFlow Lite not installed

**Solution:**
```bash
pip3 install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime

# If that fails, try:
pip3 install tflite-runtime
```

**Verify:**
```bash
python3 -c "from tflite_runtime.interpreter import Interpreter; print('✅ TFLite installed')"
```

---

### **Issue 2: "Camera not detected" / "Cannot open camera"**

**Symptoms:**
- Black screen instead of camera feed
- Error: "Failed to grab frame"
- Camera test fails

**Solutions:**

**A. Enable camera in config:**
```bash
sudo raspi-config
# Interface Options → Camera → Enable
sudo reboot
```

**B. Check camera connection:**
```bash
# Check if camera is detected
vcgencmd get_camera

# Should show: supported=1 detected=1
# If detected=0, camera cable is loose
```

**C. Test camera:**
```bash
# Take a test photo
raspistill -o test.jpg

# If this works but detection doesn't, check permissions:
ls -l /dev/video0
# Should show: crw-rw----+ 1 root video
```

**D. Check camera module:**
```bash
# List video devices
v4l2-ctl --list-devices

# Try different camera index in config.json:
"camera": {
  "device_id": 0  # Try 0, 1, or 2
}
```

**E. Power issue:**
```bash
# Camera needs stable power
# Use official 5V 3A power supply
# Check voltage:
vcgencmd get_throttled
# If output is not "0x0", power is insufficient
```

---

### **Issue 3: "No microphone detected" / "Audio device not found"**

**Symptoms:**
- Audio detection not working
- Error listing audio devices
- `arecord -l` shows no devices

**Solutions:**

**A. Check USB microphone:**
```bash
# List USB devices
lsusb

# List audio devices
arecord -l

# Should show your USB mic
```

**B. Set correct audio device:**
```bash
# Find device card number
arecord -l

# Output example:
# card 1: Device [USB Audio Device], device 0: USB Audio [USB Audio]

# Test recording:
arecord -D plughw:1,0 -d 3 test.wav
aplay test.wav
```

**C. Set default device:**
```bash
nano ~/.asoundrc
```

Add:
```
pcm.!default {
    type hw
    card 1
    device 0
}

ctl.!default {
    type hw
    card 1
}
```

**D. Fix permissions:**
```bash
sudo usermod -a -G audio pi
sudo reboot
```

---

### **Issue 4: Low FPS / Laggy Performance**

**Symptoms:**
- Less than 10 FPS
- Delayed detection
- System feels slow

**Solutions:**

**A. Reduce camera resolution:**

Edit `config.json`:
```json
{
  "camera": {
    "width": 320,
    "height": 240,
    "fps": 20
  }
}
```

**B. Check CPU usage:**
```bash
htop

# If CPU is at 100%, other processes may be interfering
```

**C. Close unnecessary programs:**
```bash
# Stop desktop environment (if running headless)
sudo systemctl stop lightdm

# Disable Bluetooth (if not needed)
sudo systemctl disable bluetooth
sudo systemctl stop bluetooth
```

**D. Increase GPU memory:**
```bash
sudo nano /boot/config.txt

# Add or modify:
gpu_mem=256

# Reboot
sudo reboot
```

**E. Enable performance mode:**
```bash
# Check current governor
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

# Set to performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

**F. Overclock (careful!):**
```bash
sudo raspi-config
# Performance Options → Overclock → Modest

# Monitor temperature:
watch -n 1 vcgencmd measure_temp
# Should stay below 80°C
```

---

### **Issue 5: Too Many False Alerts**

**Symptoms:**
- Alerts triggering on normal activities
- System too sensitive

**Solutions:**

**A. Increase confidence threshold:**

Edit `config.json`:
```json
{
  "detection": {
    "alert_threshold": 0.80  // Increase from 0.70
  }
}
```

**B. Increase cooldown:**
```json
{
  "detection": {
    "alert_cooldown_frames": 60  // Increase from 30
  }
}
```

**C. Disable audio detection temporarily:**
```json
{
  "detection": {
    "audio_enabled": false
  }
}
```

**D. Review alert logs:**
```bash
# Check what's triggering alerts
grep "ALERT" ~/violence_detection/logs/alerts_$(date +%Y%m%d).log

# View saved alert images
ls -lh ~/violence_detection/alerts/
```

---

### **Issue 6: No Alerts / System Not Detecting**

**Symptoms:**
- No alerts even during obvious violence
- System seems to run but doesn't detect anything

**Solutions:**

**A. Lower confidence threshold:**
```json
{
  "detection": {
    "alert_threshold": 0.60  // Decrease from 0.70
  }
}
```

**B. Check if detection is actually running:**
```bash
# View live output
python3 integrated_detection.py

# Watch for confidence scores in terminal
```

**C. Verify models loaded:**
```bash
python3 << EOF
from tflite_runtime.interpreter import Interpreter

v_model = Interpreter('models/vision_model.tflite')
v_model.allocate_tensors()
print("✅ Vision model OK")

a_model = Interpreter('models/audio_model.tflite')
a_model.allocate_tensors()
print("✅ Audio model OK")
EOF
```

**D. Test with known violent content:**
```bash
# Download test video
wget https://example.com/test_violence_video.mp4

# Run detection on video file
# (Modify script to accept video file input)
```

---

### **Issue 7: High Temperature / Overheating**

**Symptoms:**
- Temperature above 80°C
- System throttling
- Sudden crashes

**Solutions:**

**A. Check current temperature:**
```bash
vcgencmd measure_temp

# Also check throttling:
vcgencmd get_throttled
# 0x0 = all good
# Other values = throttling occurring
```

**B. Improve cooling:**
- Add heatsinks to CPU and RAM
- Install active cooling fan
- Ensure good airflow around Pi
- Don't enclose in tight spaces

**C. Reduce workload:**
```json
{
  "camera": {
    "fps": 15  // Reduce from 30
  },
  "performance": {
    "frame_skip": 2  // Process every 3rd frame
  }
}
```

**D. Disable overclock:**
```bash
sudo raspi-config
# Performance Options → Overclock → None
```

---

### **Issue 8: "Permission denied" Errors**

**Symptoms:**
- Cannot write to logs
- Cannot save alerts
- File access errors

**Solutions:**

**A. Fix directory permissions:**
```bash
sudo chown -R pi:pi ~/violence_detection
chmod -R 755 ~/violence_detection
```

**B. Fix camera permissions:**
```bash
sudo usermod -a -G video pi
sudo reboot
```

**C. Fix audio permissions:**
```bash
sudo usermod -a -G audio pi
sudo reboot
```

---

### **Issue 9: Disk Space Full**

**Symptoms:**
- "No space left on device"
- Cannot save alerts
- System crashes

**Solutions:**

**A. Check disk usage:**
```bash
df -h

# Check which folders are large:
du -h ~/violence_detection | sort -rh | head -20
```

**B. Clean up old files:**
```bash
# Delete old alerts (older than 7 days)
find ~/violence_detection/alerts -name "*.jpg" -mtime +7 -delete

# Delete old logs
find ~/violence_detection/logs -name "*.log" -mtime +30 -delete

# Clean package cache
sudo apt-get clean
sudo apt-get autoclean
```

**C. Enable log rotation:**
```bash
# Create cleanup script
nano ~/violence_detection/cleanup.sh
```

Add:
```bash
#!/bin/bash
find ~/violence_detection/alerts -mtime +7 -delete
find ~/violence_detection/logs -mtime +30 -delete
```

```bash
chmod +x ~/violence_detection/cleanup.sh

# Add to crontab (daily at 2 AM)
crontab -e
# Add: 0 2 * * * ~/violence_detection/cleanup.sh
```

---

### **Issue 10: System Crashes / Freezes**

**Symptoms:**
- Raspberry Pi becomes unresponsive
- Need to power cycle
- System hangs

**Solutions:**

**A. Check logs before crash:**
```bash
sudo journalctl -xe
tail -100 ~/violence_detection/logs/service_error.log
```

**B. Check power supply:**
```bash
# Undervoltage causes crashes
vcgencmd get_throttled

# If not "0x0", power supply is insufficient
# Use official 5V 3A adapter
```

**C. Check memory:**
```bash
free -h

# If swap is heavily used, add more swap:
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Change CONF_SWAPSIZE=100 to CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

**D. Add watchdog:**
```bash
# Install watchdog
sudo apt-get install watchdog

# Configure
sudo nano /etc/watchdog.conf

# Uncomment:
# watchdog-device = /dev/watchdog
# max-load-1 = 24

# Enable
sudo systemctl enable watchdog
sudo systemctl start watchdog
```

---

### **Issue 11: Network Issues (If using remote access)**

**Symptoms:**
- Cannot SSH to Pi
- Cannot transfer files
- Remote connection drops

**Solutions:**

**A. Find Pi's IP address:**
```bash
# On the Pi:
hostname -I

# On your router, check connected devices
```

**B. Use static IP:**
```bash
sudo nano /etc/dhcpcd.conf

# Add at end:
interface wlan0
static ip_address=192.168.1.100/24
static routers=192.168.1.1
static domain_name_servers=192.168.1.1 8.8.8.8

sudo reboot
```

**C. Enable SSH:**
```bash
sudo raspi-config
# Interface Options → SSH → Enable
```

---

## 🆘 Emergency Recovery

### **System Won't Boot**

1. Remove microSD card
2. Insert into computer
3. Check `/boot/config.txt` for errors
4. Remove any recent changes
5. Reinsert and boot

### **Complete Reset**

```bash
# Backup first!
cp -r ~/violence_detection ~/violence_detection_backup_$(date +%Y%m%d)

# Remove everything
rm -rf ~/violence_detection

# Re-extract package
cd ~
unzip violence_detection_package.zip
cd violence_detection_package/scripts
./rpi4_setup.sh

# Restore models
cp ~/violence_detection_backup_*/models/* ~/violence_detection/models/
```

---

## 📊 Performance Benchmarks

**Normal Performance:**
- FPS: 15-25
- CPU Usage: 40-60%
- RAM Usage: ~500MB
- Temperature: 50-70°C

**If outside these ranges:**
- <10 FPS → Check "Low FPS" section
- >80% CPU → Close other programs
- >80°C → Check "Overheating" section

---

## 📞 Getting Help

**Before contacting support, collect:**

1. **System info:**
```bash
uname -a
cat /etc/os-release
vcgencmd version
```

2. **Error logs:**
```bash
tail -100 ~/violence_detection/logs/service_error.log
sudo journalctl -u violence-detection.service -n 100
```

3. **Diagnostic output:**
```bash
python3 integrated_detection.py > debug.log 2>&1
```

4. **System status:**
```bash
vcgencmd get_throttled
vcgencmd measure_temp
free -h
df -h
```

**Contact:**
- Email: [support@yourteam.com]
- Include: All above information + description of issue

---

**Last Updated:** January 2025