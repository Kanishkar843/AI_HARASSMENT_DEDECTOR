#!/bin/bash
# ============================================================================
# RASPBERRY PI 4 - VIOLENCE DETECTION SYSTEM SETUP
# Complete automated setup script
# ============================================================================

echo "========================================================================"
echo "Violence Detection System - Automated Setup"
echo "========================================================================"
echo ""
echo "This script will install all dependencies for the violence detection"
echo "system on your Raspberry Pi 4."
echo ""
echo "Estimated time: 10-15 minutes"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# ============================================================================
# STEP 1: System Update
# ============================================================================

echo ""
echo "========================================================================"
echo "STEP 1: Updating System"
echo "========================================================================"

sudo apt-get update
sudo apt-get upgrade -y

echo "✅ System updated"

# ============================================================================
# STEP 2: Install Python and Build Tools
# ============================================================================

echo ""
echo "========================================================================"
echo "STEP 2: Installing Python and Build Tools"
echo "========================================================================"

sudo apt-get install -y python3-pip python3-dev
sudo apt-get install -y build-essential cmake pkg-config
sudo apt-get install -y libjpeg-dev libtiff5-dev libjasper-dev libpng-dev
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install -y libxvidcore-dev libx264-dev
sudo apt-get install -y libfontconfig1-dev libcairo2-dev
sudo apt-get install -y libgdk-pixbuf2.0-dev libpango1.0-dev
sudo apt-get install -y libgtk2.0-dev libgtk-3-dev
sudo apt-get install -y libatlas-base-dev gfortran

echo "✅ Build tools installed"

# ============================================================================
# STEP 3: Install OpenCV
# ============================================================================

echo ""
echo "========================================================================"
echo "STEP 3: Installing OpenCV"
echo "========================================================================"

sudo apt-get install -y python3-opencv

echo "✅ OpenCV installed"

# ============================================================================
# STEP 4: Install Audio Dependencies
# ============================================================================

echo ""
echo "========================================================================"
echo "STEP 4: Installing Audio Dependencies"
echo "========================================================================"

sudo apt-get install -y portaudio19-dev python3-pyaudio
sudo apt-get install -y libsndfile1 libsndfile1-dev
sudo apt-get install -y alsa-utils

echo "✅ Audio dependencies installed"

# ============================================================================
# STEP 5: Install Python Packages
# ============================================================================

echo ""
echo "========================================================================"
echo "STEP 5: Installing Python Packages"
echo "========================================================================"

pip3 install --upgrade pip

echo "Installing TensorFlow Lite Runtime..."
pip3 install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime

echo "Installing NumPy..."
pip3 install numpy==1.23.5

echo "Installing PyAudio..."
pip3 install pyaudio

echo "Installing additional libraries..."
pip3 install opencv-python
pip3 install soundfile
pip3 install wave

echo "✅ Python packages installed"

# ============================================================================
# STEP 6: Enable Camera
# ============================================================================

echo ""
echo "========================================================================"
echo "STEP 6: Enabling Camera"
echo "========================================================================"

# Enable camera interface
sudo raspi-config nonint do_camera 0

echo "✅ Camera enabled"

# ============================================================================
# STEP 7: Configure Audio
# ============================================================================

echo ""
echo "========================================================================"
echo "STEP 7: Configuring Audio"
echo "========================================================================"

# Set audio output to auto
sudo raspi-config nonint do_audio 0

# Test if microphone is detected
if arecord -l | grep -q "card"; then
    echo "✅ Microphone detected"
else
    echo "⚠️  Warning: No microphone detected"
    echo "   Please connect a USB microphone and reboot"
fi

# ============================================================================
# STEP 8: Create Project Directory Structure
# ============================================================================

echo ""
echo "========================================================================"
echo "STEP 8: Creating Project Directories"
echo "========================================================================"

cd ~
mkdir -p violence_detection/models
mkdir -p violence_detection/logs
mkdir -p violence_detection/alerts
mkdir -p violence_detection/scripts

echo "✅ Project directories created at ~/violence_detection/"

# ============================================================================
# STEP 9: Performance Optimization
# ============================================================================

echo ""
echo "========================================================================"
echo "STEP 9: Optimizing Performance"
echo "========================================================================"

# Increase GPU memory for camera operations
if grep -q "gpu_mem=" /boot/config.txt; then
    sudo sed -i 's/gpu_mem=.*/gpu_mem=256/' /boot/config.txt
else
    echo "gpu_mem=256" | sudo tee -a /boot/config.txt
fi

echo "✅ GPU memory set to 256MB"

# ============================================================================
# STEP 10: Verification
# ============================================================================

echo ""
echo "========================================================================"
echo "STEP 10: Verification"
echo "========================================================================"

echo ""
echo "Checking installations..."
echo ""

# Check Python
if python3 --version > /dev/null 2>&1; then
    echo "✅ Python 3: $(python3 --version)"
else
    echo "❌ Python 3: Not installed"
fi

# Check OpenCV
if python3 -c "import cv2" 2>/dev/null; then
    echo "✅ OpenCV: Installed"
else
    echo "❌ OpenCV: Not installed"
fi

# Check TFLite
if python3 -c "import tflite_runtime" 2>/dev/null; then
    echo "✅ TensorFlow Lite: Installed"
else
    echo "❌ TensorFlow Lite: Not installed"
fi

# Check NumPy
if python3 -c "import numpy" 2>/dev/null; then
    echo "✅ NumPy: Installed"
else
    echo "❌ NumPy: Not installed"
fi

# Check PyAudio
if python3 -c "import pyaudio" 2>/dev/null; then
    echo "✅ PyAudio: Installed"
else
    echo "❌ PyAudio: Not installed"
fi

# Check Camera
if vcgencmd get_camera | grep -q "detected=1"; then
    echo "✅ Camera: Detected"
else
    echo "⚠️  Camera: Not detected (may need reboot)"
fi

# ============================================================================
# FINAL SUMMARY
# ============================================================================

echo ""
echo "========================================================================"
echo "✅ SETUP COMPLETE!"
echo "========================================================================"
echo ""
echo "📁 Project Directory: ~/violence_detection/"
echo ""
echo "📋 Next Steps:"
echo "   1. Copy model files to ~/violence_detection/models/"
echo "   2. Copy detection script to ~/violence_detection/"
echo "   3. Reboot your Raspberry Pi (recommended)"
echo "   4. Run the detection system"
echo ""
echo "🔄 To reboot now, run: sudo reboot"
echo ""
echo "📞 If you encounter any issues, check the troubleshooting guide"
echo ""
echo "========================================================================"

# Save setup log
LOGFILE=~/violence_detection/setup_log_$(date +%Y%m%d_%H%M%S).txt
echo "Setup completed on $(date)" > $LOGFILE
echo "Python: $(python3 --version)" >> $LOGFILE
dpkg -l | grep -E 'python3-opencv|portaudio' >> $LOGFILE

echo "📄 Setup log saved to: $LOGFILE"
echo ""