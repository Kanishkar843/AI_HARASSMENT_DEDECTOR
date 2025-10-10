#!/usr/bin/env python3
"""
Integrated Violence Detection System for Raspberry Pi 4
Combines Visual + Audio Detection with Alert System
"""

import cv2
import numpy as np
import pyaudio
import wave
import threading
import time
from datetime import datetime
from pathlib import Path
import json
from tflite_runtime.interpreter import Interpreter

# ============================================================================
# Configuration
# ============================================================================

class Config:
    # Paths
    VISION_MODEL = 'models/violence_detection.tflite'
    AUDIO_MODEL = 'models/audio_detection.tflite'
    VISION_LABELS = 'models/vision_labels.txt'
    AUDIO_LABELS = 'models/audio_labels.txt'
    
    # Alert settings
    ALERT_THRESHOLD = 0.70  # Confidence threshold for alerts
    ALERT_COOLDOWN = 30     # Frames between alerts (~1 second at 30fps)
    LOG_DIR = 'logs'
    ALERT_DIR = 'alerts'
    
    # Camera settings
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 30
    
    # Audio settings
    AUDIO_RATE = 16000
    AUDIO_CHUNK = 1024
    AUDIO_CHANNELS = 1
    AUDIO_RECORD_SECONDS = 2

# ============================================================================
# Vision Detection Thread
# ============================================================================

class VisionDetector:
    def __init__(self, model_path, labels_path):
        print("Loading vision model...")
        self.interpreter = Interpreter(model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Load labels
        with open(labels_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]
        
        self.current_detection = None
        self.running = False
        print(f"✅ Vision model loaded. Classes: {self.labels}")
    
    def preprocess_frame(self, frame):
        """Preprocess frame for model input"""
        img = cv2.resize(frame, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return np.expand_dims(img, axis=0).astype(np.uint8)
    
    def detect(self, frame):
        """Run inference on frame"""
        try:
            input_data = self.preprocess_frame(frame)
            
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            
            output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            
            pred_idx = np.argmax(output)
            confidence = output[pred_idx] / 255.0
            
            return {
                'class': self.labels[pred_idx],
                'class_id': int(pred_idx),
                'confidence': float(confidence),
                'is_threat': pred_idx in [0, 1] and confidence > Config.ALERT_THRESHOLD
            }
        except Exception as e:
            print(f"Vision detection error: {e}")
            return None

# ============================================================================
# Audio Detection Thread
# ============================================================================

class AudioDetector:
    def __init__(self, model_path, labels_path):
        print("Loading audio model...")
        self.interpreter = Interpreter(model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Load labels
        with open(labels_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]
        
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.current_detection = None
        self.running = False
        
        print(f"✅ Audio model loaded. Classes: {self.labels}")
    
    def start_stream(self):
        """Start audio stream"""
        try:
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=Config.AUDIO_CHANNELS,
                rate=Config.AUDIO_RATE,
                input=True,
                frames_per_buffer=Config.AUDIO_CHUNK
            )
            print("✅ Audio stream started")
        except Exception as e:
            print(f"⚠️  Audio stream error: {e}")
    
    def preprocess_audio(self, audio_data):
        """Preprocess audio for model input"""
        # Convert to numpy array
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        
        # Normalize
        audio_np = audio_np.astype(np.float32) / 32768.0
        
        # Reshape for model (adjust based on your model's input shape)
        # Example: if model expects (1, num_samples)
        audio_np = np.expand_dims(audio_np, axis=0)
        
        return audio_np.astype(np.float32)
    
    def detect(self):
        """Continuously detect audio threats"""
        if not self.stream:
            return None
        
        try:
            # Read audio chunk
            audio_data = self.stream.read(Config.AUDIO_CHUNK * 32, exception_on_overflow=False)
            
            # Preprocess
            input_data = self.preprocess_audio(audio_data)
            
            # Run inference
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            
            output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            
            pred_idx = np.argmax(output)
            confidence = float(output[pred_idx])
            
            # Check if it's a threat (scream, gunshot, etc.)
            threat_classes = ['scream', 'gunshot', 'explosion', 'breaking']
            is_threat = (self.labels[pred_idx].lower() in threat_classes and 
                        confidence > Config.ALERT_THRESHOLD)
            
            return {
                'class': self.labels[pred_idx],
                'class_id': int(pred_idx),
                'confidence': confidence,
                'is_threat': is_threat
            }
            
        except Exception as e:
            print(f"Audio detection error: {e}")
            return None
    
    def stop_stream(self):
        """Stop audio stream"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()

# ============================================================================
# Alert System
# ============================================================================

class AlertSystem:
    def __init__(self):
        self.log_dir = Path(Config.LOG_DIR)
        self.alert_dir = Path(Config.ALERT_DIR)
        
        self.log_dir.mkdir(exist_ok=True)
        self.alert_dir.mkdir(exist_ok=True)
        
        self.log_file = self.log_dir / f"alerts_{datetime.now().strftime('%Y%m%d')}.log"
        
        print(f"✅ Alert system initialized")
    
    def log_alert(self, vision_result, audio_result):
        """Log alert to file"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        alert_data = {
            'timestamp': timestamp,
            'vision': vision_result,
            'audio': audio_result
        }
        
        # Write to log file
        with open(self.log_file, 'a') as f:
            f.write(f"{json.dumps(alert_data)}\n")
        
        # Console alert
        print(f"\n{'='*60}")
        print(f"🚨 ALERT TRIGGERED - {timestamp}")
        if vision_result and vision_result.get('is_threat'):
            print(f"   Vision: {vision_result['class']} ({vision_result['confidence']:.2%})")
        if audio_result and audio_result.get('is_threat'):
            print(f"   Audio: {audio_result['class']} ({audio_result['confidence']:.2%})")
        print(f"{'='*60}\n")
    
    def save_evidence(self, frame, audio_data=None):
        """Save video frame and audio as evidence"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save frame
        frame_path = self.alert_dir / f"alert_{timestamp}.jpg"
        cv2.imwrite(str(frame_path), frame)
        
        return str(frame_path)

# ============================================================================
# Main Detection System
# ============================================================================

class IntegratedDetectionSystem:
    def __init__(self):
        print("\n" + "="*60)
        print("Integrated Violence Detection System")
        print("="*60 + "\n")
        
        # Initialize components
        self.vision_detector = VisionDetector(Config.VISION_MODEL, Config.VISION_LABELS)
        self.audio_detector = AudioDetector(Config.AUDIO_MODEL, Config.AUDIO_LABELS)
        self.alert_system = AlertSystem()
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, Config.CAMERA_FPS)
        
        # Start audio stream
        self.audio_detector.start_stream()
        
        # State
        self.alert_cooldown = 0
        self.running = False
        
        print("\n✅ System initialized successfully!\n")
    
    def run(self):
        """Main detection loop"""
        self.running = True
        
        print("="*60)
        print("🎥 SYSTEM RUNNING - Press 'q' to quit")
        print("="*60 + "\n")
        
        # Audio detection in separate thread
        audio_thread = threading.Thread(target=self._audio_loop, daemon=True)
        audio_thread.start()
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Vision detection
                vision_result = self.vision_detector.detect(frame)
                
                # Get latest audio detection
                audio_result = self.audio_detector.current_detection
                
                # Check for threats
                vision_threat = vision_result and vision_result.get('is_threat', False)
                audio_threat = audio_result and audio_result.get('is_threat', False)
                
                # Trigger alert if threat detected
                if (vision_threat or audio_threat) and self.alert_cooldown == 0:
                    self.alert_system.log_alert(vision_result, audio_result)
                    self.alert_system.save_evidence(frame)
                    self.alert_cooldown = Config.ALERT_COOLDOWN
                
                # Display results on frame
                self._draw_results(frame, vision_result, audio_result)
                
                # Show frame
                cv2.imshow('Violence Detection System', frame)
                
                # Update cooldown
                if self.alert_cooldown > 0:
                    self.alert_cooldown -= 1
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("\n\nShutting down...")
        
        finally:
            self.cleanup()
    
    def _audio_loop(self):
        """Audio detection loop (runs in separate thread)"""
        while self.running:
            result = self.audio_detector.detect()
            if result:
                self.audio_detector.current_detection = result
            time.sleep(0.1)
    
    def _draw_results(self, frame, vision_result, audio_result):
        """Draw detection results on frame"""
        h, w = frame.shape[:2]
        
        # Background for text
        cv2.rectangle(frame, (5, 5), (w-5, 120), (0, 0, 0), -1)
        
        # Vision result
        if vision_result:
            v_color = (0, 0, 255) if vision_result.get('is_threat') else (0, 255, 0)
            v_text = f"Vision: {vision_result['class']} ({vision_result['confidence']:.2%})"
            cv2.putText(frame, v_text, (10, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, v_color, 2)
        
        # Audio result
        if audio_result:
            a_color = (0, 0, 255) if audio_result.get('is_threat') else (0, 255, 0)
            a_text = f"Audio: {audio_result['class']} ({audio_result['confidence']:.2%})"
            cv2.putText(frame, a_text, (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, a_color, 2)
        
        # Alert status
        if self.alert_cooldown > 0:
            cv2.putText(frame, "🚨 ALERT ACTIVE", (10, 105),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(frame, timestamp, (w-250, h-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def cleanup(self):
        """Cleanup resources"""
        print("\nCleaning up...")
        self.running = False
        self.cap.release()
        self.audio_detector.stop_stream()
        cv2.destroyAllWindows()
        print("✅ System stopped")

# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    try:
        system = IntegratedDetectionSystem()
        system.run()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()