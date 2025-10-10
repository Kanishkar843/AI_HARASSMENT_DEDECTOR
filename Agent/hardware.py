"""
Hardware abstraction layer for the harassment detection system.
Handles camera, microphone, and GPIO interfaces.
"""

import cv2
import numpy as np
import pyaudio
import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
from config import config

logger = logging.getLogger(__name__)

class CameraInterface(ABC):
    """Abstract base class for camera interfaces"""
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize camera"""
        pass
    
    @abstractmethod
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame"""
        pass
    
    @abstractmethod
    def release(self):
        """Release camera resources"""
        pass

class USBCamera(CameraInterface):
    """USB camera implementation"""
    
    def __init__(self, device_id: int, width: int, height: int):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.cap = None
        
    def initialize(self) -> bool:
        try:
            self.cap = cv2.VideoCapture(self.device_id)
            if not self.cap.isOpened():
                logger.error(f"Failed to open USB camera {self.device_id}")
                return False
                
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            return True
            
        except Exception as e:
            logger.error(f"USB camera initialization error: {e}")
            return False
    
    def capture_frame(self) -> Optional[np.ndarray]:
        if self.cap is None:
            return None
            
        try:
            ret, frame = self.cap.read()
            if ret:
                return frame
            return None
        except Exception as e:
            logger.error(f"USB camera capture error: {e}")
            return None
    
    def release(self):
        if self.cap is not None:
            self.cap.release()

class IPCamera(CameraInterface):
    """IP camera implementation"""
    
    def __init__(self, ip_address: str, width: int, height: int):
        self.ip_address = ip_address
        self.width = width
        self.height = height
        self.cap = None
        
    def initialize(self) -> bool:
        try:
            self.cap = cv2.VideoCapture(self.ip_address)
            if not self.cap.isOpened():
                logger.error(f"Failed to connect to IP camera at {self.ip_address}")
                return False
            return True
        except Exception as e:
            logger.error(f"IP camera initialization error: {e}")
            return False
    
    def capture_frame(self) -> Optional[np.ndarray]:
        if self.cap is None:
            return None
            
        try:
            ret, frame = self.cap.read()
            if ret:
                return cv2.resize(frame, (self.width, self.height))
            return None
        except Exception as e:
            logger.error(f"IP camera capture error: {e}")
            return None
    
    def release(self):
        if self.cap is not None:
            self.cap.release()

class AudioInterface(ABC):
    """Abstract base class for audio interfaces"""
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize audio device"""
        pass
    
    @abstractmethod
    def record(self, duration: float) -> Optional[np.ndarray]:
        """Record audio for specified duration"""
        pass
    
    @abstractmethod
    def release(self):
        """Release audio resources"""
        pass

class MicrophoneInterface(AudioInterface):
    """Standard microphone implementation"""
    
    def __init__(self, device_id: Optional[int], sample_rate: int, channels: int, chunk_size: int):
        self.device_id = device_id
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.audio = None
        self.stream = None
        
    def initialize(self) -> bool:
        try:
            self.audio = pyaudio.PyAudio()
            self.stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_id,
                frames_per_buffer=self.chunk_size
            )
            return True
        except Exception as e:
            logger.error(f"Microphone initialization error: {e}")
            return False
    
    def record(self, duration: float) -> Optional[np.ndarray]:
        if self.stream is None:
            return None
            
        try:
            frames = []
            n_chunks = int((self.sample_rate * duration) / self.chunk_size)
            
            for _ in range(n_chunks):
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                frames.append(np.frombuffer(data, dtype=np.float32))
            
            return np.concatenate(frames)
        except Exception as e:
            logger.error(f"Audio recording error: {e}")
            return None
    
    def release(self):
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        if self.audio is not None:
            self.audio.terminate()

class HardwareManager:
    """Manages all hardware interfaces"""
    
    def __init__(self):
        self.camera = None
        self.microphone = None
        self.config = config
    
    def initialize_hardware(self) -> bool:
        """Initialize all hardware interfaces"""
        success = True
        
        # Initialize camera
        camera_config = self.config.get("hardware.camera")
        if camera_config["type"] == "usb":
            self.camera = USBCamera(
                camera_config["device_id"],
                camera_config["width"],
                camera_config["height"]
            )
        elif camera_config["type"] == "ip":
            self.camera = IPCamera(
                camera_config["ip_address"],
                camera_config["width"],
                camera_config["height"]
            )
        
        if not self.camera.initialize():
            success = False
        
        # Initialize microphone
        mic_config = self.config.get("hardware.microphone")
        self.microphone = MicrophoneInterface(
            mic_config["device_id"],
            mic_config["sample_rate"],
            mic_config["channels"],
            mic_config["chunk_size"]
        )
        
        if not self.microphone.initialize():
            success = False
        
        return success
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get a frame from the camera"""
        return self.camera.capture_frame() if self.camera else None
    
    def get_audio(self, duration: float) -> Optional[np.ndarray]:
        """Get audio recording"""
        return self.microphone.record(duration) if self.microphone else None
    
    def release(self):
        """Release all hardware resources"""
        if self.camera:
            self.camera.release()
        if self.microphone:
            self.microphone.release()
    
    def get_status(self) -> Dict[str, Any]:
        """Get hardware status"""
        return {
            "camera": {
                "initialized": self.camera is not None,
                "type": self.config.get("hardware.camera.type")
            },
            "microphone": {
                "initialized": self.microphone is not None,
                "sample_rate": self.config.get("hardware.microphone.sample_rate")
            }
        }

# Global hardware manager instance
hardware = HardwareManager()