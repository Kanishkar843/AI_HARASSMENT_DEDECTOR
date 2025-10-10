"""
Configuration management for the harassment detection system.
Handles all configurable parameters for hardware, ML models, and system behavior.
"""

import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class SystemConfig:
    # Default configuration
    DEFAULT_CONFIG = {
        "hardware": {
            "camera": {
                "device_id": 0,
                "width": 640,
                "height": 480,
                "fps": 30,
                "type": "usb",  # usb, ip, or picamera
                "ip_address": "",  # for IP cameras
                "reconnect_attempts": 3
            },
            "microphone": {
                "device_id": None,  # None uses default device
                "sample_rate": 16000,
                "channels": 1,
                "chunk_size": 1024,
                "record_seconds": 3
            },
            "gpio": {
                "pir_pin": 18,
                "led_pin": 23,
                "buzzer_pin": 24
            }
        },
        "ml_models": {
            "vision": {
                "model_path": "microsoft/resnet-50",
                "fallback_model": "google/vit-base-patch16-224",
                "confidence_threshold": 0.7,
                "use_gpu": True
            },
            "audio": {
                "model_path": "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
                "fallback_model": "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
                "confidence_threshold": 0.6,
                "use_gpu": True
            }
        },
        "system": {
            "alert_thresholds": {
                "high": 0.8,
                "medium": 0.7,
                "low": 0.3
            },
            "fusion_weights": {
                "vision": 0.6,
                "audio": 0.4
            },
            "storage": {
                "save_alerts": True,
                "max_storage_days": 30,
                "alert_dir": "alerts",
                "snapshot_dir": "snapshots"
            },
            "api": {
                "enabled": True,
                "host": "localhost",
                "port": 5000,
                "jwt_secret": "",
                "ssl_cert": "",
                "ssl_key": ""
            }
        }
    }
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize configuration system"""
        self.config_path = config_path
        self.config = self.DEFAULT_CONFIG.copy()
        self.load_config()
    
    def load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    file_config = json.load(f)
                    # Update default config with file values
                    self._update_recursive(self.config, file_config)
                logger.info(f"Loaded configuration from {self.config_path}")
            else:
                self.save_config()  # Save default config
                logger.info(f"Created default configuration at {self.config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            logger.info(f"Saved configuration to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def _update_recursive(self, base: Dict, update: Dict):
        """Recursively update nested dictionary"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict):
                self._update_recursive(base[key], value)
            else:
                base[key] = value
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        Example: config.get("hardware.camera.device_id")
        """
        try:
            value = self.config
            for key in path.split('.'):
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, path: str, value: Any) -> bool:
        """
        Set configuration value using dot notation
        Example: config.set("hardware.camera.device_id", 1)
        """
        try:
            keys = path.split('.')
            current = self.config
            for key in keys[:-1]:
                current = current[key]
            current[keys[-1]] = value
            return True
        except (KeyError, TypeError):
            return False
    
    def get_hardware_config(self) -> Dict[str, Any]:
        """Get all hardware-related configuration"""
        return self.config["hardware"]
    
    def get_ml_config(self) -> Dict[str, Any]:
        """Get all ML model configuration"""
        return self.config["ml_models"]
    
    def get_system_config(self) -> Dict[str, Any]:
        """Get all system configuration"""
        return self.config["system"]
    
    def validate(self) -> bool:
        """
        Validate current configuration
        Returns True if valid, logs errors if invalid
        """
        try:
            # Validate camera config
            camera = self.get("hardware.camera")
            if camera["type"] == "ip" and not camera["ip_address"]:
                logger.error("IP camera selected but no IP address provided")
                return False
            
            # Validate ML models
            for model_type in ["vision", "audio"]:
                model_config = self.get(f"ml_models.{model_type}")
                if not model_config["model_path"]:
                    logger.error(f"No model path provided for {model_type}")
                    return False
            
            # Validate alert thresholds
            thresholds = self.get("system.alert_thresholds")
            if not (0 <= thresholds["low"] <= thresholds["medium"] <= thresholds["high"] <= 1):
                logger.error("Invalid alert thresholds. Must be in range [0,1] and low <= medium <= high")
                return False
            
            # Validate fusion weights
            weights = self.get("system.fusion_weights")
            if abs((weights["vision"] + weights["audio"]) - 1.0) > 0.001:
                logger.error("Fusion weights must sum to 1.0")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return False

# Global configuration instance
config = SystemConfig()