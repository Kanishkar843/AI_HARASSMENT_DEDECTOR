# context_agent.py - LangChain Version
"""
Context Agent for environmental context detection.
Handles PIR sensor data and time-of-day context for harassment detection.
Optimized for Raspberry Pi 4 GPIO integration.
"""

import time
import json
from datetime import datetime
from typing import Dict, Any
import logging

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Use version-adaptive import for Tool
try:
    from langchain_core.tools import Tool
except ImportError:
    try:
        from langchain.tools.base import Tool
    except ImportError:
        from langchain.tools import Tool

# GPIO handling (mock for development, real for Raspberry Pi)
RASPBERRY_PI = False
GPIO = None

# Only try to import GPIO if we're on Linux (likely Raspberry Pi)
import platform
if platform.system() == 'Linux':
    try:
        # Defer import to runtime only on Linux systems
        GPIO = __import__('RPi.GPIO')
        RASPBERRY_PI = True
        logger.info("RPi.GPIO imported successfully")
    except ImportError:
        logger.info("RPi.GPIO not available - running in development mode")
else:
    logger.info(f"Running on {platform.system()} - GPIO support disabled")

class ContextAgent:
    def __init__(self, pir_pin: int = 18):
        """
        Initialize Context Agent for environmental monitoring
        
        Args:
            pir_pin: GPIO pin number for PIR sensor
        """
        self.pir_pin = pir_pin
        self.motion_history = []  # Store recent motion events
        self.max_history = 10
        self._mock_last_update = time.time()  # For mock motion detection
        
        # Initialize GPIO if on Raspberry Pi
        if RASPBERRY_PI and GPIO is not None:
            try:
                GPIO.setwarnings(False)  # Disable warnings
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(self.pir_pin, GPIO.IN)
                logger.info(f"PIR sensor initialized on pin {self.pir_pin}")
            except Exception as e:
                logger.error(f"GPIO initialization failed: {e}")
                global RASPBERRY_PI
                RASPBERRY_PI = False
        else:
            logger.info("Running in development mode - PIR sensor mocked")
    
    def read_pir_sensor(self) -> bool:
        """
        Read PIR motion sensor status
        
        Returns:
            Boolean indicating motion detection
        """
        try:
            if RASPBERRY_PI and GPIO is not None:
                # Read actual GPIO pin
                try:
                    motion_detected = GPIO.input(self.pir_pin) == GPIO.HIGH
                except Exception as e:
                    logger.error(f"Error reading GPIO pin: {e}")
                    motion_detected = False
            else:
                # Mock PIR sensor for development/testing
                # Simulate more realistic motion patterns
                current_time = time.time()
                time_diff = current_time - self._mock_last_update
                
                # Update mock state every 2-3 seconds
                if time_diff >= 2:
                    self._mock_last_update = current_time
                    hour = datetime.now().hour
                    
                    # Higher chance of motion during day hours
                    base_probability = 0.3  # 30% base chance
                    if 8 <= hour <= 20:  # Daytime
                        base_probability = 0.4
                    elif 0 <= hour <= 5:  # Late night
                        base_probability = 0.1
                        
                    motion_detected = (time.time() * 1000) % 100 < (base_probability * 100)
                else:
                    # Keep previous state if not enough time has passed
                    motion_detected = bool(self.motion_history[-1]["motion"]) if self.motion_history else False
            
            # Store in motion history
            timestamp = time.time()
            self.motion_history.append({
                "timestamp": timestamp,
                "motion": motion_detected
            })
            
            # Keep only recent history
            if len(self.motion_history) > self.max_history:
                self.motion_history = self.motion_history[-self.max_history:]
            
            logger.info(f"PIR sensor: {'Motion detected' if motion_detected else 'No motion'}")
            
            return motion_detected
            
        except Exception as e:
            logger.error(f"PIR sensor read error: {e}")
            return False
    
    def get_time_context(self) -> Dict[str, Any]:
        """
        Analyze time-based context factors
        
        Returns:
            Dictionary with time-based risk factors
        """
        now = datetime.now()
        
        # Time-based risk factors
        hour = now.hour
        day_of_week = now.weekday()  # 0=Monday, 6=Sunday
        
        # Risk levels based on time patterns
        time_risk = 0.0
        
        # Higher risk during late night/early morning hours
        if 22 <= hour or hour <= 6:
            time_risk += 0.3
        
        # Moderate risk during evening hours
        elif 18 <= hour <= 22:
            time_risk += 0.2
        
        # Weekend nights might have higher risk
        if day_of_week >= 5 and (20 <= hour or hour <= 2):  # Friday/Saturday night
            time_risk += 0.2
        
        # Working hours generally lower risk
        if 9 <= hour <= 17 and day_of_week <= 4:
            time_risk = max(0.0, time_risk - 0.1)
        
        time_risk = min(time_risk, 1.0)  # Cap at 1.0
        
        return {
            "hour": hour,
            "day_of_week": day_of_week,
            "time_risk": time_risk,
            "time_category": self._get_time_category(hour),
            "is_weekend": day_of_week >= 5
        }
    
    def _get_time_category(self, hour: int) -> str:
        """Categorize time of day"""
        if 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 18:
            return "afternoon"
        elif 18 <= hour < 22:
            return "evening"
        else:
            return "night"
    
    def get_motion_patterns(self) -> Dict[str, Any]:
        """
        Analyze motion patterns from recent history
        
        Returns:
            Dictionary with motion pattern analysis
        """
        if len(self.motion_history) < 2:
            return {
                "motion_frequency": 0.0,
                "continuous_motion": False,
                "motion_count": 0
            }
        
        # Calculate motion frequency (motions per minute)
        recent_motions = [m for m in self.motion_history 
                         if time.time() - m["timestamp"] <= 60]  # Last minute
        
        motion_count = sum(1 for m in recent_motions if m["motion"])
        # Normalize motion frequency to [0,1] range over last minute
        motion_frequency = min(motion_count / 30.0, 1.0)  # Cap at 30 motions per minute = 1.0
        
        # Check for continuous motion (multiple detections in sequence)
        continuous_motion = False
        if len(recent_motions) >= 3:
            last_three = recent_motions[-3:]
            continuous_motion = all(m["motion"] for m in last_three)
        
        return {
            "motion_frequency": motion_frequency,
            "continuous_motion": continuous_motion,
            "motion_count": motion_count,
            "total_history": len(self.motion_history)
        }
    
    def analyze_context(self) -> Dict[str, Any]:
        """
        Comprehensive context analysis
        
        Returns:
            Dictionary with all context factors
        """
        try:
            # Get current PIR reading
            current_motion = self.read_pir_sensor()
            
            # Get time-based context
            time_context = self.get_time_context()
            
            # Get motion patterns
            motion_patterns = self.get_motion_patterns()
            
            # Calculate overall context risk
            context_risk = 0.0
            
            # PIR sensor contributes to risk
            if current_motion:
                context_risk += 0.4
            
            # Time-based risk
            context_risk += time_context["time_risk"] * 0.3
            
            # Motion pattern risk
            if motion_patterns["continuous_motion"]:
                context_risk += 0.3
            elif motion_patterns["motion_frequency"] > 0.5:
                context_risk += 0.2
            
            context_risk = min(context_risk, 1.0)
            
            result = {
                "pir_triggered": current_motion,
                "context_risk": context_risk,
                "timestamp": datetime.now().isoformat(),
                "time_context": time_context,
                "motion_patterns": motion_patterns,
                "status": "success"
            }
            
            logger.info(f"Context analysis: PIR={current_motion}, Risk={context_risk:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Context analysis error: {e}")
            return {
                "pir_triggered": False,
                "context_risk": 0.0,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def process_input(self, input_data: str = "") -> str:
        """
        LangChain Tool interface - analyzes current environmental context
        
        Args:
            input_data: Tool input (unused)
            
        Returns:
            JSON string with context analysis results
        """
        results = self.analyze_context()
        return json.dumps(results)
    
    def cleanup(self):
        """Clean up GPIO resources"""
        if RASPBERRY_PI and GPIO is not None:
            try:
                GPIO.cleanup(self.pir_pin)  # Clean up only our pin
                logger.info("GPIO cleanup completed")
            except Exception as e:
                logger.warning(f"GPIO cleanup failed: {e}")
        self.motion_history.clear()

# Create LangChain Tool wrapper
def create_context_tool() -> Tool:
    """Create LangChain Tool for Context Agent"""
    agent = ContextAgent()
    
    return Tool(
        name="ContextAnalyzer",
        description="Analyzes environmental context including PIR motion sensor and time-based risk factors. Returns PIR status and context risk score (0-1).",
        func=agent.process_input
    )