# alert_agent.py - LangChain Version
"""
Alert Agent for harassment detection system.
Handles arbitration rules, alert generation, and logging.
Optimized for Raspberry Pi 4.
"""

import json
import time
import os
import logging
from typing import Dict, Any, List
from datetime import datetime

try:
    from langchain_core.tools import Tool
except ImportError:
    try:
        from langchain.tools.base import Tool
    except ImportError:
        from langchain.tools import Tool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("alert_agent")

class AlertAgent:
    # Alert thresholds
    NORMAL_THRESHOLD = 0.2   # Below this is NORMAL
    MEDIUM_THRESHOLD = 0.5   # Below this is MEDIUM, above is HIGH
    
    def __init__(self, log_file: str = "incidents.log"):
        """Initialize Alert Agent with log file"""
        self.log_file = log_file
        # Add file handler for incident logging
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
        logger.info(f"Alert Agent initialized, logging to: {log_file}")
    
    def evaluate_threat(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate threat level from multiple inputs
        
        Args:
            inputs: Dict with:
                - vision_confidence: Float 0-1
                - audio_confidence: Float 0-1
                - pir_triggered: Boolean
                - context_risk: Float 0-1
        
        Returns:
            Dict with:
                - alert_level: "HIGH"/"MEDIUM"/"LOW"/"NORMAL"
                - alert_score: Float 0-1
                - reasoning: List of explanation strings
        """
        # Extract inputs with defaults
        vision_conf = float(inputs.get("vision_confidence", 0.0))
        audio_conf = float(inputs.get("audio_confidence", 0.0))
        pir_triggered = bool(inputs.get("pir_triggered", False))
        context_risk = float(inputs.get("context_risk", 0.0))
        
        # Calculate final threat score
        # If either vision or audio shows strong signal, maintain that high score
        if vision_conf > 0.7 or audio_conf > 0.7:
            final_score = max(vision_conf, audio_conf)
        else:
            # Otherwise use weighted combination
            final_score = (
                vision_conf * 0.5 +    # Vision has highest weight
                audio_conf * 0.4 +     # Audio is strong secondary
                context_risk * 0.1     # Context adds support
            )
        
        # Add bonus for multiple indicators
        if vision_conf > 0.4 and audio_conf > 0.4:
            final_score += 0.1  # Boost when both main sensors trigger
        if pir_triggered and (vision_conf > 0.3 or audio_conf > 0.3):
            final_score += 0.1  # Boost when motion confirms other signals
            
        final_score = min(final_score, 1.0)  # Cap at 1.0
        
        # Determine alert level and reasoning
        reasoning = []
        
        if final_score >= self.MEDIUM_THRESHOLD:
            if final_score >= 0.8:
                alert_level = "HIGH"
                if vision_conf > 0.7 and audio_conf > 0.6:
                    reasoning.append(f"HIGH: Strong vision ({vision_conf:.1%}) + audio ({audio_conf:.1%}) signals")
                elif vision_conf > 0.8:
                    reasoning.append(f"HIGH: Very strong vision signal ({vision_conf:.1%})")
                elif audio_conf > 0.8:
                    reasoning.append(f"HIGH: Very strong audio signal ({audio_conf:.1%})")
                else:
                    reasoning.append(f"HIGH: Multiple strong indicators")
            else:
                alert_level = "MEDIUM"
                if vision_conf > audio_conf:
                    reasoning.append(f"MEDIUM: High vision {vision_conf:.1%} + context risk {context_risk:.1%}")
                else:
                    reasoning.append(f"MEDIUM: High audio {audio_conf:.1%} + context risk {context_risk:.1%}")
        
        elif final_score >= self.NORMAL_THRESHOLD:
            alert_level = "LOW"
            reasoning.append("LOW: Moderate detection levels")
        else:
            alert_level = "NORMAL"
            reasoning.append("NORMAL: No significant threat detected")
        
        result = {
            "alert_level": alert_level,
            "alert_score": float(final_score),
            "reasoning": reasoning,
            "timestamp": datetime.now().isoformat()
        }
        
        # Log the alert
        self._log_alert(result)
        
        # Emit appropriate log level
        if alert_level == "HIGH":
            logger.critical(f"🚨 HIGH ALERT - Score: {final_score:.2f}")
        elif alert_level == "MEDIUM":
            logger.warning(f"⚠️  MEDIUM ALERT - Score: {final_score:.2f}")
        elif alert_level == "LOW":
            logger.info(f"ℹ️  LOW ALERT - Score: {final_score:.2f}")
        else:
            logger.info(f"Alert decision: {alert_level}")
        
        return result
    
    def _log_alert(self, result: Dict[str, Any]):
        """Log alert to incidents file"""
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(result) + "\n")
            logger.info(f"Incident logged: {result['alert_level']}")
            
            # Send notification for MEDIUM/HIGH alerts
            if result["alert_level"] in ["MEDIUM", "HIGH"]:
                logger.info(f"📱 Notification sent: {result['alert_level']} ALERT")
                
        except Exception as e:
            logger.error(f"Failed to log alert: {e}")

def create_alert_tool() -> Tool:
    """Create LangChain Tool for Alert Agent"""
    agent = AlertAgent()
    
    return Tool(
        name="AlertArbitrator",
        description="Evaluates multi-sensor inputs and determines alert level.",
        func=lambda x: json.dumps(agent.evaluate_threat(json.loads(x)))
    )
from datetime import datetime
from typing import Dict, Any, List
from langchain.tools import Tool
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertAgent:
    def __init__(self, log_file: str = "incidents.log"):
        """
        Initialize Alert Agent for decision making and logging
        
        Args:
            log_file: Path to incidents log file
        """
        self.log_file = log_file
        self.alert_history = []
        self.max_history = 100
        
        # Ensure alerts directory exists
        os.makedirs("alerts", exist_ok=True)
        
        # Ensure log file exists
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                pass  # Create empty file
        
        logger.info(f"Alert Agent initialized, logging to: {self.log_file}")
    
    def apply_arbitration_rules(self, 
                              vision_confidence: float,
                              audio_confidence: float,
                              pir_triggered: bool,
                              context_risk: float = 0.0) -> Dict[str, Any]:
        """
        Apply arbitration rules to determine alert level
        
        Args:
            vision_confidence: Vision agent confidence (0-1)
            audio_confidence: Audio agent confidence (0-1)
            pir_triggered: PIR motion sensor status
            context_risk: Additional context risk factor (0-1)
            
        Returns:
            Dictionary with alert decision and reasoning
        """
        try:
            alert_level = "NORMAL"
            reasoning = []
            alert_score = 0.0
            
            # Rule 1: HIGH Alert - Vision ≥80% + Audio ≥70%
            if vision_confidence >= 0.8 and audio_confidence >= 0.7:
                alert_level = "HIGH"
                alert_score = max(vision_confidence, audio_confidence)
                reasoning.append(f"HIGH: Vision {vision_confidence:.1%} + Audio {audio_confidence:.1%}")
            
            # Rule 2: MEDIUM Alert - Vision ≥70% OR Audio ≥60% with PIR triggered
            elif ((vision_confidence >= 0.7 or audio_confidence >= 0.6) and pir_triggered):
                alert_level = "MEDIUM"
                alert_score = max(vision_confidence, audio_confidence) * 0.8  # Slightly lower than input
                reasoning.append(f"MEDIUM: Vision {vision_confidence:.1%} OR Audio {audio_confidence:.1%} + PIR active")
            
            # Additional contextual rules for enhanced detection
            elif vision_confidence >= 0.75 and context_risk >= 0.5:
                alert_level = "MEDIUM"
                alert_score = vision_confidence * 0.9
                reasoning.append(f"MEDIUM: High vision {vision_confidence:.1%} + context risk {context_risk:.1%}")
            
            elif audio_confidence >= 0.65 and context_risk >= 0.6:
                alert_level = "MEDIUM"
                alert_score = audio_confidence * 0.8
                reasoning.append(f"MEDIUM: High audio {audio_confidence:.1%} + context risk {context_risk:.1%}")
            
            # Low-level alerts for monitoring
            elif vision_confidence >= 0.5 or audio_confidence >= 0.5:
                alert_level = "LOW"
                alert_score = max(vision_confidence, audio_confidence) * 0.6
                reasoning.append(f"LOW: Moderate detection levels")
            
            else:
                alert_level = "NORMAL"
                alert_score = max(vision_confidence, audio_confidence) * 0.3
                reasoning.append("NORMAL: No significant threat detected")
            
            decision = {
                "alert_level": alert_level,
                "alert_score": alert_score,
                "reasoning": reasoning,
                "timestamp": datetime.now().isoformat(),
                "input_data": {
                    "vision": vision_confidence,
                    "audio": audio_confidence,
                    "pir": pir_triggered,
                    "context_risk": context_risk
                }
            }
            
            logger.info(f"Alert decision: {alert_level} (score: {alert_score:.2f})")
            
            return decision
            
        except Exception as e:
            logger.error(f"Arbitration error: {e}")
            return {
                "alert_level": "ERROR",
                "alert_score": 0.0,
                "reasoning": [f"Error in arbitration: {str(e)}"],
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def log_incident(self, decision: Dict[str, Any]) -> bool:
        """
        Log incident to file and memory
        
        Args:
            decision: Alert decision dictionary
            
        Returns:
            Success status
        """
        try:
            # Create log entry
            log_entry = {
                "timestamp": decision["timestamp"],
                "vision": decision["input_data"]["vision"],
                "audio": decision["input_data"]["audio"],
                "pir": decision["input_data"]["pir"],
                "context_risk": decision["input_data"].get("context_risk", 0.0),
                "decision": decision["alert_level"],
                "score": decision["alert_score"],
                "reasoning": decision["reasoning"]
            }
            
            # Write to file (append mode)
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
            
            # Store in memory history
            self.alert_history.append(log_entry)
            
            # Keep history manageable
            if len(self.alert_history) > self.max_history:
                self.alert_history = self.alert_history[-self.max_history:]
            
            logger.info(f"Incident logged: {decision['alert_level']}")
            return True
            
        except Exception as e:
            logger.error(f"Logging error: {e}")
            return False
    
    def trigger_alert_actions(self, decision: Dict[str, Any]):
        """
        Trigger appropriate actions based on alert level
        
        Args:
            decision: Alert decision dictionary
        """
        try:
            alert_level = decision["alert_level"]
            
            if alert_level == "HIGH":
                self._handle_high_alert(decision)
            elif alert_level == "MEDIUM":
                self._handle_medium_alert(decision)
            elif alert_level == "LOW":
                self._handle_low_alert(decision)
            
        except Exception as e:
            logger.error(f"Alert action error: {e}")
    
    def _handle_high_alert(self, decision: Dict[str, Any]):
        """Handle HIGH alert actions"""
        logger.critical(f"🚨 HIGH ALERT TRIGGERED - Score: {decision['alert_score']:.2f}")
        
        # Here you would add:
        # - Send notification to security personnel
        # - Activate emergency protocols  
        # - Start continuous recording
        # - Sound local alarms
        
        # Mock notification
        self._send_notification("HIGH ALERT", decision)
    
    def _handle_medium_alert(self, decision: Dict[str, Any]):
        """Handle MEDIUM alert actions"""
        logger.warning(f"⚠️  MEDIUM ALERT - Score: {decision['alert_score']:.2f}")
        
        # Here you would add:
        # - Increase monitoring frequency
        # - Send notification to monitoring staff
        # - Start selective recording
        
        self._send_notification("MEDIUM ALERT", decision)
    
    def _handle_low_alert(self, decision: Dict[str, Any]):
        """Handle LOW alert actions"""
        logger.info(f"ℹ️  LOW ALERT - Score: {decision['alert_score']:.2f}")
        
        # Here you would add:
        # - Log for trend analysis
        # - Increase sensor sensitivity temporarily
    
    def _send_notification(self, alert_type: str, decision: Dict[str, Any]):
        """Mock notification system (replace with real implementation)"""
        notification = {
            "type": alert_type,
            "timestamp": decision["timestamp"],
            "score": decision["alert_score"],
            "reasoning": decision["reasoning"]
        }
        
        # In production, this would send to:
        # - Email/SMS alerts
        # - Slack/Teams notifications  
        # - Security system integration
        # - Mobile app push notifications
        
        logger.info(f"📱 Notification sent: {alert_type}")
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent alerts for analysis"""
        try:
            cutoff_time = time.time() - (hours * 3600)
            recent_alerts = []
            
            for alert in self.alert_history:
                alert_time = datetime.fromisoformat(alert["timestamp"]).timestamp()
                if alert_time >= cutoff_time:
                    recent_alerts.append(alert)
            
            return recent_alerts
            
        except Exception as e:
            logger.error(f"Error retrieving recent alerts: {e}")
            return []
    
    def process_input(self, input_data: str) -> str:
        """
        LangChain Tool interface - processes agent outputs and makes decisions
        
        Args:
            input_data: JSON string with agent outputs containing:
                - vision_confidence: float
                - audio_confidence: float
                - pir_triggered: bool
                - context_risk: float
            
        Returns:
            JSON string with alert decision
        """
        try:
            if not input_data:
                return json.dumps({
                    "alert_level": "ERROR",
                    "error": "No input data provided"
                })
            
            # Parse input data
            try:
                data = json.loads(input_data)
            except json.JSONDecodeError:
                return json.dumps({
                    "alert_level": "ERROR",
                    "error": "Invalid JSON input"
                })
            
            # Extract required fields with defaults
            vision_conf = float(data.get("vision_confidence", 0.0))
            audio_conf = float(data.get("audio_confidence", 0.0))
            pir_triggered = bool(data.get("pir_triggered", False))
            context_risk = float(data.get("context_risk", 0.0))
            # Parse input data (should contain outputs from other agents)
            data = json.loads(input_data) if input_data else {}
            
            # Extract confidence scores
            vision_conf = data.get("vision_confidence", 0.0)
            audio_conf = data.get("audio_confidence", 0.0)
            pir_triggered = data.get("pir_triggered", False)
            context_risk = data.get("context_risk", 0.0)
            
            # Apply arbitration rules
            decision = self.apply_arbitration_rules(
                vision_conf, audio_conf, pir_triggered, context_risk
            )
            
            # Log the incident
            self.log_incident(decision)
            
            # Trigger appropriate actions
            self.trigger_alert_actions(decision)
            
            return json.dumps(decision)
            
        except Exception as e:
            logger.error(f"Alert processing error: {e}")
            error_decision = {
                "alert_level": "ERROR",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            return json.dumps(error_decision)

# Create LangChain Tool wrapper
def create_alert_tool() -> Tool:
    """Create LangChain Tool for Alert Agent"""
    agent = AlertAgent()
    
    return Tool(
        name="AlertArbitrator",
        description="Processes vision, audio, and context data to make alert decisions using arbitration rules. Input should be JSON with confidence scores and sensor data.",
        func=agent.process_input
    )