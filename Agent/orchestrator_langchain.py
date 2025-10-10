# orchestrator_langchain.py - LangChain Multi-Agent System
"""
LangChain orchestrator for harassment detection multi-agent system.
Coordinates Vision, Audio, Context, and Alert agents using LangChain framework.
Optimized for Raspberry Pi 4.
"""

import json
import time
import logging
from datetime import datetime
from typing import Dict, Any, List
from langchain.agents import initialize_agent, AgentType
from langchain.schema import AgentAction, AgentFinish
from langchain.callbacks.base import BaseCallbackHandler

# Import our custom agents
from vision_agent import create_vision_tool
from audio_agent import create_audio_tool
from context_agent import create_context_tool
from alert_agent import create_alert_tool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('harassment_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HarassmentDetectionCallback(BaseCallbackHandler):
    """Custom callback handler to monitor agent execution"""
    
    def __init__(self):
        self.agent_results = {}
        self.execution_times = {}
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """Called when tool starts executing"""
        tool_name = serialized.get("name", "Unknown")
        tool_id = kwargs.get("run_id", str(time.time()))  # Use run_id for unique tracking
        self.execution_times[tool_id] = {"name": tool_name, "start": time.time()}
        logger.info(f"🔧 Starting {tool_name} (ID: {tool_id})")
    
    def on_tool_end(self, output: str, **kwargs) -> None:
        """Called when tool finishes executing"""
        tool_id = kwargs.get("run_id", None)
        if tool_id and tool_id in self.execution_times:
            tool_info = self.execution_times[tool_id]
            tool_name = tool_info["name"]
            
            try:
                # Parse output if it's JSON
                result = json.loads(output)
            except json.JSONDecodeError:
                result = {"raw_output": output}
            
            self.agent_results[tool_name] = result
            duration = time.time() - tool_info["start"]
            logger.info(f"✅ {tool_name} completed in {duration:.2f}s")
            del self.execution_times[tool_id]  # Cleanup

class LangChainOrchestrator:
    """
    LangChain-based orchestrator for harassment detection system
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the LangChain orchestrator
        
        Args:
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        self.callback_handler = HarassmentDetectionCallback()
        
        # Create tools from our agents
        self.tools = [
            create_vision_tool(),
            create_audio_tool(), 
            create_context_tool(),
            create_alert_tool()
        ]
        
        # Initialize LangChain agent - using ZERO_SHOT_REACT_DESCRIPTION for simplicity
        # Note: We'll use a mock LLM since we don't have access to OpenAI/Anthropic APIs
        # In production, you would use a real LLM
        self.agent = self._create_mock_agent()
        
        logger.info("LangChain Orchestrator initialized with 4 agents")
    
    def _create_mock_agent(self):
        """
        Create a mock agent that follows our specific workflow
        Since we don't have LLM access, we'll simulate the agent behavior
        """
        class MockAgent:
            def __init__(self, tools, callback_handler):
                self.tools = tools
                self.callback_handler = callback_handler
            
            def run(self, query):
                """Execute all tools in sequence and combine results"""
                results = {}
                
                # Execute each tool
                for tool in self.tools:
                    try:
                        self.callback_handler.on_tool_start({"name": tool.name}, "")
                        output = tool.func("")
                        self.callback_handler.on_tool_end(output)
                        results[tool.name] = output
                    except Exception as e:
                        logger.error(f"Error executing {tool.name}: {e}")
                        results[tool.name] = f"Error: {str(e)}"
                
                return results
        
        return MockAgent(self.tools, self.callback_handler)
    
    def run_detection_cycle(self) -> Dict[str, Any]:
        """
        Run a complete detection cycle using all agents
        
        Returns:
            Dictionary with results from all agents and final decision
        """
        try:
            logger.info("🚀 Starting harassment detection cycle")
            cycle_start_time = time.time()
            
            # Reset callback handler
            self.callback_handler.agent_results = {}
            self.callback_handler.execution_times = {}
            
            # Run all tools and collect results
            results = self.agent.run("")
            
            # Parse the results
            parsed_results = self._parse_agent_results(results)
            
            # Extract individual results for alert arbitration
            vision_result = parsed_results.get("VisionAnalyzer", {})
            audio_result = parsed_results.get("AudioAnalyzer", {})
            context_result = parsed_results.get("ContextAnalyzer", {})
            alert_input = {
                "vision_confidence": vision_result.get("violence_confidence", 0.0),
                "audio_confidence": audio_result.get("audio_confidence", 0.0),
                "pir_triggered": context_result.get("pir_triggered", False),
                "context_risk": context_result.get("context_risk", 0.0)
            }
            
            # Run alert arbitration with combined results
            alert_result = json.loads(
                self.tools[-1].func(json.dumps(alert_input))  # AlertArbitrator is last tool
            )
            
            # Create final alert decision
            final_decision = self._create_final_decision(parsed_results)
            
            cycle_time = time.time() - cycle_start_time
            
            # Create comprehensive result
            detection_result = {
                "timestamp": datetime.now().isoformat(),
                "cycle_time": cycle_time,
                "agent_results": parsed_results,
                "final_decision": final_decision,
                "system_status": "operational"
            }
            
            # Log structured result
            self._log_detection_result(detection_result)
            
            logger.info(f"🎯 Detection cycle completed in {cycle_time:.2f}s - Decision: {final_decision.get('alert_level', 'UNKNOWN')}")
            
            return detection_result
            
        except Exception as e:
            logger.error(f"Detection cycle error: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "system_status": "error"
            }
    
    def _parse_agent_results(self, results: Dict[str, str]) -> Dict[str, Any]:
        """Parse and clean results from each agent"""
        parsed = {}
        
        for agent_name, result_str in results.items():
            try:
                # Try to parse as JSON
                if result_str.startswith('{') or result_str.startswith('['):
                    parsed[agent_name] = json.loads(result_str)
                else:
                    # Handle non-JSON results
                    parsed[agent_name] = {"raw_output": result_str}
                    
            except json.JSONDecodeError:
                # Handle malformed JSON
                parsed[agent_name] = {"raw_output": result_str, "parse_error": True}
                logger.warning(f"Failed to parse JSON from {agent_name}")
        
        return parsed
    
    def _create_final_decision(self, parsed_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create final alert decision by combining agent outputs
        This simulates what the Alert Agent would do in the workflow
        """
        try:
            # Extract key metrics
            vision_conf = 0.0
            audio_conf = 0.0
            pir_triggered = False
            context_risk = 0.0
            
            # Extract vision confidence
            if "VisionAnalyzer" in parsed_results:
                vision_data = parsed_results["VisionAnalyzer"]
                vision_conf = vision_data.get("violence_confidence", 0.0)
            
            # Extract audio confidence  
            if "AudioAnalyzer" in parsed_results:
                audio_data = parsed_results["AudioAnalyzer"]
                audio_conf = audio_data.get("aggression_confidence", 0.0)
            
            # Extract context data
            if "ContextAnalyzer" in parsed_results:
                context_data = parsed_results["ContextAnalyzer"]
                pir_triggered = context_data.get("pir_triggered", False)
                context_risk = context_data.get("context_risk", 0.0)
            
            # Extract alert decision if available
            if "AlertArbitrator" in parsed_results:
                alert_data = parsed_results["AlertArbitrator"]
                return alert_data
            
            # Manual arbitration if alert agent failed
            return self._manual_arbitration(vision_conf, audio_conf, pir_triggered, context_risk)
            
        except Exception as e:
            logger.error(f"Final decision error: {e}")
            return {
                "alert_level": "ERROR",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _manual_arbitration(self, vision_conf: float, audio_conf: float, 
                          pir_triggered: bool, context_risk: float) -> Dict[str, Any]:
        """Manual arbitration rules as fallback"""
        
        alert_level = "NORMAL"
        reasoning = []
        
        # Apply the same rules as Alert Agent
        if vision_conf >= 0.8 and audio_conf >= 0.7:
            alert_level = "HIGH"
            reasoning.append(f"HIGH: Vision {vision_conf:.1%} + Audio {audio_conf:.1%}")
        elif (vision_conf >= 0.7 or audio_conf >= 0.6) and pir_triggered:
            alert_level = "MEDIUM" 
            reasoning.append(f"MEDIUM: High confidence + PIR triggered")
        elif vision_conf >= 0.5 or audio_conf >= 0.5:
            alert_level = "LOW"
            reasoning.append("LOW: Moderate detection levels")
        
        return {
            "alert_level": alert_level,
            "alert_score": max(vision_conf, audio_conf),
            "reasoning": reasoning,
            "timestamp": datetime.now().isoformat(),
            "fallback_arbitration": True
        }
    
    def _log_detection_result(self, result: Dict[str, Any]):
        """Log detection result in structured JSON format"""
        try:
            with open("detection_results.log", "a") as f:
                f.write(json.dumps(result) + "\n")
        except Exception as e:
            logger.error(f"Failed to log detection result: {e}")
    
    def run_continuous_monitoring(self, interval_seconds: int = 30):
        """
        Run continuous monitoring loop
        
        Args:
            interval_seconds: Time between detection cycles
        """
        logger.info(f"🔄 Starting continuous monitoring (interval: {interval_seconds}s)")
        
        try:
            while True:
                # Run detection cycle
                result = self.run_detection_cycle()
                
                # Check for high alerts
                if result.get("final_decision", {}).get("alert_level") == "HIGH":
                    logger.critical("🚨 HIGH ALERT DETECTED - Consider immediate action!")
                
                # Wait for next cycle
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            logger.info("🛑 Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Continuous monitoring error: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "timestamp": datetime.now().isoformat(),
            "agents_available": len(self.tools),
            "tools": [tool.name for tool in self.tools],
            "last_callback_results": len(self.callback_handler.agent_results),
            "system_status": "operational"
        }

def main():
    """
    Main function demonstrating the LangChain harassment detection system
    """
    print("🤖 LangChain Harassment Detection System - Raspberry Pi 4")
    print("=" * 60)
    
    # Initialize orchestrator
    orchestrator = LangChainOrchestrator(verbose=True)
    
    # Show system status
    status = orchestrator.get_system_status()
    print(f"System Status: {json.dumps(status, indent=2)}")
    print()
    
    # Run sample detection cycle
    print("Running sample detection cycle...")
    result = orchestrator.run_detection_cycle()
    
    print("\n" + "="*60)
    print("DETECTION RESULTS:")
    print("="*60)
    print(f"Alert Level: {result['final_decision'].get('alert_level', 'UNKNOWN')}")
    print(f"Cycle Time: {result.get('cycle_time', 0):.2f} seconds")
    
    if 'agent_results' in result:
        for agent_name, agent_result in result['agent_results'].items():
            print(f"\n{agent_name}:")
            if isinstance(agent_result, dict):
                for key, value in agent_result.items():
                    if key not in ['raw_results', 'error']:  # Skip verbose data
                        print(f"  {key}: {value}")
    
    print("\n" + "="*60)
    
    # Ask user if they want continuous monitoring
    try:
        response = input("\nStart continuous monitoring? (y/N): ").lower()
        if response == 'y':
            orchestrator.run_continuous_monitoring(interval_seconds=30)
    except KeyboardInterrupt:
        print("\nShutdown complete.")

if __name__ == "__main__":
    main()