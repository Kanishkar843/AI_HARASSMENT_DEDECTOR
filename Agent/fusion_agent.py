"""
Fusion Agent for combining Vision and Audio detections
Integrates harassment detections from multiple sensors into a final probability.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any
from vision_agent import VisionAgent
from audio_agent import AudioAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('incidents.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FusionAgent:
    # Confidence weights
    VISION_WEIGHT = 0.6
    AUDIO_WEIGHT = 0.4
    
    # Alert thresholds
    HIGH_ALERT_THRESHOLD = 0.8
    MEDIUM_ALERT_THRESHOLD = 0.7
    
    def __init__(self):
        """Initialize Fusion Agent with vision and audio detectors"""
        self.vision_agent = VisionAgent()
        self.audio_agent = AudioAgent()
        
        # Create snapshots directory
        self.snapshots_dir = os.path.join(os.path.dirname(__file__), "snapshots")
        os.makedirs(self.snapshots_dir, exist_ok=True)
    
    def fuse_detections(self, vision_results: Dict[str, Any], audio_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fuse vision and audio detections into final harassment probability
        
        Args:
            vision_results: Results from vision agent
            audio_results: Results from audio agent
            
        Returns:
            Dict with fusion results including final probability and alert status
        """
        try:
            # Extract confidence scores
            vision_conf = vision_results.get("vision_confidence", 0.0)
            audio_conf = audio_results.get("audio_confidence", 0.0)
            
            # Calculate weighted fusion score
            fusion_score = (self.VISION_WEIGHT * vision_conf) + (self.AUDIO_WEIGHT * audio_conf)
            
            # Determine alert level
            alert_level = "none"
            if fusion_score >= self.HIGH_ALERT_THRESHOLD:
                alert_level = "high"
            elif fusion_score >= self.MEDIUM_ALERT_THRESHOLD:
                alert_level = "medium"
            
            # Log fusion results
            logger.info(
                f"Fusion results: score={fusion_score:.2f}, "
                f"vision={vision_conf:.2f}, audio={audio_conf:.2f}, "
                f"alert={alert_level}"
            )
            
            # Save snapshot if high alert
            snapshot_path = None
            if alert_level == "high" and vision_results.get("frame") is not None:
                snapshot_path = self._save_snapshot(vision_results["frame"])
            
            return {
                "fusion_score": fusion_score,
                "alert_level": alert_level,
                "vision_confidence": vision_conf,
                "audio_confidence": audio_conf,
                "saved_snapshot": snapshot_path,
                "saved_audio": audio_results.get("saved_audio"),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Fusion error: {e}")
            return {
                "fusion_score": 0.0,
                "alert_level": "error",
                "error": str(e)
            }
    
    def _save_snapshot(self, frame) -> str:
        """Save video frame on high alert"""
        try:
            timestamp = int(datetime.now().timestamp())
            filename = os.path.join(self.snapshots_dir, f"alert_snapshot_{timestamp}.jpg")
            
            # Save using OpenCV
            import cv2
            cv2.imwrite(filename, frame)
            
            logger.info(f"Saved alert snapshot: {filename}")
            return filename
        except Exception as e:
            logger.error(f"Error saving snapshot: {e}")
            return None

    def test_fusion(self, video_file: str = None, audio_file: str = None):
        """Run fusion test with sample files"""
        logger.info(f"Testing fusion with: video={video_file}, audio={audio_file}")
        
        # Get vision results
        if video_file:
            vision_results = self.vision_agent.analyze_video(video_file)
        else:
            vision_results = {"vision_confidence": 0.0}
        
        # Get audio results
        if audio_file:
            audio_data = self.audio_agent.load_audio_file(audio_file)
            audio_results = self.audio_agent.analyze_audio(audio_data)
        else:
            audio_results = {"audio_confidence": 0.0}
        
        # Run fusion
        fusion_results = self.fuse_detections(vision_results, audio_results)
        
        # Log detailed results
        print("\nFusion Test Results")
        print("=" * 50)
        print(f"Vision confidence: {vision_results.get('vision_confidence', 0.0):.2f}")
        print(f"Audio confidence: {audio_results.get('audio_confidence', 0.0):.2f}")
        print(f"Final fusion score: {fusion_results['fusion_score']:.2f}")
        print(f"Alert level: {fusion_results['alert_level'].upper()}")
        
        if fusion_results.get("saved_snapshot"):
            print(f"⚠️ Snapshot saved: {fusion_results['saved_snapshot']}")
        if fusion_results.get("saved_audio"):
            print(f"⚠️ Audio saved: {fusion_results['saved_audio']}")
        print("=" * 50)
        
        return fusion_results

def main():
    """Run fusion tests with sample files"""
    fusion = FusionAgent()
    
    # Test cases
    print("\nTest Case 1: Aggressive scream + normal video")
    fusion.test_fusion(
        video_file="test_normal_video.mp4",
        audio_file="test_aggressive_scream.wav"
    )
    
    print("\nTest Case 2: Harassment video + quiet audio")
    fusion.test_fusion(
        video_file="test_harassment_video.mp4",
        audio_file="test_quiet_background.wav"
    )
    
    print("\nTest Case 3: Harassment video + aggressive scream")
    fusion.test_fusion(
        video_file="test_harassment_video.mp4",
        audio_file="test_aggressive_scream.wav"
    )

if __name__ == "__main__":
    main()
