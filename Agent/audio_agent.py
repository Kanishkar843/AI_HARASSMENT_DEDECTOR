# audio_agent.py - LangChain Version
"""
Audio Agent for harassment detection using Hugging Face audio classification.
Detects screaming, shouting, and aggressive audio patterns.
Optimized for Raspberry Pi 4.
"""

import wave
import json
import numpy as np
import torch
from transformers import pipeline
import librosa

# Use version-adaptive import for Tool
try:
    from langchain_core.tools import Tool
except ImportError:
    try:
        from langchain.tools.base import Tool
    except ImportError:
        from langchain.tools import Tool

try:
    import pyaudio
except ImportError:
    raise ImportError(
        "PyAudio is required. Install it using: pip install pyaudio"
    )
import logging
import time
import os
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioAgent:
    # Threshold constants for easier tuning
    VOLUME_HIGH = 0.7     # High volume threshold
    VOLUME_VERY_HIGH = 0.8  # Very high volume threshold
    PITCH_VAR_HIGH = 0.2   # High pitch variation threshold
    PITCH_VAR_VERY_HIGH = 0.4  # Very high pitch variation threshold
    
    def __init__(self, model_name: str = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"):
        """
        Initialize Audio Agent for aggression/violence detection
        
        Args:
            model_name: Hugging Face model for audio classification
                Supported models:
                - audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim (default)
                - MIT/ast-finetuned-speech-commands-v2
                
        Notes:
            - On Raspberry Pi, install system dependencies first:
              sudo apt-get install portaudio19-dev python3-pyaudio
            - Model loading includes fallback chain:
              1. Try primary model on GPU
              2. Try primary model on CPU
              3. Try fallback model on CPU
            
            Alert Thresholds:
            - Volume > 0.8 or Pitch Variation > 0.4: High risk
            - Volume > 0.7 or Pitch Variation > 0.2: Medium risk
            - Model confidence > 0.5: Add to risk score
        """
        self.model_name = model_name
        self.sample_rate = 16000
        self.duration = 3  # seconds to record
        
        # Audio recording settings
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        
        try:
            # Device handling matching VisionAgent pattern
            device = torch.cuda.current_device() if torch.cuda.is_available() else -1
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            
            # Try loading primary model
            try:
                self.classifier = pipeline(
                    "audio-classification",
                    model=model_name,
                    device=device,
                    dtype=dtype  # Updated from torch_dtype to dtype
                )
                logger.info(f"Audio Agent initialized with model: {model_name} on {'cuda:'+str(device) if device >= 0 else 'cpu'}")
                
            except Exception as e:
                logger.warning(f"Failed to load primary model {model_name}, trying fallback. Error: {e}")
                
                # First fallback - try same model on CPU
                try:
                    self.classifier = pipeline(
                        "audio-classification",
                        model=model_name,
                        device=-1,
                        torch_dtype=torch.float32
                    )
                    logger.info(f"Using primary model on CPU: {model_name}")
                    
                except Exception as e2:
                    logger.warning(f"Failed to load primary model on CPU: {e2}")
                    
                    # Second fallback - try simpler model
                    try:
                        fallback_model = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
                        self.classifier = pipeline(
                            "audio-classification",
                            model=fallback_model,
                            device=-1,  # Force CPU for fallback
                            torch_dtype=torch.float32
                        )
                        logger.info(f"Using fallback audio emotion model on CPU: {fallback_model}")
                    except Exception as e3:
                        logger.error(f"Failed to load any models: {e3}")
                        self.classifier = None
                        
        except Exception as e:
            logger.error(f"Critical error initializing AudioAgent: {e}")
            self.classifier = None
    
    def record_audio(self) -> np.ndarray:
        """
        Record audio sample from microphone
        
        Returns:
            numpy array of float32 values normalized to [-1, 1]
            empty array if recording fails
        """
        audio = None
        stream = None
        try:
            # Initialize PyAudio
            try:
                audio = pyaudio.PyAudio()
            except Exception as e:
                if "PortAudio" in str(e):
                    logger.error("PortAudio error. On Raspberry Pi, run: sudo apt-get install portaudio19-dev")
                    return np.array([])
                raise e
            
            # Get default input device info
            try:
                device_info = audio.get_default_input_device_info()
                logger.info(f"Using audio input device: {device_info.get('name', 'Unknown')}")
            except Exception as e:
                logger.error(f"No audio input device found: {e}")
                return np.array([])
            
            # Start recording with error handling
            try:
                stream = audio.open(
                    format=self.format,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    frames_per_buffer=self.chunk,
                    input_device_index=device_info.get('index', None)
                )
            except Exception as e:
                logger.error(f"Failed to open audio stream: {e}")
                return np.array([])
            
            logger.info(f"Recording {self.duration} seconds of audio...")
            frames = []
            
            # Record with overflow handling
            for _ in range(0, int(self.sample_rate / self.chunk * self.duration)):
                try:
                    data = stream.read(self.chunk, exception_on_overflow=False)
                    frames.append(data)
                except Exception as e:
                    logger.warning(f"Frame capture error: {e}")
                    continue
            
            if not frames:
                logger.error("No audio frames captured")
                return np.array([])
            
            # Convert to numpy array
            audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
            audio_data = audio_data.astype(np.float32) / 32768.0  # Normalize to [-1, 1]
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Audio recording error: {e}")
            return np.array([])
    
    def analyze_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Analyze audio for aggression/harassment indicators
        
        Args:
            audio_data: Raw audio data as float32 numpy array normalized to [-1,1]
            sample_rate: Sample rate of the audio data (default 16kHz)
            
        Returns:
            Dict with:
                - audio_confidence: Float 0-1 indicating violence/harassment likelihood
                  Based on emotion model + acoustic features. Values mean:
                  > 0.7: High risk (screaming, shouting, clear aggression)
                  0.3-0.7: Medium risk (raised voices, potential aggression)
                  < 0.3: Low risk (normal conversation, background noise)
                
                - volume_level: Float 0-1 indicating loudness intensity
                  < 0.1: Very quiet/background noise
                  0.3-0.5: Normal conversation
                  > 0.7: Loud voices/shouting
                
                - pitch_variation: Float 0-1 indicating vocal stress
                  < 0.3: Steady speech/monotone
                  0.3-0.6: Normal conversation variation
                  > 0.6: High variation (stress/panic indicators)
                
                - raw_results: List of emotion model predictions, each with:
                  - label: Emotion category (ang=anger, neu=neutral, etc)
                  - score: Confidence 0-1 for that emotion
                
                - status: "success" or error description
                
                - saved_audio: Path to saved WAV if audio_confidence > threshold
                  (default threshold 0.6, adjust for testing)
        """
        try:
            if len(audio_data) == 0 or self.classifier is None:
                return {"audio_confidence": 0.0, "error": "No audio or classifier"}
            
            # Ensure proper sampling rate
            if len(audio_data) < sample_rate * 0.5:  # Less than 0.5 seconds
                return {"audio_confidence": 0.0, "error": "Audio too short"}
            
            # Get predictions
            results = self.classifier(audio_data, sampling_rate=self.sample_rate)
            
            # Map model predictions to aggression confidence
            label_map = {
                # High-risk labels (direct indicators)
                'angry': 1.0,
                'anger': 1.0,
                'scream': 1.0,
                'shout': 0.9,
                'yell': 0.9,
                'fear': 0.8,
                'panic': 0.8,
                
                # Medium-risk labels (potential indicators)
                'loud': 0.6,
                'tense': 0.5,
                'stress': 0.5,
                'irritated': 0.5,
                
                # Low-risk labels (contextual)
                'neutral': 0.1,
                'calm': 0.0,
                'happy': 0.0
            }
            
            audio_confidence = 0.0
            
            # Process model predictions
            for result in results:
                label = result['label'].lower()
                score = result['score']
                
                # Apply label-specific risk weights
                if label in label_map:
                    weighted_score = score * label_map[label]
                    audio_confidence = max(audio_confidence, weighted_score)
                    if weighted_score > 0.5:
                        logger.info(f"Detected {label} with confidence {score:.2f}")
                        
            # Add context from acoustic features
            volume_risk = 0.0
            pitch_risk = 0.0
            
            # Additional audio feature analysis
            volume_level = self._analyze_volume(audio_data)
            pitch_variation = self._analyze_pitch_variation(audio_data)
            
            # Heuristic-based audio confidence scoring
            
            # Calculate base audio confidence from acoustic features
            if volume_level > 0.75 and pitch_variation > 0.2:
                audio_confidence = 0.8   # Aggressive scream → High alert
                logger.info(f"HIGH ALERT - Aggressive scream detected: vol={volume_level:.2f}, pitch_var={pitch_variation:.2f}")
            elif volume_level > 0.7 or pitch_variation > 0.2:
                audio_confidence = 0.5   # Loud shouting or unstable tone → Medium alert
                if volume_level > 0.7:
                    logger.info(f"MEDIUM ALERT - Loud shouting: vol={volume_level:.2f}")
                else:
                    logger.info(f"MEDIUM ALERT - Unstable vocal tone: pitch_var={pitch_variation:.2f}")
            else:
                audio_confidence = 0.0   # Quiet background / normal conversation → Safe
                logger.info(f"Normal audio levels: vol={volume_level:.2f}, pitch_var={pitch_variation:.2f}")
            
            # Use emotion model outputs to refine confidence
            for result in results[:3]:  # Look at top 3 predictions
                label = result['label'].lower()
                score = result['score']
                
                # Boost confidence based on emotional arousal/stress
                if label == 'arousal' and score > 0.6:
                    audio_confidence = max(audio_confidence, 0.6)
                    logger.info(f"High arousal detected: {score:.2f}")
                elif label == 'dominance' and score > 0.7:
                    audio_confidence = max(audio_confidence, 0.5)
                    logger.info(f"High dominance detected: {score:.2f}")
                    
            # If significant volume but no other indicators, maintain minimum alert level
            if volume_level > 0.6 and audio_confidence < 0.3:
                audio_confidence = 0.3
                logger.info("Setting minimum alert level due to elevated volume")
            
            # Log detailed analysis
            logger.info(
                f"Audio analysis: conf={audio_confidence:.2f}, "
                f"vol={volume_level:.2f}, pitch_var={pitch_variation:.2f}"
            )
            
            # Save audio clip if confidence is high
            saved_path = None
            if audio_confidence > 0.6:
                saved_path = self.save_audio_clip(audio_data, audio_confidence)
            
            return {
                "audio_confidence": float(audio_confidence),
                "volume_level": float(volume_level),
                "pitch_variation": float(pitch_variation),
                "raw_results": [
                    {"label": str(r["label"]), "score": float(r["score"])} 
                    for r in results[:3]
                ],
                "status": "success",
                "saved_audio": saved_path if saved_path else None
            }
            
        except Exception as e:
            logger.error(f"Audio analysis error: {e}")
            return {"aggression_confidence": 0.0, "error": str(e)}
    
    def _analyze_volume(self, audio_data: np.ndarray) -> float:
        """Analyze volume level (0-1)"""
        try:
            rms = np.sqrt(np.mean(audio_data ** 2))
            # Normalize to 0-1 scale (assuming max RMS of 0.5 is very loud)
            return min(rms / 0.5, 1.0)
        except:
            return 0.0
    
    def _analyze_pitch_variation(self, audio_data: np.ndarray) -> float:
        """Analyze pitch variation (higher = more chaotic/aggressive)"""
        try:
            # Use librosa for pitch analysis
            pitches, magnitudes = librosa.piptrack(
                y=audio_data, 
                sr=self.sample_rate,
                threshold=0.1
            )
            
            # Extract fundamental frequencies
            pitch_values = []
            for frame in range(pitches.shape[1]):
                index = magnitudes[:, frame].argmax()
                pitch = pitches[index, frame]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if len(pitch_values) > 1:
                pitch_std = np.std(pitch_values)
                pitch_mean = np.mean(pitch_values)
                
                # Calculate coefficient of variation (normalized std)
                if pitch_mean > 0:
                    variation = pitch_std / pitch_mean
                    return min(variation / 2.0, 1.0)  # Normalize
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Pitch analysis error: {e}")
            return 0.0
    
    def process_input(self, input_data: str = "") -> str:
        """
        LangChain Tool interface - processes live audio input or file path
        
        Args:
            input_data: Optional file path to analyze instead of recording
            
        Returns:
            JSON string with analysis results including:
            - audio_confidence: Float 0-1 indicating aggression level
            - volume_level: Float 0-1 indicating volume intensity
            - pitch_variation: Float 0-1 indicating pitch variability
            - raw_results: List of emotion classification results
            - status: "success" or error message
            - saved_audio: Path to saved clip if significant event detected
        """
        if input_data and os.path.exists(input_data):
            # Load audio file
            audio_data = self.load_audio_file(input_data)
        else:
            # Record from microphone
            audio_data = self.record_audio()
            
        results = self.analyze_audio(audio_data)
        
        # Save audio for any significant event (confidence > 0.5 or very loud)
        if (results.get("audio_confidence", 0) > 0.5 or 
            results.get("volume_level", 0) > 0.7):
            
            # Get top emotion from raw results
            top_emotion = ""
            if results.get("raw_results"):
                top_result = results["raw_results"][0]
                if top_result["score"] > 0.3:  # Only include if somewhat confident
                    top_emotion = f"emotion_{top_result['label']}"
            
            # Build metadata string
            metadata = [
                f"vol={results.get('volume_level', 0):.2f}",
                f"pitch={results.get('pitch_variation', 0):.2f}"
            ]
            if top_emotion:
                metadata.append(top_emotion)
                
            saved_path = self.save_audio_clip(
                audio_data,
                results["audio_confidence"],
                note=",".join(metadata)
            )
            results["saved_audio"] = saved_path
            logger.info(f"Saved audio for review: {saved_path}")
            
        return json.dumps(results)
    
    def save_audio_clip(self, audio_data: np.ndarray, confidence: float, note: str = ""):
        """
        Save audio clip with metadata
        
        Args:
            audio_data: Audio data to save
            confidence: Confidence score for the event
            note: Additional metadata to include in filename
        """
        try:
            if len(audio_data) == 0:
                logger.error("No audio data to save")
                return
                
            alerts_dir = os.path.join(os.path.dirname(__file__), "alerts")
            os.makedirs(alerts_dir, exist_ok=True)
            
            # Create filename with metadata
            timestamp = int(time.time())
            if note:
                filename = os.path.join(alerts_dir, 
                    f"alert_audio_{timestamp}_conf{confidence:.2f}_{note}.wav"
                )
            else:
                filename = os.path.join(alerts_dir, 
                    f"alert_audio_{timestamp}_conf{confidence:.2f}.wav"
                )
            
            # Convert back to int16 for saving
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
            
            logger.info(f"Saved alert audio with metadata: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error saving audio clip: {e}")
            return None

    @staticmethod
    def load_audio_file(file_path: str, target_sr: int = 16000) -> np.ndarray:
        """
        Load and preprocess an audio file for analysis
        
        Args:
            file_path: Path to audio file (WAV, MP3, etc)
            target_sr: Target sample rate (default 16kHz for most models)
            
        Returns:
            Normalized audio data as numpy array
        """
        try:
            # Load audio file with resampling
            y, sr = librosa.load(file_path, sr=target_sr)
            
            # Ensure float32 and normalize to [-1, 1]
            if y.dtype != np.float32:
                y = y.astype(np.float32)
            if np.abs(y).max() > 1.0:
                y /= np.abs(y).max()
                
            logger.info(f"Loaded audio file: {file_path} (length: {len(y)/target_sr:.1f}s)")
            return y
            
        except Exception as e:
            logger.error(f"Error loading audio file {file_path}: {e}")
            return np.array([])

# Create LangChain Tool wrapper
def create_audio_tool() -> Tool:
    """Create LangChain Tool for Audio Agent"""
    agent = AudioAgent()
    
    return Tool(
        name="AudioAnalyzer",
        description="Analyzes microphone input for aggressive audio patterns, screaming, or shouting. Returns confidence score (0-1) for aggression detection.",
        func=agent.process_input
    )