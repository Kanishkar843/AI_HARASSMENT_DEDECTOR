# Audio Classification Model for Raspberry Pi 4

## Model Information
- **Test Accuracy:** 92.82%
- **Model Size:** 0.50 MB (Quantized TFLite)
- **Classes:** Scream, Aggressive_Speech, Normal_Talk, Environmental_Noise

## Audio Configuration
- **Sample Rate:** 22050 Hz
- **Duration:** 3 seconds
- **MFCC Features:** 40
- **Hop Length:** 512
- **FFT Size:** 2048

## Files Included
1. `audio_classification_quantized.tflite` - Quantized model for RPi4
2. `audio_model.h5` - Full Keras model
3. `audio_labels.txt` - Class labels
4. `audio_config.json` - Model configuration
5. `README.txt` - This file

## Usage on Raspberry Pi 4

```python
import tensorflow as tf
import numpy as np
import librosa

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="audio_classification_quantized.tflite")
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and process audio
audio, sr = librosa.load("audio_file.wav", sr=22050, duration=3)
mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40, 
                            hop_length=512, n_fft=2048)
mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
mfcc = mfcc[..., np.newaxis]  # Add channel dimension
mfcc = np.expand_dims(mfcc, axis=0)  # Add batch dimension

# Quantize input
input_scale = input_details[0]['quantization'][0]
input_zero_point = input_details[0]['quantization'][1]
input_data = mfcc / input_scale + input_zero_point
input_data = np.clip(input_data, 0, 255).astype(np.uint8)

# Run inference
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])

# Get prediction
predicted_class = np.argmax(output[0])
confidence = output[0][predicted_class]

print(f"Prediction: {class_names[predicted_class]}")
print(f"Confidence: {confidence:.2f}")
```

## Performance
- Inference time on RPi4: ~50-100ms per audio clip
- Memory usage: ~1 MB

## Training Details
- Training samples: 2534
- Validation samples: 543
- Test samples: 543
- Total epochs: 46

Generated: 2025-10-08T10:34:16
