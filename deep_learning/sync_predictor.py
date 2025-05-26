import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.python.keras import models, layers
import librosa
import collections
import gc
import tensorflow as tf
from deep_learning.losses import hybrid_loss
class HybridPredictor:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={'hybrid_loss': hybrid_loss},  # Explicitly reference
            compile=True
        )
        try:
            # Load with BOTH custom objects and compilation
            self.model = tf.keras.models.load_model(
                model_path,
                custom_objects={
                    'hybrid_loss': hybrid_loss,  # Explicit mapping
                    # Add other custom objects here if needed
                },
                compile=True  # Keep optimizer/compilation intact
            )
            # Manually compile with custom loss
            self.model = tf.keras.models.load_model(
                model_path,
                custom_objects={'hybrid_loss': hybrid_loss},  # Register during loading
                compile=True  # Keep compilation intact [7][8]
            )
            self.model.make_predict_function()
        except Exception as e:
            raise RuntimeError(f"Model loading failed: {str(e)}")

        
        # Initialize audio buffer
        self.audio_window = collections.deque(maxlen=100)
        
        # Warm-up with proper cleanup
        self.warmup_model()
        
        physical_devices = tf.config.list_physical_devices('GPU')
        layers.Conv2D(64, (3,3), kernel_initializer='he_normal', use_bias=False)

    def warmup_model(self):
        """Warmup with proper input shape handling for AMD GPUs"""
        try:
            # Handle multi-input models (visual + audio)
            dummy_inputs = []
            for input_spec in self.model.inputs:
                # Get shape without batch dimension [1:]
                input_shape = [dim if dim is not None else 224 for dim in input_spec.shape[1:]]
                
                # Generate dummy input with batch dimension=1
                dummy_input = np.random.rand(1, *input_shape).astype(np.float32)
                dummy_inputs.append(dummy_input)
            
            # Warmup prediction
            self.model.predict(dummy_inputs, batch_size=1)
        except Exception as e:
            print(f"Warmup failed: {str(e)}")
        finally:
            tf.keras.backend.clear_session()
            gc.collect()



            
    def predict_sync(self, frame_batch):
        try:
            return self.model(frame_batch)
        except tf.errors.ResourceExhaustedError:
            tf.keras.backend.clear_session()
            gc.collect()
            return self.model(frame_batch[:16])  # Retry with smaller batch

    @tf.function(jit_compile=True)  # Enable XLA compilation
    def preprocess_batch(self, frames):
        # Keep preprocessing on GPU
        resized = tf.image.resize(frames, [320, 180])
        return tf.cast(resized, tf.float32) / 255.0

    def preprocess_frame(self, frame):
        if cv2.ocl.haveOpenCL():
            gpu_frame = cv2.UMat(frame)
            resized = cv2.resize(gpu_frame, (320, 180))
            return cv2.UMat.get(resized) / 255.0
        return cv2.resize(frame, (320, 180)) / 255.0

    def cleanup_resources(self):
        tf.keras.backend.clear_session()
        gc.collect()
        if cv2.ocl.haveOpenCL():
            cv2.ocl.finish()

    def __del__(self):
        self.cleanup_resources()
       
    
    def hybrid_loss(y_true, y_pred):
        audio_weight = HYBRID_PARAMS["audio_weight"]
        return (audio_weight * tf.keras.losses.binary_crossentropy(y_true, y_pred) +
                (1 - audio_weight) * tf.keras.losses.binary_crossentropy(y_true, y_pred))