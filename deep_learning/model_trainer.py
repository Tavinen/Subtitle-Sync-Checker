import tensorflow as tf
tf.config.run_functions_eagerly(True)
import os
import logging
from logging.handlers import RotatingFileHandler
import sys
import psutil
import tensorflow as tf
import datetime
import cv2
import numpy as np

def extract_frames(video_path, target_frames=32):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames-1, target_frames, dtype=np.int32)
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        else:
            frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
    cap.release()
    frames = np.array(frames, dtype=np.float32) / 255.0 # (target_frames, 224, 224, 3)
    return frames

# Configure logging: keep last 5 log files, each up to 5MB
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_file = 'app.log'
file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5)
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)

logger = logging.getLogger('crash_logger')
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def log_system_usage(filename="crash_resource_usage.txt"):
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(f"{now}_{filename}", "w") as f:
        cpu = psutil.cpu_percent(interval=1)
        ram = psutil.virtual_memory()
        try:
            gpu_stats = tf.config.experimental.get_memory_info('GPU:0')
            gpu_usage = (gpu_stats['current'] / gpu_stats['limit']) * 100
        except Exception:
            gpu_usage = 'Unavailable'
        f.write(f"Timestamp: {now}\n")
        f.write(f"CPU Usage: {cpu}%\n")
        f.write(f"RAM Usage: {ram.percent}% (Used: {ram.used // (1024**2)} MB / Total: {ram.total // (1024**2)} MB)\n")
        f.write(f"GPU Usage: {gpu_usage}%\n")

def log_uncaught_exceptions(exctype, value, tb):
    logger.critical("Uncaught exception", exc_info=(exctype, value, tb))
    log_system_usage()  # Save resource usage snapshot

sys.excepthook = log_uncaught_exceptions

logger.info("Application started")
from tensorflow.keras import mixed_precision


policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'



print(os.getcwd())
import cv2
import tensorflow as tf


tf.config.optimizer.set_jit(False)  # Enable XLA globally

opt = tf.keras.optimizers.Adam()
tf.config.optimizer.set_jit(False)  # Enable XLA

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow log spam
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'  # Optimize GPU threading for DirectML

# Limit GPU memory to 70% (leaves 30% free)
import psutil
import time
import gc

dataset_path = "training_data/dataset"
abs_dataset_path = os.path.abspath(dataset_path)

# For glob






def memory_safe(threshold_percent=80, check_interval=2):
    """Decorator to pause processing if RAM usage exceeds threshold"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            while psutil.virtual_memory().percent > threshold_percent:
                print(f"Memory {psutil.virtual_memory().percent}% > {threshold_percent}% - waiting...")
                time.sleep(check_interval)
            
            result = func(*args, **kwargs)
            
            # Force garbage collection after each batch
            tf.keras.backend.clear_session()
            gc.collect()
            
            return result
        return wrapper
    return decorator

import scipy
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import librosa
from hybrid_config import HYBRID_PARAMS
import tensorflow as tf

# After GPU configuration
from tensorflow.python.client import device_lib
print("GPU configuration:", device_lib.list_local_devices())

# During training, monitor usage
def get_gpu_memory():
    stats = tf.config.experimental.get_memory_info('GPU:0')
    return stats['current'] / (1024**30)  # Usage in GB




def extract_audio_features(video_path, stream=False, chunk_size=10):
    import subprocess
    import tempfile
    import os
    import librosa
    import numpy as np

    # Create temporary WAV file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
        temp_name = temp_audio.name

    # Extract audio using ffmpeg
    subprocess.call([
        'ffmpeg', '-y', '-i', video_path,
        '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
        temp_name
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        audio, sr = librosa.load(temp_name, sr=16000)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        # Ensure shape (40, 100) for model compatibility
        if mfccs.shape[1] < 100:
            mfccs = np.pad(mfccs, ((0, 0), (0, 100 - mfccs.shape[1])), mode='constant')
        elif mfccs.shape[1] > 100:
            mfccs = mfccs[:, :100]
        if mfccs.shape[1] < 100:
            mfccs = np.pad(mfccs, ((0, 0), (0, 100 - mfccs.shape[1])), mode='constant')
        elif mfccs.shape[1] > 100:
            mfccs = mfccs[:, :100]
        print("extract_audio_features: mfccs.shape =", mfccs.shape)
        return mfccs.astype(np.float32)
    finally:
        os.unlink(temp_name)



policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
optimizer = mixed_precision.LossScaleOptimizer(optimizer)
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

def preprocess_frame(frame):
    height, width = frame.shape[:2]
    roi = frame[-int(height*0.2):, :]  # Bottom 20% for subtitles
    resized = cv2.resize(roi, (224, 224))
    return resized / 255.0

def retain_buffer_samples(dataset_path, keep_ratio=0.1):
    """Retain a percentage of samples in the training buffer"""
    import os
    import random
    import shutil
    
    buffer_dir = os.path.join(dataset_path, "buffer")
    os.makedirs(buffer_dir, exist_ok=True)

    # Clean up excess files first
    buffer_files = []
    for root, _, files in os.walk(buffer_dir):
        buffer_files.extend([os.path.join(root, f) for f in files])
    
    if len(buffer_files) > HYBRID_PARAMS["buffer_size"]:
        # Remove oldest 20% of excess files
        excess = len(buffer_files) - HYBRID_PARAMS["buffer_size"]
        to_remove = sorted(buffer_files, key=os.path.getctime)[:int(excess*0.2)]
        for f in to_remove:
            os.remove(f)

    # Copy samples from main dataset
    for class_name in ["synced", "unsynced"]:
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.exists(class_dir):
            continue
            
        samples = [os.path.join(class_dir, f) for f in os.listdir(class_dir)]
        keep = max(1, int(len(samples) * keep_ratio))
        
        # Prioritize medium-sized files (avoid extremes)
        samples.sort(key=lambda x: os.path.getsize(x))
        start_idx = int(len(samples)*0.2)
        end_idx = int(len(samples)*0.8)
        selected = samples[start_idx:end_idx][:keep]
        
        for src in selected:
            dst = os.path.join(buffer_dir, os.path.basename(src))
            if not os.path.exists(dst):
                shutil.move(src, dst)



def memory_monitor(threshold_percent=80, check_interval=5):
    """Monitor memory usage during training"""
    import psutil
    if psutil.virtual_memory().percent > threshold_percent:
        print(f"Warning: High memory usage detected ({psutil.virtual_memory().percent}%)")
        print("Forcing garbage collection...")
        gc.collect()
        tf.keras.backend.clear_session()
        time.sleep(check_interval)
        return True
    return False
from tensorflow.keras import layers, models

def create_multimodal_model():
    # Visual branch
    visual_input = layers.Input(shape=(None, 224, 224, 3), name='visual_input')
    
    # Use larger filter sizes and more filters for better GPU utilization
    x = layers.TimeDistributed(layers.Conv2D(64, (5,5), activation='relu'))(visual_input)
    x = layers.TimeDistributed(layers.MaxPooling2D((2,2)))(x)
    x = layers.TimeDistributed(layers.Conv2D(128, (3,3), activation='relu'))(x)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
    
    # Replace LSTM with Conv1D but use larger filters
    x = layers.Conv1D(256, 5, activation='relu', padding='same')(x)
    x = layers.GlobalMaxPooling1D()(x)

    # Audio branch
    audio_input = layers.Input(shape=(40, 100), name='audio_input')

    y = layers.Conv1D(64, 3, activation='relu')(audio_input)
    y = layers.GlobalMaxPooling1D()(y)

    # Larger fusion layer
    fused = layers.concatenate([x, y])
    z = layers.Dense(512, activation='relu')(fused)
    output = layers.Dense(1, activation='sigmoid')(z)
    
    return models.Model(inputs=[visual_input, audio_input], outputs=output)

    
# Hybrid loss function
# Ensure this is the ONLY place hybrid_loss is defined
@tf.keras.utils.register_keras_serializable(package='Custom')  # REQUIRED DECORATOR
def hybrid_loss(y_true, y_pred):
    audio_weight = HYBRID_PARAMS["audio_weight"]  # Replace with your actual params
    return (audio_weight * tf.keras.losses.binary_crossentropy(y_true, y_pred) +
            (1 - audio_weight) * tf.keras.losses.binary_crossentropy(y_true, y_pred))

# Later in your training code, save the model WITH the optimizer state
def create_dataset_from_files(file_paths, batch_size=8):

    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset.batch(batch_size)
    
    
import os
import numpy as np
import tensorflow as tf

def get_video_paths_and_labels(dataset_path):
    abs_dataset_path = os.path.abspath(dataset_path)
    all_paths = []
    all_labels = []
    for class_name, label in [("synced", 1), ("unsynced", 0)]:
        class_dir = os.path.join(abs_dataset_path, class_name)
        if not os.path.exists(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.mp4', '.avi', '.mkv', '.webm')):
                all_paths.append(os.path.join(class_dir, fname))
                all_labels.append(label)
    return np.array(all_paths), np.array(all_labels)
def video_generator(video_paths, labels, num_frames=32):
    for path, label in zip(video_paths, labels):
        frames = extract_frames(path, target_frames=num_frames)  # shape: (num_frames, 224, 224, 3)
        audio_features = extract_audio_features(path)            # shape: (40, 100)
        yield (frames, audio_features), label
            
def create_streaming_dataset(video_paths, labels, batch_size=4, num_frames=32):
    dataset = tf.data.Dataset.from_generator(
        lambda: video_generator(video_paths, labels, num_frames),
        output_signature=(
            (tf.TensorSpec(shape=(num_frames, 224, 224, 3), dtype=tf.float32),
             tf.TensorSpec(shape=(40, 100), dtype=tf.float32)),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
    

@memory_safe(threshold_percent=70)
def train_model(dataset_path, model_save_path, existing_model=None):
    import os
    import tensorflow as tf
    import numpy as np
    from deep_learning.losses import hybrid_loss
    from deep_learning.model_trainer import create_multimodal_model
    from deep_learning.model_trainer import extract_audio_features, extract_audio_features
    from deep_learning.model_trainer import extract_audio_features

    # Enable memory optimizations
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    
    # Mixed precision setup
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

    # In model_trainer.py's create_streaming_dataset
    import tensorflow as tf
    import numpy as np

    def create_streaming_dataset(video_paths, labels, batch_size=4, num_frames=32):
        dataset = tf.data.Dataset.from_generator(
            lambda: video_generator(video_paths, labels, num_frames),
            output_signature=(
                tf.TensorSpec(shape=(num_frames, 224, 224, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32)
            )
        )
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset


    # Use this function in your training code:
    dataset_path = "training_data/dataset"  # or your dataset directory
    all_paths, all_labels = get_video_paths_and_labels(dataset_path)

    # Shuffle and split
    indices = np.arange(len(all_paths))
    np.random.shuffle(indices)
    all_paths = all_paths[indices]
    all_labels = all_labels[indices]

    train_size = int(0.8 * len(all_paths))
    train_paths, val_paths = all_paths[:train_size], all_paths[train_size:]
    train_labels, val_labels = all_labels[:train_size], all_labels[train_size:]

    # Create streaming datasets (these will have correct batch dimension)
    from deep_learning.model_trainer import create_streaming_dataset

    batch_size = 4  # or whatever you want
    num_frames = 32 # or whatever your model expects

    train_dataset = create_streaming_dataset(train_paths, train_labels, batch_size=batch_size, num_frames=num_frames)
    val_dataset = create_streaming_dataset(val_paths, val_labels, batch_size=batch_size, num_frames=num_frames)

    # Model setup
    if existing_model and os.path.exists(existing_model):
        model = tf.keras.models.load_model(
            existing_model,
            custom_objects={'hybrid_loss': hybrid_loss},
            compile=False  # MUST SET TO FALSE
        )
        # Recompile with current configuration
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=hybrid_loss,
            metrics=['accuracy'],
            jit_compile=True  # Keep this if using XLA
        )
    else:
        model = create_multimodal_model()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=hybrid_loss,
            metrics=['accuracy'],
            jit_compile=True
        )
    for batch in train_dataset.take(1):
        print("Batch:", batch)
        if isinstance(batch, tuple):
            x, y = batch
            if isinstance(x, (tuple, list)):
                for i, inp in enumerate(x):
                    print(f"x[{i}] shape:", getattr(inp, "shape", None))
            else:
                print("x shape:", getattr(x, "shape", None))
            print("y shape:", getattr(y, "shape", None))
        else:
            print("Batch shape:", getattr(batch, "shape", None))
    # Training loop with memory cleanup
    for epoch in range(10):
        tf.keras.backend.clear_session()
        gc.collect()
        model.fit(
            train_dataset.map(lambda x, y: ((x[0], x[1]), y)),
            validation_data=val_dataset.map(lambda x, y: ((x[0], x[1]), y)),
            epochs=10
        )
        model.save('deep_learning/models/trained_model.h5')
        print("Model saved to trained_model.h5")


    model.save(model_save_path)
    
    # Final cleanup
    tf.keras.backend.clear_session()
    gc.collect()


def train_hybrid(dataset_path, buffer_path, model_save_path, existing_model=None):
    from deep_learning.losses import hybrid_loss
    # Load previous model if exists
    if existing_model and tf.io.gfile.exists(existing_model):
        from deep_learning.losses import hybrid_loss
        model = tf.keras.models.load_model(
            existing_model,
            custom_objects={'hybrid_loss': hybrid_loss},
            compile=False  # CRITICAL CHANGE
        )
        # Freeze 75% of layers
        for layer in model.layers[:int(len(model.layers)*HYBRID_PARAMS["freeze_layers"])]:
            layer.trainable = False
        model.compile(
            optimizer='adam',
            loss=hybrid_loss,
            metrics=['accuracy']
        )
    else:
        model = create_hybrid_model()
    
    # Load new data
    new_data = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        label_mode='binary',
        image_size=(224,224),
        batch_size=32
    )
    
    # Load buffer data
    buffer_data = tf.keras.utils.image_dataset_from_directory(
        buffer_path,
        label_mode='binary',
        image_size=(224,224),
        batch_size=32
    )
    
    # Combine datasets with 70-30 ratio
    final_dataset = new_data.concatenate(buffer_data).shuffle(1000)
    
    
    model.compile(optimizer='adam', loss=hybrid_loss, metrics=['accuracy'])
    model.fit(final_dataset, epochs=10)
    model.save(model_save_path)
    policy = tf.keras.mixed_precision.Policy('mixed_bfloat16')
    tf.keras.mixed_precision.set_global_policy(policy)
    import tensorflow.keras.backend as K
    K.clear_session()
    if tf.config.list_physical_devices('GPU'):
        for device in tf.config.experimental.get_visible_devices('GPU'):
            tf.config.experimental.set_memory_growth(device, True)

# Usage example before training:
dataset_path = "training_data/dataset"  # Your dataset root folder
all_paths, all_labels = get_video_paths_and_labels(dataset_path)

# Shuffle and split dataset
indices = np.arange(len(all_paths))
np.random.shuffle(indices)
all_paths = all_paths[indices]
all_labels = all_labels[indices]

train_size = int(0.8 * len(all_paths))
train_paths, val_paths = all_paths[:train_size], all_paths[train_size:]
train_labels, val_labels = all_labels[:train_size], all_labels[train_size:]

batch_size = 4
num_frames = 32

train_dataset = create_streaming_dataset(train_paths, train_labels, batch_size=batch_size, num_frames=num_frames)
val_dataset = create_streaming_dataset(val_paths, val_labels, batch_size=batch_size, num_frames=num_frames)
# Then train your model with these batched datasets:


