import cv2
import subprocess
import concurrent.futures
import multiprocessing
import os
os.environ["DML_VISIBLE_DEVICES"] = "0"  # Use first AMD GPU
os.environ["DML_GRAPHICS_COMPUTE_ONLY"] = "1"  # Dedicate to compute
os.environ["DML_GPU_FENCE_POLICY"] = "1"      # Enable async memory management
os.environ["OPENCV_OPENCL_DEVICE"] = "AMD:GPU"  # Force OpenCL AMD selection
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Enable oneDNN optimizations [2]
os.environ["OPENCV_OPENCL_DEVICE"] = "AMD:GPU"  # Explicit AMD selection
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow log spam
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'  # Optimize GPU threading for DirectML
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '0'  # Add to data_collector.py
import tensorflow as tf
import numpy as np

from deep_learning.model_trainer import create_multimodal_model, hybrid_loss
from deep_learning.model_trainer import create_multimodal_model
import tensorflow as tf
tf.config.optimizer.set_jit(False) 


gpus = tf.config.list_physical_devices('GPU')
import gc
import shutil
import sys
import os
import logging
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import tensorflow.keras.optimizers as optimizers

# Instead of simply using 'adam'
optimizer = optimizers.Adam(learning_rate=0.001)
optimizer.jit_compile = False  # Explicitly disable JIT
# Function that returns a model
model = create_multimodal_model() 
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=hybrid_loss,  # Your custom loss function
    metrics=['accuracy']
)

from deep_learning.model_trainer import create_streaming_dataset
import logging
import time
import psutil
import tensorflow as tf
from tensorflow.keras.callbacks import Callback


import cv2
import numpy as np
import tensorflow as tf

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
    frames = np.array(frames, dtype=np.float32) / 255.0  # (target_frames, 224, 224, 3)
    return frames

def load_and_preprocess_video(video_path, label, num_frames=32, output_size=(224, 224)):
    frames = tf.py_function(
        func=extract_frames,
        inp=[video_path, num_frames],
        Tout=tf.float32
    )
    frames.set_shape([num_frames, output_size[0], output_size[1], 3])
    return frames, label

def create_video_dataset(video_paths, labels, batch_size=8, num_frames=32, output_size=(224, 224)):
    dataset = tf.data.Dataset.from_tensor_slices((video_paths, labels))
    dataset = dataset.map(
        lambda path, label: load_and_preprocess_video(path, label, num_frames, output_size),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# === Example usage ===
if __name__ == "__main__":
    # Flexible: fill these from your GUI or any file dialog
    video_paths = [
        r"D:\Videos\movie1.mp4",
        r"E:\Clips\clip2.avi",
        r"F:\Anime\ep03.mkv"
    ]
    labels = [1, 0, 1]  # 1=synced, 0=unsynced

    batch_size = 4
    num_frames = 32

    train_dataset = create_video_dataset(
        video_paths,
        labels,
        batch_size=batch_size,
        num_frames=num_frames
    )

    # Example: test batch shapes
    for batch_frames, batch_labels in train_dataset.take(1):
        print("Batch frames shape:", batch_frames.shape)
        print("Batch labels:", batch_labels)
    model.fit(train_dataset, epochs=10)

class EnhancedLogging:
    def __init__(self, log_level=logging.INFO):
        self.logger = logging.getLogger('DLVideoLogger')
        self.logger.setLevel(log_level)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(ch)
        
        # File handler
        fh = logging.FileHandler('training.log')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(fh)
        
        # Initialize resource monitoring
        self.start_memory_monitor()

    def start_memory_monitor(self):
        """Log system resources every 5 seconds"""
        def monitor():
            while True:
                mem = psutil.virtual_memory()
                self.logger.info(
                    f"Memory Usage: {mem.percent}% | "
                    f"Available: {mem.available/(1024**3):.1f}GB | "
                    f"CPU: {psutil.cpu_percent()}%"
                )
                time.sleep(5)
        
        threading.Thread(target=monitor, daemon=True).start()

class VideoProgressLogger(Callback):
    def __init__(self, total_frames, log_interval=10):
        super().__init__()
        self.total_frames = total_frames
        self.log_interval = log_interval
        self.start_time = None
        
    def on_predict_batch_begin(self, batch, logs=None):
        if batch % self.log_interval == 0:
            fps = self.log_interval / (time.time() - self.last_log_time)
            logging.info(
                f"Processing batch {batch}/{self.total_frames} | "
                f"Estimated FPS: {fps:.1f} | "
                f"ETA: {(self.total_frames-batch)/fps/60:.1f} mins"
            )
            self.last_log_time = time.time()

    def on_predict_begin(self, logs=None):
        self.start_time = time.time()
        self.last_log_time = time.time()
        logging.info("Starting video processing...")

    def on_predict_end(self, logs=None):
        total_time = time.time() - self.start_time
        logging.info(
            f"Processing complete! Total time: {total_time/60:.1f} mins | "
            f"Average FPS: {self.total_frames/total_time:.1f}"
        )

# Usage in your video processing code:
def process_video(video_path):
    logger = EnhancedLogging()
    try:
        from deep_learning.losses import hybrid_loss
        model = tf.keras.models.load_model(
            'deep_learning/models/trained_model.h5', 
            compile=False
        )
        # Manual compilation required
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=hybrid_loss,
            metrics=['accuracy'],
            run_eagerly=False  # Better for GPU performance
        )
        logger.logger.info("Loaded pre-trained model successfully")
        
        # Create video processing pipeline
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize progress logger
        progress_logger = VideoProgressLogger(total_frames)
        
        # Create prediction pipeline with logging
        logger.logger.info(f"Starting processing on {video_path}")
        logger.logger.info(f"Total frames: {total_frames}")
        
        # Batch processing with logging
        batch_size = 128
        for batch_start in range(0, total_frames, batch_size):
            batch_frames = []
            for _ in range(batch_size):
                ret, frame = cap.read()
                if not ret: break
                batch_frames.append(preprocess_frame(frame))
            
            # Process batch with timing
            start_time = time.time()
            predictions = model.predict(np.array(batch_frames))
            batch_time = time.time() - start_time
            
            logger.logger.debug(
                f"Processed batch {batch_start}-{batch_start+len(batch_frames)} | "
                f"Batch time: {batch_time:.2f}s | "
                f"FPS: {len(batch_frames)/batch_time:.1f}"
            )
            
    except Exception as e:
        logger.logger.error(f"Processing failed: {str(e)}", exc_info=True)
        raise
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        tf.keras.backend.clear_session()
        gc.collect()

# Add this to your GUI initialization
class TrainingGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.log_area = ScrolledText(self, state='disabled')
        self.log_area.pack(fill=tk.BOTH, expand=True)
        
        # Redirect logs to GUI
        self.log_handler = TextHandler(self.log_area)
        logging.getLogger().addHandler(self.log_handler)

class TextHandler(logging.Handler):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
        
    def emit(self, record):
        msg = self.format(record)
        self.text_widget.configure(state='normal')
        self.text_widget.insert(tk.END, msg + '\n')
        self.text_widget.configure(state='disabled')
        self.text_widget.yview(tk.END)


class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))

class ConsoleUi:
    def __init__(self, parent):
        self.parent = parent
        self.log_queue = queue.Queue()
        self.queue_handler = QueueHandler(self.log_queue)
        self.queue_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.logger = logging.getLogger()
        self.logger.addHandler(self.queue_handler)
        self.logger.setLevel(logging.INFO)
        self.log_area = ScrolledText(parent, state='disabled', height=5)
        self.log_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        parent.after(100, self.poll_log_queue)

    def poll_log_queue(self):
        while True:
            try:
                msg = self.log_queue.get(block=False)
            except queue.Empty:
                break
            self.log_area.configure(state='normal')
            self.log_area.insert(tk.END, msg + '\n')
            self.log_area.configure(state='disabled')
            self.log_area.yview(tk.END)
        self.parent.after(100, self.poll_log_queue)


class DataCollector(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.status_queue = queue.Queue()
        self.title("Deep Learning Training Interface")
        self.geometry("1000x562")
        

        # Video list setup
        self.video_list = ttk.Treeview(self, columns=("Path", "Status"), show="headings", height=4)
        self.video_list.heading("Path", text="Video Path")
        self.video_list.heading("Status", text="Status")
        self.video_list.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.video_data = {}
        
        # Scrollbar for video_list
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.video_list.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.video_list.configure(yscrollcommand=scrollbar.set)

        # Button frame
        self.btn_frame = tk.Frame(self)
        self.btn_frame.pack(fill=tk.X, padx=10, pady=10)

        self.add_btn = tk.Button(self.btn_frame, text="Add Videos", command=self.add_videos)
        self.add_btn.pack(side=tk.LEFT, padx=5)

        self.complete_btn = tk.Button(self.btn_frame, text="Mark as Synced", command=lambda: self.set_status("Complete"))
        self.complete_btn.pack(side=tk.LEFT, padx=5)

        self.failed_btn = tk.Button(self.btn_frame, text="Mark as Failed", command=lambda: self.set_status("Failed"))
        self.failed_btn.pack(side=tk.LEFT, padx=5)
        self.preprocess_btn = tk.Button(self.btn_frame, text="Preprocess Data", command=self.preprocess_data)
        self.preprocess_btn.pack(side=tk.RIGHT, padx=5)
        self.train_btn = tk.Button(self.btn_frame, text="Train Model", command=self.start_training)
        self.train_btn.pack(side=tk.RIGHT, padx=5)

        self.pause_btn = tk.Button(self.btn_frame, text="Pause", command=self.toggle_pause)
        self.pause_btn.pack(side=tk.RIGHT, padx=5)
        import multiprocessing
        self.pause_event = multiprocessing.Event()

        self.progress = ttk.Progressbar(self.btn_frame, mode='determinate')
        self.progress.pack(side=tk.RIGHT, padx=5)

        self.log_frame = tk.LabelFrame(self, text="Training Log")
        self.log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.enable_ctrl_a_on_treeview()

        self.console = ConsoleUi(self)
        self.status_label = tk.Label(self.btn_frame, text="Status: Ready")
        self.status_label.pack(side=tk.LEFT, padx=5)

    def preprocess_data(self):
        import threading
        from deep_learning.model_trainer import train_model  # Remove preprocess_and_save
        import os
        import shutil

        # Gather all videos marked as Complete or Failed
        training_data = []
        for item_id, info in self.video_data.items():
            status = info.get("status", "")
            if status in ["Complete", "Failed"]:
                label = 1 if status == "Complete" else 0
                training_data.append((info["path"], label))

        if not training_data:
            self.console.logger.info("No videos marked as Complete or Failed for training.")
            return

        dataset_path = os.path.join("training_data", "dataset")
        os.makedirs(dataset_path, exist_ok=True)


    def add_videos(self):
        files = filedialog.askopenfilenames(filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.webm")])
        for file in files:
            item = self.video_list.insert("", "end", values=(file, "Pending"))
            self.video_data[item] = {"path": file, "status": "Pending"}

    def set_status(self, status):
        selected = self.video_list.selection()
        for item in selected:
            self.video_list.item(item, values=(self.video_data[item]["path"], status))
            self.video_data[item]["status"] = status

    def toggle_pause(self):
        if self.pause_event.is_set():
            self.pause_event.clear()
            self.pause_btn.config(text="Pause")
            logging.info("Training resumed")
            # Force memory cleanup
            tf.keras.backend.clear_session()
            gc.collect()
        else:
            self.pause_event.set()
            self.pause_btn.config(text="Resume")
            logging.info("Training paused")

    def enable_ctrl_a_on_treeview(self):
        def select_all(event):
            try:
                self.video_list.selection_set(self.video_list.get_children())
            except Exception as e:
                print(f"Selection error: {e}")
            return "break"
        # Fix event bindings
        self.video_list.bind("<Control-a>", select_all)  # Windows/Linux
        self.video_list.bind("<Command-a>", select_all)  # macOS


    def start_training(self):
        training_data = []
        for item in self.video_data.values():
            if item["status"] in ["Complete", "Failed"]:
                training_data.append((item["path"], 1 if item["status"] == "Complete" else 0))
        threading.Thread(target=self.run_training, args=(training_data,)).start()

    def run_training(self, training_data):
        """Main training loop with direct video streaming (no copying, no dataset_path dependency)"""
        import gc
        import tensorflow as tf
        from deep_learning.model_trainer import create_streaming_dataset, create_multimodal_model, hybrid_loss

        self.status_label.config(text="Initializing training...")

        try:
            # Prepare lists of paths and labels
            video_paths = [item[0] for item in training_data]
            labels = [item[1] for item in training_data]

            # Check for empty list
            if not video_paths:
                self.console.logger.error("No videos selected for training!")
                self.after(0, messagebox.showerror, "Error", "No videos selected for training!")
                return

            self.console.logger.info(f"Starting training on {len(video_paths)} videos")

            batch_size = 8  # number of videos to be processed at once
            num_frames = 32

            # Create dataset directly from video paths and labels
            train_dataset = create_streaming_dataset(video_paths, labels, batch_size=batch_size, num_frames=num_frames)

            # Debug: print first batch shape to catch empty dataset or shape mismatch
            for batch in train_dataset.take(1):
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

            # Build and compile the model
            model = create_multimodal_model()
            model.compile(
                optimizer=tf.keras.optimizers.Adam(),
                loss=hybrid_loss,
                metrics=['accuracy'],
                run_eagerly=True  # Enable eager for better error messages during debugging
            )

            # Train the model
            model.fit(train_dataset, epochs=10)
            model.save("deep_learning/models/trained_model.h5")

            self.status_label.config(text="Training complete!")
            self.progress['value'] = 100

        except Exception as e:
            self.console.logger.error(f"Training failed: {str(e)}")
            self.after(0, messagebox.showerror, "Error", f"Training failed: {str(e)}")
        finally:
            # Memory cleanup
            tf.keras.backend.clear_session()
            gc.collect()



    def get_gpu_usage(self):
        """Get current GPU memory usage percentage"""
        try:
            stats = tf.config.experimental.get_memory_info('GPU:0')
            return (stats['current'] / stats['limit']) * 100
        except Exception as e:
            self.console.logger.error(f"GPU usage error: {str(e)}")
            return 0
                
def retain_buffer_samples(self, dataset_path, keep_ratio=0.1):
    buffer_dir = os.path.join(dataset_path, "buffer")
    try:
        # Add buffer size limit check and cleanup
        if os.path.exists(buffer_dir):
            buffer_files = [os.path.join(root, f) for root, _, files in os.walk(buffer_dir) for f in files]
            if len(buffer_files) > HYBRID_PARAMS["buffer_size"]:  # [1]
                # Remove oldest 20% of excess files
                excess = len(buffer_files) - HYBRID_PARAMS["buffer_size"]
                to_remove = sorted(buffer_files, key=os.path.getctime)[:int(excess*0.2)]
                for f in to_remove:
                    os.remove(f)
                    self.console.logger.info(f"Removed old buffer sample: {os.path.basename(f)}")

        # Copy with size-aware selection
        for class_name in ["synced", "unsynced"]:
            class_dir = os.path.join(dataset_path, class_name)
            if os.path.exists(class_dir):
                videos = sorted(os.listdir(class_dir), key=lambda x: os.path.getsize(os.path.join(class_dir, x)))
                keep = max(1, int(len(videos) * keep_ratio))
                
                # Select medium-sized files (avoid extremes)
                start_idx = int(len(videos)*0.2)
                end_idx = int(len(videos)*0.8)
                selected = videos[start_idx:end_idx][:keep]
                
                for video in selected:
                    src = os.path.join(class_dir, video)
                    dst = os.path.join(buffer_dir, class_name, video)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    if not os.path.exists(dst):
                        shutil.move(src, dst)  # Move instead of copy to save space
                        self.console.logger.info(f"Buffered sample: {video}")
                    
        # Explicit memory cleanup
        del buffer_files, videos, selected
        gc.collect()
        
    except Exception as e:
        self.console.logger.error(f"Buffer update error: {str(e)}")
        self.after(0, messagebox.showerror, "Error", f"Buffer update failed: {str(e)}")



def update_buffer(dataset_path, buffer_size=5000):
    buffer_dir = os.path.join(dataset_path, "buffer")
    os.makedirs(buffer_dir, exist_ok=True)
    
    # Get all samples and select most informative
    all_samples = []
    for class_name in ["synced", "unsynced"]:
        class_path = os.path.join(dataset_path, class_name)
        samples = [os.path.join(class_path, f) for f in os.listdir(class_path)]
        all_samples.extend(random.sample(samples, min(len(samples), buffer_size//2)))
    
    # Update buffer
    for sample in all_samples:
        shutil.copy(sample, os.path.join(buffer_dir, os.path.basename(sample)))
def video_generator(video_paths, batch_size=4):
    for path in video_paths:
        cap = cv2.VideoCapture(path)
        while cap.isOpened():
            frames = []
            for _ in range(batch_size):
                ret, frame = cap.read()
                if not ret: break
                frames.append(cv2.resize(frame[-100:], (224,224))/255.0)  # Bottom crop
            if len(frames) == batch_size:
                yield (np.array(frames, dtype=np.float16), 
                       extract_audio_features(path))
        cap.release()
        
        
def extract_audio_features(video_path, stream=False, chunk_size=10):
    import subprocess
    import tempfile
    import os
    import librosa
    
    # Create temporary WAV file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
        temp_name = temp_audio.name
        
    # Extract audio using ffmpeg
    subprocess.call([
        'ffmpeg', '-y', '-i', video_path,
        '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
        temp_name
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Process with librosa
    try:
        audio, sr = librosa.load(temp_name, sr=16000)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        
        # Ensure shape (40, 100) for model compatibility
        if mfccs.shape[1] < 100:
            mfccs = np.pad(mfccs, ((0, 0), (0, 100 - mfccs.shape[1])), mode='constant')
        elif mfccs.shape[1] > 100:
            mfccs = mfccs[:, :100]
            
        return mfccs
    finally:
        os.unlink(temp_name)
# Add this to data_collector.py
def extract_audio_features(video_path, stream=False, chunk_size=10):
    import subprocess
    import tempfile
    import os
    import librosa
    
    # Create temporary WAV file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
        temp_name = temp_audio.name
        
    # Extract audio using ffmpeg
    subprocess.call([
        'ffmpeg', '-y', '-i', video_path,
        '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
        temp_name
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Process with librosa
    try:
        audio, sr = librosa.load(temp_name, sr=16000)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        
        # Ensure shape (40, 100) for model compatibility
        if mfccs.shape[1] < 100:
            mfccs = np.pad(mfccs, ((0, 0), (0, 100 - mfccs.shape[1])), mode='constant')
        elif mfccs.shape[1] > 100:
            mfccs = mfccs[:, :100]
            
        return mfccs
    finally:
        os.unlink(temp_name)