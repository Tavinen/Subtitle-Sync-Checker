import os
import tensorflow as tf

# Set environment variables (if needed)
os.environ["DML_VISIBLE_DEVICES"] = "0"
os.environ["DML_GRAPHICS_COMPUTE_ONLY"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

# Configure GPU BEFORE importing anything else that uses TensorFlow
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_visible_devices(gpus[0], 'GPU')
    except RuntimeError as e:
        print(f"GPU config error: {e}")

# Now import the rest of your code
from deep_learning.sync_predictor import HybridPredictor
from data_collector import DataCollector
from utils.memory_monitor import MemoryMonitor
import sys
import os
import time
import threading
import multiprocessing
import cv2
import numpy as np
import datetime
import traceback
import psutil
import queue
import subprocess
import re
import tensorflow as tf
print("Visible devices:", tf.config.get_visible_devices())

import platform
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tkinter as tk
import psutil
import time

def wait_for_resources(cpu_limit=80, ram_limit=80):
    while True:
        cpu = psutil.cpu_percent(interval=1)
        ram = psutil.virtual_memory().percent
        if cpu < cpu_limit and ram < ram_limit:
            break
        print(f"High resource usage (CPU: {cpu}%, RAM: {ram}%), waiting...")
        time.sleep(2)
from tkinter import ttk, filedialog, messagebox, scrolledtext


import sys
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"  # For gfx1201 (RDNA3)
os.environ["MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP16_ALT_IMPL"] = "1"

class SubtitleSyncChecker:
    def __init__(self):
        self.worker_to_item = {}

        self.worker_to_item = {}
        self.configure_hardware()
        self.memory_monitor = MemoryMonitor(threshold=70)
        self.memory_monitor.start()
        # Add this after GPU initialization
        self.configure_gpu_performance()
        import os
        import tkinter as tk
        from tkinter import ttk, scrolledtext
        import multiprocessing
        # Add after GPU initialization
        os.environ["OPENCV_OPENCL_DEVICE"] = "AMD:GPU"  # Force AMD selection [4]


        # Environment variables to improve OpenCL performance
        os.environ["OPENCV_OPENCL_CACHE_ENABLE"] = "1"
        os.environ["OPENCV_OPENCL_CACHE_WRITE"] = "1"
        

        # Initialize flags and queues
        self.gpu_available = False
        self.stop_requested = False
        self.is_processing = False
        self.is_paused = False
        self.pause_event = multiprocessing.Event()
        self.task_queue = multiprocessing.Queue()
        self.task_queue = multiprocessing.Queue()
        self.task_queue = multiprocessing.Queue()
        self.worker_processes = []

        self.status_dict = {}


        # GUI initialization
        self.root = tk.Tk()
        self.root.title("Subtitle Sync Checker")
        self.root.geometry("1260x720")  # Extended width for larger file list

        # Store file paths with their respective tree IDs
        self.file_paths = {}

        # Main layout frames
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel (file list and video info)
        self.left_panel = tk.Frame(self.main_frame)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right panel (buttons and processing options)
        

        
        self.right_panel = tk.Frame(self.main_frame)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0), pady=10)

        self.buttons_frame = tk.Frame(self.right_panel)
        self.buttons_frame.pack(pady=20)  # Adjust padding as needed

        # File list (table-like structure)
        self.file_list_frame = tk.Frame(self.left_panel)
        self.file_list_frame.pack(fill=tk.BOTH, expand=True)
        style = ttk.Style()
        style.configure("Treeview", rowheight=40)  # Doubled the default height

        columns = ("Name", "Duration", "Format", "Status")
        self.file_list = ttk.Treeview(
            self.file_list_frame,
            columns=columns,
            show="headings",
            height=7,
            selectmode="extended"
        )  # <-- Added this line
        self.enable_ctrl_a_on_treeview(self.file_list)

        self.file_list.tag_configure("pending", background="white")
        self.file_list.tag_configure("checking", background="light blue", foreground="black")
        self.file_list.tag_configure("completed", background="green", foreground="white")
        self.file_list.tag_configure("failed", background="red", foreground="white")

        for col in columns:
            self.file_list.heading(col, text=col)
            self.file_list.column(col, width=300 if col == "Name" else 100)  # Extended width for Name column

        scrollbar = ttk.Scrollbar(self.file_list_frame, orient="vertical", command=self.file_list.yview)
        self.file_list.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.file_list.bind("<Double-1>", self.open_video_directory)
        self.file_list.bind('<<TreeviewSelect>>', self.show_file_info)

        # --- Improved File Information Display ---
        from tkinter import scrolledtext  # Make sure this is at the top of your file

        # --- Improved File Information Display ---
        self.info_frame = tk.LabelFrame(self.left_panel, text="File Information")
        self.info_frame.pack(fill=tk.X, pady=(10, 0), padx=10)

        self.info_text = scrolledtext.ScrolledText(
            self.info_frame, 
            height=6,
            wrap=tk.WORD,
            font=("Consolas", 10),
            state=tk.DISABLED
        )
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Log panel for detailed processing information (moved to bottom)
        self.log_frame = tk.LabelFrame(self.left_panel, text="Processing Log")
        self.log_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        self.log_text = scrolledtext.ScrolledText(self.log_frame, height=8, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_text.config(state=tk.DISABLED)

        # Performance chart frame - will only be visible when benchmark is checked
        self.perf_frame = tk.LabelFrame(self.left_panel, text="Performance Comparison")
        # We'll initialize the matplotlib components only when needed

        # Processing mode selection
        self.mode_frame = tk.LabelFrame(self.right_panel, text="Processing Mode")
        self.mode_frame.pack(fill=tk.X, pady=10, padx=5)

        self.processing_mode = tk.StringVar(value="gpu_optimized")
        tk.Radiobutton(self.mode_frame, text="Auto", variable=self.processing_mode, 
                      value="auto").pack(anchor="w")
        tk.Radiobutton(self.mode_frame, text="Force GPU (Standard)", variable=self.processing_mode, 
                      value="gpu").pack(anchor="w")
        tk.Radiobutton(self.mode_frame, text="Force GPU (Optimized)", variable=self.processing_mode, 
                      value="gpu_optimized").pack(anchor="w")
        tk.Radiobutton(self.mode_frame, text="Force CPU", variable=self.processing_mode, 
                      value="cpu").pack(anchor="w")

        # Multi-Video Processing Settings
        self.multi_frame = tk.LabelFrame(self.right_panel, text="Multi-Video Processing")
        self.multi_frame.pack(fill=tk.X, pady=10, padx=5)

        # Enable Multi-Video toggle
        self.enable_multi_var = tk.BooleanVar(value=True)
        enable_check = tk.Checkbutton(self.multi_frame, text="Enable Multi-Video Processing", 
                                     variable=self.enable_multi_var)
        enable_check.pack(anchor="w", pady=(5,0))

        # Number of concurrent videos
        parallel_frame = tk.Frame(self.multi_frame)
        parallel_frame.pack(fill=tk.X, pady=(5,0))

        tk.Label(parallel_frame, text="Concurrent Videos:").pack(side=tk.LEFT)
        
        parallel_frame = tk.Frame(self.multi_frame)
        parallel_frame.pack(fill=tk.X, pady=(5,0))


        # Calculate based on available RAM
        import psutil
        available_ram_gb = psutil.virtual_memory().total / (1024**3)
        max_concurrent = max(1, int(available_ram_gb // 2.5))  # 2.5GB per video

        self.concurrent_var = tk.IntVar(value=min(2, max_concurrent))  # Start with 2

        # Create spinbox
        concurrent_spinner = tk.Spinbox(
            parallel_frame, 
            from_=1, 
            to=max_concurrent,
            textvariable=self.concurrent_var, 
            width=5
        )
        concurrent_spinner.pack(side=tk.LEFT, padx=5)
      

        concurrent_spinner = tk.Spinbox(parallel_frame, from_=1, to=max_concurrent,
            textvariable=self.concurrent_var, width=5)

        # Benchmark mode - process all videos with CPU, then with GPU
        benchmark_frame = tk.Frame(self.multi_frame)
        benchmark_frame.pack(fill=tk.X, pady=5)

        self.benchmark_var = tk.BooleanVar(value=False)
        self.benchmark_check = tk.Checkbutton(benchmark_frame, text="Run CPU/GPU Benchmark", 
                                             variable=self.benchmark_var, command=self.toggle_performance_frame)
        self.benchmark_check.pack(anchor="w")

        # Buttons panel
        self.buttons_frame = tk.Frame(self.right_panel)
        self.buttons_frame.pack(pady=20)
        
        self.add_button = tk.Button(self.buttons_frame, text="Add Files", width=15, command=self.add_files)
        self.add_button.pack(pady=5)
        
        self.dl_button = tk.Button(self.buttons_frame, text="Deep Learning", 
                                  width=15, command=self.open_dl_interface)
        self.dl_button.pack(pady=5)

        self.check_button = tk.Button(self.buttons_frame, text="Start Check", width=15, command=self.start_check)
        self.check_button.pack(pady=5)

        self.stop_button = tk.Button(self.buttons_frame, text="Stop Check", width=15, 
                                   command=self.stop_check, state=tk.DISABLED)
        self.stop_button.pack(pady=5)

        self.remove_button = tk.Button(self.buttons_frame, text="Remove Selected", width=15, 
                                     command=self.remove_selected)
        self.remove_button.pack(pady=5)
        
        self.monitor_btn = tk.Button(self.buttons_frame, text="GPU Monitor", 
                             width=15, command=self.open_gpu_monitor)
        self.monitor_btn.pack(pady=5)

        # Progress windows for individual videos
        self.progress_windows = {}

        # Summary frame (appears after completion)
        self.summary_frame = None

        # Performance data for benchmarking
        self.performance_data = {
            "CPU": {"videos": [], "times": []},
            "GPU": {"videos": [], "times": []}
        }

        # Initialize GPU after UI components are created
        self.initialize_gpu()


        # Start the resource monitor
        self.start_resource_monitor()
        
        has_ffmpeg = self.check_ffmpeg_available()
        if not has_ffmpeg:
            self.log("WARNING: ffmpeg not found. Advanced video formats won't be processed.")
        
        # Placeholder for worker processes
        self.worker_processes = []

        # Placeholder for worker processes
        self.worker_processes = []
        self.task_queue = multiprocessing.Queue()
        self.task_queue = multiprocessing.Queue()
        
        self.status_dict = {}
        self.video_start_times = {}    # item_id: start_time
        self.video_progress = {}       # item_id: (current_progress, total)
        self.root.after(100, self.poll_status_queue)  # Start polling
        
        self.task_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        self.status_queue = multiprocessing.Queue()
        
        self.predictor = None  # or skip loading the model for now
        

    import os
    model_path = os.path.abspath("deep_learning/models/trained_model.h5")
    if os.path.exists(model_path):
        predictor = HybridPredictor(model_path)
    else:
        print(f"Model file not found at {model_path}. You need to train the model first.")  
    def poll_status_queue(self):
        """Periodically check the status queue"""
        if not hasattr(self, 'status_queue') or self.status_queue is None:  # Safer check
           return
        """Periodically check the status queue"""
        if self.status_queue is None:  # Safety check
            return
        
        try:
            while True:
                msg = self.status_queue.get_nowait()
                if msg[0] == "status_update":
                    _, item_id, status = msg
                    self._safe_update_status(item_id, status)
                elif msg[0] == "map_worker":
                    _, worker_id, item_id = msg
                    self.worker_to_item[worker_id] = item_id
                elif isinstance(msg[0], int):
                    worker_id, item_id, filename, progress, status_msg = msg
                    self.root.after(0, self.update_progress_content, worker_id, filename, progress, status_msg)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.poll_status_queue)

    def update_status(self, item_id, status):
                values = list(self.file_list.item(item_id)["values"])
                values[3] = status
                self.root.after(0, self._safe_update_status, item_id, status)
                self.file_list.item(item_id, values=values)
                # Apply the appropriate tag for color coding
                if self.root.winfo_exists():  # Check if window still exists
                    self.root.after(0, self._safe_update_status, item_id, status)
                if status == "Pending":
                    self.file_list.item(item_id, tags=("pending",))
                elif status == "Checking":
                    self.file_list.item(item_id, tags=("checking",))
                elif status == "Completed":
                    self.file_list.item(item_id, tags=("completed",))
                elif status in ["Failed", "Error", "Stopped"]:
                    self.file_list.item(item_id, tags=("failed",))
                self.root.update_idletasks()
                # Keep status_dict updated
                self.status_dict[item_id] = status

        
    def _safe_update_status(self, item_id, status):
        """Actual UI update method (main thread only)"""
        if item_id not in self.file_list.get_children():
            return

        values = list(self.file_list.item(item_id)["values"])
        values[3] = status
        self.file_list.item(item_id, values=values)

        # Update tags for color coding
        tags = ()
        if status == "Checking":
            tags = ("checking",)
        elif status == "Completed":
            tags = ("completed",)
        elif status in ["Failed", "Error", "Stopped"]:
            tags = ("failed",)
        self.file_list.item(item_id, tags=tags)
        self.status_dict[item_id] = status
        self.root.update_idletasks()

        
    @staticmethod
    def video_worker(worker_id, task_queue, result_queue, status_queue, use_gpu, optimized_mode, pause_event):
        """Worker process with x265/HEVC handling and AMD GPU optimizations"""
        import subprocess
        import tempfile
        import os
        import cv2
        import time
        import queue
        import gc

        # AMD-specific OpenCL configuration
        cv2.ocl.setUseOpenCL(True)
        
        # Safely configure logging with cross-version compatibility
        try:
            # Try OpenCV 4.x+ approach
            import cv2.utils.logging
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        except (ImportError, AttributeError):
            try:
                # Try direct numeric approach (2 = ERROR level in OpenCV)
                cv2.setLogLevel(2)
            except (AttributeError, TypeError):
                try:
                    # Try older style constant
                    cv2.setLogLevel(cv2.CV_LOG_LEVEL_ERROR)
                except (AttributeError, TypeError):
                    # Fallback: just suppress console output
                    print(f"Worker {worker_id}: OpenCV logging configuration skipped")
        
        try:
            device = cv2.ocl.Device_getDefault()
            print(f"Worker {worker_id} using OpenCL device: {device.name()}")
        except Exception as e:
            print(f"Worker {worker_id} OpenCL error: {str(e)}")
        
        while True:
            try:
                # Check for pause before getting new task
                while pause_event.is_set():
                    time.sleep(0.5)
                    gc.collect()
                    cv2.ocl.finish()  # Release AMD GPU resources

                # Get a video to process with timeout
                idx, item_id, file_path, filename = task_queue.get(timeout=1.0)
                status_queue.put(("map_worker", worker_id, item_id))
            except queue.Empty:
                break  # Exit when queue is empty

            process_path = file_path
            temp_path = None
            cap = None

            try:
                # --- Video Conversion Phase ---
                needs_conversion = any(s in file_path for s in {'x265', 'HEVC', '10Bit', 'FLAC'})
                
                if needs_conversion:
                    status_queue.put(("status_update", item_id, "Converting"))
                    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
                        temp_path = temp_video.name
                    
                    
                    try:
                        # Try AMD GPU accelerated encoding
                        subprocess.run([
                            'ffmpeg', '-y', '-hwaccel', 'auto', '-i', file_path,
                            '-c:v', 'h264_amf', '-pix_fmt', 'yuv420p', '-c:a', 'aac',
                            '-q:v', '23', '-q:a', '3',
                            temp_path
                        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                    except subprocess.CalledProcessError:
                        # Fallback to CPU encoding if h264_amf fails
                        subprocess.run([
                            'ffmpeg', '-y', '-hwaccel', 'auto', '-i', file_path,
                            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac',
                            '-q:v', '23', '-q:a', '3',
                            temp_path
                        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                    
                    process_path = temp_path

                # --- Video Processing Phase ---
                status_queue.put((worker_id, item_id, filename, 0, 
                            f"Processing: {filename}"))

                cap = cv2.VideoCapture(process_path)
                
                if not cap.isOpened():
                    raise RuntimeError(f"Failed to open video: {filename}")

                # Get video properties
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                subtitle_frames = 0
                speech_frames = 0
                matched_frames = 0

                # Memory-optimized batch processing
                batch_size = 64 if use_gpu else 8  # Smaller batches for GPU stability
                total_samples = min(frame_count, 500)
                frame_step = max(1, frame_count // total_samples)

                for batch_start in range(0, frame_count, frame_step * batch_size):
                    # Pause handling with AMD GPU cleanup
                    while pause_event.is_set():
                        print(f"Worker {worker_id} paused")
                        time.sleep(0.5)
                        gc.collect()
                        if hasattr(cv2, "ocl"):
                            cv2.ocl.finish()


                    # Batch processing with UMat for AMD GPU
                    batch_frames = []
                    for i in range(batch_size):
                        frame_idx = batch_start + (i * frame_step)
                        if frame_idx >= frame_count:
                            break
                        if pause_event.is_set():
                            print(f"Worker {worker_id} paused during frame")
                            while pause_event.is_set():
                                time.sleep(0.5)
                                gc.collect()
                                if hasattr(cv2, "ocl"):
                                    cv2.ocl.finish()
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, frame = cap.read()
                        
                        if not ret:
                            frame = np.zeros((height, width, 3), dtype=np.uint8)
                        
                        # Use OpenCL-accelerated processing
                        gpu_frame = cv2.UMat(frame) if use_gpu else frame
                        batch_frames.append(gpu_frame)
                        frame_progress = ((frame_idx + 1) / frame_count) * 100
                        status_queue.put((worker_id, item_id, filename, frame_progress, f"Analyzing frame {frame_idx+1}/{frame_count} ({frame_progress:.1f}%)"))

                    # Process batch
                    for frame in batch_frames:
                        # AMD-optimized processing
                        has_subtitle = process_frame(
                            frame,
                            use_gpu=use_gpu,
                            optimized_mode=optimized_mode
                        )
                        
                        # Simulated speech detection (replace with actual audio processing)
                        has_speech = (frame_idx // 30) % 5 != 0
                        
                        # Update counters
                        subtitle_frames += int(has_subtitle)
                        speech_frames += int(has_speech)
                        matched_frames += int(has_subtitle and has_speech)

                    # Explicit memory cleanup
                    del batch_frames
                    if use_gpu:
                        cv2.ocl.finish()  # Release AMD GPU memory
                    gc.collect()

                # Final results
                sync_ratio = matched_frames / speech_frames if speech_frames > 0 else 0
                status = "Completed" if sync_ratio > 0.8 else "Failed"
                status_queue.put((worker_id, item_id, filename, 100,
                                f"Completed: Sync ratio {sync_ratio:.2f} ({'PASS' if status == 'Completed' else 'FAIL'})"))
                result_queue.put((idx, item_id, status, sync_ratio))

            except Exception as e:
                error_msg = f"Worker {worker_id} error: {str(e)}"
                print(error_msg)
                status_queue.put((worker_id, item_id, filename, 100, f"Error: {error_msg}"))
                result_queue.put((idx, item_id, "Error", 0.0))
            
            finally:
                # Cleanup resources
                if cap and cap.isOpened():
                    cap.release()
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except Exception as e:
                        print(f"Error deleting temp file: {str(e)}")
                
                # Force cleanup of AMD GPU resources
                cv2.ocl.finish()
                gc.collect()

        print(f"Worker {worker_id} finished")
                

        def update_performance_chart(self, num_videos, cpu_time, gpu_time):
            """Update the performance comparison chart"""
            try:
                # Ensure matplotlib components are initialized
                if not hasattr(self, 'figure'):
                    from matplotlib.figure import Figure
                    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
                    
                    self.figure = Figure(figsize=(6, 3), dpi=100)
                    self.plot = self.figure.add_subplot(111)
                    self.canvas = FigureCanvasTkAgg(self.figure, self.perf_frame)
                    self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                
                # Clear the plot
                self.plot.clear()
                
                # Add CPU and GPU times
                labels = ['CPU', 'GPU']
                times = [cpu_time, gpu_time]
                bar_colors = ['blue', 'green']
                
                bars = self.plot.bar(labels, times, color=bar_colors)
                
                # Add data labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    self.plot.text(bar.get_x() + bar.get_width()/2., height,
                                  f'{height:.2f}s',
                                  ha='center', va='bottom')
                
                # Calculate speedup
                speedup = cpu_time / gpu_time if gpu_time > 0 else 0
                speedup_text = f"GPU {speedup:.2f}x faster" if speedup > 1 else f"CPU {1/speedup:.2f}x faster"
                
                # Add title and labels
                self.plot.set_title(f"Performance Comparison ({num_videos} videos)")
                self.plot.set_ylabel('Processing Time (seconds)')
                
                # Add speedup text as annotation
                self.plot.annotate(speedup_text, xy=(0.5, 0.9), xycoords='axes fraction',
                                 ha='center', va='center',
                                 bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.8))
                
                # Refresh canvas
                self.canvas.draw()
                
            except Exception as e:
                self.log(f"Error updating performance chart: {str(e)}")

        def stop_check(self):
            """Force stop processing and cleanup resources"""
            if not self.is_processing:
                return

            # Reset pause state
            self.is_paused = False
            self.pause_event.clear()

            # Set flags
            self.stop_requested = True
            self.is_processing = False

            # Update pause button text if window exists
            if hasattr(self, 'progress_win') and "pause_btn" in self.progress_win:
                self.progress_win["pause_btn"].config(text="Pause")

            # Terminate worker processes forcefully
            for p in self.worker_processes:
                if p.is_alive():
                    p.kill()  # More aggressive than terminate()

            # Clear all queues
            for q in [self.task_queue, self.result_queue, self.status_queue]:
                if q is not None:
                    while not q.empty():
                        try:
                            q.get_nowait()
                        except queue.Empty:
                            break

            # Close progress window
            if hasattr(self, 'progress_win'):
                self.progress_win["window"].destroy()
                del self.progress_win  # Force cleanup

            # Reset UI
            self.check_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.log("Processing forcefully stopped")


        def process_files(self, items):
            import time
            for item_id in items:
                self.video_start_times[item_id] = time.time()
                self.video_progress[item_id] = (0, 100)
            """Process files and update UI (single/sequential mode)."""
            try:
                total_items = len(items)
                # Create a progress window for single-file processing
                video_names = [os.path.basename(self.file_paths[item_id]) for item_id in items if item_id in self.file_paths]
                self.root.after(0, self.create_progress_window, video_names)

                for idx, item_id in enumerate(items):
                    # Check if stop was requested
                    if self.stop_requested:
                        self.log("Processing stopped by user")
                        return  # <-- This ensures summary is NOT shown if stopped

                    if item_id not in self.file_paths:
                        continue

                    file_path = self.file_paths[item_id]
                    values = self.file_list.item(item_id)["values"]
                    filename = values[0]

                    # Update progress and status in the progress window
                    progress_pct = (idx / total_items) * 100
                    self.worker_to_item[0] = item_id
                    self.update_progress_content(0, filename, progress_pct, f"Processing {idx+1}/{total_items}: {filename}")

                    # Update status to 'Checking' with light blue color
                    self.status_queue.put(("status_update", item_id, status))
                    self.log(f"Processing file: {filename}")

                    try:
                        # Actual subtitle synchronization check logic
                        self.check_subtitle_sync(file_path, item_id)
                    except Exception as e:
                        error_message = f"Error processing {filename}: {str(e)}"
                        self.log(f"ERROR: {error_message}")
                        self.log(traceback.format_exc())
                        print(f"DEBUG: Calling update_status on {self} of type {type(self)}")
                        self.update_status(item_id, "Error")
                        self.update_progress_content(0, filename, progress_pct, "Error occurred")

                # Complete progress bar and update status
                self.update_progress_content(0, filename, 100, "Processing complete" if not self.stop_requested else "Processing stopped")
                self.log("All files processed")
                    
                # Always close the progress window at the end
                if hasattr(self, 'progress_win'):
                    try:
                        self.progress_win["window"].destroy()
                    except Exception:
                        pass
                    del self.progress_win

            except Exception as e:
                self.log(f"Processing error: {str(e)}")
                self.log(traceback.format_exc())
                self.update_progress_content(0, filename if 'filename' in locals() else '', 0, "Error occurred")
            finally:
                self.cleanup_after_processing()


        import librosa
        import numpy as np
        def check_subtitle_sync(self, file_path, item_id):
            """Check subtitle synchronization for a file using the deep-learned model."""
            if self.predictor is None:
                self.log("No trained model loaded. Deep-learned checking unavailable.")
                self.update_status(item_id, "Error")
                return

            filename = os.path.basename(file_path)
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                self.log(f"Could not open video file: {file_path}")
                self.update_status(item_id, "Error")
                return

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # 1. Extract 32 evenly spaced frames
            num_frames = 32
            frame_indices = np.linspace(0, frame_count-1, num_frames, dtype=np.int32)
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
            frames = np.array(frames, dtype=np.float32) / 255.0  # Normalize

            # 2. Extract MFCC audio features
            audio, sr = librosa.load(file_path, sr=16000)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
            if mfccs.shape[1] < 100:
                mfccs = np.pad(mfccs, ((0, 0), (0, 100 - mfccs.shape[1])), mode='constant')
            elif mfccs.shape[1] > 100:
                mfccs = mfccs[:, :100]
            audio_features = mfccs.astype(np.float32)

            # 3. Prepare batch dimensions
            frames_batch = np.expand_dims(frames, axis=0)         # (1, 32, 224, 224, 3)
            audio_batch = np.expand_dims(audio_features, axis=0)  # (1, 40, 100)

            # 4. Run prediction
            pred = self.predictor.model.predict([frames_batch, audio_batch])[0][0]
            self.log(f"Deep model sync prediction: {pred:.3f}")

            # 5. Decide pass/fail (threshold can be tuned)
            if pred > 0.5:
                self.update_status(item_id, "Completed")
                self.log(f"Subtitle check PASSED: {filename} (score: {pred:.2f})")
            else:
                self.update_status(item_id, "Failed")
                self.log(f"Subtitle check FAILED: {filename} (score: {pred:.2f})")

    def cleanup_after_processing(self):
        """Clean up after processing is complete"""
        self.is_processing = False
        self.stop_requested = False
        self.check_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        # self.gpu_var.set("0.0%")  # <-- REMOVE or comment out this line
        self.gpu_progress["value"] = 0

        # Close any progress windows
        for window_info in self.progress_windows.values():
            if window_info and "window" in window_info:
                try:
                    window_info["window"].destroy()
                except:
                    pass
        self.progress_windows = {}
        self.check_button.config(state=tk.NORMAL)
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        
    def get_gpu_usage(self):
        """Get current GPU compute utilization percentage for AMD/DirectML"""
        try:
            # DirectML/TensorFlow method
            stats = tf.config.experimental.get_memory_info('GPU:0')
            return (stats['current'] / stats['limit']) * 100
        except AttributeError:
            # Fallback for non-DirectML configurations
            try:
                from tensorflow.python.client import device_lib
                devices = device_lib.list_local_devices()
                gpu_stats = [d for d in devices if 'GPU' in d.name]
                if gpu_stats:
                    return float(gpu_stats[0].memory_limit)  # Simplified metric
            except:
                return 0.0
        except Exception:
            return 0.0

    def update_progress_window_display(self):
                """Update visible progress components for the active tab"""
                if not hasattr(self, 'progress_win'):
                    return
                
                # Hide all content frames
                for frame in self.progress_win["content_frames"]:
                    frame.pack_forget()
                
                # Show active tab's content
                active_idx = min(self.progress_win["current_tab"], 
                                len(self.progress_win["content_frames"]) - 1)
                self.progress_win["content_frames"][active_idx].pack(fill=tk.BOTH, expand=True)
                
                # Update tab button states
                for idx, btn in enumerate(self.progress_win["tab_buttons"]):
                    btn.config(relief=tk.SUNKEN if idx == active_idx else tk.RAISED)

        
    def set_status(self, status):
        selected_items = self.video_list.selection()
        for item in selected_items:
            self.video_list.item(item, values=(self.video_data[item]["path"], status))
            self.video_data[item]["status"] = status
                
    def add_files(self):
            """Add video files to the file list."""
            files = filedialog.askopenfilenames(filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.webm")])
            if not files:
                return
            self.log(f"Adding {len(files)} file(s) to the list")
            for file in files:
                filename = os.path.basename(file)
                file_format = os.path.splitext(file)[1][1:].upper()
                try:
                    self.log(f"Reading file information: {filename}")
                    cap = cv2.VideoCapture(file)
                    if not cap.isOpened():
                        raise Exception("Could not open file")
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if fps > 0 and frame_count > 0:
                        duration_sec = frame_count / fps
                        hours, remainder = divmod(duration_sec, 3600)
                        minutes, seconds = divmod(remainder, 60)
                        duration = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
                    else:
                        duration = "00:00:00"
                    cap.release()
                except Exception as e:
                    self.log(f"Error getting duration: {str(e)}")
                    duration = "00:00:00"
                status = "Pending"
                item_id = self.file_list.insert("", "end", values=(filename, duration, file_format, status), tags=("pending",))
                self.file_paths[item_id] = file
                self.log(f"Added: {filename} ({duration}, {file_format})")
                
    def open_dl_interface(self):
        """Open the Deep Learning training interface"""
        DataCollector(self.root)  # Creates the DataCollector window
    def configure_gpu_performance(self):
        """NVIDIA-specific optimizations"""
        if hasattr(tf.config.optimizer, 'set_jit'):
            tf.config.optimizer.set_jit(False)
            
        # Enable cuDNN auto-tuner
        os.environ['TF_CUDNN_DETERMINISTIC'] = '0'
        os.environ['TF_CUDNN_USE_FRONTEND'] = '1'
        cv2.ocl.setUseOpenCL(True)
        os.environ["OPENCV_OPENCL_DEVICE"] = "AMD:GPU"
        if cv2.ocl.haveOpenCL():
            device = cv2.ocl.Device_getDefault()
            print(f"Using OpenCL device: {device.name()}")
        
        # Set data layout to channels_last
        tf.keras.backend.set_image_data_format('channels_last')
    def enable_ctrl_a_on_treeview(self, treeview):
        """Enable Ctrl+A to select all items in a Treeview"""
        def select_all(event):
            try:
                treeview.selection_set(treeview.get_children())
            except Exception as e:
                print(f"Selection error: {str(e)}")
            return "break"  # Prevent default behavior

        # Bind Ctrl+A (Windows/Linux) and Command+A (macOS)
        treeview.bind("<Control-a>", select_all)
        treeview.bind("<Command-a>", select_all)
    def open_video_directory(self, event):
        """Open file explorer to the directory of the selected video on double-click"""
        selected_items = self.file_list.selection()
        if not selected_items:
            return
            
        item_id = selected_items[0]
        if item_id not in self.file_paths:
            return
            
        file_path = self.file_paths[item_id]
        directory = os.path.dirname(file_path)
        
        try:
            # Open file explorer to the directory
            if os.name == 'nt':  # Windows
                os.startfile(directory)
            elif os.name == 'posix':  # macOS, Linux
                subprocess.Popen(['xdg-open', directory])
        except Exception as e:
            self.log(f"Error opening directory: {str(e)}")
    def show_file_info(self, event):
        """Display detailed file information when a Treeview item is selected."""
        selected_items = self.file_list.selection()
        if not selected_items:
            return
        
        item_id = selected_items[0]
        if item_id not in self.file_paths:
            return
        
        file_path = self.file_paths[item_id]
        values = self.file_list.item(item_id)["values"]
        
        try:
            # Get detailed file information
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
            cap = cv2.VideoCapture(file_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            codec = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec_str = "".join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])
            cap.release()

            info = f"""File Name: {values[0]}
    Path: {file_path}
    Size: {file_size:.2f} MB
    Resolution: {width}x{height}
    FPS: {fps:.2f}
    Format: {values[2]} ({codec_str})
    Duration: {values[1]}
    Status: {values[3]}"""

            # Update the info text box
            self.info_text.config(state=tk.NORMAL)
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(tk.END, info)
            self.info_text.config(state=tk.DISABLED)

        except Exception as e:
            self.log(f"Error showing file info: {str(e)}")
    def toggle_performance_frame(self):
        """Show/hide performance comparison frame based on benchmark checkbox"""
        if self.benchmark_var.get():
            # Initialize performance frame components if needed
            if not hasattr(self, 'perf_frame'):
                from matplotlib.figure import Figure
                from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
                
                self.perf_frame = tk.LabelFrame(self.left_panel, text="Performance Comparison")
                self.figure = Figure(figsize=(6, 3), dpi=100)
                self.plot = self.figure.add_subplot(111)
                self.canvas = FigureCanvasTkAgg(self.figure, self.perf_frame)
                self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Show the frame
            self.perf_frame.pack(fill=tk.X, pady=(10, 0), padx=10)
        else:
            # Hide the frame
            if hasattr(self, 'perf_frame'):
                self.perf_frame.pack_forget()
    
    def start_check(self, items=None):
        """Start subtitle synchronization check for videos."""
        # Reset pause state when starting new check
        self.is_paused = False
        self.pause_event.clear()
        if self.status_queue is None:
            self.status_queue = multiprocessing.Queue()

        if self.is_processing:
            messagebox.showinfo("Processing", "Already processing files. Please wait.")
            return

        if items is None:
            items = self.file_list.get_children()
        if not items:
            messagebox.showinfo("No Files", "Please add video files first.")
            return

        # Update GPU availability based on selected mode
        mode = self.processing_mode.get()
        if mode == "gpu" or mode == "gpu_optimized":
            if not cv2.ocl.haveOpenCL():
                messagebox.showwarning("GPU Not Available", "OpenCL is not available. Using CPU instead.")
                self.gpu_available = False
            else:
                cv2.ocl.setUseOpenCL(True)
                self.gpu_available = True
                self.log(f"Forcing GPU processing mode: {'Optimized' if mode == 'gpu_optimized' else 'Standard'}")
        elif mode == "cpu":
            cv2.ocl.setUseOpenCL(False)
            self.gpu_available = False
            self.log("Forcing CPU processing mode")
        else:  # auto
            cv2.ocl.setUseOpenCL(True)
            self.gpu_available = cv2.ocl.haveOpenCL() and cv2.ocl.useOpenCL()
            self.log(f"Auto processing mode: Using {'GPU' if self.gpu_available else 'CPU'}")

        self.is_processing = True
        self.stop_requested = False
        self.check_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        # Clear log
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)

        self.log("Starting subtitle synchronization check")
        self.log(f"Using {'GPU' if self.gpu_available else 'CPU'} acceleration")

        # Check if multi-video processing is enabled
        if self.enable_multi_var.get() and len(items) > 1:
            self.log(f"Multi-video processing enabled with {self.concurrent_var.get()} concurrent videos")
            # Run multi-processing with current mode
            threading.Thread(target=self.run_multi_processing, args=(items,), daemon=True).start()
        else:
            # Run single-video processing
            self.log("Multi-video processing disabled - processing files sequentially")
            threading.Thread(target=self.process_files, args=(items,), daemon=True).start()
    def stop_check(self):
        """Force stop processing and cleanup resources"""
        if not self.is_processing:
            return

        # Reset pause state
        self.is_paused = False
        self.pause_event.clear()

        # Set flags
        self.stop_requested = True
        self.is_processing = False

        # Update pause button text if window exists
        if hasattr(self, 'progress_win') and "pause_btn" in self.progress_win:
            self.progress_win["pause_btn"].config(text="Pause")

        # Terminate worker processes forcefully
        for p in self.worker_processes:
            if p.is_alive():
                p.kill()  # More aggressive than terminate()

        # Clear all queues
        for q in [self.task_queue, self.result_queue, self.status_queue]:
            if q is not None:
                while not q.empty():
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        break

        # Close progress window
        if hasattr(self, 'progress_win'):
            try:
                self.progress_win["window"].destroy()
            except:
                pass
            del self.progress_win  # Force cleanup

        # Reset UI
        self.check_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.log("Processing forcefully stopped")
    def remove_selected(self):
        """Remove selected files from the file list."""
        selected_items = self.file_list.selection()
        if not selected_items:
            return
        for item_id in selected_items:
            filename = self.file_list.item(item_id)["values"][0]
            self.log(f"Removing file: {filename}")
            if item_id in self.file_paths:
                del self.file_paths[item_id]
            self.file_list.delete(item_id)
    def initialize_gpu(self):
        """Check for GPU availability and set OpenCL if available."""
        try:
            # Enable OpenCL
            cv2.ocl.setUseOpenCL(True)
            # Check if OpenCL is available
            self.gpu_available = cv2.ocl.haveOpenCL() and cv2.ocl.useOpenCL()
            if self.gpu_available:
                device = cv2.ocl.Device_getDefault()
                device_name = device.name()
                print(f"OpenCL device: {device_name}")
                # Detect AMD GPU specifically
                if "AMD" in device_name or "Radeon" in device_name or "gfx" in device_name.lower():
                    print(f"AMD GPU detected: {device_name}")
            else:
                print("OpenCL not available or not enabled.")
        except Exception as e:
            print(f"Error initializing GPU: {str(e)}")
            self.gpu_available = False  # Fix indentation here
    def start_resource_monitor(self):
        """Start a thread to monitor CPU and GPU usage"""
        threading.Thread(target=self.update_resource_usage, daemon=True).start()
    def update_resource_usage(self):
        while True:
            try:
                self.root.update_idletasks()
            except Exception as e:
                print(f"Resource update error: {str(e)}")
            time.sleep(1)


    def get_directml_gpu_usage(self):
        """Get current GPU usage percentage for DirectML workloads"""
        try:
            if os.name == 'nt':  # Windows
                import subprocess
                result = subprocess.check_output(
                    ['powershell', 
                     '"Get-Counter \\"\\GPU Engine(*engtype_Compute)\\Utilization Percentage\\" -ErrorAction SilentlyContinue | ' +
                     'Select-Object -ExpandProperty CounterSamples | ' +
                     'Select-Object -ExpandProperty CookedValue"'],
                    stderr=subprocess.DEVNULL,
                    shell=True
                ).decode().strip()
                
                try:
                    values = [float(x) for x in result.split('\n') if x.strip()]
                    return max(values) if values else 0
                except:
                    return 0
            return 0
        except:
            return 0

    def log(self, message):
        """Add a message to the log with timestamp"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        
        # Print to console always
        print(log_message.strip())
        
        # Update log widget if it exists
        if hasattr(self, 'log_text') and self.log_text is not None:
            self.log_text.config(state=tk.NORMAL)
            self.log_text.insert(tk.END, log_message)
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)
            self.root.update_idletasks()

    def show_file_info(self, event):
        """Display detailed file information in a modern scrollable box"""
        selected = self.file_list.selection()
        if not selected:
            return
        
        item_id = selected[0]
        if item_id not in self.file_paths:
            return
        
        file_path = self.file_paths[item_id]
        values = self.file_list.item(item_id)["values"]
        
        try:
            # Get detailed info
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            cap = cv2.VideoCapture(file_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            codec = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec_str = "".join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])
            cap.release()

            info = f"""File Name: {values[0]}
    Path: {file_path}
    Size: {file_size:.2f} MB
    Resolution: {width}x{height}
    FPS: {fps:.2f}
    Format: {values[2]} ({codec_str})
    Duration: {values[1]}
    Status: {values[3]}"""

            # Update the info box
            self.info_text.config(state=tk.NORMAL)
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(tk.END, info)
            self.info_text.config(state=tk.DISABLED)

        except Exception as e:
            self.log(f"Error showing file info: {str(e)}")


    def toggle_pause(self):
        """Pause/Resume processing when pause button is clicked"""
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_event.set()
            self.log("Processing paused")
        else:
            self.pause_event.clear()
            self.log("Processing resumed")
        
        # Update button text safely
        if hasattr(self, 'progress_win') and "pause_btn" in self.progress_win:
            self.progress_win["pause_btn"].config(
                text="Resume" if self.is_paused else "Pause"
            )

    def show_file_info(self, event):
        selected = self.file_list.selection()
        if not selected:
            return
        item_id = selected[0]
        if item_id not in self.file_paths:
            return
        file_path = self.file_paths[item_id]
        values = self.file_list.item(item_id)["values"]
        info_text = (
            f"File Name: {values[0]}\n"
            f"Duration: {values[1]}\n"
            f"Format: {values[2]}\n"
            f"Status: {values[3]}\n"
            f"Path: {file_path}"
        )
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert(tk.END, info_text)
        self.info_text.config(state=tk.DISABLED)

 

    def recheck_failed_videos(self, summary_window):
        """Recheck videos that failed synchronization"""
        summary_window.destroy()
        
        # Find all failed videos
        failed_items = []
        for item_id in self.file_list.get_children():
            status = self.file_list.item(item_id)["values"][3]
            if status == "Failed" or status == "Error":
                failed_items.append(item_id)
                # Reset status to Pending
                print(f"DEBUG: Calling update_status on {self} of type {type(self)}")
                self.update_status(item_id, "Pending")
        
        # Start processing just the failed videos
        if failed_items:
            self.start_check(failed_items)

    def run_benchmark(self, items):
        """Run CPU and GPU benchmark on the same set of videos"""
        self.log("Starting CPU/GPU benchmark...")
        
        # First run with CPU
        self.log("Phase 1: CPU processing")
        original_mode = self.processing_mode.get()
        self.processing_mode.set("cpu")
        cv2.ocl.setUseOpenCL(False)
        self.gpu_available = False
        
        # Reset status
        for item_id in items:
            print(f"DEBUG: Calling update_status on {self} of type {type(self)}")
            self.update_status(item_id, "Pending")
        
        # Run CPU benchmark
        cpu_start_time = time.time()
        self.run_multi_processing(items, "CPU")
        cpu_end_time = time.time()
        cpu_total_time = cpu_end_time - cpu_start_time
        self.log(f"CPU processing completed in {cpu_total_time:.2f} seconds")
        
        # Check if stop was requested
        if self.stop_requested:
            self.cleanup_after_processing()
            return
        
        # Now run with GPU if available
        if cv2.ocl.haveOpenCL():
            self.log("Phase 2: GPU processing")
            self.processing_mode.set("gpu_optimized")
            cv2.ocl.setUseOpenCL(True)
            self.gpu_available = True
            
            # Reset status
            for item_id in items:
                print(f"DEBUG: Calling update_status on {self} of type {type(self)}")
                self.update_status(item_id, "Pending")
            
            # Run GPU benchmark
            gpu_start_time = time.time()
            self.run_multi_processing(items, "GPU")
            gpu_end_time = time.time()
            gpu_total_time = gpu_end_time - gpu_start_time
            self.log(f"GPU processing completed in {gpu_total_time:.2f} seconds")
            
            # Update performance chart
            self.update_performance_chart(len(items), cpu_total_time, gpu_total_time)
        else:
            self.log("GPU not available for benchmark comparison")
        
        # Restore original mode
        self.processing_mode.set(original_mode)
        self.cleanup_after_processing()

    def handle_queue_updates(self):
        while self.is_processing:
            try:
                worker_id, item_id, filename, progress, status = self.status_queue.get_nowait()
                self.root.after(0, self.update_progress_content, 
                              worker_id, filename, progress, status)
            except queue.Empty:
                time.sleep(0.1)
            except Exception as e:
                self.log(f"Queue error: {str(e)}")
                
    def process_results(self, items, total_items, mode_label):
                """Process results from worker processes"""
                self.processed_count = 0
                
                while self.processed_count < total_items and not self.stop_requested:
                    try:
                        # Skip when paused
                        if hasattr(self, 'is_paused') and self.is_paused:
                            time.sleep(0.2)
                            continue
                            
                        # Get results with timeout
                        try:
                            result = self.result_queue.get(timeout=0.5)
                            self.processed_count += 1
                            
                            # Unpack result
                            idx, item_id, status, sync_ratio = result
                            
                            # Update status in UI
                            print(f"DEBUG: Calling update_status on {self} of type {type(self)}")

                            self.update_status(item_id, status)
                            self.update_progress_window_display()

                            
                            # Log result
                            self.log(f"Completed video {idx+1}/{total_items}: {os.path.basename(self.file_paths[item_id])}")
                            self.log(f"Sync ratio: {sync_ratio:.2f}")
                            
                            
                        except queue.Empty:
                            continue
                            
                    except Exception as e:
                        self.log(f"Error processing results: {str(e)}")
                
                # Clean up processing resources
                self.cleanup_after_processing()
                
                # Always close the progress window at the end
                if hasattr(self, 'progress_win'):
                    try:
                        self.progress_win["window"].destroy()
                    except Exception:
                        pass
                    del self.progress_win
                
    def run_multi_processing(self, items, mode_label=None):
        import time
        for item_id in items:
            self.video_start_times[item_id] = time.time()
            self.video_progress[item_id] = (0, 100)
        concurrent_videos = min(self.concurrent_var.get(), len(items))
        if not mode_label:
            mode_label = "GPU" if self.gpu_available else "CPU"
        self.log(f"Starting multi-video processing with {concurrent_videos} concurrent videos")

        # Get list of video filenames for the progress window
        video_names = [os.path.basename(self.file_paths[item_id]) for item_id in items if item_id in self.file_paths]

        # Create the progress window in the main thread
        self.root.after(0, self.create_progress_window, video_names)
        """Process multiple videos concurrently"""
        concurrent_videos = min(self.concurrent_var.get(), len(items))
        if not mode_label:
            mode_label = "GPU" if self.gpu_available else "CPU"
        
        self.log(f"Starting multi-video processing with {concurrent_videos} concurrent videos")
        
        # Get list of video filenames for the progress window
        video_names = [self.file_paths[item_id] for item_id in items[:self.concurrent_var.get()] if item_id in self.file_paths]
        for item_id in items[:self.concurrent_var.get()]:
            if item_id in self.file_paths:
                video_names.append(os.path.basename(self.file_paths[item_id]))
        video_names = [os.path.basename(self.file_paths[item_id]) for item_id in items if item_id in self.file_paths]
        # Create a single progress window with tabs for each video
        
        # Set up pause functionality
        self.is_paused = False
        
        # Start time tracking for time-left calculation
        self.processing_start_time = time.time()
        self.processed_count = 0

        self.worker_processes = [
            multiprocessing.Process(
                target=self.video_worker,
                args=(i, self.task_queue, self.result_queue, 
                      self.status_queue, 
                      self.gpu_available,  # use_gpu
                      (self.processing_mode.get() == "gpu_optimized"),  # optimized_mode
                      self.pause_event)
            ) for i in range(self.concurrent_var.get())
        ]
        for i, item_id in enumerate(items[:concurrent_videos]):
            self.worker_to_item[i] = item_id
        # Add all items to the task queue
        for idx, item_id in enumerate(items):
            if item_id in self.file_paths:
                file_path = self.file_paths[item_id]
                filename = os.path.basename(file_path)
                self.task_queue.put((idx, item_id, file_path, filename))
        
        # Determine processing mode
        use_gpu = self.gpu_available
        opt_mode = self.processing_mode.get() == "gpu_optimized"

        # Start worker processes
        self.worker_processes = []
        for i in range(concurrent_videos):
            p = multiprocessing.Process(
            target=self.video_worker,
            args=(i, self.task_queue, self.result_queue, self.status_queue, use_gpu, opt_mode, self.pause_event)
            )
            p.daemon = True
            p.start()
            self.worker_processes.append(p)
        
        # Start status update and result processing threads
        threading.Thread(target=self.handle_status_updates, daemon=True).start()
        threading.Thread(target=self.process_results, args=(items, len(items), mode_label), daemon=True).start()
        
    def create_progress_window(self, video_names):
        """Create a progress window with tabs for monitoring multiple video processing"""
        # Destroy existing window if any
        if hasattr(self, 'progress_win'):
            try:
                self.progress_win["window"].destroy()
            except:
                pass

        # Create new window
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Video Processing")
        progress_window.resizable(False, False)

        # Center the window
        width = 540
        height = 360
        x = (progress_window.winfo_screenwidth() // 2) - (width // 2)
        y = (progress_window.winfo_screenheight() // 2) - (height // 2)
        progress_window.geometry(f'{width}x{height}+{x}+{y}')

        # Initialize progress window data
        self.progress_win = {
            "window": progress_window,
            "tab_buttons": [],
            "content_frames": [],
            "file_labels": [],
            "progress_bars": [],
            "status_labels": [],
            "current_tab": 0
        }

        # Create tab container at the top
        tab_frame = tk.Frame(progress_window)
        tab_frame.pack(fill=tk.X, padx=2, pady=2)

        # Create content frame (holds all video frames)
        content_frame = tk.Frame(progress_window)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        num_boxes = self.concurrent_var.get()
        for i, name in enumerate(video_names[:num_boxes]):
            short_name = os.path.basename(name)
            if len(short_name) > 15:
                short_name = short_name[:12] + "..."

            # Create tab button
            tab = tk.Button(tab_frame, text=short_name, relief=tk.RAISED,
                            command=lambda idx=i: self.switch_tab(idx))
            tab.pack(side=tk.LEFT, padx=1, pady=2)
            self.progress_win["tab_buttons"].append(tab)

            # Create individual content frame for this tab
            vid_frame = tk.Frame(content_frame)

            # Add widgets to vid_frame
            processing_label = tk.Label(vid_frame, text="Processing:", anchor="w")
            processing_label.pack(fill=tk.X, padx=10, pady=(15, 0))

            file_label = tk.Label(vid_frame, text=os.path.basename(name), anchor="w", font=("Arial", 10, "bold"))
            file_label.pack(fill=tk.X, padx=10, pady=(0, 10))

            progress_label = tk.Label(vid_frame, text="Progress:", anchor="w")
            progress_label.pack(fill=tk.X, padx=10, pady=0)

            progress_bar = ttk.Progressbar(vid_frame, length=480)
            progress_bar.pack(padx=10, pady=(0, 10))

            eta_label = tk.Label(vid_frame, text="ETA: --:--", anchor="w")
            eta_label.pack(fill=tk.X, padx=10, pady=(0, 5))

            status_label = tk.Label(vid_frame, text="Starting...", anchor="w")
            status_label.pack(fill=tk.X, padx=10, pady=5)



            # Store components
            self.progress_win["content_frames"].append(vid_frame)
            self.progress_win["file_labels"].append(file_label)
            self.progress_win["progress_bars"].append(progress_bar)
            self.progress_win.setdefault("eta_labels", []).append(eta_label)
            self.progress_win["status_labels"].append(status_label)


        # Create control buttons at the bottom
        buttons_frame = tk.Frame(progress_window)
        buttons_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=(15, 10))

        self.progress_win["pause_btn"] = tk.Button(buttons_frame, text="Pause", width=10,
                                                   command=self.toggle_pause)
        self.progress_win["pause_btn"].pack(side=tk.LEFT, padx=(80, 10))

        self.progress_win["stop_btn"] = tk.Button(buttons_frame, text="Stop checking", width=15,
                                                  command=self.stop_check)
        self.progress_win["stop_btn"].pack(side=tk.RIGHT, padx=(10, 80))

        # Show first tab's content
        if self.progress_win["content_frames"]:
            self.progress_win["content_frames"][0].pack(fill=tk.BOTH, expand=True)
            self.progress_win["tab_buttons"][0].config(relief=tk.SUNKEN)

        return self.progress_win


    def switch_tab(self, tab_index):
        """Switch between video tabs in the progress window"""
        if not hasattr(self, 'progress_win'):
            return

        # Hide all frames
        for frame in self.progress_win["content_frames"]:
            frame.pack_forget()

        # Show the selected tab's frame
        self.progress_win["content_frames"][tab_index].pack(fill=tk.BOTH, expand=True)

        # Update tab button appearance
        for idx, btn in enumerate(self.progress_win["tab_buttons"]):
            btn.config(relief=tk.SUNKEN if idx == tab_index else tk.RAISED)

        # Update current tab index
        self.progress_win["current_tab"] = tab_index


        
    def update_progress_content(self, worker_id, filename, progress, status):
        print(f"update_progress_content called: worker_id={worker_id}, progress={progress}, filename={filename}")
        item_id = self.worker_to_item.get(worker_id)
        print(f"Mapped item_id: {item_id}")
        if item_id is None:
            return

        # ETA calculation
        import time
        if not hasattr(self, "video_start_times"):
            self.video_start_times = {}
        start_time = self.video_start_times.get(item_id)
        if start_time is None:
            start_time = time.time()
            self.video_start_times[item_id] = start_time
        elapsed = time.time() - start_time
        pct = float(progress) / 100.0 if progress > 0 else 0.0001
        remaining = (elapsed / pct) - elapsed if pct < 1.0 else 0
        mins, secs = divmod(int(remaining), 60)
        eta_str = f"ETA: {mins:02d}:{secs:02d}" if progress < 100 else "ETA: 00:00"

        # Update widgets
        if not hasattr(self, 'progress_win'):
            return
        if worker_id >= len(self.progress_win["file_labels"]):
            return
        try:
            self.progress_win["file_labels"][worker_id].config(text=filename)
            self.progress_win["progress_bars"][worker_id]["value"] = progress
            # Update ETA label
            if "eta_labels" in self.progress_win and worker_id < len(self.progress_win["eta_labels"]):
                self.progress_win["eta_labels"][worker_id].config(text=eta_str)
            self.progress_win["status_labels"][worker_id].config(text=status)
        except tk.TclError:
            pass


    def handle_status_updates(self):
        while True:
            try:
                msg = self.status_queue.get_nowait()
                if msg[0] == "map_worker":
                    _, worker_id, item_id = msg
                    self.worker_to_item[worker_id] = item_id
                    continue
            except queue.Empty:
                break
        while True:
            try:
                msg = self.status_queue.get_nowait()
                if msg[0] == "status_update":
                    _, item_id, status = msg
                    print(f"DEBUG: Calling update_status on {self} of type {type(self)}")
                    self.update_status(item_id, status)
            except queue.Empty:
                break
        while self.is_processing:
            try:
                msg_type, item_id, status = self.status_queue.get_nowait()
                if msg_type == "status_update":
                    self.root.after(0, self.update_status, item_id, status)  # GUI update

                try:
                    status_update = self.status_queue.get(timeout=0.1)
                    worker_id, item_id, filename, progress, status_msg = status_update

                    # Update progress window in main thread
                    self.root.after(0, self.update_progress_content,
                                    worker_id, filename, progress, status_msg)

                    # Update main file list status
                    if status_msg.startswith("Processing"):
                        self.status_queue.put(("status_update", item_id, status))

                except queue.Empty:
                    time.sleep(0.1)

            except Exception as e:
                self.log(f"Error handling status updates: {str(e)}")
                break

            time.sleep(0.1)
    def check_ffmpeg_available(self):
        """Check if ffmpeg is available and supports required codecs"""
        import subprocess
        import shutil
        
        # Check if ffmpeg is in PATH
        if not shutil.which('ffmpeg'):
            print("WARNING: ffmpeg not found in PATH. Video conversion will fail.")
            return False
        
        # Check version and codec support
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE,
                                    text=True)
            if result.returncode == 0:
                print(f"Found ffmpeg: {result.stdout.splitlines()[0]}")
                return True
        except Exception as e:
            print(f"Error checking ffmpeg: {e}")
        
        return False
    def open_gpu_monitor(self):
        """Open the GPU monitoring window"""
        GPUMonitor(self.root)
        
    def configure_hardware(self):
        """Configure hardware settings for AMD/DirectML"""
        system = platform.system()
        if system == "Windows":
            try:
                import tensorflow_directml as dml
                dml.set_default_device()
            except ImportError:
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:  # <-- ADD THIS LINE
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.optimizer.set_experimental_options({
                    'layout_optimizer': True,
                    'arithmetic_optimization': True
                })
                tf.config.set_visible_devices(gpus[0], 'GPU')
            except RuntimeError as e:  # <-- KEEP THIS LINE
                print(f"GPU config error: {e}")


@staticmethod
def process_frame(frame, use_gpu=False, optimized_mode=False):
    def get_shape(frame):
        # Handles both UMat and ndarray
        if isinstance(frame, cv2.UMat):
            return frame.get().shape
        return frame.shape
    try:
        small_frame = cv2.resize(frame, (320, 180))
        yuv_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2YUV)

        # Extract ROI safely for GPU/CPU
        if use_gpu:
            # Convert UMat to NumPy for slicing
            yuv_np = yuv_frame.get()
            roi_np = yuv_np[-60:, :]
            roi = cv2.UMat(roi_np)  # Back to UMat
        else:
            roi = yuv_frame[-60:, :]  # CPU path uses normal slicing

        # Rest of GPU/CPU logic remains unchanged
        if use_gpu:
            gpu_roi = roi
            y_channel = cv2.split(gpu_roi)[0]
            _, gpu_thresh = cv2.threshold(y_channel, 200, 255, cv2.THRESH_BINARY)
            thresh_np = cv2.UMat.get(gpu_thresh)
            white_pixels = cv2.countNonZero(thresh_np)
        else:
            y_channel = roi[:, :, 0]
            _, thresh = cv2.threshold(y_channel, 200, 255, cv2.THRESH_BINARY)
            white_pixels = cv2.countNonZero(thresh)

        return white_pixels > (get_shape(roi)[1] * 60 * 0.05)
    except Exception as e:
        print(f"Frame processing error: {str(e)}")
        return False



# Add the parent directory to Python's search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deep_learning.sync_predictor import HybridPredictor

class GPUMonitor(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("DirectML GPU Monitor")
        self.geometry("600x400")
        
        # Initialize data storage
        self.timestamps = []
        self.gpu_usage = []
        self.running = True
        
        # Create the figure and canvas
        self.figure = plt.Figure(figsize=(5, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Current usage
        self.usage_frame = tk.Frame(self)
        self.usage_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(self.usage_frame, text="Current Compute Usage:").pack(side=tk.LEFT)
        self.usage_var = tk.StringVar(value="0.0%")
        tk.Label(self.usage_frame, textvariable=self.usage_var, 
                 font=("Arial", 16, "bold")).pack(side=tk.LEFT, padx=10)
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self.update_data, daemon=True)
        self.monitor_thread.start()
        
        # Set up a periodic callback to update the plot
        self.after(1000, self.update_plot)
        
        # Ensure we stop the thread when window closes
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def get_gpu_compute_usage(self):
        """Get DirectML compute usage percentage"""
        # Use the method from the parent class
        return SubtitleSyncChecker.get_directml_gpu_usage(self)
    
    def update_data(self):
        """Update data in the background thread"""
        while self.running:
            usage = self.get_gpu_compute_usage()
            
            # Update data
            self.timestamps.append(time.time())
            self.gpu_usage.append(usage)
            
            # Keep only the most recent 60 seconds
            if len(self.timestamps) > 60:
                self.timestamps.pop(0)
                self.gpu_usage.pop(0)
            
            # Update the current usage variable
            self.usage_var.set(f"{usage:.1f}%")
            
            time.sleep(1)
    
    def update_plot(self):
        """Update the plot with the latest data"""
        if not self.running:
            return
            
        # Clear the plot
        self.ax.clear()
        
        # Plot the data
        if self.timestamps:
            # Convert to relative time for x-axis
            relative_times = [t - self.timestamps[0] for t in self.timestamps]
            self.ax.plot(relative_times, self.gpu_usage, 'b-')
            self.ax.fill_between(relative_times, self.gpu_usage, alpha=0.3)
            
            # Set labels and limits
            self.ax.set_xlabel('Time (s)')
            self.ax.set_ylabel('GPU Compute Utilization (%)')
            self.ax.set_ylim(0, 100)
            self.ax.set_title('DirectML GPU Compute Usage')
            self.ax.grid(True)
        
        # Redraw the canvas
        self.canvas.draw()
        
        # Schedule the next update
        self.after(1000, self.update_plot)
    
    def on_closing(self):
        """Handle window closing"""
        self.running = False
        self.destroy()

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    app = SubtitleSyncChecker()
    app.root.mainloop()
    # Set multiprocessing start method
    if hasattr(multiprocessing, 'set_start_method'):
        try:
            multiprocessing.set_start_method('spawn')
        except RuntimeError:
            pass  # Method already set
            
    app = SubtitleSyncChecker()
    app.root.mainloop()

class HybridPredictor:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.audio_window = []
        
    def process_audio(self, audio_chunk):
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=audio_chunk, sr=16000, n_mfcc=40)
        self.audio_window.append(mfcc)
        if len(self.audio_window) > 100:
            self.audio_window.pop(0)
            
    def predict_sync(self, frame):
        # Visual processing
        processed_frame = preprocess_frame(frame)  # Your existing frame processing
        
        # Audio processing
        audio_input = np.array(self.audio_window[-100:]).T  # Last 100 frames
        
        # Hybrid prediction
        return self.model.predict([np.expand_dims(processed_frame, 0), 
                                 np.expand_dims(audio_input, 0)])

def convert_video_for_processing(self, file_path, target_format='h264'):
    """Convert x265/HEVC video to more compatible format using ffmpeg"""
    import subprocess
    import tempfile
    import os
    
    # Create a temporary converted video file
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
        temp_name = temp_video.name
        
    # Use ffmpeg to convert the video
    self.log(f"Converting video format using ffmpeg...")
    try:
        if target_format == 'h264':
            # Convert to h264 with AAC audio
            subprocess.call([
                'ffmpeg', '-y', '-i', file_path, 
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac',
                temp_name
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif target_format == 'wav_extract':
            # Extract audio only
            subprocess.call([
                'ffmpeg', '-y', '-i', file_path, 
                '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                temp_name.replace('.mp4', '.wav')
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
        return temp_name
    except Exception as e:
        self.log(f"Conversion error: {str(e)}")
        return file_path  # Return original if conversion fails