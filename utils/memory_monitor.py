import psutil
import threading
import gc
import cv2
from tensorflow.python.keras import backend as K
import time
import tensorflow as tf


class MemoryMonitor:
    def __init__(self, threshold=70):
        self.threshold = threshold
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.monitor, daemon=True)
        
    def start(self):
        self.thread.start()
        
    def monitor(self):
        while not self.stop_event.is_set():
            # Windows-compatible monitoring
            cpu = psutil.cpu_percent()
            ram = psutil.virtual_memory().percent
            gpu = self.get_directml_gpu_usage()  # From Option 2
            
            time.sleep(1)
    
    def get_directml_gpu_usage(self):
        """For DirectML/TensorFlow on Windows"""
        try:
            stats = tf.config.experimental.get_memory_info('GPU:0')
            return stats['current'] / stats['limit'] * 100
        except:
            return 0
    
    def stop(self):
        self.stop_event.set()
        self.thread.join()
    def get_gpu_usage():
        stats = tf.config.experimental.get_memory_info('GPU:0')
        return stats['current'] / stats['limit'] * 100