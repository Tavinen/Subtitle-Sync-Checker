HYBRID_PARAMS = {
    "audio_weight": 0.6,
    "replay_ratio": 0.3,
    "buffer_size": 5000,
    "freeze_layers": 0.75,
    "gpu_thread_count": 16, 
    "mixed_precision": True,
    "tensor_layout": "NHWC",
    "ocl_device": "AMD:GPU",  # Added comma
    "compute_dtype": "bfloat16"
}