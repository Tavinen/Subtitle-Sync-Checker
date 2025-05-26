import os
import subprocess
import cv2
import numpy as np
import soundfile as sf
from pathlib import Path
import shutil
from tqdm import tqdm
import logging
import shutil

def check_ffmpeg():
    if not shutil.which('ffmpeg'):
        raise RuntimeError(
            "FFmpeg not found! Install via:\n"
            "Windows: winget install ffmpeg\n"
            "Linux: sudo apt install ffmpeg\n"
            "Mac: brew install ffmpeg"
        )

# Call this before processing
check_ffmpeg()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_dependencies():
    """Verify required system dependencies."""
    if not shutil.which('ffmpeg'):
        raise EnvironmentError(
            "ffmpeg not found in PATH.\n"
            "Install via:\n"
            "Linux: sudo apt install ffmpeg\n"
            "Mac: brew install ffmpeg\n"
            "Windows: Download from https://ffmpeg.org/"
        )

def preprocess_and_save(video_path: str, label: int, dataset_path: str, idx: int):
    """Process a video file into training samples."""
    try:
        # Convert to Path objects
        video_path = Path(video_path)
        dataset_path = Path(dataset_path)
        
        # Create output directory
        output_dir = dataset_path / "buffer"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate input path
        if not video_path.exists():
            raise FileNotFoundError(f"Video path not found: {video_path}")
            
        # Process video files in directory
        if video_path.is_dir():
            video_files = list(video_path.glob('*.[mM][pP][4]')) + list(video_path.glob('*.[mM][kK][vV]'))
            if not video_files:
                raise ValueError(f"No video files found in directory: {video_path}")
        else:
            video_files = [video_path]

        # Process each video file
        for vf in video_files:
            process_single_video(vf, label, output_dir, idx)

    except Exception as e:
        logging.error(f"Failed to process {video_path}: {str(e)}")
        raise

def process_single_video(video_file: Path, label: int, output_dir: Path, idx: int):
    """Process individual video file."""
    temp_dir = output_dir / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Extract frames and audio using FFmpeg
        frame_pattern = temp_dir / "frame_%04d.png"
        audio_path = temp_dir / "audio.wav"
        
        # FFmpeg command (cross-platform paths)
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-hwaccel', 'auto',
            '-i', video_file.as_posix(),
            '-vf', 'fps=30,scale=224:224',
            '-q:v', '2',
            frame_pattern.as_posix(),
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            audio_path.as_posix()
        ]
        
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        
        # Load and process frames
        frame_files = sorted(temp_dir.glob('frame_*.png'))
        frames = [cv2.imread(str(f)) for f in frame_files]
        frames = np.array(frames, dtype=np.float32) / 255.0  # Normalize
        
        # Load and process audio
        audio, sr = sf.read(audio_path)
        audio = audio.mean(axis=1) if audio.ndim > 1 else audio  # Mono conversion
        audio = audio[::2]  # Downsample to 8kHz
        
        # Create sample chunks
        sample_length = 16000 * 5  # 5-second chunks
        for i in range(0, len(audio), sample_length):
            audio_chunk = audio[i:i+sample_length]
            if len(audio_chunk) < sample_length:
                break  # Discard partial chunks
     
    finally:
        # Cleanup temporary files
        for f in temp_dir.glob('*'):
            f.unlink()
        temp_dir.rmdir()

if __name__ == "__main__":
    check_dependencies()
    
    # === Configuration ===
    video_label_list = [
        (r"S:\Anime\Japanese - Sub\For a while\Order\Render\[Moozzi2] Kimetsu no Yaiba Katanak", 1),
        (r"S:\Anime\Japanese - Sub\For a while\Order\Deep Learn\[Moozzi2] Kono Subarashii Seka", 0),
    ]
    
    dataset_path = Path(r"T:\Deeplearn\training_data\dataset")
    
    # === Processing ===
    logging.info("Starting video preprocessing")
    for idx, (video_path, label) in enumerate(tqdm(video_label_list, desc="Processing videos")):
        try:
            preprocess_and_save(video_path, label, dataset_path, idx)
        except Exception as e:
            logging.error(f"Critical error processing {video_path}: {str(e)}")
            continue
            
    logging.info("Preprocessing complete. You can now start training.")