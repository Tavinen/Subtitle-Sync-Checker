Subtitle Sync Checker Model

📝 Description

Subtitle Sync Checker is a deep learning-based tool designed to automatically verify whether subtitles are synchronized with the spoken audio in video files. It uses a hybrid multimodal neural network that analyzes both the visual frames (to detect subtitle overlays) and the audio track (to extract speech features). The model is optimized for AMD GPUs and supports both CPU and GPU processing. This project is ideal for anime, movies, or any video content where subtitle accuracy is crucial.

Visual branch: Processes 32 evenly spaced video frames (224×224 pixels) per sample.

Audio branch: Extracts MFCC features from the audio track for robust speech analysis.

Hybrid loss function: Balances audio and visual cues for improved accuracy.

Performance: Supports batch processing, memory monitoring, and mixed precision for fast and stable training and inference.

🚀 Installation

Requirements:

Python 3.8+

TensorFlow (with DirectML or CUDA support for GPU acceleration)

OpenCV

librosa

ffmpeg (must be installed and in your PATH)


Clone the repository:

bash
git clone https://github.com/your-username/subtitle-sync-checker.git
cd subtitle-sync-checker
Install dependencies:

bash
pip install -r requirements.txt
Check ffmpeg installation:

Windows:
winget install ffmpeg

Linux:
sudo apt install ffmpeg

Mac:
brew install ffmpeg

🎯 Usage
1. Preprocess Videos for Training
Prepare your dataset by labeling videos as synced (label=1) or unsynced (label=0). Use the provided preprocessing script:

bash
python preprocess_videos.py
This extracts frames and audio, storing them in training_data/dataset/.

2. Train the Model
bash
python -m deep_learning.model_trainer
The script will automatically split your data, train the hybrid model, and save it to deep_learning/models/trained_model.h5.

3. Check Subtitle Synchronization
To check subtitle sync on new videos:

bash
python subtitle_sync_checker.py
Use the GUI to add video files and start the check.

The tool will color-code results for easy review (green: synced, red: failed).

4. Programmatic Inference
You can also use the model in your own scripts:

python
from deep_learning.sync_predictor import HybridPredictor

predictor = HybridPredictor("deep_learning/models/trained_model.h5")
score = predictor.model.predict([frames_batch, audio_batch])[0][0]
print("Sync score:", score)
frames_batch shape: (1, 32, 224, 224, 3)

audio_batch shape: (1, 40, 100)

♿ Accessibility & Usability
Clear headings and bold important steps.

Short sentences and bulleted lists for readability.

Color-coded status in the GUI for quick understanding.

Font recommendations: Use dyslexia-friendly fonts like Arial, Verdana, or OpenDyslexic in the GUI settings.

High-contrast colors are used for status indicators (green = synced, red = failed, blue = checking).

🛠️ Troubleshooting
ffmpeg not found:
Make sure ffmpeg is installed and available in your system PATH.

GPU errors:
Ensure you have the correct drivers and TensorFlow version for your hardware.

Memory issues:
The tool monitors system resources and will pause if memory is low.

📚 More Information
Model architecture and training details: See deep_learning/model_trainer.py and hybrid_config.py.

Custom loss function: See deep_learning/losses.py.

GUI usage: Run subtitle_sync_checker.py and follow on-screen instructions.

Tip: For best accessibility, increase font size in your terminal or GUI, and use the color-blind mode in your OS if needed.

Feel free to open an issue or pull request if you have suggestions or need help!
