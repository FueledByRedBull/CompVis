# Facial Expression Analyzer

A real-time facial expression analysis system using a 3-model ensemble approach with dlib's 68-point facial landmarks for enhanced accuracy.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-~83%25-orange.svg)

## Features

- **3-Model Ensemble**: Combines predictions from three HSEmotion models (enet_b2, vgaf, afew) with per-emotion weighted voting
- **68-Point Facial Landmarks**: Uses dlib for precise facial feature analysis
- **Smart Refinements**: Targeted corrections for common confusion patterns (Fear↔Surprise, Sad↔Angry)
- **Modern GUI**: CustomTkinter-based interface with glassmorphism effects, bracket corners, and ghost transparency
- **Live Webcam Support**: Real-time emotion detection from webcam feed
- **Head Pose Estimation**: Detects if subject is facing camera or turned left/right
- **Face Enumeration**: Numbers faces in reading order (top-to-bottom, left-to-right)

## Supported Emotions

| Emotion | Description |
|---------|-------------|
| Happy | Happiness, joy, smiling |
| Sad | Sadness, melancholy |
| Angry | Anger, frustration |
| Fear | Fear, anxiety |
| Surprise | Surprise, astonishment |
| Disgust | Disgust, displeasure |
| Neutral | Calm, neutral state |
| Contempt | Contempt, disdain |

## Installation

### Prerequisites

- Python 3.10 or higher
- CMake (required for dlib compilation)
- Visual Studio Build Tools (Windows) or GCC (Linux/Mac)

### Step 1: Clone the Repository

```bash
git clone https://github.com/FueledByRedBull/CompVis.git
cd CompVis
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Shape Predictor

Download the dlib 68-point facial landmark predictor:

1. Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
2. Extract the `.bz2` file
3. Place `shape_predictor_68_face_landmarks.dat` in the project root directory

## Usage

### GUI Application (Recommended)

```bash
python gui_app.py
```

**Features:**
- Click "Select Folder" to choose a directory of images
- Use "Start Webcam" for real-time analysis
- Navigate between images with Previous/Next buttons
- View detailed emotion breakdowns in the results panel

### Command Line Interface

**Analyze a directory:**
```bash
python main.py ./photos
```

**Save results to output directory:**
```bash
python main.py ./photos --output ./results
```

**Save annotated images:**
```bash
python main.py ./photos --output ./results --save-annotated
```

**Analyze a single image:**
```bash
python main.py ./photo.jpg --single
```

### Programmatic Usage

```python
from emotion_analyzer import EmotionAnalyzer
import cv2

# Initialize analyzer
analyzer = EmotionAnalyzer()

# Load and analyze image
image = cv2.imread("photo.jpg")
results = analyzer.analyze_image(image)

# Process results
for face in results:
    print(f"Face #{face['face_number']}: {face['dominant_emotion']} ({face['confidence']:.1f}%)")
    print(f"  Head pose: {face['head_pose']}")
    print(f"  Model agreement: {face['backend_agreement']:.0f}%")
```

## Project Structure

```
CompVis/
├── emotion_analyzer.py      # Core analysis engine
├── gui_app.py               # CustomTkinter GUI application
├── main.py                  # CLI interface
├── requirements.txt         # Python dependencies
├── shape_predictor_68_face_landmarks.dat  # dlib landmark model (download separately)
├── README.md                # This file
├── report.md                # Detailed project analysis
└── LICENSE                  # MIT License
```

## Technical Details

### Model Ensemble

The system uses three HSEmotion ONNX models:

| Model | Strengths |
|-------|-----------|
| enet_b2 | Best overall, especially Happy/Neutral |
| vgaf | Natural expressions (Sad, Surprise) |
| afew | Acted expressions (Fear, Angry, Disgust) |

Each emotion uses optimized weights:
```python
EMOTION_WEIGHTS = {
    'angry':    {'enet_b2': 0.3, 'vgaf': 0.2, 'afew': 0.5},
    'happy':    {'enet_b2': 0.5, 'vgaf': 0.3, 'afew': 0.2},
    'surprise': {'enet_b2': 0.3, 'vgaf': 0.5, 'afew': 0.2},
    # ... etc
}
```

### Landmark Analysis

Uses dlib's 68-point facial landmarks for:
- **Mouth Opening**: Detects open mouth for Surprise vs Fear
- **Mouth Corners**: Detects smile/frown for Happy vs Sad vs Angry
- **Head Pose**: Estimates yaw from eye centers and nose tip

### Refinement Logic

Post-processing corrections for common confusions:
- **Fear → Surprise**: If mouth is open, bias toward Surprise (more common in photos)
- **Sad → Angry**: If mouth isn't downturned, bias toward Angry
- **Majority Override**: If 2/3 models agree, boost that emotion

## Performance

| Metric | Value |
|--------|-------|
| Basic Emotions Accuracy | ~83% |
| Face Detection | dlib HOG |
| Models Used | 3 (ensemble) |
| Landmark Points | 68 |
| GUI Framework | CustomTkinter |

## Requirements

```
opencv-python>=4.8.0
numpy>=1.24.0
Pillow>=10.0.0
hsemotion-onnx>=0.3
customtkinter>=5.2.0
dlib>=19.24.0
```

## Troubleshooting

### "Shape predictor not found"
Download `shape_predictor_68_face_landmarks.dat` from dlib.net and place it in the project directory.

### dlib installation fails
Ensure CMake is installed:
- Windows: Download from cmake.org or `choco install cmake`
- Linux: `sudo apt install cmake`
- Mac: `brew install cmake`

### Low accuracy on certain emotions
The system is optimized for posed photographs. Genuine subtle emotions may be less accurate.

### Webcam not detected
Ensure no other application is using the webcam. Try restarting the application.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [HSEmotion](https://github.com/HSE-asavchenko/face-emotion-recognition) for the emotion recognition models
- [dlib](http://dlib.net/) for facial landmark detection
- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) for the modern GUI framework
