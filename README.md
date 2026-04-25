# Facial Expression Analyzer

A real-time facial expression analysis system using a 3-model ensemble approach with dlib's 68-point facial landmarks for enhanced accuracy.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-~83%25-orange.svg)

## Features

- **3-Model Ensemble**: Combines predictions from three HSEmotion models (enet_b2, vgaf, afew) with per-emotion weighted voting
- **68-Point Facial Landmarks**: Uses dlib for precise facial feature analysis
- **Smart Refinements**: Targeted corrections for common confusion patterns (Fear/Surprise, Sad/Angry)
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

## Privacy and Consent

Run webcam and image analysis only on media you are allowed to process. Emotion labels are model estimates, not clinical or identity claims, and should not be used for high-stakes decisions.

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/FueledByRedBull/CompVis.git
cd CompVis

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
# Note: requirements.txt uses dlib-bin (pre-built, Windows-optimized)
# Linux/Mac users may need: pip install dlib (requires CMake + C++ compiler)
pip install -r requirements.txt

# Download shape predictor
# Get from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# Extract and place in project root

# Run GUI
python gui_app.py
```

---

## Installation

### Prerequisites

| Platform | Requirements |
|----------|-------------|
| Windows | Python 3.10+, Visual Studio Build Tools, CMake |
| Linux | Python 3.10+, build-essential, cmake |
| macOS | Python 3.10+, Xcode Command Line Tools, cmake |

### Windows Installation

#### Step 1: Install Visual Studio Build Tools

1. Download **Visual Studio Build Tools** from:
   https://visualstudio.microsoft.com/visual-cpp-build-tools/

2. Run the installer and select:
   - **"Desktop development with C++"** workload
   - Make sure **Windows 10/11 SDK** is checked
   - Make sure **MSVC v143 (or latest)** is checked

3. Click **Install** and wait for completion (may take 10-20 minutes)

4. **Restart your computer** after installation

#### Step 2: Install CMake

**Option A: Download installer (Recommended)**
1. Download from: https://cmake.org/download/
2. Choose **"Windows x64 Installer"**
3. During installation, select **"Add CMake to the system PATH"**

**Option B: Using package managers**
```bash
# Using Chocolatey
choco install cmake

# Using winget
winget install Kitware.CMake
```

**Verify CMake installation:**
```bash
cmake --version
# Should show: cmake version 3.x.x
```

#### Step 3: Create Virtual Environment

```bash
# Navigate to project directory
cd CompVis

# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate
```

#### Step 4: Install Python Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** dlib compilation may take 5-10 minutes. This is normal.

#### Step 5: Download Shape Predictor

1. Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
2. Extract the `.bz2` file (use 7-Zip or similar)
3. Place `shape_predictor_68_face_landmarks.dat` in the project root directory

---

### Linux Installation

#### Step 1: Install Build Tools

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install build-essential cmake python3-dev python3-venv

# Fedora
sudo dnf install gcc gcc-c++ cmake python3-devel

# Arch Linux
sudo pacman -S base-devel cmake python
```

#### Step 2: Create Virtual Environment

```bash
cd CompVis
python3 -m venv venv
source venv/bin/activate
```

#### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

#### Step 4: Download Shape Predictor

```bash
# Download and extract
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```

---

### macOS Installation

#### Step 1: Install Xcode Command Line Tools

```bash
xcode-select --install
```

#### Step 2: Install CMake

```bash
# Using Homebrew (recommended)
brew install cmake

# Or download from cmake.org
```

#### Step 3: Create Virtual Environment

```bash
cd CompVis
python3 -m venv venv
source venv/bin/activate
```

#### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

#### Step 5: Download Shape Predictor

```bash
# Download and extract
curl -O http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```

---

## Troubleshooting dlib Installation

### "CMake not found" or "cmake is not recognized"

**Windows:**
1. Ensure CMake is installed from cmake.org
2. During installation, check "Add CMake to system PATH"
3. Restart your terminal/command prompt
4. If still failing, manually add CMake to PATH:
   - Open System Properties > Environment Variables
   - Add `C:\Program Files\CMake\bin` to PATH

**Linux/Mac:**
```bash
# Verify cmake is in PATH
which cmake
cmake --version
```

### "No Visual Studio C++ Build Tools found" (Windows)

1. Ensure you installed **"Desktop development with C++"** workload
2. Restart your computer after installation
3. Open a **new** command prompt after restart
4. Try installing in **Developer Command Prompt for VS**:
   - Search for "Developer Command Prompt" in Start Menu
   - Run `pip install dlib` from there

### "error: Microsoft Visual C++ 14.0 or greater is required"

1. Install/repair Visual Studio Build Tools with C++ workload
2. Make sure Windows SDK is included
3. Restart computer and try again

### dlib compilation takes too long

This is normal - dlib compiles from source and may take 5-15 minutes depending on your system. The compilation only happens once.

### Alternative: Pre-built dlib wheel

If compilation fails, try a pre-built wheel:

```bash
pip install dlib-bin
```

> **Note:** Pre-built wheels may not be available for all Python versions.

---

## Docker (Recommended for Easy Setup)

Skip all the dlib compilation hassle by using Docker:

```bash
# Build the image (downloads shape predictor automatically)
docker build -t emotion-analyzer .

# Analyze images in a folder
docker run -v /path/to/images:/data emotion-analyzer python main.py /data

# Save results
docker run -v /path/to/images:/data -v /path/to/output:/output \
    emotion-analyzer python main.py /data --output /output --save-annotated
```

> **Note:** GUI mode (`gui_app.py`) requires X11 forwarding and is not recommended in Docker.

---

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

```bash
# Analyze a directory
python main.py ./photos

# Save results to output directory
python main.py ./photos --output ./results

# Save annotated images
python main.py ./photos --output ./results --save-annotated

# Analyze a single image
python main.py ./photo.jpg --single

# Quiet mode (less output)
python main.py ./photos --quiet
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

---

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

---

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

---

## Performance

| Metric | Value |
|--------|-------|
| Basic Emotions Accuracy | ~83% |
| Face Detection | dlib HOG |
| Models Used | 3 (ensemble) |
| Landmark Points | 68 |
| GUI Framework | CustomTkinter |

---

## Requirements

```
opencv-python>=4.8.0
numpy>=1.24.0
Pillow>=10.0.0
hsemotion-onnx>=0.3
customtkinter>=5.2.0
dlib-bin>=19.24.0  # Windows pre-built; Linux/Mac: pip install dlib
```

> **Platform Note:** The `requirements.txt` uses `dlib-bin` which provides pre-built wheels for Windows. Linux/Mac users should replace with `dlib` (requires CMake and C++ compiler).

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [HSEmotion](https://github.com/HSE-asavchenko/face-emotion-recognition) for the emotion recognition models
- [dlib](http://dlib.net/) for facial landmark detection
- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) for the modern GUI framework
