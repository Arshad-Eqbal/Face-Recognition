# Face Recognition System

A real-time face recognition system built with Python, DeepFace, and MTCNN. This system detects, recognizes, and identifies faces from a webcam feed with high accuracy using deep learning models.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![DeepFace](https://img.shields.io/badge/DeepFace-Latest-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📋 Table of Contents

- [Features](#features)
- [System Architecture](#system-architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [How It Works](#how-it-works)
- [Troubleshooting](#troubleshooting)
- [Performance Tips](#performance-tips)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- **Real-time Face Detection**: Uses MTCNN for accurate face detection and alignment
- **Deep Learning Embeddings**: Leverages Facenet for generating 128-dimensional face embeddings
- **K-Nearest Neighbors Classification**: Fast and accurate face identification
- **Data Augmentation**: Automatic horizontal flipping to double training data
- **Confidence Scoring**: Displays confidence percentages for each recognition
- **Live Webcam Processing**: Real-time recognition with optimized frame processing
- **Unknown Face Detection**: Identifies unrecognized faces with configurable threshold

## 🏗️ System Architecture

```
┌─────────────────┐
│   Raw Images    │
│   (Dataset)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessing  │  (MTCNN Detection & Alignment)
│  preprocess.py  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Embeddings    │  (Facenet 128D Vectors)
│  embeddings.py  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  KNN Training   │  (K-Nearest Neighbors)
│ train_model.py  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Real-time     │  (Webcam Recognition)
│   webcam.py     │
└─────────────────┘
```

## 📦 Prerequisites

- Python 3.8 or higher
- Webcam/Camera device
- 4GB+ RAM recommended
- Windows, macOS, or Linux

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/Arshad-Eqbal/Face-Recognition
cd face-recognition-system
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
numpy>=1.21.0
opencv-python>=4.5.0
deepface>=0.0.75
scikit-learn>=1.0.0
tensorflow>=2.8.0
mtcnn>=0.1.1
pillow>=9.0.0
```

Or install individually:

```bash
pip install numpy opencv-python deepface scikit-learn tensorflow mtcnn pillow
```

## 📁 Dataset Preparation

### Directory Structure

Create the following folder structure:

```
project_root/
│
├── dataset/                    # Your raw images go here
│   ├── Person1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── Person2/
│   │   ├── image1.jpg
│   │   └── ...
│   └── ...
│
├── processed_dataset/          # Auto-generated after preprocessing
├── preprocess.py
├── embeddings.py
├── train_classifier.py
├── webcam.py
└── README.md
```

### Dataset Guidelines

1. **Folder Naming**: Create one folder per person with their name
2. **Image Requirements**:
   - At least 5-10 images per person for better accuracy
   - Clear, well-lit face photos
   - Different angles and expressions recommended
   - Supported formats: JPG, PNG, JPEG
3. **Image Quality**:
   - Minimum resolution: 224x224 pixels
   - Face should be clearly visible
   - Avoid heavy occlusions (sunglasses, masks)

## 🎯 Usage

### Step 1: Preprocess Dataset

Detect and align faces from your raw dataset:

```bash
python preprocess.py
```

**What it does:**
- Detects faces using MTCNN
- Aligns faces for consistency
- Creates original + flipped versions (data augmentation)
- Saves to `processed_dataset/` folder

**Expected Output:**
```
Using mtcnn for face detection and alignment...
Processed Person1: 5 source images → 10 total images.
Processed Person2: 8 source images → 16 total images.
Done. Preprocessed dataset ready for embedding generation.
```

### Step 2: Generate Embeddings

Convert preprocessed faces into 128D feature vectors:

```bash
python embeddings.py
```

**What it does:**
- Loads processed face images
- Generates 128D embeddings using Facenet
- Encodes labels numerically
- Saves to `embeddings_facenet.pkl`

**Expected Output:**
```
Generating embeddings using Facenet...
Generated 26 embeddings with shape: (26, 128)
✓ Saved 26 embeddings to embeddings_facenet.pkl
✓ Classes: ['Person1', 'Person2', ...]
```

### Step 3: Train Classifier

Train the K-Nearest Neighbors model:

```bash
python train_classifier.py
```

**What it does:**
- Loads embeddings and labels
- Trains KNN classifier (k=5)
- Saves model to `knn_model.pkl`

**Expected Output:**
```
Loading embeddings...
Training on 26 samples across 3 classes...
Training KNN classifier with 5 neighbors...
✓ KNN classifier trained and saved to knn_model.pkl
✓ Ready for real-time recognition
```

### Step 4: Run Real-Time Recognition

Start the webcam face recognition:

```bash
python webcam.py
```

**What it does:**
- Opens your webcam
- Detects faces in real-time
- Identifies known faces with confidence scores
- Displays results with bounding boxes

**Controls:**
- Press **'q'** to quit

**Expected Output:**
```
✓ Loaded classifier with 3 classes
Starting real-time face recognition...
Press 'q' to quit
```

## 📂 Project Structure

```
face-recognition-system/
│
├── dataset/                        # Raw training images
├── processed_dataset/              # Preprocessed & aligned faces
│
├── preprocess.py                   # Face detection & alignment
├── embeddings.py                   # Feature extraction
├── train_classifier.py             # KNN model training
├── webcam.py                       # Real-time recognition
│
├── embeddings_facenet.pkl          # Generated embeddings
├── knn_model.pkl                   # Trained classifier
│
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## ⚙️ Configuration

### Adjustable Parameters

#### `preprocess.py`
```python
INPUT_DIR = "dataset"              # Raw images folder
OUTPUT_DIR = "processed_dataset"   # Output folder
DETECTOR_BACKEND = "mtcnn"         # Face detector
```

#### `embeddings.py`
```python
INPUT_DIR = "processed_dataset"    # Preprocessed images
MODEL_NAME = "Facenet"             # Embedding model
OUTPUT_FILE = "embeddings_facenet.pkl"
```

#### `train_classifier.py`
```python
N_NEIGHBORS = 5                    # KNN neighbors (3-7 recommended)
```

#### `webcam.py`
```python
CONFIDENCE_THRESHOLD = 0.60        # Recognition threshold (0.0-1.0)
DETECTION_INTERVAL = 5             # Process every Nth frame
MODEL_NAME = "Facenet"             # Embedding model
DETECTOR_BACKEND = "mtcnn"         # Face detector
```

### Tuning Tips

- **Lower confidence threshold** (0.5): More lenient, may have false positives
- **Higher confidence threshold** (0.7): Stricter, fewer false positives
- **Increase DETECTION_INTERVAL**: Better performance, less responsive
- **Decrease DETECTION_INTERVAL**: More responsive, higher CPU usage
- **Adjust N_NEIGHBORS**: Higher = smoother decisions, lower = more sensitive

## 🔬 How It Works

### 1. Face Detection (MTCNN)
MTCNN (Multi-task Cascaded Convolutional Networks) detects faces through three stages:
- **P-Net**: Proposes candidate windows
- **R-Net**: Refines candidates
- **O-Net**: Final detection with facial landmarks

### 2. Face Alignment
Detected faces are aligned using facial landmarks to ensure consistent orientation.

### 3. Feature Extraction (Facenet)
Facenet converts aligned faces into 128-dimensional embeddings where:
- Similar faces have close vectors (small Euclidean distance)
- Different faces have distant vectors (large Euclidean distance)

### 4. Classification (KNN)
K-Nearest Neighbors finds the k=5 closest training embeddings and:
- Uses distance-weighted voting
- Returns the most common identity
- Provides confidence score based on probability distribution

## 🐛 Troubleshooting

### Common Issues

#### "No module named 'cv2'"
```bash
pip install opencv-python
```

#### "No faces detected"
- Ensure good lighting
- Face should be clearly visible
- Try different angles
- Check if images are too small

#### "Low confidence scores"
- Add more training images per person
- Ensure diverse angles and expressions
- Check image quality
- Verify proper lighting

#### "Webcam not opening"
```python
# In webcam.py, try different camera indices:
cap = cv2.VideoCapture(0)  # Try 0, 1, 2, etc.
```

#### "Slow performance"
- Increase `DETECTION_INTERVAL` in webcam.py
- Close other applications
- Use GPU-enabled TensorFlow (if available)

## ⚡ Performance Tips

1. **More Training Data**: 10+ images per person improves accuracy
2. **Quality over Quantity**: Clear, well-lit photos work best
3. **Varied Poses**: Include different angles and expressions
4. **Consistent Lighting**: Similar lighting in training and testing
5. **GPU Acceleration**: Install `tensorflow-gpu` for faster processing
6. **Frame Skipping**: Increase `DETECTION_INTERVAL` for smoother performance

## 📊 Model Alternatives

You can experiment with different embedding models by changing `MODEL_NAME`:

| Model | Embedding Size | Speed | Accuracy |
|-------|---------------|-------|----------|
| Facenet | 128D | Fast | High |
| Facenet512 | 512D | Medium | Very High |
| ArcFace | 512D | Medium | Very High |
| OpenFace | 128D | Fast | Good |
| VGG-Face | 2622D | Slow | High |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [DeepFace](https://github.com/serengil/deepface) - Face recognition framework
- [MTCNN](https://github.com/ipazc/mtcnn) - Face detection
- [FaceNet](https://arxiv.org/abs/1503.03832) - Face embedding model
- [scikit-learn](https://scikit-learn.org/) - Machine learning tools

## 📧 Contact

For questions or support, please open an issue on GitHub.

---
