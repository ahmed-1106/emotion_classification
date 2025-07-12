# emotion_classification

# Emotion Detection System

A deep learning system for real-time emotion detection using facial expressions. The project consists of two main components:
1. `test_feelings.py` - Training script for the emotion detection model
2. `run_feelings.py` - Real-time emotion detection using webcam

## Table of Contents
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [License](#license)

## Project Structure
emotion-detection/
├── test_feelings.py # Training script
├── run_feelings.py # Real-time detection script
├── emotion_model.pth # Trained model weights
├── train/ # Training dataset directory
│ ├── angry/ # Angry emotion images
│ ├── happy/ # Happy emotion images
│ └── neutral/ # Neutral emotion images
└── test/ # Test dataset directory
├── angry/
├── happy/
└── neutral/
## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/emotion-detection.git
cd emotion-detection
Install required dependencies:
pip install torch torchvision opencv-python pandas numpy pillow matplotlib
Usage
1. Training the Model (test_feelings.py)
This script trains the emotion detection model on your dataset.

python test_feelings.py
Requirements:

Dataset folders (train/ and test/) with subdirectories for each emotion class

CUDA-enabled GPU recommended for faster training

Features:

Custom Dataset classes for loading images

Data augmentation and normalization

Training loop with loss tracking

Model saving functionality

2. Real-time Detection (run_feelings.py)
This script runs real-time emotion detection using your webcam.
python run_feelings.py
Requirements:

Trained model file (emotion_model.pth)

Webcam

OpenCV with Haar cascades

Features:

Real-time face detection

Emotion classification

Visual feedback with bounding boxes and labels

Press 'q' to quit

Model Architecture
The system uses a CNN (Convolutional Neural Network) with the following architecture:

EmotionCNN(
  (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (dropout): Dropout(p=0.25, inplace=False)
  (fc1): Linear(in_features=9216, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=3, bias=True)
)

Dataset
The model is trained to recognize three emotion classes:

Angry 😠

Happy 😊

Neutral 😐

Dataset Structure:

Images should be organized in separate folders for each emotion class

Supported image formats: JPG, JPEG, PNG

Recommended image size: 48x48 pixels (grayscale)

License
This project is licensed under the MIT License - see the LICENSE file for details.








