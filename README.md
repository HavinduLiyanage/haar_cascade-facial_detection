# Face Recognition System using OpenCV

This project implements a simple face recognition system using the Local Binary Patterns Histogram (LBPH) algorithm with OpenCV. It includes two main scripts: one for training the recognizer (`faces_train.py`) and one for recognizing faces from images (`face_reco.py`).

## Features

- **Face Training**: Train the model using a set of face images organized by person.
- **Face Recognition**: Identify faces in images using the trained model.
- **Haar Cascade Classifier**: Detects faces in images before recognition.
- **LBPH Algorithm**: A powerful and simple algorithm for face recognition.

## Requirements

- Python 3.x
- OpenCV
- NumPy

Install dependencies using:

```bash
pip install opencv-python numpy
