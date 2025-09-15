# Sign Language Detection using LSTM and MediaPipe
 Real-time Sign Language Recognition system built using Python, MediaPipe, OpenCV, and TensorFlow.
 This project detects and classifies human sign language gestures (like Hello, Thanks, I Love You) in real-time webcam feed, using keypoint landmark extraction and an LSTM deep learning model.

# Demo


# Tech Stack
 - Python
 - TensorFlow (LSTM Neural Network)
 - MediaPipe Holistic – Face, Pose, and Hand landmarks
 - OpenCV – Video frame processing
 - scikit-learn – Data preprocessing & evaluation
 - NumPy, Matplotlib – Utilities and visualization

# Project Workflow
 1. Import and Install Dependencies
  pip install tensorflow==2.4.1 opencv-python mediapipe-silicon sklearn matplotlib

  import cv2, numpy as np, os, time 
  import mediapipe as mp 
  from matplotlib import pyplot as plt

2. Keypoint Extraction using MediaPipe Holistic
 - Detects face, pose, left-hand, right-hand landmarks on webcam frames
 - Draws styled landmarks using mp_drawing utils
 - Converts each frame → 1662-dim feature vector

3. Data Collection & Preprocessing
 - Collects 30 videos × 30 frames per action
 - Saves each frame's keypoints as .npy files
 - Prepares features and labels for model training
 - Splits into training and testing sets

4. Build & Train LSTM Model
 - LSTM architecture with 3 LSTM layers and Dense layers
 - Trained to classify actions: hello, thanks, iloveyou
 - Compiled using categorical_crossentropy loss and Adam optimizer

  model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])

5. Evaluate Model
 - Confusion Matrix and Accuracy Score using sklearn
 - Achieves high accuracy on test set

6. Real-Time Prediction
 - Uses webcam to get live feed
 - Predicts gesture in real-time
 - Displays predicted sign + live confidence bars on the screen

# Project Structure

`SignLanguageDetection/
 │
 ├── MP_Data/                 # Collected keypoint data
 │   ├── hello/
 │   ├── thanks/
 │   └── iloveyou/
 │
 ├── Logs/                    # TensorBoard logs
 ├── action.h5                # Saved LSTM model
 ├── sign_language.ipynb      # Main project notebook
 └── README.md`

# Getting Started
 - Clone this repository
 git clone https://github.com/badri-sirimalla/Sign-Language-Detection-Project.git
 cd Sign-Language-Detection

# Run the Notebook
 - Open sign_language.ipynb in VS Code or Jupyter
 - Execute cells step-by-step from Dependencies → Real-Time Testing

# Results
| Action     | Accuracy |
| ---------- | -------- |
| Hello      | 97%      |
| Thanks     | 95%      |
| I Love You | 94%      |

# Future Improvements
 - Add more gestures
 - Deploy as a web application
 - Integrate with real-time translation

# Contributions
 Contributions, issues, and feature requests are welcome.
 Feel free to fork this repo and submit a pull request.

# Author
 - Badri Sirimalla
 - badrisirimalla2003@gmail.com











