# Nationality_Detection_Model-Task-6
A Deep Learning-based computer vision application that predicts a person's nationality, age, emotion, and dress color from a facial image using MobileNetV2 and a Streamlit-based graphical user interface.


# Project Overview

The Nationality Detection Model is a Deep Learning-based facial analysis application developed as part of an internship project. The system analyzes a facial image and predicts the person's nationality, age, emotion, and dress color. The application uses trained MobileNetV2-based models for prediction and provides an interactive Streamlit graphical user interface (GUI) for users to upload images and view results.

The project fulfills the internship requirement of displaying different outputs depending on the predicted nationality.

# Project Objectives
Detect the nationality of a person from a facial image.
Predict the age group of the detected person.
Recognize the facial emotion.
Detect the dominant dress color.
Display different outputs based on the predicted nationality.
Provide a simple and user-friendly graphical interface using Streamlit.
T
# Technologies Used
Programming Language
Python 3.11
Frameworks
TensorFlow
Keras
Streamlit
Libraries
OpenCV
NumPy
Pandas
Pillow
Matplotlib
Scikit-learn
Ultralytics (YOLOv8)
MobileNetV2

# Development Environment
Visual Studio Code
Windows 10/11

# Project Workflow
- User Uploads Image

- Image Preprocessing
        
- Nationality Prediction
        
- Age Prediction
        
- Emotion Prediction
        
- Dress Color Detection
        
- Display Output Based on Nationality

# Datasets Used
1. FairFace Dataset

Used for:

Nationality Prediction
Age Prediction

The FairFace dataset provides labeled facial images containing:

Age
Race
Gender

For this project, race labels were mapped into four nationality categories:

Indian
United States
African
Others

2. RAF-DB Dataset

Used for:

Emotion Prediction

Emotion classes:

Angry
Disgust
Fear
Happy
Neutral
Sad
Surprise

# Model Architecture

All prediction models are based on MobileNetV2 using transfer learning.

- Nationality Model
Base Model: MobileNetV2
Dataset: FairFace
Output Classes:
Indian
United States
African
Others
Age Model
- Base Model: MobileNetV2
- Dataset: FairFace
Output: Age Groups
- Emotion Model
- Base Model: MobileNetV2
- Dataset: RAF-DB
Output: Seven emotions

# Folder Structure
- Nationality_Detection_Model
- dataset/
      - FairFace/
             - fairface_label_train.csv
             - fairface_label_val.cs
             - Images
      - RAF-DB/
             - train/
             - test/
- models/
     - nationality_model.keras
     - age_model.keras
     - emotion_model.keras

- src/
      - gui.py
      - predict_nationality.py
      - predict_age.py
      - predict_emotion.py
      - dress_color.py
      - utils.py

- test_images/
