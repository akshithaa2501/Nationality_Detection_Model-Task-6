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
         - tarin
         - val 
         - fairface_label_train.csv
         - fairface_label_val.cs
         - Images
     - RAF-DB/
         - train/
         - test/
         - train_labels.csv
         - test_labels.csv
           
- models/
     - nationality_model.keras
     - age_model.keras
     - emotion_model.keras

- src/
     - gui.py
     - predict_nationality.py
     - predict_age.py
     - predict_emotion.py
     - predict_all.py
     - dress_color.py
     - train_age.py
     - train_emotion.py
     - train_nationality.py

- test_images/

# Install Required Libraries

- Install all dependencies using:

pip install -r requirements.txt

Or 
- install manually:

pip install tensorflow
pip install keras
pip install streamlit
pip install opencv-python
pip install numpy
pip install pandas
pip install pillow
pip install matplotlib
pip install scikit-learn
pip install ultralytics

# Training Process
- Age Model
1. Load FairFace dataset.
2. Preprocess images.
3. Resize images to 224 × 224.
4. Normalize pixel values.
5. Train MobileNetV2 model.
6. Save model as:
    - age_model.keras

- Emotion Model
1. Load RAF-DB dataset.
2. Apply image augmentation.
3. Train MobileNetV2.
4. Save model as:
    - emotion_model.keras

- Nationality Model
1. Load FairFace dataset.
2. Convert race labels into nationality classes.
3. Resize images.
4. Train MobileNetV2.
5. Save model as:
    - nationality_model.keras

- Dress Color Detection

The dress color detection module:

Reads the uploaded image.
Detects the clothing region.
Uses OpenCV and color analysis to determine the dominant dress color.

Supported colors include:

Red
Blue
Green
Yellow
Orange
Purple
Brown
Black
White
Gray


# Running the Project

- Navigate to the project folder:

     cd src

- Run the Streamlit application:

     streamlit run gui.py

The application will automatically open in your default web browser.



# Conclusion

The Nationality Detection Model demonstrates how Deep Learning can be applied to facial analysis tasks. By combining multiple prediction modules with an interactive Streamlit interface, the project provides a practical solution for detecting nationality, age, emotion, and dress color from facial images while satisfying the internship project requirements.
