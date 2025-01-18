Group Emotion Detection and Music Recommendation
Overview
This project focuses on detecting group emotions from facial expressions using deep learning and recommending music tailored to the detected emotions. It integrates a Convolutional Neural Network (CNN) for emotion detection and a music recommendation system for therapeutic purposes.

Features
Emotion Detection: Detects emotions with a trained CNN model using the FER2013 dataset.
Music Recommendation: Provides personalized playlists based on detected emotions.
AI & Deep Learning: Utilizes data augmentation and hyperparameter tuning to enhance performance.
Project Achievements
Training Accuracy: Achieved 97.42% accuracy for emotion classification.
Dataset: Trained on over 28,000 grayscale images from the FER2013 dataset.
Application: Combines AI-based emotion detection with music recommendations to promote emotional well-being.

Folder Structure

.
├── .ipynb_checkpoints/    # Checkpoints for Jupyter Notebooks
├── app.py                 # Main script to run the Streamlit app
├── emotion_models.h5      # Trained CNN model for emotion detection
├── Image/                 # input image
├── musics/                # Contains music files for different emotions
└── train/                 # Training data and scripts includes [ happy,sad,angry,neutral]

Running the Project

Clone the repository:
Copy code
git clone <repository-url>
cd <repository-folder>


Install required dependencies:
bash
Copy code
pip install -r requirements.txt


Run the Streamlit app:
bash
Copy code
streamlit run app.py


How It Works
Input: Upload a group image via the Streamlit app.
Emotion Detection: The CNN model (emotion_models.h5) processes the image to classify emotions.
Music Recommendation: Based on the detected group emotion, the system recommends appropriate music from the musics/ folder.


Key Technologies
AI and Deep Learning
Computer Vision
Python (Streamlit, Keras, TensorFlow)

Installation Requirements
Python 3.7+
Libraries: TensorFlow, Keras, Streamlit, OpenCV, NumPy, Pandas


Dataset
FER2013: A dataset of facial emotion recognition containing 28,000+ grayscale images.
Applications
Therapeutic and mental health support.
Personalized music systems for relaxation and entertainment.
