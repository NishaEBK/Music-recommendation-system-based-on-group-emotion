import streamlit as st
import cv2
import numpy as np
import os
import pygame
from tensorflow.keras.models import load_model

# Initialize pygame mixer for audio playback
pygame.mixer.init()

# Define the directory where your music files are located
music_folder = 'musics'

# Load pre-trained emotion recognition model
emotion_model = load_model('emotion_models.h5')

# Define simplified emotion labels
emotion_labels = ['Happy', 'Sad', 'Angry', 'Neutral']

# Function to preprocess image for emotion recognition
def preprocess_image(image):
    resized_image = cv2.resize(image, (48, 48))
    grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    normalized_image = grayscale_image / 255.0
    processed_image = np.expand_dims(normalized_image, axis=-1)
    processed_image = np.expand_dims(processed_image, axis=0)
    return processed_image

# Function to predict emotion from face image
def predict_emotion(image):
    preprocessed_image = preprocess_image(image)
    emotion_probabilities = emotion_model.predict(preprocessed_image)
    predicted_emotion_index = np.argmax(emotion_probabilities)
    if predicted_emotion_index < len(emotion_labels):
        return emotion_labels[predicted_emotion_index]
    else:
        return "Unknown"

# Function to aggregate emotions from detected faces
def aggregate_emotions(detected_emotions):
    emotion_counts = {emotion: 0 for emotion in emotion_labels}
    for emotion in detected_emotions:
        if emotion in emotion_labels:
            emotion_counts[emotion] += 1
    dominant_emotion = max(emotion_counts, key=emotion_counts.get)
    return dominant_emotion

# Function to create playlist based on overall group emotion
def create_playlist_based_on_emotion(emotion):
    playlist = []
    emotion_folder = os.path.join(music_folder, emotion.lower())
    if os.path.isdir(emotion_folder):
        for root, _, files in os.walk(emotion_folder):
            for file in files:
                if file.endswith('.mp3'):
                    playlist.append(os.path.join(root, file))
    return playlist

# Function to stop music playback
def stop_music():
    pygame.mixer.music.stop()
    st.write("Music playback stopped.")

def main():
    st.title("Emotion-Based Music Player")
    
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Detect faces in the image
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Process each detected face
        detected_emotions = []
        for (x, y, w, h) in faces:
            face_image = image[y:y+h, x:x+w]
            emotion = predict_emotion(face_image)
            detected_emotions.append(emotion)
            cv2.putText(image, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Aggregate emotions of detected faces
        if detected_emotions:
            overall_emotion = aggregate_emotions(detected_emotions)
            st.write(f"Overall Emotion: {overall_emotion}")
            playlist = create_playlist_based_on_emotion(overall_emotion)
            st.write(f"Playlist for {overall_emotion}:")
            for i, song in enumerate(playlist, start=1):
                st.write(f"{i}. {os.path.basename(song)}")

            # Prompt user to select a song from the playlist
            selected_song_index = st.number_input("Enter the number of the song you want to play (0 to stop): ", min_value=0, max_value=len(playlist))
            if selected_song_index != 0:
                selected_song_path = playlist[selected_song_index - 1]
                pygame.mixer.music.load(selected_song_path)
                pygame.mixer.music.play()

            # Prompt user to stop music playback
            stop_option = st.radio("Do you want to stop the music?", ('Yes', 'No'))
            if stop_option == 'Yes':
                stop_music()

        # Display the result
        st.image(image, caption='Emotion-Based Music Player', use_column_width=True)


if __name__ == "__main__":
    main()
