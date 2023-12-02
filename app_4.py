#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import cv2
import dlib
import numpy as np
import time
import pygame

def calculate_eye_aspect_ratio(eye_landmarks):
    vertical_dist1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    vertical_dist2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    horizontal_dist = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    aspect_ratio = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)
    return aspect_ratio

def play_sound(filename):
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

left_eye_landmarks = [36, 37, 38, 39, 40, 41]
right_eye_landmarks = [42, 43, 44, 45, 46, 47]

start_time_left = None
start_time_right = None
eye_closed_duration_threshold = 2

# Initialize Streamlit app
st.title("Eye Blink Detection with Streamlit")
st.session_state['answer'] = ''!

st.write(st.session_state)

realans = ['', 'abc', 'edf']

if  st.session_state['answer'] in realans:
    answerStat = "correct"
elif st.session_state['answer'] not in realans:
    answerStat = "incorrect"

st.write(st.session_state)
st.write(answerStat)
# Function to display video stream and eye blink detection results
def display_video():
    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray)
        for face in faces:
            landmarks = predictor(gray, face)
            
            left_eye_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in left_eye_landmarks], dtype=np.int32)
            aspect_ratio_left = calculate_eye_aspect_ratio(left_eye_points)

            right_eye_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in right_eye_landmarks], dtype=np.int32)
            aspect_ratio_right = calculate_eye_aspect_ratio(right_eye_points)

            if aspect_ratio_left < 0.2 or aspect_ratio_right < 0.2:
                if aspect_ratio_left < 0.2 and start_time_left is None:
                    start_time_left = time.time()
                elif aspect_ratio_right < 0.2 and start_time_right is None:
                    start_time_right = time.time()
                else:
                    elapsed_time_left = time.time() - start_time_left if start_time_left is not None else 0
                    elapsed_time_right = time.time() - start_time_right if start_time_right is not None else 0

                    if elapsed_time_left > eye_closed_duration_threshold or elapsed_time_right > eye_closed_duration_threshold:
                        st.error("ALERT! Eyes Closed")
                        play_sound("alarm_1.wav")

            else:
                start_time_left = None
                start_time_right = None

        # Display video frame
#         st.image(frame, channels="BGR", use_column_width=True)

# Display the video stream and eye blink detection results in the Streamlit app
if __name__ == "__main__":
    display_video()


# In[ ]:




