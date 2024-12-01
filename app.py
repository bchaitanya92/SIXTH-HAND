from flask import Flask, jsonify, request
import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model
import os

app = Flask(__name__)

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the trained model
model = load_model('smnist.h5')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Define the letters for prediction
letterpred = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

@app.route('/recognize', methods=['POST'])
def recognize_sign_language():
    # Start video capture
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        return jsonify({'error': 'Could not access webcam'})

    predicted_text = "apple"

    for _ in range(50):  # Process for a limited number of frames
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame for hand landmarks
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(framergb)

        if result.multi_hand_landmarks:
            for handLMs in result.multi_hand_landmarks:
                x_max, y_max, x_min, y_min = 0, 0, frame.shape[1], frame.shape[0]

                for lm in handLMs.landmark:
                    x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                    x_max, x_min = max(x, x_max), min(x, x_min)
                    y_max, y_min = max(y, y_max), min(y, y_min)

                # Extract and preprocess ROI
                roi = frame[y_min:y_max, x_min:x_max]
                if roi.size == 0:
                    continue

                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                roi_resized = cv2.resize(roi_gray, (28, 28))
                pixeldata = roi_resized.flatten().reshape(-1, 28, 28, 1) / 255.0

                # Predict using the model
                prediction = model.predict(pixeldata)
                predicted_class = np.argmax(prediction)
                predicted_letter = letterpred[predicted_class]

                if np.max(prediction) > 0.7:  # Threshold
                    predicted_text += predicted_letter

    cap.release()
    return jsonify({'recognized_text': predicted_text})

if __name__ == '__main__':
    app.run(debug=True)
