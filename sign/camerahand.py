import os
import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the trained model
model = load_model('smnist.h5')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get frame dimensions
_, frame = cap.read()
h, w, _ = frame.shape

# Define the letters for prediction
letterpred = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

# Variable to store the predicted text
predicted_text = ""

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    k = cv2.waitKey(1)
    if k % 256 == 27:  # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:  # SPACE pressed
        analysis_frame = frame.copy()
        cv2.imshow("Frame", analysis_frame)

        # Process the frame for hand landmarks
        framergb_analysis = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2RGB)
        result_analysis = hands.process(framergb_analysis)
        hand_landmarks_analysis = result_analysis.multi_hand_landmarks

        if hand_landmarks_analysis:
            for handLMs_analysis in hand_landmarks_analysis:
                x_max = 0
                y_max = 0
                x_min = w
                y_min = h
                for lm_analysis in handLMs_analysis.landmark:
                    x, y = int(lm_analysis.x * w), int(lm_analysis.y * h)
                    x_max = max(x, x_max)
                    x_min = min(x, x_min)
                    y_max = max(y, y_max)
                    y_min = min(y, y_min)

                # Adjust bounding box
                y_min -= 20
                y_max += 20
                x_min -= 20
                x_max += 20

                # Ensure bounding box is within frame dimensions
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(w, x_max)
                y_max = min(h, y_max)

                # Extract the region of interest (ROI) for analysis
                analysis_frame = frame[y_min:y_max, x_min:x_max]

                # Check if the region is valid (not empty)
                if analysis_frame.size == 0:
                    print("Invalid region for resizing")
                    continue

                # Convert to grayscale and resize
                analysis_frame = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2GRAY)
                analysis_frame = cv2.resize(analysis_frame, (28, 28))

                # Prepare input for the model
                nlist = analysis_frame.flatten() / 255.0  # Normalize pixel values
                pixeldata = nlist.reshape(-1, 28, 28, 1)  # Reshape for model input

                # Make prediction
                prediction = model.predict(pixeldata)
                predicted_class = np.argmax(prediction)
                predicted_letter = letterpred[predicted_class]
                predicted_probability = np.max(prediction)

                if predicted_probability > 0.7:  # Adjust this threshold based on your model's performance
                    predicted_letter = letterpred[predicted_class]
                    # Append the predicted letter to the accumulated text
                    predicted_text += predicted_letter
                else:
                    predicted_letter = ""  # Reset if confidence is low

                # Display the predicted letter on the frame
                cv2.putText(frame, predicted_letter, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the predicted text continuously
    cv2.putText(frame, predicted_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the frame with hand landmarks
    cv2.imshow("Frame", frame)

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
