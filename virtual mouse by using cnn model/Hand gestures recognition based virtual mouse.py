import cv2
import os

# Define the gestures and the number of images to capture for each gesture
gestures = ['click', 'right_click', 'scroll_up', 'scroll_down']
num_images_per_gesture = 1000

# Create a directory for storing the collected data
data_dir = 'data'
os.makedirs(data_dir, exist_ok=True)

# Function to capture images for a specific gesture
def capture_images_for_gesture(gesture, num_images):
    gesture_dir = os.path.join(data_dir, gesture)
    os.makedirs(gesture_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(f"Error: Could not open camera for {gesture}.")
        return

    print(f"Collecting images for gesture: {gesture}")
    img_num = 0
    while img_num < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Display the frame
        cv2.imshow('Frame', frame)

        # Save the frame to the appropriate directory
        img_path = os.path.join(gesture_dir, f'{img_num}.jpg')
        cv2.imwrite(img_path, frame)

        img_num += 1
        print(f"Captured image {img_num}/{num_images} for gesture '{gesture}'", end='\r')

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinished capturing images for gesture: {gesture}")

# Main loop to collect data for each gesture
for gesture in gestures:
    capture_images_for_gesture(gesture, num_images_per_gesture)

print("Data collection complete.")


import cv2
import numpy as np
import os
from sklearn.preprocessing import LabelBinarizer

gestures = ['click', 'right_click', 'scroll_up', 'scroll_down']
data = []
labels = []

for gesture in gestures:
    images = os.listdir(f'data/{gesture}')
    for img in images:
        img_path = os.path.join(f'data/{gesture}', img)
        image = cv2.imread(img_path)
        image = cv2.resize(image, (64, 64))  # Resize images to a fixed size
        data.append(image)
        labels.append(gesture)

data = np.array(data, dtype='float32') / 255.0  # Normalize images
labels = np.array(labels)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

pip install tensorflow

"""Building and Training the CNN Model"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(gestures), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val), batch_size=32)

# Save the model
model.save('virtual_mouse_model.h5')

pip install mediapipe

pip install pyautogui



import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Get the size of the screen
screen_width, screen_height = pyautogui.size()

# Function to map the finger positions to screen coordinates
def map_coordinates(x, y):
    """Map the normalized position of the finger to screen position."""
    screen_x = np.interp(x, [0, 1], [0, screen_width])
    screen_y = np.interp(y, [0, 1], [0, screen_height])
    return screen_x, screen_y

# Function to move the mouse safely within screen boundaries
def safe_move_to(x, y, buffer=10):
    """Move the cursor to (x, y) coordinates while ensuring it stays within a safe buffer from screen edges."""
    safe_x = np.clip(x, buffer, screen_width - buffer)
    safe_y = np.clip(y, buffer, screen_height - buffer)
    pyautogui.moveTo(safe_x, safe_y)

# Function to calculate the Euclidean distance between two landmarks
def calculate_distance(landmark1, landmark2):
    return np.sqrt((landmark1.x - landmark2.x) ** 2 + (landmark1.y - landmark2.y) ** 2)

# OpenCV capture device setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Initialize variables for debouncing actions
last_click_time = 0
last_scroll_time = 0
debounce_time = 0.5  # 500ms debounce time

# Variables to calculate FPS
fps = 0
frame_count = 0
start_time = time.time()

# Main loop for processing video input
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    frame_count += 1
    current_time = time.time()
    elapsed_time = current_time - start_time

    # Calculate FPS every second
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = current_time

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # Convert back to BGR for displaying
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    current_action = "None"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get fingertip positions
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

            # Move cursor with the index finger
            index_x, index_y = map_coordinates(index_tip.x, index_tip.y)
            safe_move_to(index_x, index_y)

            current_time = time.time()

            # Detect clicks and scrolls
            if calculate_distance(thumb_tip, index_tip) < 0.05 and current_time - last_click_time > debounce_time:
                pyautogui.click()
                current_action = "Left Click"
                last_click_time = current_time
            elif calculate_distance(thumb_tip, middle_tip) < 0.05 and current_time - last_click_time > debounce_time:
                pyautogui.click(button='right')
                current_action = "Right Click"
                last_click_time = current_time

            if calculate_distance(thumb_tip, pinky_tip) < 0.05 and current_time - last_scroll_time > debounce_time:
                pyautogui.scroll(25)  # Scroll up
                current_action = "Scroll Up"
                last_scroll_time = current_time
            elif calculate_distance(thumb_tip, ring_tip) < 0.05 and current_time - last_scroll_time > debounce_time:
                pyautogui.scroll(-25)  # Scroll down
                current_action = "Scroll Down"
                last_scroll_time = current_time

    # Display the image with annotations and debug information
    cv2.putText(image_bgr, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(image_bgr, f'Action: {current_action}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('MediaPipe Hands', image_bgr)
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Clean up
cap.release()
cv2.destroyAllWindows()

