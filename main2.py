import streamlit as st
import numpy as np
import math
from PIL import Image
import cv2
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Ideal angles for different poses
ideal_angles = {
    'Tree': {
        'right_elbow': 144,
        'right_shoulder': 179,
        'left_shoulder': 171,
        'right_knee': 180,
    },
    'Mountain': {
        'right_elbow': 169,
        'right_shoulder': 269,
        'left_shoulder': 104,
        'right_knee': 114,
    },
    'Warrior2': {
        'right_elbow': 175,
        'right_shoulder': 171,
        'left_shoulder': 194,
        'right_knee': 16,
    }
}

# Helper function to calculate the angle between three points
def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    return round(ang + 360 if ang < 0 else ang)

# Helper function to calculate overall accuracy based on the angle differences
def calculate_accuracy(user_angles, ideal_angles):
    total_difference = 0
    for joint, user_angle in user_angles.items():
        ideal_angle = ideal_angles.get(joint)
        if ideal_angle is not None:
            total_difference += abs(user_angle - ideal_angle)
    
    accuracy = 100 - (total_difference / len(ideal_angles))
    return round(accuracy, 2)

# Streamlit UI and video processing logic
st.set_page_config(layout="wide")
st.sidebar.title('Team karma')
app_mode = st.sidebar.selectbox('Select The Pose', ['Tree', 'Mountain', 'Warrior2'])

col1, col2 = st.columns([2, 3])

with col1:
    st.write(f"{app_mode} Pose")
    image = Image.open(f'{app_mode.lower()}.jpg')
    st.image(image, caption=f'{app_mode} Pose')

with col2:
    st.write("Local Webcam Feed")
    button = st.empty()
    start = button.button('Start')

    if start:
        stop = button.button('Stop')
        FRAME_WINDOW = st.image([])
        instructions_box = st.empty()
        angle_diff_box = st.empty()
        accuracy_box = st.empty()

        # Access the local webcam
        cap = cv2.VideoCapture(0)  # Change from ESP32 stream to local webcam

        if cap.isOpened():
            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Unable to retrieve frame. Retrying...")
                        time.sleep(1)
                        continue

                    h, w, c = frame.shape

                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    results = pose.process(image)
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    # Drawing pose landmarks
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

                    FRAME_WINDOW.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                    # Extract pose landmarks and calculate angles
                    pose_landmarks = []
                    if results.pose_landmarks:
                        for lm in results.pose_landmarks.landmark:
                            pose_landmarks.append((int(lm.x * w), int(lm.y * h)))

                    if len(pose_landmarks) == 0:
                        instructions_box.text("Body Not Visible")
                        continue
                    else:
                        # Calculate user's joint angles
                        user_angles = {
                            'right_elbow': getAngle(pose_landmarks[16], pose_landmarks[14], pose_landmarks[12]),
                            'right_shoulder': getAngle(pose_landmarks[14], pose_landmarks[12], pose_landmarks[24]),
                            'left_shoulder': getAngle(pose_landmarks[13], pose_landmarks[11], pose_landmarks[23]),
                            'right_knee': getAngle(pose_landmarks[24], pose_landmarks[26], pose_landmarks[28]),
                        }

                        # Generate correction instructions based on the pose and deviations
                        instructions = []
                        angle_diff_text = []

                        for joint, user_angle in user_angles.items():
                            ideal_angle = ideal_angles[app_mode].get(joint)
                            deviation = abs(user_angle - ideal_angle)
                            
                            if deviation > 10:
                                if user_angle < ideal_angle:
                                    direction = 'increase'
                                else:
                                    direction = 'decrease'
                                instructions.append(f"Adjust your {joint.replace('_', ' ')}: {direction} the angle by {deviation} degrees.")
                            
                            # Update angle difference text
                            angle_diff_text.append(f"{joint.replace('_', ' ')}: {user_angle}° (Ideal: {ideal_angle}°)")

                        # Display angle differences
                        angle_diff_box.text("\n".join(angle_diff_text))

                        # Calculate and display overall accuracy
                        accuracy = calculate_accuracy(user_angles, ideal_angles[app_mode])
                        accuracy_box.text(f"Overall Accuracy: {accuracy}%")

                        # Display correction instructions
                        if instructions:
                            instructions_box.text("\n".join(instructions))
                        else:
                            instructions_box.text("Great job! Your posture looks good.")

                    if stop:
                        cap.release()
                        cv2.destroyAllWindows()
                        break
        else:
            st.error("Failed to access the webcam.")
