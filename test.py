# test.py
# Driver Drowsiness Detection Testing System
# Measures Accuracy + Latency + Saves All Test Results

import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import os
from datetime import datetime

# ---------------- SETTINGS ----------------

EAR_THRESHOLD = 0.25
DROWSY_TIME = 3

# ---------------- MEDIAPIPE ----------------

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    max_num_faces=1
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ---------------- EAR FUNCTION ----------------

def eye_aspect_ratio(landmarks, eye):

    p1 = landmarks[eye[0]]
    p2 = landmarks[eye[1]]
    p3 = landmarks[eye[2]]
    p4 = landmarks[eye[3]]
    p5 = landmarks[eye[4]]
    p6 = landmarks[eye[5]]

    v1 = np.linalg.norm(np.array(p2) - np.array(p6))
    v2 = np.linalg.norm(np.array(p3) - np.array(p5))
    h = np.linalg.norm(np.array(p1) - np.array(p4))

    ear = (v1 + v2) / (2 * h)

    return ear

# ---------------- CAMERA ----------------

cap = cv2.VideoCapture(0)

eye_start = None

# ---------------- TEST VARIABLES & HISTORY ----------------

frame_count = 0
latency_list = []

true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0

print("\n========== TESTING MODE ==========")
print("Press:")
print("D = Actual Drowsy")
print("N = Actual Normal")
print("Q = Quit")
print("==================================\n")

# ---------------- CSV FILE SETUP & READ ----------------

csv_filename = "test_results.csv"
file_exists = os.path.exists(csv_filename)

# 1. Read existing data if the file already exists
if file_exists:
    with open(csv_filename, "r") as read_file:
        reader = csv.reader(read_file)
        next(reader, None)  # Skip the header row
        
        for row in reader:
            if len(row) >= 4:  # Ensure the row has enough columns
                actual = row[1]
                predicted = row[2]
                
                # Load past latencies to get all-time average
                try:
                    past_latency = float(row[3])
                    latency_list.append(past_latency)
                except ValueError:
                    pass
                
                # Tally up historical confusion matrix
                if actual == "Drowsy" and predicted == "Drowsy":
                    true_positive += 1
                elif actual == "Drowsy" and predicted == "Normal":
                    false_negative += 1
                elif actual == "Normal" and predicted == "Normal":
                    true_negative += 1
                elif actual == "Normal" and predicted == "Drowsy":
                    false_positive += 1

# 2. Open file in append mode for the current session's new data
csv_file = open(csv_filename, "a", newline="")
writer = csv.writer(csv_file)

# Write header if the file was just created
if not file_exists:
    writer.writerow([
        "DateTime",
        "Actual",
        "Predicted",
        "Latency(ms)",
        "EAR"
    ])
    
# ---------------- MAIN LOOP ----------------

while True:

    start_time = time.time()

    ret, frame = cap.read()

    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    h, w, _ = frame.shape

    prediction = "Normal"
    ear_value = 0

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            landmarks = []

            for lm in face_landmarks.landmark:

                x = int(lm.x * w)
                y = int(lm.y * h)

                landmarks.append((x, y))

            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)

            ear = (left_ear + right_ear) / 2

            ear_value = round(ear, 3)

            # ---------------- DROWSINESS CHECK ----------------

            if ear < EAR_THRESHOLD:

                if eye_start is None:
                    eye_start = time.time()

            else:
                eye_start = None

            eye_time = 0

            if eye_start:
                eye_time = time.time() - eye_start

            if eye_time >= DROWSY_TIME:
                prediction = "Drowsy"

            # ---------------- DISPLAY ----------------

            cv2.putText(
                frame,
                f"EAR: {ear_value}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Prediction: {prediction}",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

    # ---------------- LATENCY ----------------

    end_time = time.time()

    latency = (end_time - start_time) * 1000

    latency_list.append(latency)

    frame_count += 1

    cv2.putText(
        frame,
        f"Latency: {latency:.2f} ms",
        (30, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    # ---------------- SHOW FRAME ----------------

    cv2.imshow("Testing System", frame)

    # ---------------- KEY INPUT ----------------

    key = cv2.waitKey(1) & 0xFF

    # ACTUAL DROWSY TEST

    if key == ord('d'):

        actual = "Drowsy"

        if prediction == "Drowsy":
            true_positive += 1
        else:
            false_negative += 1

        writer.writerow([
            datetime.now(),
            actual,
            prediction,
            round(latency, 2),
            ear_value
        ])

        print(f"[DROWSY TEST]")
        print(f"Actual: {actual}")
        print(f"Predicted: {prediction}\n")

    # ACTUAL NORMAL TEST

    elif key == ord('n'):

        actual = "Normal"

        if prediction == "Normal":
            true_negative += 1
        else:
            false_positive += 1

        writer.writerow([
            datetime.now(),
            actual,
            prediction,
            round(latency, 2),
            ear_value
        ])

        print(f"[NORMAL TEST]")
        print(f"Actual: {actual}")
        print(f"Predicted: {prediction}\n")

    # QUIT

    elif key == ord('q'):
        break

# ---------------- FINAL RESULTS ----------------

cap.release()
cv2.destroyAllWindows()
csv_file.close()

total = (
    true_positive
    + true_negative
    + false_positive
    + false_negative
)

if total > 0:

    accuracy = (
        (true_positive + true_negative) / total
    ) * 100

else:
    accuracy = 0

average_latency = sum(latency_list) / len(latency_list)

result_text = f"""
========== FINAL RESULTS ==========

Total Tests            : {total}

Average Latency        : {average_latency:.2f} ms

True Positive          : {true_positive}
True Negative          : {true_negative}
False Positive         : {false_positive}
False Negative         : {false_negative}

Model Accuracy         : {accuracy:.2f}%

===================================
"""

print(result_text)

# SAVE FINAL REPORT

with open("final_report.txt", "a") as f:

    f.write("\n")
    f.write(str(datetime.now()))
    f.write("\n")
    f.write(result_text)

print("Final report saved in final_report.txt")