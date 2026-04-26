import cv2
import mediapipe as mp
import numpy as np
import simpleaudio as sa
import time
import threading
import tkinter as tk
from PIL import Image, ImageTk
import os
from datetime import datetime

# ---------------- SETTINGS ----------------

BG_COLOR = "#0f0f0f"
PANEL_COLOR = "#1c1c1c"

GREEN = "#2ecc71"
RED = "#e74c3c"
BLUE = "#4aa3ff"
GRAY = "#444444"

show_landmarks = False
alarm_enabled = True
alarm_playing = False
alarm_source = None

drowsy_logged = False
micro_logged = False

# ---------------- LOGGING ----------------


def log_event(event):

    file = "driver_log.txt"

    now = datetime.now()

    date = now.strftime("%Y-%m-%d")
    time_now = now.strftime("%H:%M:%S")

    line = f"{date},{time_now},{event}\n"

    if not os.path.exists(file):

        with open(file, "w") as f:
            f.write("Date,Time,Event\n")

    with open(file, "a") as f:
        f.write(line)


# ---------------- ALARM ----------------


def play_alarm(source):

    global alarm_playing, alarm_source

    if alarm_playing or not alarm_enabled:
        return

    alarm_playing = True
    alarm_source = source

    wave_obj = sa.WaveObject.from_wave_file("alarm.wav")

    for _ in range(3):

        play = wave_obj.play()

        while play.is_playing():
            time.sleep(0.1)

    alarm_playing = False
    alarm_source = None


# ---------------- MEDIAPIPE ----------------

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


# ---------------- EAR ----------------


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

    return (v1 + v2) / (2 * h)


# ---------------- GUI ----------------

root = tk.Tk()
root.title("Driver Monitoring System")
root.configure(bg=BG_COLOR)

main = tk.Frame(root, bg=BG_COLOR)
main.pack(padx=10, pady=10)

video_label = tk.Label(main)
video_label.grid(row=0, column=0, padx=10)

panel = tk.Frame(main, bg=PANEL_COLOR, padx=20, pady=20)
panel.grid(row=0, column=1, sticky="n")

title = tk.Label(
    panel, text="Driver Status", bg=PANEL_COLOR, fg=BLUE, font=("Segoe UI", 18, "bold")
)

title.grid(row=0, column=0, columnspan=2, pady=(0, 15))


def toggle_alarm():

    global alarm_enabled

    alarm_enabled = not alarm_enabled

    if alarm_enabled:
        alarm_button.config(text="Alarm System: ON", bg=GREEN)
    else:
        alarm_button.config(text="Alarm System: OFF", bg=GRAY)


alarm_button = tk.Button(
    panel,
    text="Alarm System: ON",
    bg=GREEN,
    font=("Segoe UI", 11, "bold"),
    width=18,
    height=2,
    command=toggle_alarm,
)

alarm_button.grid(row=1, column=0, columnspan=2, pady=10)

tk.Label(
    panel, text="Detection", bg=PANEL_COLOR, fg="white", font=("Segoe UI", 11, "bold")
).grid(row=2, column=0, pady=10)

tk.Label(
    panel, text="Action", bg=PANEL_COLOR, fg="white", font=("Segoe UI", 11, "bold")
).grid(row=2, column=1, pady=10)

drowsy_btn = tk.Label(
    panel,
    text="Drowsiness",
    bg=GREEN,
    fg="black",
    width=14,
    height=2,
    font=("Segoe UI", 11, "bold"),
)

drowsy_btn.grid(row=3, column=0, pady=10, padx=5)

alarm_btn = tk.Label(
    panel,
    text="Alarm",
    bg=GREEN,
    fg="black",
    width=14,
    height=2,
    font=("Segoe UI", 11, "bold"),
)

alarm_btn.grid(row=3, column=1, pady=10, padx=5)

micro_btn = tk.Label(
    panel,
    text="Micro Sleep",
    bg=GREEN,
    fg="black",
    width=14,
    height=2,
    font=("Segoe UI", 11, "bold"),
)

micro_btn.grid(row=4, column=0, pady=10, padx=5)

action_frame = tk.Frame(panel, bg=PANEL_COLOR)
action_frame.grid(row=4, column=1)

alarm2_btn = tk.Label(
    action_frame,
    text="Alarm",
    bg=GREEN,
    fg="black",
    width=6,
    height=2,
    font=("Segoe UI", 11, "bold"),
)

alarm2_btn.pack(side="left", padx=3)

brake_btn = tk.Label(
    action_frame,
    text="Brake",
    bg=GREEN,
    fg="black",
    width=6,
    height=2,
    font=("Segoe UI", 11, "bold"),
)

brake_btn.pack(side="left", padx=3)

exit_label = tk.Label(
    panel, text="Press Q to Exit", bg=PANEL_COLOR, fg="#888888", font=("Segoe UI", 9)
)

exit_label.grid(row=5, column=0, columnspan=2, pady=20)


# ---------------- CAMERA ----------------

cap = cv2.VideoCapture(0)

eye_start = None
head_start = None


def update():

    global eye_start, head_start, drowsy_logged, micro_logged

    ret, frame = cap.read()

    if not ret:
        root.after(10, update)
        return

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    h, w, _ = frame.shape

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            if show_landmarks:

                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_style.get_default_face_mesh_tesselation_style(),
                )

            landmarks = []

            for lm in face_landmarks.landmark:

                x = int(lm.x * w)
                y = int(lm.y * h)

                landmarks.append((x, y))

            ear = (
                eye_aspect_ratio(landmarks, LEFT_EYE)
                + eye_aspect_ratio(landmarks, RIGHT_EYE)
            ) / 2

            nose = landmarks[1]
            chin = landmarks[152]

            head_drop = chin[1] - nose[1]

            if ear < 0.25:

                if eye_start is None:
                    eye_start = time.time()

            else:
                eye_start = None
                drowsy_logged = False

            if head_drop < 60:

                if head_start is None:
                    head_start = time.time()

            else:
                head_start = None
                micro_logged = False

            eye_time = 0
            head_time = 0

            if eye_start:
                eye_time = time.time() - eye_start

            if head_start:
                head_time = time.time() - head_start

            cv2.putText(
                frame,
                f"EyeClosed: {eye_time:.1f}s",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"HeadDrop: {head_drop}",
                (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"HeadDown: {head_time:.1f}s",
                (50, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 200, 0),
                2,
            )

            if eye_time >= 3:

                cv2.putText(
                    frame,
                    "DROWSY",
                    (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

                if not drowsy_logged:

                    log_event("Drowsiness")
                    drowsy_logged = True

                if not alarm_playing:
                    threading.Thread(target=play_alarm, args=("drowsy",)).start()

            if head_time >= 3:

                cv2.putText(
                    frame,
                    "MICRO SLEEP DETECTED",
                    (50, 190),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

                if not micro_logged:

                    log_event("Micro Sleep")
                    micro_logged = True

                if not alarm_playing:
                    threading.Thread(target=play_alarm, args=("micro",)).start()

    if alarm_playing:

        if alarm_source == "drowsy":

            drowsy_btn.config(bg=RED)
            alarm_btn.config(bg=RED)

        elif alarm_source == "micro":

            micro_btn.config(bg=RED)
            alarm2_btn.config(bg=RED)
            brake_btn.config(bg=RED)

    else:

        drowsy_btn.config(bg=GREEN)
        alarm_btn.config(bg=GREEN)

        micro_btn.config(bg=GREEN)
        alarm2_btn.config(bg=GREEN)
        brake_btn.config(bg=GREEN)

    display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img = Image.fromarray(display)
    imgtk = ImageTk.PhotoImage(image=img)

    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(10, update)


update()


def key_event(e):

    global show_landmarks

    if e.char.lower() == "l":
        show_landmarks = not show_landmarks

    elif e.char.lower() == "o":
        toggle_alarm()

    elif e.char.lower() == "q":
        on_close()


root.bind("<Key>", key_event)


def on_close():

    cap.release()
    root.destroy()


root.protocol("WM_DELETE_WINDOW", on_close)

root.mainloop()
