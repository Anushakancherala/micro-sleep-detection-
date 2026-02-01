import cv2
import os
import mediapipe as mp
import numpy as np
import threading
import winsound
import time

# ---------------- CONFIG ---------------- #
EAR_THRESH = 0.23
FRAME_LIMIT = 20
ALARM_FILE = "alarm.wav"
# ---------------------------------------- #

counter = 0
alarm_on = False

# ---------------- ALARM ---------------- #
def play_alarm():
    if os.path.exists(ALARM_FILE):
        winsound.PlaySound(ALARM_FILE, winsound.SND_FILENAME | winsound.SND_ASYNC)
# --------------------------------------- #

# ---------------- EAR ---------------- #
def ear(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)
# ------------------------------------- #

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

cap = cv2.VideoCapture(0)
time.sleep(1)

print("Press ESC or Q to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        for face in result.multi_face_landmarks:
            h, w, _ = frame.shape
            pts = [(int(l.x * w), int(l.y * h)) for l in face.landmark]

            left = np.array([pts[i] for i in LEFT_EYE])
            right = np.array([pts[i] for i in RIGHT_EYE])

            ear_val = (ear(left) + ear(right)) / 2

            cv2.putText(frame, f"EAR: {round(ear_val,2)}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if ear_val < EAR_THRESH:
                counter += 1
                if counter >= FRAME_LIMIT:
                    cv2.putText(frame, "MICRO-SLEEP ALERT!", (50, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                    if not alarm_on:
                        alarm_on = True
                        threading.Thread(target=play_alarm, daemon=True).start()
            else:
                counter = 0
                alarm_on = False

    cv2.imshow("Micro Sleep Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        print("Exit key pressed")
        break

print("Releasing camera...")
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
winsound.PlaySound(None, winsound.SND_PURGE)
print("Camera closed safely")
