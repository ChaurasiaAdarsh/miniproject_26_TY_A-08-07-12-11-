import cv2
import mediapipe as mp
import serial
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

SERIAL_PORT = 'COM7'    
BAUD_RATE = 115200
COOLDOWN = 1.2           # sec change
last_action = ""
last_time = 0

esp = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
time.sleep(2)  # wait for ESP to reset

# MEDIAPIPE setup
base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)

#fun
def send(cmd: str):
    """Send command to ESP with cooldown"""
    global last_action, last_time
    now = time.time()
    if cmd != last_action and now - last_time > COOLDOWN:
        try:
            esp.write((cmd + "\n").encode())
            last_action = cmd
            last_time = now
            print(f"Sent: {cmd}")
        except Exception as e:
            print(f"Error sending: {cmd}, {e}")

def count_fingers(lm, handedness):
    """Return list of booleans: fingers extended"""
    fingers = []

    # thumb
    if handedness == "Right":
        fingers.append(lm[4].x < lm[3].x)
    else:
        fingers.append(lm[4].x > lm[3].x)

    # other fin
    tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
    pips = [6, 10, 14, 18]
    for t, p in zip(tips, pips):
        fingers.append(lm[t].y < lm[p].y)

    return fingers

def gesture_from_landmarks(fingers, lm):
    """
    Accurate gesture mapping:
    - Fist → ALL_OFF
    - Thumb up → LOCK
    - Thumb flat/side → UNLOCK
    - Fingers 1-5 → LED1–LED5
    """
    thumb_tip = lm[4]
    thumb_ip = lm[3]

    cnt_fingers = sum(fingers)  # include thumb for LED counting

    # Fist - ALL_OFF
    if cnt_fingers == 0:
        return "ALL_OFF", "✊"

    # Thumb gestures - LOCK / UNLOCK
    if cnt_fingers == 1 and fingers[0]:  # only thumb
        delta_y = thumb_ip.y - thumb_tip.y
        delta_x = thumb_tip.x - thumb_ip.x
        if delta_y > 0.04:  # thumb up
            return "LOCK", "🤙🏻"
        elif abs(delta_y) < 0.02 and abs(delta_x) > 0.02:  # flat/thumb sideways
            return "UNLOCK", "🔓"

    # Finger count - LED1–LED5
    mapping = {1: ("LED1", "1"),
               2: ("LED2", "2"),
               3: ("LED3", "3"),
               4: ("LED4", "4"),
               5: ("LED5", "5")}
    return mapping.get(cnt_fingers, ("NO_HAND", "?"))

# main handletr
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = detector.detect(mp_image)
    action = "NO_HAND"
    symbol = "?"

    if result.hand_landmarks:
        lm = result.hand_landmarks[0]
        handedness = result.handedness[0][0].category_name

        fingers = count_fingers(lm, handedness)
        action, symbol = gesture_from_landmarks(fingers, lm)

        # Send command to ESP
        send(action)

        # Draw landmarks
        for p in lm:
            x = int(p.x * frame.shape[1])
            y = int(p.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # Display gesture symbol and ESP command
    cv2.putText(frame, f"Gesture: {symbol}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    cv2.putText(frame, f"ESP Cmd: {action}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 0), 3)

    cv2.imshow("Hand Controlled Home Automation", frame)
    if cv2.waitKey(1) & 0xFF == 27:  
        break

cap.release()
cv2.destroyAllWindows()
esp.close()
