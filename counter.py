import time
import cv2
from module.landmark import Value
from module import tool as get
from module.tool import Play_buzzer
import enum

# Define states for the FSM
class State(enum.Enum):
    INITIAL = 1
    KUDA2 = 2
    TRANSISI = 3
    TENDANG = 4
    AKHIR = 5

# Global variables to store results of landmark processing
l_knee_angle = 0
r_knee_angle = 0
r_hip_angle = 0
foot_distance = 0
r_knee = None
l_knee = None
r_hip = None
l_hip = None
kick_type = "not detected"
position = "not detected"

# Helper function to calculate all necessary landmarks and store globally
def process_landmarks(frame):
    global l_knee_angle, r_knee_angle, r_hip_angle, foot_distance, r_knee, l_hip, l_knee, r_hip
    foot_distance = get.Distance(Value(frame, "l_ankle"), Value(frame, "r_ankle"))

    l_knee_angle = get.Angle(Value(frame, "l_hip"), Value(frame, "l_knee"), Value(frame, "l_ankle"))
    r_knee_angle = get.Angle(Value(frame, "r_hip"), Value(frame, "r_knee"), Value(frame, "r_ankle"))
    r_hip_angle = get.Angle(Value(frame, "r_shoulder"), Value(frame, "r_hip"), Value(frame, "r_knee"))

    r_knee = Value(frame, "r_knee")
    l_knee = Value(frame, "l_knee")

    r_hip = Value(frame, "r_hip")
    l_hip = Value(frame, "l_hip")

# Function to check for face and body landmarks
def Check(frame, name):
    buffer = Value(frame, name)
    if buffer is None or buffer[3] is False:
        return False
    return buffer[3]

# Functions for each state (KUDA2, TRANSISI, TENDANG, etc.)
def kuda_kuda():
    global position
    if r_knee and l_knee and r_knee[0] > l_knee[0]:
        position = "right"
    elif r_knee and l_knee and r_knee[0] < l_knee[0]:
        position = "left"
    if 60 < l_knee_angle < 90 and 60 < r_knee_angle < 160 and foot_distance > 0.20:
        return True
    return False

def transisi():
    global kick_type
    if position == "right":
        if r_knee and r_hip and r_knee[1] <= r_hip[1]:  # y-coordinates, smaller means higher
            if 100 < r_knee_angle < 160:  # Kaki terangkat tapi belum sepenuhnya lurus
                kick_type = "front kick"
                return True
    elif position == "left":
        if l_knee and l_hip and l_knee[1] <= l_hip[1]:  # y-coordinates, smaller means higher
            if 100 < l_knee_angle < 160:  # Kaki terangkat tapi belum sepenuhnya lurus
                kick_type = "front kick"
                return True
    return False

def kick():
    if position + " " + kick_type == "right front kick":
        if r_hip_angle < 100 and r_knee_angle > 100:  # Atur threshold sesuai kebutuhan
            return True
    elif position + " " + kick_type == "left front kick":
        if r_hip_angle < 100 and r_knee_angle > 100:  # Atur threshold sesuai kebutuhan
            return True
    # Add additional conditions for other kick types if necessary
    return False

# Main function
def main():
    global l_knee_angle, r_knee_angle, r_hip_angle, foot_distance, r_knee, r_hip
    face = False
    body = False
    leg = False
    current_state = State.INITIAL
    tendangan_counter = 0  # Counter untuk menghitung jumlah tendangan

    # Menggunakan video file sebagai input
    video_path = "B:/Abiyu/PA/silat-kicking-counter-asisst/video_testing/Tendangan_Depan_Kanan.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Tidak dapat membuka video dari {video_path}.")
        return
    
    prev_frame_time = 0
    frame_skip = 3  # Skipping every 3rd frame to improve performance
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Tidak dapat membaca frame.")
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue  # Skip frames to improve performance

        # Resize the frame for faster processing
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Process landmarks in the same loop without threading
        process_landmarks(frame)

        # FSM to manage different states
        match current_state:
            case State.INITIAL:
                if Check(frame, "nose") and Check(frame, "r_ear") and Check(frame, "l_ear"):
                    face = True
                if Check(frame, "l_shoulder") and Check(frame, "r_shoulder") and Check(frame, "l_hip") and Check(frame, "r_hip"):
                    body = True
                if Check(frame, "l_knee") and Check(frame, "r_knee") and Check(frame, "l_ankle") and Check(frame, "r_ankle"):
                    leg = True

                if face and body and leg:
                    current_state = State.KUDA2

            case State.KUDA2:
                if kuda_kuda():
                    print("Kuda-kuda terdeteksi!")
                    current_state = State.TRANSISI

            case State.TRANSISI:
                if transisi():
                    print("Transisi terdeteksi!")
                    current_state = State.TENDANG

            case State.TENDANG:
                if kick():
                    tendangan_counter += 1
                    print(f"Tendangan terdeteksi: {tendangan_counter}")
                    Play_buzzer()  # Suara jika perlu
                    current_state = State.AKHIR

            case State.AKHIR:
                if not kick():
                    current_state = State.INITIAL

        # Calculate FPS and display it on the frame
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time != 0 else 0
        prev_frame_time = new_frame_time

        # Display text information on the frame
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"l_knee_angle: {l_knee_angle:.2f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, f"r_knee_angle: {r_knee_angle:.2f}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, f"foot_distance: {foot_distance:.2f}", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.putText(frame, f"Tendangan: {tendangan_counter}", (650, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"position: {position}", (650, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"kick type: {kick_type}", (650, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Show the processed video frame
        cv2.imshow('Video Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
