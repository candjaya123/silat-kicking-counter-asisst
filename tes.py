import time
import cv2
import mediapipe as mp
import imutils
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


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Helper function to calculate all necessary landmarks and store globally
def process_landmarks(landmarks, w, h):
    global l_knee_angle, r_knee_angle, r_hip_angle, foot_distance, r_knee, l_hip, l_knee, r_hip
    LHip  = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].y * h]
    LKnee = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].y * h]
    LAnkle = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].y * h]

    RHip  = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].y * h]
    RKnee = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value].y * h]
    RAnkle = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value].y * h]

    # Calculate distances and angles
    foot_distance = get.Distance(LAnkle, RAnkle)
    l_knee_angle = get.Angle(LHip, LKnee, LAnkle)
    r_knee_angle = get.Angle(RHip, RKnee, RAnkle)
    r_hip_angle = get.Angle([landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y * h], RHip, RKnee)

    r_knee, l_knee = RKnee, LKnee
    r_hip, l_hip = RHip, LHip

# Functions for each state (KUDA2, TRANSISI, TENDANG, etc.)
def kuda_kuda():
    global position
    if r_knee and l_knee and r_knee[0] > l_knee[0]:
        position = "left"
        if 90 < r_knee_angle < 150 and 150 < l_knee_angle < 180 and foot_distance > 180:
            return True
    elif r_knee and l_knee and l_knee[0] > r_knee[0]:
        position = "right"
        if 90 < l_knee_angle < 150 and 150 < r_knee_angle < 180 and foot_distance > 180:
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

def back(leg_position):
    if leg_position == "right":
        if r_knee and l_knee and r_knee[0] < l_knee[0]:  # y-coordinates, smaller means higher
            return True
    if leg_position == "left":
        if r_knee and l_knee and r_knee[0] > l_knee[0]: # y-coordinates, smaller means higher
            return True

# Main function
def main():
    global l_knee_angle, r_knee_angle, r_hip_angle, foot_distance, r_knee, r_hip
    face = False
    body = False
    leg = False
    current_state = State.INITIAL
    tendangan_counter = 0  # Counter untuk menghitung jumlah tendangan

    # Menggunakan video file sebagai input
    video_path = "B:/Abiyu/PA/silat-kicking-counter-asisst/raw_video/Tendangan_Depan_Kiri.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Tidak dapat membuka video dari {video_path}.")
        return
    
    frame_skip = 1  # Skipping every 3rd frame to improve performance
    frame_count = 0
    prev_frame_time = 0

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set up MediaPipe Pose
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Tidak dapat membaca frame.")
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue  # Skip frames to improve performance

            # Resize the frame for faster processing
            frame = imutils.resize(frame, width=1080)

            # Get frame dimensions
            h, w = frame.shape[:2]

            # Recolor the image to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process landmarks using MediaPipe Pose
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                process_landmarks(results.pose_landmarks.landmark, w, h)

                # Use mp_drawing to draw the landmarks
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(66, 245, 230), thickness=2, circle_radius=2)
                )

                match current_state:
                    case State.INITIAL:
                        face = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE.value].visibility > 0.5
                        body = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility > 0.5
                        leg = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility > 0.5
                        if face and body and leg:
                            current_state = State.KUDA2

                    case State.KUDA2:
                        print("Kuda!")
                        if kuda_kuda():
                            print("Kuda-kuda terdeteksi!")
                            current_state = State.TRANSISI

                    case State.TRANSISI:
                        print("Transisi!")
                        if transisi():
                            print("Transisi terdeteksi!")
                            current_state = State.TENDANG

                    case State.TENDANG:
                        print("Tendang!")
                        if kick():
                            tendangan_counter += 1
                            print(f"Tendangan terdeteksi: {tendangan_counter}")
                            Play_buzzer()  # Suara jika perlu
                            current_state = State.AKHIR

                    case State.AKHIR:
                        print("BACK!")
                        if back(position):
                            print("back terdeteksi!")
                            current_state = State.INITIAL


            # cv2.putText(frame, f"FPS: {fps:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"l_knee_angle: {l_knee_angle:.2f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(frame, f"r_knee_angle: {r_knee_angle:.2f}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(frame, f"foot_distance: {foot_distance:.2f}", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            cv2.putText(frame, f"Tendangan: {tendangan_counter}", (650, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, f"position: {position}", (650, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, f"kick type: {kick_type}", (650, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Show processed frame
            cv2.imshow("Kicking Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
