import os
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
l_hip_angle = 0
foot_distance = 0
r_knee = None
l_knee = None
r_hip = None
l_hip = None
r_ankle = None
l_ankle = None
kick_type = "not detected"
position = "not detected"
facing = "not detected"

shoulder_slope = 0
hip_slope = 0
shoulder_distance = 0
crossing_leg = False
hip_distance = 0

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Helper function to calculate all necessary landmarks and store globally
def process_landmarks(landmarks, w, h):
    global l_knee_angle, r_knee_angle, r_hip_angle, l_hip_angle, foot_distance, shoulder_slope, hip_slope, shoulder_distance, crossing_leg, hip_distance, facing, l_hip, l_knee, l_ankle
    global l_hip, l_knee, l_ankle, r_hip, r_knee, r_ankle
    
    # Left side landmarks
    LHip  = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].y * h]
    LKnee = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].y * h]
    LAnkle = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value].y * h]

    # Right side landmarks
    RHip  = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].y * h]
    RKnee = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value].y * h]
    RAnkle = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value].y * h]

    # Shoulder landmarks
    LShoulder = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
    RShoulder = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].x * w, landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]

    l_hip, l_knee, l_ankle = LHip, LKnee, LAnkle
    r_hip, r_knee, r_ankle = RHip, RKnee, RAnkle

    # Calculate distances and angles
    foot_distance = get.Distance(LAnkle, RAnkle)
    l_knee_angle = get.Angle(LHip, LKnee, LAnkle)
    r_knee_angle = get.Angle(RHip, RKnee, RAnkle)
    r_hip_angle = get.Angle(RShoulder, RHip, RKnee)
    l_hip_angle = get.Angle(LShoulder, LHip, LKnee)

    # Calculate shoulder and hip slopes
    shoulder_slope = get.Y_angle(LShoulder, RShoulder)
    hip_slope = get.Y_angle(LHip, RHip)

    # Calculate shoulder distance
    shoulder_distance = get.Distance(LShoulder, RShoulder)
    hip_distance = get.Distance(LHip, RHip)

    # Detect crossing leg
    if hip_distance > 30:
        if (RHip[0] < LHip[0] and RAnkle[0] > LAnkle[0]):  # Check if left ankle is opposite to the right hip and vice versa
            crossing_leg = True
        else:
            crossing_leg = False

    # Landmark for detecting facing direction
    nose = landmarks[mp.solutions.pose.PoseLandmark.NOSE.value]
    l_ear = landmarks[mp.solutions.pose.PoseLandmark.LEFT_EAR.value]
    r_ear = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_EAR.value]

    # Determine facing direction based on nose position
    if nose.x > l_ear.x and nose.x > r_ear.x:
        facing = "right"  # Nose closer to left shoulder, facing right
    else:
        facing = "left"   # Nose closer to right shoulder, facing left

# Functions for each state (KUDA2, TRANSISI, TENDANG, etc.)
def kuda_kuda():
    global position
    if l_ankle[1] > l_knee[1] and r_ankle[1] > r_knee[1] and l_ankle[1] > r_knee[1] and r_ankle[1] > l_knee[1]:
        if facing == "right":
            if r_knee and l_knee and r_knee[0] > l_knee[0]: #kanan depan
                position = "left"
                if 90 < r_knee_angle < 150 and 150 < l_knee_angle < 180 and foot_distance > 140:
                    return True
            elif r_knee and l_knee and l_knee[0] > r_knee[0]: #kiri depan
                position = "right"
                if 120 < l_knee_angle < 165 and 150 < r_knee_angle < 180 and foot_distance > 140:
                    return True

        elif facing == "left":
            if r_knee and l_knee and l_knee[0] > r_knee[0]: #kanan depan
                position = "left"
                if 100 < r_knee_angle < 165 and 150 < l_knee_angle < 180 and foot_distance > 140:
                    return True
            elif r_knee and l_knee and r_knee[0] > l_knee[0]: #kiri depan
                position = "right"
                if 100 < l_knee_angle < 165 and 150 < r_knee_angle < 180 and foot_distance > 140:
                    return True
    return False

def transisi():
    global kick_type
    print(f"crossing_leg = {crossing_leg}")
    if crossing_leg:
        kick_type = "T kick"
        return True
    else:
        if position == "right":
            if r_knee and r_hip and r_knee[1] <= r_hip[1] and r_knee_angle < 150 and r_hip_angle < 150:  # y-coordinates, smaller means higher
                if  hip_distance < 20 and shoulder_distance < 30:  # Kaki terangkat tapi belum sepenuhnya lurus
                    kick_type = "front kick"
                    return True
                elif hip_distance > 20 and shoulder_distance > 30:
                    kick_type = "sickle kick"
                    return True
        elif position == "left":
            if l_knee and l_hip and l_knee[1] <= l_hip[1] and l_knee_angle < 150 and l_hip_angle < 150:  # y-coordinates, smaller means higher
                if hip_distance < 20 and shoulder_distance < 40:  # Kaki terangkat tapi belum sepenuhnya lurus
                    kick_type = "front kick"
                    return True
                elif hip_distance > 20 and shoulder_distance > 40:
                    kick_type = "sickle kick"
                    return True
    return False

def kick():
    if position + " " + kick_type == "right front kick":
        if l_hip_angle > 150 and r_hip_angle < 100 and l_knee_angle > 150 and r_knee_angle > 160:  # Atur threshold sesuai kebutuhan
            return True
    elif position + " " + kick_type == "left front kick":
        if r_hip_angle > 150 and l_hip_angle < 100 and r_knee_angle > 150 and l_knee_angle > 160:  # Atur threshold sesuai kebutuhan
            return True
    elif position + " " + kick_type == "right sickle kick":
        if 150 < r_hip_angle > 110 and l_hip_angle < 110 and r_knee_angle > 160 and l_knee_angle > 160:  # Atur threshold sesuai kebutuhan
            return True
    elif position + " " + kick_type == "left sickle kick":
        if 150 < l_hip_angle > 110 and r_hip_angle < 110 and r_knee_angle > 160 and l_knee_angle > 160:  # Atur threshold sesuai kebutuhan
            return True
    elif position + " " + kick_type == "right T kick" or position + " " + kick_type == "left T kick" :
        if r_knee_angle > 160 and l_knee_angle > 160:  # Atur threshold sesuai kebutuhan
            return True
    
    # Add additional conditions for other kick types if necessary
    return False

def back(position):
    if l_knee_angle > 150 and r_knee_angle > 150 and foot_distance > 150 and shoulder_distance > 40 and hip_distance > 40 and crossing_leg == False:
        if position == "right" :
            if r_hip[0] < l_hip[0] and r_ankle[0] < l_ankle[0]:
                return True
        elif position == "left" :
            if r_hip[0] > l_hip[0] and r_ankle[0] > l_ankle[0]:
                return True
        return True
    return False

# Main function
def main():

    global l_knee_angle, r_knee_angle, r_hip_angle, foot_distance, r_knee, r_hip, facing
    face = False
    body = False
    leg = False
    current_state = State.INITIAL
    tendangan_counter = 0  # Counter untuk menghitung jumlah tendangan

    # Menggunakan video file sebagai input
    video_path = "B:/Abiyu/PA/silat-kicking-counter-asisst/raw_video/Tendangan_Sabit_Kiri.mp4"
    output_folder = "B:/Abiyu/PA/silat-kicking-counter-asisst/output_video/"
    
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    output_path = os.path.join(output_folder, "processed_video.mp4")
    
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Tidak dapat membuka video dari {video_path}.")
        return

    # Define the codec and create VideoWriter object to save video as MP4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) // 2)  # Setengah lebar
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) // 2)  # Setengah tinggi
    
    # Video writer object
    out = cv2.VideoWriter(output_path, fourcc, 20, (1080, 607))

    frame_skip = 2  # Skipping every 3rd frame to improve performance
    frame_count = 0

    # Set up MediaPipe Pose
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Tidak dapat membaca frame.")
                break
            
            resized_frame = cv2.resize(frame, (frame_width, frame_height))
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue  # Skip frames to improve performance

            # Get frame dimensions
            h, w = resized_frame.shape[:2]

            # Recolor the image to RGB
            image_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

            # Process landmarks using MediaPipe Pose
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                process_landmarks(results.pose_landmarks.landmark, w, h)

                # Use mp_drawing to draw the landmarks
                mp_drawing.draw_landmarks(
                    resized_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
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
                            current_state = State.TRANSISI

                    case State.TRANSISI:
                        print("Transisi!")
                        if transisi():
                            current_state = State.TENDANG

                    case State.TENDANG:
                        print("Tendang!")
                        if kick():
                            tendangan_counter += 1
                            print(f"Tendangan terdeteksi! Total tendangan: {tendangan_counter}")
                            current_state = State.AKHIR

                    case State.AKHIR:
                        print("back!")
                        if back(position):
                            current_state = State.INITIAL

            # Display the frame with data
            cv2.putText(resized_frame, f"l_knee_angle: {l_knee_angle:.2f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(resized_frame, f"r_knee_angle: {r_knee_angle:.2f}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(resized_frame, f"l_hip_angle: {l_hip_angle:.2f}", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(resized_frame, f"r_hip_angle: {r_hip_angle:.2f}", (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(resized_frame, f"foot_distance: {foot_distance:.2f}", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(resized_frame, f"hip_distance: {hip_distance:.2f}", (30, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(resized_frame, f"shoulder_distance: {shoulder_distance:.2f}", (30, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            cv2.putText(resized_frame, f"Crossing Leg: {'Yes' if crossing_leg else 'No'}", (30, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(resized_frame, f"Tendangan: {tendangan_counter}", (650, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(resized_frame, f"position: {position}", (650, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(resized_frame, f"kick type: {kick_type}", (650, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(resized_frame, f"facing: {facing}", (650, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Display the frame
            cv2.imshow('Silat Kick Counter', resized_frame)

            # Write the frame to the output video
            out.write(resized_frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()  # Make sure to release the VideoWriter when done
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()