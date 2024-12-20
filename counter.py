import os
import time
import cv2
import mediapipe as mp
import imutils
from module import tool as get
from module.tool import Play_buzzer
import threading
import enum
import csv

def save_to_csv(file_path, jenis, tendangan_ke, st, r_knee, l_knee, r_hip, l_hip, 
                foot_dist, hip_dist, shoulder_dist):

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write data row
        writer.writerow([
            tendangan_ke,jenis, st, r_knee, l_knee, r_hip, l_hip, 
            foot_dist, hip_dist, shoulder_dist
        ])

def save_header_to_csv(file_path):
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write header if file does not exist
        writer.writerow([
            "Tendangan Ke", "Jenis Tendangna" ,"State", "R Knee Angle", "L Knee Angle", "R Hip Angle", "L Hip Angle", 
            "Foot Distance", "Hip Distance", "Shoulder Distance"
        ])

# Define states for the FSM
class State(enum.Enum):
    INITIAL = 1
    KUDA2 = 2
    TRANSISI = 3
    TENDANG = 4
    AKHIR = 5

# Global variables to store results of landmark processing

csv_path = "./data.csv"
l_knee_angle = 0
r_knee_angle = 0
r_hip_angle = 0
l_hip_angle = 0
foot_distance = 0
nose = 0
l_ear = 0
r_ear = 0
r_knee = None
l_knee = None
r_hip = None
l_hip = None
r_ankle = None
l_ankle = None
kick_type = "not detected"
position = "not detected"
facing = "not detected"
feedback = "none"

shoulder_slope = 0
hip_slope = 0
shoulder_distance = 0
crossing_leg = False
hip_distance = 0

kuda = False
tendang = False
transisi = False

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Helper function to calculate all necessary landmarks and store globally
def process_landmarks(landmarks, w, h):
    global l_knee_angle, r_knee_angle, r_hip_angle, l_hip_angle, foot_distance, shoulder_slope, hip_slope, shoulder_distance, crossing_leg, hip_distance, facing,l_hip, l_knee, l_ankle
    global l_hip, l_knee, l_ankle, r_hip, r_knee, r_ankle
    global nose, l_ear, r_ear
    
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

# Functions for each state (KUDA2, TRANSISI, TENDANG, etc.)
def kuda_kuda():
    global feedback, feedback, kuda, facing, position

    ############## untuk keluar dari fungsi ini ##############
    if crossing_leg:
        print("keluar kuda paksa")
        return True
    ##########################################################

    ############## untuk keluar dari fungsi ini ##############
    if ((r_knee_angle < 140 or l_knee_angle < 140) and foot_distance > 200) or r_ankle[1] <= l_knee[1] or l_ankle[1] <= r_knee[1]:
        print("keluar kuda paksa")
        return True
    ##########################################################

    # Determine facing direction based on nose position
    if nose.x > l_ear.x and nose.x > r_ear.x:
        facing = "right"  # Nose closer to left shoulder, facing right
    else:
        facing = "left"   # Nose closer to right shoulder, facing left

    # print(f'HUAAA = {abs(l_ankle[1] - r_ankle[1])}')
    
    if foot_distance > 170 and r_ankle[1] > l_knee[1] and l_ankle[1] > r_knee[1]:
        position = "right" if facing == "right" else "left"
        # if r_ankle[0] > l_ankle[0] :
        #     position = "left" if facing == "right" else "right"
        # elif l_ankle[0] > r_ankle[0] :
        #     position = "right" if facing == "right" else "left"

    if foot_distance < 160 and crossing_leg == False or foot_distance < 190:
        return False

    if position == "left" and crossing_leg == False:
        if 100 < r_knee_angle < 150 and 150 < l_knee_angle < 180 :
            feedback = "pas" 
            kuda = True
            return True
        elif r_knee_angle > 150:
            feedback = "kaki kanan kurang nekuk"
            return False
        elif r_knee_angle < 100:
            feedback = "kaki kanan terlalu nekuk"
            return False
        elif l_knee_angle < 150:
            feedback = "kaki kiri kurang lurus"
            return False
        else:
            feedback = "salah"
        
    elif position == "right" and crossing_leg == False:
        if 100 < l_knee_angle < 150 and 150 < r_knee_angle < 180:
            feedback = "pas" 
            kuda = True
            return True
        elif l_knee_angle > 150:
            feedback = "kaki kiri kurang nekuk"
            return False
        elif l_knee_angle < 100:
            feedback = "kaki kiri terlalu nekuk"
            return False
        elif r_knee_angle < 150:
            feedback = "kaki kanan kurang lurus"
            return False
        else:
            feedback = "salah"
    
    return False

def transition():
    global kick_type, feedback, feedback, transisi

    # print(f"crossing_leg = {crossing_leg}")
    if crossing_leg:
        kick_type = "T kick"
        feedback = "pas"
        return True
    
    else:
        if r_knee[1] <= r_hip[1] or l_knee[1] <= l_hip[1]: 
            kick_type = "front kick" if hip_distance < 20 and shoulder_distance < 30 else "sickle kick"

        if position == "right":
            if r_knee and r_hip and r_knee[1] <= r_hip[1] and r_knee_angle < 160 and r_hip_angle < 110:  # y-coordinates, smaller means higher
                transisi = True
                feedback = "pas"
                return True
            elif r_knee_angle > 150:
                feedback = "kaki kanan kurang nekuk"
                return False
            elif r_hip_angle > 110:
                feedback = "kaki kanan kurang naik"
                return False
            else:
                feedback = "salah"
            
        elif position == "left":
            if l_knee and l_hip and l_knee[1] <= l_hip[1] and l_knee_angle < 160 and l_hip_angle < 110:  # y-coordinates, smaller means higher
                transisi = True
                feedback = "pas"
                return True
            elif l_knee_angle > 150:
                feedback = "kaki kiri kurang nekuk"
                return False
            elif l_hip_angle > 110:
                feedback = "kaki kiri kurang naik"
                return False
            else:
                feedback = "salah"

            
    ################### untuk keluar dari fungsi ini ###################
    if foot_distance > 190 and (l_knee_angle > 150 and r_knee_angle > 150) :
        return True
    ####################################################################
    return False

def kick():
    global feedback, feedback, tendang

    if crossing_leg:
        return False

    if l_knee_angle > 160 and r_knee_angle > 160 and crossing_leg == False:
        tendang = True
        feedback = "pas"
        return True
    elif l_knee_angle < 160:
        feedback = "kaki kiri kurang lurus"
        return False
    elif r_knee_angle < 160:
        feedback = "kaki kanan kurang lurus"
        return False
    else:
        feedback = "salah"
    
    ######### untuk keluar dari fungsi ini ###########
    if back(True):
        return True
    ##################################################
    
    return False

def back(skip):
    global kuda, transisi, tendang

    if skip == False:
        kuda = False
        transisi = False
        tendang = False

    if l_ankle[1] < l_knee[1] or l_ankle[1] < r_knee[1] or r_ankle[1] < l_knee[1] or r_ankle[1] < r_knee[1]:
        return False

    if r_hip[0] < l_hip[0] and r_ankle[0] < l_ankle[0]:
        if kick_type == "T kick":
            if foot_distance > 160:
                return True 
        else:
            return True
    return False

def draw(frame, text, text_x, text_y, text_color, rect_color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2

    # Get text size
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    rect_x1 = text_x - 10
    rect_y1 = text_y - text_size[1] - 10
    rect_x2 = text_x + text_size[0] + 10
    rect_y2 = text_y + 10

    cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), rect_color, -1)

    # Put the text on top of the rectangle
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, font_thickness)


# Main function
def main():
    global l_knee_angle, r_knee_angle, r_hip_angle, foot_distance, r_knee, r_hip, facing
    face = False
    body = False
    leg = False
    current_state = State.INITIAL
    tendangan_counter = 0  # Counter untuk menghitung jumlah tendangan
    tendang_benar_count = 0
    tendang_salah_count = 0
    state = "none"
    feedback_kuda = "none"
    feedback_transisi = "none"
    feedback_kick = "none"

    # Menggunakan video file sebagai input
    video_path = "B:/Abiyu/PA/silat-kicking-counter-asisst/raw_video/Tendangan_T_Kiri2.mp4"
    output_folder = "B:/Abiyu/PA/silat-kicking-counter-asisst/output_video/"
    
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    output_path = os.path.join(output_folder, "uhuy.mp4")
    
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Tidak dapat membuka video dari {video_path}.")
        return

    # Define the codec and create VideoWriter object to save video as MP4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Setengah lebar
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Setengah tinggi
    
    # Video writer object
    out = cv2.VideoWriter(output_path, fourcc, 20, (1280, 720))

    frame_skip = 1  # Skipping every 3rd frame to improve performance
    fps = 0
    frame_count = 0
    start_time = time.time()

    save_header_to_csv(csv_path)

    # Set up MediaPipe Pose
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            # print(f'KUDAAAAAAAAAAAAAAAAAAAAAAAA = {kuda}')
            # print(f'TRANSISIIIIIIIIIIIIIIIIIIII = {transisi}')
            # print(f'KICKKKKKKKKKKKKKKKKKKKKKKKK = {tendang}')
            ret, frame = cap.read()
            if not ret:
                print("Tidak dapat membaca frame.")
                break
            
            
            resized_frame = cv2.resize(frame, (1280, 720))
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()

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
                        face = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE.value].visibility > 0.8
                        body = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility > 0.8
                        leg = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility > 0.8
                        ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].visibility > 0.8
                        if face and body and leg and ankle:
                            current_state = State.KUDA2

                    case State.KUDA2:
                        # print("Kuda!")
                        state = "kuda-kuda"
                        if kuda_kuda():
                            feedback_kuda = feedback
                            threading.Thread(target=Play_buzzer, daemon=True).start()
                            save_to_csv(csv_path, position + " " + kick_type, tendangan_counter + 1, state, r_knee_angle, l_knee_angle, r_hip_angle, l_hip_angle, 
                                        foot_distance, hip_distance, shoulder_distance)
                            current_state = State.TRANSISI

                    case State.TRANSISI:
                        # print("Transisi!")
                        state = "Transisi"
                        if transition():
                            save_to_csv(csv_path, position + " " + kick_type, tendangan_counter + 1, state, r_knee_angle, l_knee_angle, r_hip_angle, l_hip_angle, 
                                        foot_distance, hip_distance, shoulder_distance)
                            feedback_transisi = feedback
                            current_state = State.TENDANG

                    case State.TENDANG:
                        state = "Tendang"
                        # print("Tendang!")
                        if kick():
                            feedback_kick = feedback
                            save_to_csv(csv_path, position + " " + kick_type, tendangan_counter + 1, state, r_knee_angle, l_knee_angle, r_hip_angle, l_hip_angle, 
                                        foot_distance, hip_distance, shoulder_distance)
                            tendangan_counter += 1
                            if transisi and tendang:
                                tendang_benar_count += 1
                            else :
                                tendang_salah_count += 1
                            print(f"Tendangan terdeteksi! Total tendangan: {tendangan_counter}")
                            current_state = State.AKHIR

                    case State.AKHIR:
                        state = "Back"
                        # print("back!")
                        if back(False):
                            current_state = State.INITIAL

            # Display the frame with data
            c_text_color = (0, 255, 255) #warna text di tengah
            r_text_color = (0, 255, 255) #warna text di kanan
            l_text_color = (0, 255, 255) #warna text di kiri

            c_rect_color = (0, 0, 0) #warna rect di tengah
            r_rect_color = (0, 0, 0) #warna rect di kanan
            l_rect_color = (0, 0, 0) #warna rect di kiri

            draw(resized_frame, f"kuda: {feedback_kuda}", 400, 30, c_text_color, c_rect_color) 
            draw(resized_frame, f"transisi: {feedback_transisi}", 400, 70, c_text_color, c_rect_color) 
            draw(resized_frame, f"kick: {feedback_kick}", 400, 110, c_text_color, c_rect_color) 

            if fps > 0:
                draw(resized_frame, f"FPS: {fps - 10:.0f}", 20, 30, l_text_color, l_rect_color)

            draw(resized_frame, f"jumlah Tendangan: {tendangan_counter}", 20, 100, l_text_color, l_rect_color)
            draw(resized_frame, f"Tendangan benar : {tendang_benar_count}", 20, 140, l_text_color, l_rect_color)
            draw(resized_frame, f"Tendangan salah : {tendang_salah_count}", 20, 180, l_text_color, l_rect_color)
            draw(resized_frame, f"State: {state}", 20, 220, l_text_color, l_rect_color)
            draw(resized_frame, f"jenis tendangan: {position} {kick_type}", 20, 260, l_text_color, l_rect_color)
            draw(resized_frame, f"facing: {facing}", 20, 300, l_text_color, l_rect_color)

            draw(resized_frame, f"r_knee_angle     : {r_knee_angle:.0f}", 900, 100, r_text_color, r_rect_color)
            draw(resized_frame, f"l_knee_angle     : {l_knee_angle:.0f}", 900, 140, r_text_color, r_rect_color)
            draw(resized_frame, f"r_hip_angle      : {r_hip_angle:.0f}", 900, 180, r_text_color, r_rect_color)
            draw(resized_frame, f"l_hip_angle      : {l_hip_angle:.0f}", 900, 220, r_text_color, r_rect_color)
            draw(resized_frame, f"foot_distance    : {foot_distance:.0f}", 900, 260, r_text_color, r_rect_color)
            draw(resized_frame, f"hip_distance     : {hip_distance:.0f}", 900, 300, r_text_color, r_rect_color)
            draw(resized_frame, f"shoulder_distance: {shoulder_distance:.0f}", 900, 340, r_text_color, r_rect_color)
            draw(resized_frame, f"crossing_leg     : {crossing_leg}", 900, 380, r_text_color, r_rect_color)

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