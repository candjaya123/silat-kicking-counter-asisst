import os
import time
import cv2
import mediapipe as mp
import imutils
from module import tool as get
from module.tool import Play_buzzer
import threading
import enum
import time
from threading import Timer

# Define states for the FSM
class State(enum.Enum):
    INITIAL = 1
    KUDA2 = 2
    TRANSISI = 3
    TENDANG = 4
    AKHIR = 5

class SilatCounter:
    def __init__(self, video_path, output_folder):
        self.video_path = video_path
        self.output_folder = output_folder
        self.current_state = State.INITIAL
        self.tendangan_counter = 0
        self.tendang_benar_count = 0
        self.tendang_salah_count = 0
        self.feedback_kuda = "none"
        self.feedback_transisi = "none"
        self.feedback_kick = "none"
        self.kuda = False
        self.transisi = False
        self.tendang = False

        self.l_knee_angle = 0
        self.r_knee_angle = 0
        self.r_hip_angle = 0
        self.l_hip_angle = 0
        self.foot_distance = 0
        self.shoulder_distance = 0
        self.hip_distance = 0
        self.crossing_leg = False
        self.facing = "not detected"
        self.l_hip = None
        self.l_knee = None
        self.l_ankle = None
        self.r_hip = None
        self.r_knee = None
        self.r_ankle = None
        self.position = "not detected"
        self.kick_type = "not detected"
        self.timer = None  # Timer untuk state reset

        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        # Ensure output folder exists
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        self.output_path = os.path.join(self.output_folder, "processed_video.mp4")

        # Helper function to calculate all necessary landmarks and store globally
    def process_landmarks(self, landmarks, w, h):
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
        self.l_hip, self.l_knee, self.l_ankle = LHip, LKnee, LAnkle
        self.r_hip, self.r_knee, self.r_ankle = RHip, RKnee, RAnkle
        # Calculate distances and angles
        self.foot_distance = get.Distance(LAnkle, RAnkle)
        self.l_knee_angle = get.Angle(LHip, LKnee, LAnkle)
        self.r_knee_angle = get.Angle(RHip, RKnee, RAnkle)
        self.r_hip_angle = get.Angle(RShoulder, RHip, RKnee)
        self.l_hip_angle = get.Angle(LShoulder, LHip, LKnee)
        # Calculate shoulder distance
        self.shoulder_distance = get.Distance(LShoulder, RShoulder)
        self.hip_distance = get.Distance(LHip, RHip)
        # Detect crossing leg
        if self.hip_distance > 30:
            if (RHip[0] < LHip[0] and RAnkle[0] > LAnkle[0]):  # Check if left ankle is opposite to the right hip and vice versa
                self.crossing_leg = True
            else:
                self.crossing_leg = False
        # Landmark for detecting facing direction
        nose = landmarks[mp.solutions.pose.PoseLandmark.NOSE.value]
        l_ear = landmarks[mp.solutions.pose.PoseLandmark.LEFT_EAR.value]
        r_ear = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_EAR.value]
        # Determine facing direction based on nose position
        if nose.x > l_ear.x and nose.x > r_ear.x:
            self.facing = "right"  # Nose closer to left shoulder, facing right
        else:
            self.facing = "left"   # Nose closer to right shoulder, facing left

    def kuda_kuda(self):
        ############## untuk keluar dari fungsi ini ##############
        # if crossing_leg or r_knee_angle < 140 or l_knee_angle < 140:
        #     return True
        ##########################################################
        if self.r_knee[0] > self.l_knee[0] :
            self.position = "left" if self.facing == "right" else "right"
        elif self.l_knee[0] > self.r_knee[0] :
            self.position = "right" if self.facing == "right" else "left"
        if (self.l_ankle[1] < self.l_knee[1] and self.r_ankle[1] < self.r_knee[1] and self.l_ankle[1] < self.r_knee[1] and self.r_ankle[1] < self.l_knee[1]) or self.foot_distance < 140:
            return False
        if self.position == "left":
            if 100 < self.r_knee_angle < 140 and 140 < self.l_knee_angle < 180 :
                self.feedback = "pas" 
                self.kuda = True
                return True
            elif self.r_knee_angle > 150:
                self.feedback = "kaki kanan kurang nekuk"
                return False
            elif self.r_knee_angle < 100:
                self.feedback = "kaki kanan terlalu nekuk"
                return False
            elif self.l_knee_angle < 150:
                self.feedback = "kaki kiri kurang lurus"
                return False
        elif self.position == "right":
            if 100 < self.l_knee_angle < 140 and 140 < self.r_knee_angle < 180:
                self.feedback = "pas" 
                self.kuda = True
                return True
            elif self.l_knee_angle > 150:
                self.feedback = "kaki kiri kurang nekuk"
                return False
            elif self.l_knee_angle < 100:
                self.feedback = "kaki kiri terlalu nekuk"
                return False
            elif self.r_knee_angle < 150:
                self.feedback = "kaki kanan kurang lurus"
                return False
        return False
    
    def transition(self):
        ################### untuk keluar dari fungsi ini ###################
        # if foot_distance > 200:
        #     return True
        ####################################################################

        # print(f"crossing_leg = {crossing_leg}")
        if self.crossing_leg:
            self.kick_type = "T kick"
            self.feedback = "pas"
            return True

        else:
            if self.l_ankle[1] < 600 or self.r_ankle[1] < 600:
                self.kick_type = "front kick" if self.hip_distance < 20 and self.shoulder_distance < 30 else "sickle kick"

            if self.position == "right":
                if self.r_knee and self.r_hip and self.r_knee[1] <= self.r_hip[1] and self.r_knee_angle < 150 and self.r_hip_angle < 110:  # y-coordinates, smaller means higher
                    self.transisi = True
                    self.feedback = "pas"
                    return True
                elif self.r_knee_angle > 150:
                    self.feedback = "kaki kanan kurang nekuk"
                    return False
                elif self.r_hip_angle > 110:
                    self.feedback = "kaki kanan kurang naik"
                    return False

            elif self.position == "left":
                if self.l_knee and self.l_hip and self.l_knee[1] <= self.l_hip[1] and self.l_knee_angle < 150 and self.l_hip_angle < 110:  # y-coordinates, smaller means higher
                    self.transisi = True
                    self.feedback = "pas"
                    return True
                elif self.l_knee_angle > 150:
                    self.feedback = "kaki kiri kurang nekuk"
                    return False
                elif self.l_hip_angle > 110:
                    self.feedback = "kaki kiri kurang naik"
                    return False
        return False

    def kick(self):
        ######### untuk keluar dari fungsi ini ###########
        # if back(True):
        #     return True
        ##################################################

        if self.l_knee_angle > 160 and self.r_knee_angle > 160:
            self.tendang = True
            self.feedback = "pas"
            return True
        elif self.l_knee_angle < 160:
            self.feedback = "kaki kiri kurang lurus"
            return False
        elif self.r_knee_angle < 160:
            self.feedback = "kaki kanan kurang lurus"
            return False

        return False

    def back(self, skip):
        if skip == False:
            self.kuda = False
            self.transisi = False
            self.tendang = False

        if self.l_knee_angle > 150 and self.r_knee_angle > 150 and self.foot_distance > 150 and self.crossing_leg == False:
            if self.position == "right" :
                if self.r_hip[0] < self.l_hip[0] and self.r_ankle[0] < self.l_ankle[0]:
                    return True
            elif self.position == "left" :
                if self.r_hip[0] > self.l_hip[0] and self.r_ankle[0] > self.l_ankle[0]:
                    return True
        return False

    def draw(self, frame, text, text_x, text_y, text_color, rect_color):
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

    def start_timer(self, timeout, reset_state):
        """Memulai timer untuk mereset state setelah timeout."""
        if self.timer:
            self.timer.cancel()  # Batalkan timer sebelumnya jika ada
        self.timer = threading.Timer(timeout, self.reset_state, args=(reset_state,))
        self.timer.start()

    def reset_state(self, state):
        """Reset ke state tertentu."""
        print(f"State di-reset ke {state.name} karena timeout.")
        self.current_state = state

    def cancel_timer(self):
        """Batalkan timer jika ada."""
        if self.timer:
            self.timer.cancel()
            self.timer = None


# Main function
    def run(self):
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
        video_path = "B:/Abiyu/PA/silat-kicking-counter-asisst/raw_video/Tendangan_Depan_Kiri.mp4"
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
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Setengah lebar
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Setengah tinggi

        # Video writer object
        out = cv2.VideoWriter(output_path, fourcc, 20, (1280, 720))

        frame_skip = 2  # Skipping every 3rd frame to improve performance
        fps = 0
        frame_count = 0
        start_time = time.time()

        # Set up MediaPipe Pose
        with self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                print(f'KUDAAAAAAAAAAAAAAAAAAAAAAAA = {self.kuda}')
                print(f'TRANSISIIIIIIIIIIIIIIIIIIII = {self.transisi}')
                print(f'KICKKKKKKKKKKKKKKKKKKKKKKKK = {self.tendang}')
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
                    self.process_landmarks(results.pose_landmarks.landmark, w, h)

                    # Use mp_drawing to draw the landmarks
                    self.mp_drawing.draw_landmarks(
                        resized_frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                        self.mp_drawing.DrawingSpec(color=(66, 245, 230), thickness=2, circle_radius=2)
                    )

                    match current_state:
                        case State.INITIAL:
                            face = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE.value].visibility > 0.5
                            body = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility > 0.5
                            leg = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility > 0.5
                            if face and body and leg:
                                current_state = State.KUDA2

                        case State.KUDA2:
                            # print("Kuda!")
                            state = "kuda-kuda"
                            self.start_timer(5, State.INITIAL)
                            if self.kuda_kuda():
                                feedback_kuda = self.feedback
                                threading.Thread(target=Play_buzzer, daemon=True).start()
                                current_state = State.TRANSISI

                        case State.TRANSISI:
                            # print("Transisi!")
                            self.start_timer(5, State.INITIAL)
                            state = "Transisi"
                            if self.transition():
                                feedback_transisi = self.feedback
                                current_state = State.TENDANG

                        case State.TENDANG:
                            state = "Tendang"
                            # print("Tendang!")
                            self.start_timer(5, State.INITIAL)
                            if self.kick():
                                feedback_kick = self.feedback
                                tendangan_counter += 1
                                if self.kuda and self.transisi and self.tendang:
                                    tendang_benar_count += 1
                                else :
                                    tendang_salah_count += 1
                                print(f"Tendangan terdeteksi! Total tendangan: {tendangan_counter}")
                                current_state = State.AKHIR

                        case State.AKHIR:
                            self.cancel_timer()
                            # state = "Back"
                            # print("back!")
                            if self.back(False):
                                current_state = State.INITIAL

                # Display the frame with data
                c_text_color = (0, 255, 255) #warna text di tengah
                r_text_color = (0, 255, 255) #warna text di kanan
                l_text_color = (0, 255, 255) #warna text di kiri

                c_rect_color = (0, 0, 0) #warna rect di tengah
                r_rect_color = (0, 0, 0) #warna rect di kanan
                l_rect_color = (0, 0, 0) #warna rect di kiri

                self.draw(resized_frame, f"kuda: {feedback_kuda}", 400, 30, c_text_color, c_rect_color) 
                self.draw(resized_frame, f"transisi: {feedback_transisi}", 400, 70, c_text_color, c_rect_color) 
                self.draw(resized_frame, f"kick: {feedback_kick}", 400, 110, c_text_color, c_rect_color) 

                self.draw(resized_frame, f"FPS: {fps:.0f}", 20, 30, l_text_color, l_rect_color)

                self.draw(resized_frame, f"jumlah Tendangan: {tendangan_counter}", 20, 100, l_text_color, l_rect_color)
                self.draw(resized_frame, f"Tendangan benar : {tendang_benar_count}", 20, 140, l_text_color, l_rect_color)
                self.draw(resized_frame, f"Tendangan salah : {tendang_salah_count}", 20, 180, l_text_color, l_rect_color)
                self.draw(resized_frame, f"State: {state}", 20, 220, l_text_color, l_rect_color)

                self.draw(resized_frame, f"r_knee_angle     : {self.r_knee_angle:.0f}", 900, 100, r_text_color, r_rect_color)
                self.draw(resized_frame, f"l_knee_angle     : {self.l_knee_angle:.0f}", 900, 140, r_text_color, r_rect_color)
                self.draw(resized_frame, f"r_hip_angle      : {self.r_hip_angle:.0f}", 900, 180, r_text_color, r_rect_color)
                self.draw(resized_frame, f"l_hip_angle      : {self.l_hip_angle:.0f}", 900, 220, r_text_color, r_rect_color)
                self.draw(resized_frame, f"foot_distance    : {self.foot_distance:.0f}", 900, 260, r_text_color, r_rect_color)
                self.draw(resized_frame, f"hip_distance     : {self.hip_distance:.0f}", 900, 300, r_text_color, r_rect_color)
                self.draw(resized_frame, f"shoulder_distance: {self.shoulder_distance:.0f}", 900, 340, r_text_color, r_rect_color)
                self.draw(resized_frame, f"crossing_leg     : {self.crossing_leg}", 900, 380, r_text_color, r_rect_color)

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
    video_path = "path_to_video.mp4"
    output_folder = "output_video_folder"
    pose_estimator = SilatCounter(video_path, output_folder)
    SilatCounter.run()