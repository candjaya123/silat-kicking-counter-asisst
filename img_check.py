import cv2
import mediapipe as mp
from module import tool as get  # Assuming this module has necessary utility functions like get.Angle, get.Distance, get.Y_angle

# Global variables to store results of landmark processing
l_knee_angle = 0
r_knee_angle = 0
r_hip_angle = 0
l_hip_angle = 0
foot_distance = 0
shoulder_slope = 0
hip_slope = 0
shoulder_distance = 0
facing = "not detected"
crossing_leg = False  # New variable for crossing leg detection

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Helper function to calculate angles, foot distance, and slopes
def process_landmarks(landmarks, w, h):
    global l_knee_angle, r_knee_angle, r_hip_angle, l_hip_angle, foot_distance, shoulder_slope, hip_slope, shoulder_distance, crossing_leg, hip_distance, facing

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

# Main function to process a single image
def main():
    global l_knee_angle, r_knee_angle, r_hip_angle, l_hip_angle, foot_distance, shoulder_slope, hip_slope, shoulder_distance, crossing_leg

    image_path = "./tes_img/sabit_kiri/tendang.png"
    output_path = "./tes_img/output/tendang_sabit_kiri.png"  # Path to save the processed image

    frame = cv2.imread(image_path)

    if frame is None:
        print(f"Error: Tidak dapat membuka gambar dari {image_path}.")
        return

    # Resize the frame for better processing
    frame = cv2.resize(frame, (1080, int(frame.shape[0] * 1080 / frame.shape[1])))

    # Get frame dimensions
    h, w = frame.shape[:2]

    # Recolor the image to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Set up MediaPipe Pose
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
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

        # Display the angle, distance, and slope information on the image
        cv2.putText(frame, f"l_knee_angle: {l_knee_angle:.2f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, f"r_knee_angle: {r_knee_angle:.2f}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, f"foot_distance: {foot_distance:.2f}", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, f"l_hip_angle: {l_hip_angle:.2f}", (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, f"r_hip_angle: {r_hip_angle:.2f}", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, f"shoulder_slope: {shoulder_slope:.2f}", (30, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, f"hip_slope: {hip_slope:.2f}", (30, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, f"shoulder_distance: {shoulder_distance:.2f}", (30, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, f"hip_distance: {hip_distance:.2f}", (30, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, f"facing: {facing}", (30, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, f"Crossing Leg: {'Yes' if crossing_leg else 'No'}", (30, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Save the processed image
        cv2.imwrite(output_path, frame)
        print(f"Processed image saved to {output_path}")

        # Display the processed image
        cv2.imshow('Angle, Distance & Slope Info', frame)

        # Wait indefinitely until a key is pressed
        cv2.waitKey(0)

    # Close windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
