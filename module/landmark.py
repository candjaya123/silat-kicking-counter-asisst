import mediapipe as mp
import cv2

# Initialize MediaPipe pose solution
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

keypoint_mapping = {
    # All keypoint names and their indices
    "nose": 0, "l_eye_i": 1, "l_eye": 2, "l_eye_o": 3, "r_eye_i": 4, "r_eye": 5,
    "l_ear": 7, "r_ear": 8, "l_shoulder": 11, "r_shoulder": 12, "l_elbow": 13, 
    "r_elbow": 14, "l_wrist": 15, "r_wrist": 16, "l_hip": 23, "r_hip": 24, 
    "l_knee": 25, "r_knee": 26, "l_ankle": 27, "r_ankle": 28, "l_foot_index": 31,
    "r_foot_index": 32
}

def Value(image, keypoint_name):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = pose.process(image_rgb)

    if result.pose_landmarks and keypoint_name in keypoint_mapping:
        keypoint_index = keypoint_mapping[keypoint_name]
        keypoint = result.pose_landmarks.landmark[keypoint_index]
        visibility = keypoint.visibility >= 0.8
        
        mp_drawing.draw_landmarks(
            image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )

        return (keypoint.x, keypoint.y, keypoint.z, visibility) if visibility else (0, 0, 0, False)

    return None
