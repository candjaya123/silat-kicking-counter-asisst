# landmark.py
import mediapipe as mp
import cv2

# Initialize MediaPipe pose solution
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Create a mapping between keypoint names and their corresponding indices
keypoint_mapping = {
    "nose": 0,
    "l_eye_i": 1,
    "l_eye": 2,
    "l_eye_o": 3,
    "r_eye_i": 4,
    "r_eye": 5,
    "r_eye_o": 6,
    "l_ear": 7,
    "r_ear": 8,
    "mouth_l": 9,
    "mouth_r": 10,
    "l_shoulder": 11,
    "r_shoulder": 12,
    "l_elbow": 13,
    "r_elbow": 14,
    "l_wrist": 15,
    "r_wrist": 16,
    "l_pinky": 17,
    "r_pinky": 18,
    "l_index": 19,
    "r_index": 20,
    "l_thumb": 21,
    "r_thumb": 22,
    "l_hip": 23,
    "r_hip": 24,
    "l_knee": 25,
    "r_knee": 26,
    "l_ankle": 27,
    "r_ankle": 28,
    "l_heel": 29,
    "r_heel": 30,
    "l_foot_index": 31,
    "r_foot_index": 32
}

def Value(image, keypoint_name):
    """
    Returns the value of a specific keypoint by name from MediaPipe pose landmarks.
    
    Args:
    - image: Input image in which keypoints will be detected.
    - keypoint_name: The name of the keypoint to return.
    
    Returns:
    - A tuple (x, y, z, visibility) representing the coordinates of the keypoint
      and its visibility status (True if visibility >= 0.8, False otherwise),
      or None if the keypoint is not detected.
    """
    # Convert the image color from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image to detect landmarks
    result = pose.process(image_rgb)
    
    # Check if landmarks are detected
    if result.pose_landmarks and keypoint_name in keypoint_mapping:
        keypoint_index = keypoint_mapping[keypoint_name]
        keypoint = result.pose_landmarks.landmark[keypoint_index]
        
        # Check visibility (True if visibility >= 0.8, False otherwise)
        if keypoint.visibility >= 0.8 :
            visibility = True
        else :
            visibility = False
        # Draw landmarks and connections on the frame
        mp_drawing.draw_landmarks(image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2))
        
        if visibility == True:
            return (keypoint.x, keypoint.y, keypoint.z, visibility)
        else :
            return (0, 0, 0, False)
    
    # Return None if no keypoints are detected or invalid keypoint name is provided
    return None
