import mediapipe as mp
import cv2

# Initialize MediaPipe pose solution
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Create a pose object that can be reused across frames
pose = mp_pose.Pose()

# Define a dictionary mapping keypoint names to their indices
keypoint_mapping = {
    "nose": 0, "l_ear": 7, "r_ear": 8, "l_shoulder": 11, "r_shoulder": 12,
    "l_hip": 23, "r_hip": 24, "l_knee": 25, "r_knee": 26, "l_ankle": 27, "r_ankle": 28
}

def Value(image, keypoint_name, process_image=True, draw_landmarks=False):
    """
    Extracts the value of the specified keypoint from the given image.

    Args:
        image (np.ndarray): The input image.
        keypoint_name (str): The name of the keypoint to extract.
        process_image (bool): If True, process the image for landmarks; 
                              if False, assumes image is pre-processed.
        draw_landmarks (bool): If True, draw landmarks on the image.
    
    Returns:
        Tuple: (x, y, z, visibility) for the keypoint, or None if not found.
    """
    # Convert image to RGB only if needed (i.e., when processing the image)
    if process_image:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = pose.process(image_rgb)
    else:
        result = pose.process(image)  # Assume it's already in RGB format

    if result.pose_landmarks and keypoint_name in keypoint_mapping:
        keypoint_index = keypoint_mapping[keypoint_name]
        keypoint = result.pose_landmarks.landmark[keypoint_index]
        visibility = keypoint.visibility >= 0.8  # You can lower this threshold based on needs
        
        if draw_landmarks:
            mp_drawing.draw_landmarks(
                image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

        return (keypoint.x, keypoint.y, keypoint.z, visibility) if visibility else (0, 0, 0, False)

    return None
