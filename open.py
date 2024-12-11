import cv2

def play_video(video_path=None):
    """
    Opens and displays a video file or webcam feed.

    Parameters:
    video_path (str): Path to the video file. If None, it will use the webcam.
    """
    # Open video file or webcam
    if video_path is None:
        cap = cv2.VideoCapture(0)  # Webcam
    else:
        cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Unable to open video or webcam.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("End of video or no frame captured.")
            break

        # Display the frame
        cv2.imshow('Video Playback', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Example usage:
# To play a video file, replace 'path_to_video.mp4' with the actual file path.
# play_video('path_to_video.mp4')

# To use the webcam, call the function without arguments.
play_video("./output_video/kanan_depan.mp4")
