import cv2
import numpy as np
 
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
video_path = "B:/Abiyu/PA/silat-kicking-counter-asisst/video_testing/Tendangan_Depan_Kanan.mp4"
cap = cv2.VideoCapture(video_path)
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

while(cap.isOpened()):
    ret, frame = cap.read()
    height, width, _ = frame.shape

    # Resize the frame
    scale_percent = 50
    new_width = int(width * scale_percent / 100)
    new_height = int(height * scale_percent / 100)
    dim = (new_width, new_height)
    resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

    # Display the resized frame
    cv2.imshow('Resized Frame', resized)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()