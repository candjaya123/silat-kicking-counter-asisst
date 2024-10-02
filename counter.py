# main.py
from module.landmark import Value
from module import tool as get
from module.tool import Play_buzzer
import cv2

def main():
    # Open the webcam (0 is the default camera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        ################################ Defining Angle Varibble ############################################
        r_hip_angle = get.Angle(Value(frame,"r_shoulder"),Value(frame,"r_hip"),Value(frame,"r_knee"))
        l_hip_angle = get.Angle(Value(frame,"l_shoulder"),Value(frame,"l_hip"),Value(frame,"l_knee"))

        r_knee_angle = get.Angle(Value(frame,"r_hip"),Value(frame,"r_knee"),Value(frame,"r_angkle"))
        l_knee_angle = get.Angle(Value(frame,"l_hip"),Value(frame,"l_knee"),Value(frame,"l_angkle"))

        ear_nose_angle = get.Angle(Value(frame,"l_ear"),Value(frame,"nose"),Value(frame,"r_ear"))
        sholder_nose_angle = get.Angle(Value(frame,"r_shoulder"),Value(frame,"nose"),Value(frame,"l_shoulder"))
        
        r_hip_knee_angleY = get.Y_angle(Value(frame,"r_hip"),Value(frame,"r_knee"))
        l_hip_knee_angleY = get.Y_angle(Value(frame,"l_hip"),Value(frame,"l_knee"))

        r_hip_sholder_angleY = get.Y_angle(Value(frame,"r_hip"),Value(frame,"r_shoulder"))
        l_hip_sholder_angleY = get.Y_angle(Value(frame,"l_hip"),Value(frame,"l_shoulder"))

        r_sholder_nose_distance = get.Distance(Value(frame,"nose"),Value(frame,"r_shoulder"))
        l_sholder_nose_distance = get.Distance(Value(frame,"nose"),Value(frame,"l_shoulder"))


        # angel_test = get.Angle(Value(frame,"r_shoulder"),Value(frame,"r_elbow"),Value(frame,"r_wrist"))
        # cv2.putText(frame, f"Angle: {angel_test:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)



        cv2.imshow('Webcam Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
