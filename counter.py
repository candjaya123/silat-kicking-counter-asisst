# main.py
from module.landmark import Value
from module import tool as get
from module.tool import Play_buzzer
import cv2
import enum 

class State(enum.Enum):
    INITIAL = 1
    KUDA2 = 2
    TRANSISI = 3
    AKHIR = 4

def Check(frame, name):
    buffer = Value(frame,name)
    if buffer == None or buffer[3] == False:
        return False
    
    return buffer[3]

def main():
    face = False
    body = False
    leg = False
    current_state = State.INITIAL   
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

        match current_state:
            case State.INITIAL:
                print("Current state: Initial")
                if Check(frame, "nose") and Check(frame, "r_ear") and Check(frame, "l_ear") and Check(frame, "r_ear"):
                    face = True
                if Check(frame, "l_shoulder") and Check(frame, "r_shoulder") and Check(frame, "l_hip") and Check(frame, "r_hip"):
                    body = True
                if Check(frame, "nose") and Check(frame, "r_ear") and Check(frame, "l_ear") and Check(frame, "r_ear"):
                    leg = True
                if face and body and leg:
                    current_state = State.KUDA2

            case State.KUDA2:
                print("Current state: Kuda2")
                # Logic for kuda2 state
                current_state = State.TRANSISI
kvjbdfsdfjhbg
            case State.TRANSISI:
                print("Current state: Transisi")
                # Logic for transisi state
                current_state = State.AKHIR

            case State.AKHIR:
                print("Current state: Akhir")
                # Logic for akhir state
                current_state = State.INITIAL

            case _:
                print("Error: Unrecognized state.")
                break  # Optional: exit loop on unrecognized state


        cv2.imshow('Webcam Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
