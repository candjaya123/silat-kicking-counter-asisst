import cv2
import mediapipe as mp
import numpy as np
import imutils
import time 
from playsound import playsound
import threading

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# 3 points exis

def calculate_angel (var1,var2,var3):
    A = np.array(var1) #first
    B = np.array(var2) #second
    C = np.array(var3) #third
    
    ab = (((A[0]-B[0])*(C[0]-B[0])) + ((A[1]-B[1])*(C[1]-B[1])))

    a = np.sqrt(np.power((A[0]-B[0]),2)+ np.power((A[1]-B[1]),2))
    b = np.sqrt(np.power((C[0]-B[0]),2)+ np.power((C[1]-B[1]),2)) 
    theta = np.arccos((ab)/ (a*b))

    angle = np.abs(theta*180.0/np.pi)

    
    return angle 

# 2 points exis but 1 point only use y axis 
def calculate_angley (b,c):
    
    B = np.array(b) #second
    C = np.array(c) #third
    
    ab = (((-B[1])*(C[1]-B[1])))

    a = B[1]
    b = np.sqrt(np.power((C[0]-B[0]),2)+ np.power((C[1]-B[1]),2)) 
    theta = np.arccos((ab)/ (a*b))

    angle = np.abs(theta*180.0/np.pi)

    
    return angle 

def jarak(var1,var2):
    A = np.array(var1) #first
    B = np.array(var2) #second
    panj = np.sqrt(np.power((A[0]-B[0]),2)+ np.power((A[1]-B[1]),2))
    return panj

def play_buzzer():
    playsound('D:\PENS\PROYEK AKHIR\CODING\\buzzer1.mp3')

video_path = "B:/Abiyu/PA/silat-kicking-counter-asisst/raw_video/Tendangan_Depan_Kanan.mp4"
# Inisialisasi kamera
cap = cv2.VideoCapture(video_path)

## deklaasi
#state 
state = 1
current_state = None
prev_state = None 
hem = 0

buzzCount = 0
buzzState = True

#count

count_D_kanan = 0
count_T_kanan = 0
count_S_kanan = 0
count_D_kiri = 0
count_T_kiri = 0
count_S_kiri = 0

count_false_D_kanan = 0
count_false_T_kanan = 0
count_false_S_kanan = 0
count_false_D_kiri = 0
count_false_T_kiri = 0
count_false_S_kiri = 0

#pengondisian 
kondisi = None

# feedback 
feedback = None

# time aktif 
angkat = 0           
good_time = 0            

# previous time
prev_time_q3_1 = None 
prev_time_q3_2 = None 
prev_time_q3_3 = None 

prev_time_q4_1 = None 
prev_time_q4_2 = None 
prev_time_q4_3 = None 

prev_time_p1_1 = None 
prev_time_p1_2 = None 
prev_time_p1_3 = None 

prev_time_q2_1 = None 
prev_time_q2_2 = None 
prev_time_q2_3 = None

prev_time_XRHip_1 = None 
prev_time_XRHip_2 = None 
prev_time_XRHip_3 = None

prev_time_XRShoulder_1 = None 
prev_time_XRShoulder_2 = None 
prev_time_XRShoulder_3 = None

prev_time_XLHip_1 = None 
prev_time_XLHip_2 = None 
prev_time_XLHip_3 = None

prev_time_YLAnkle_1 = None 
prev_time_YLAnkle_2 = None 
prev_time_YLAnkle_3 = None

prev_time_p4_1 = None 
prev_time_p4_2 = None 
prev_time_p4_3 = None

prev_time_q5_1 = None 
prev_time_q5_2 = None 
prev_time_q5_3 = None

prev_time_YRAnkle_1 = None 
prev_time_YRAnkle_2 = None 
prev_time_YRAnkle_3 = None

feed_count = None
feed_count1 = None
feed_count2 = None
feed_count3 = None
feed_count4 = None
feed_count5 = None
feed_count6 = None
feed_count7 = None
feed_count8 = None
feed_count9 = None
feed_count10 = None
feed_count11 = None
feed_count12 = None
feed_count13 = None
feed_count14 = None
feed_count15 = None
feed_count16 = None
feed_count17 = None
feed_count18 = None
feed_count19 = None

# var baru 
q4_bar = None

# pengondisian 
ayunan_dec = None
XRHip_bar = None
XRKnee_bar = None
p1_bar = None

# tubuh
tubuh_ber = None

# kepala 
had_kepala = None

#langkah 
Langkah = None

# Angkat
ngangkat = None 

# visibility
wajah = None
badan = None 
kaki = None
v_state2 = None

# posisi
posisi = None

# time state 
time_state3 = 0
time_state4 = None
good_time3 = None
good_time4 = None

pos_had = None
## setup mediapipe  instance
with mp_pose.Pose(static_image_mode=False, min_detection_confidence = 0.5, min_tracking_confidence = 0.5, model_complexity=1) as pose:
     while cap.isOpened():
        ret,frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
        frame = imutils.resize(frame,width=640,height=480)
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Get height and width of the frame.
        h, w = frame.shape[:2]


        #frame = imutils.resize(frame, width = 750, height = 750)

        # Recolor image to RGB
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        # Make detection
        results = pose.process(image)

        #Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        #image = cv2.flip(image,1)

        #Ekstrak landmark
        try:
            landmarks = results.pose_landmarks.landmark
            # xy axis
            # Left
            LHip  = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x*w,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y*h]
            LKnee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x*w,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y*h]
            LShoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x*w,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y*h]
            LAnkle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x*w,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y*h]
            LEye = [landmarks[mp_pose.PoseLandmark.LEFT_EYE_INNER.value].x*w,landmarks[mp_pose.PoseLandmark.LEFT_EYE_INNER.value].y*h] 
            LEar = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x*w,landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y*h] 

            #Right
            RHip  = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x*w,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y*h]
            RKnee  = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x*w,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y*h]
            REye  = [landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER.value].x*w,landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER.value].y*h]
            REar  = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x*w,landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y*h]
            RShoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x*w,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y*h]
            RAnkle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x*w,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y*h]
            RFIndex = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x*w,landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y*h]
            RHeel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x*w,landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y*h]
            
            # Center
            Nose  = [landmarks[mp_pose.PoseLandmark.NOSE.value].x*w,landmarks[mp_pose.PoseLandmark.NOSE.value].y*h]

            ## Visibility
            # Right
            VRShoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility]
            VRHip = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].visibility]
            VRKnee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility]
            VRAnkle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].visibility]

            # Left
            VLShoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility]
            VLHip = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].visibility]
            VLKnee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility]
            VLAnkle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].visibility]
            
            # Center
            VNose = [landmarks[mp_pose.PoseLandmark.NOSE.value].visibility]
            VREar = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].visibility]
            VLEar = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].visibility]


            #Angle
                #ARHip = calculate_angel(LShoulder,RHip,RKnee)

            # Angle 3 points
           
            q1 = calculate_angel(RShoulder,RHip,RKnee)
            q2 = calculate_angel(RHip,RKnee,RAnkle)
            q3 = calculate_angel(RShoulder,Nose,LShoulder)
            q4 = calculate_angel(RHip,Nose,LHip)
            q5 = calculate_angel(LHip,LKnee,LAnkle)
            q6 = calculate_angel(LShoulder,LHip,LKnee)
            q7 = calculate_angel(LEar,Nose,REar)

            # Angle 2 points
            p1 = calculate_angley(RHip,RKnee)
            p2 = calculate_angley(RHip,RShoulder)
            p3 = calculate_angley(LHip,LShoulder)
            p4 = calculate_angley(LHip,LKnee)
            
            # Distance
            j1 = jarak(Nose,RShoulder)
            j2 = jarak(Nose,LShoulder)

            ## pengondisian
            if state == 1:
               kondisi = None
               angkat =0
               ngangkat = None
               
               # wajah 
               if ((VNose[0] > 0.9) and (VREar[0] > 0.9) and (VLEar[0] > 0.9)):
                    wajah = "ada"
                    
               if ((VNose[0] < 0.9) and (VREar[0] < 0.9) and (VLEar[0] < 0.9)):
                    wajah = "tidak ada"
                    
               
               #badan 
               if (((VRShoulder[0] > 0.6) and (VLShoulder[0] > 0.6)) and ((VRHip[0] > 0.6) and (VLHip[0] >0.6))):
                    badan = "ada"
                    
               if (((VRShoulder[0] < 0.6) and (VLShoulder[0] < 0.6)) and ((VRHip[0] < 0.6) and (VLHip[0] <0.6))):
                    badan = "tidak ada"
                    

               #kaki
               if (((VRKnee[0] > 0.7) and (VLKnee[0] > 0.7)) and ((VRAnkle[0] > 0.7) and (VLAnkle[0] >0.7))):
                    kaki = "ada"
          
               if (((VRKnee[0] < 0.7) and (VLKnee[0] < 0.7)) and ((VRAnkle[0] < 0.7) and (VLAnkle[0] <0.7))):
                    kaki = "tidak ada"
                    

               # kondisi 
               if ((wajah == "ada") and (badan == "ada") and (kaki == "ada")):
                    state = 2
                    
               
            #kuda
            if state == 2 :
               ayunan_dec =None
               tubuh_ber = None
               Langkah = None
               if ((RShoulder[0]<=LShoulder[0]) and (RHip[0]<=LHip[0])) and  ((q3 >= 25) and (q4>= 10)) and ( v_state2 == "ada") : # posisi badan
                    pos_had = "Tubuh menghadap kamera"

                    if ((q7 <= 100) and (j2 <=j1)): # posisi kepala 
                         if (q5 <=155 and q6 <=155): # badan dan kaki 
                              if (RShoulder[1]<= RHip[1] <= RKnee[1] <= RAnkle[1]):
                                   posisi = "kiri"           
                                   state = 3
                                   buzzCount += 1

                    if ((q7 <= 100) and (j2 >=j1)):  #posisi kepala
                         if (q2 <=155 and q1 <=155): # badan dan kaki
                              if (LShoulder[1]<= LHip[1] <= LKnee[1] <= LAnkle[1]):
                                   posisi = "kanan"
                                   state = 3
                                   buzzCount += 1
               
               if ((q4 < 9)):          
                    pos_had = "Badan menghadap samping kamera"
                    state = 2

               if (RShoulder[0]>=LShoulder[0]) and (RHip[0]>=LHip[0]) and (q4 > 9) :
                    pos_had = "posisi membelakangi kamera"
                    state = 2   

               ## Visibility
               # wajah     
               if ((VNose[0] > 0.7) and (VREar[0] > 0.7) and (VLEar[0] > 0.7)):
                    wajah = "ada"
                    feedback = None
                    
               if ((VNose[0] < 0.7) and (VREar[0] < 0.7) and (VLEar[0] < 0.7)):
                    wajah = "wajah tidak ada"
                    feedback = "wajah tidak ada"
                    state =1 

               #badan 
               if (((VRShoulder[0] > 0.6) and (VLShoulder[0] > 0.6)) and ((VRHip[0] > 0.6) and (VLHip[0] >0.6))):
                    badan = "ada"
                    feedback = None
                    
               if (((VRShoulder[0] < 0.6) and (VLShoulder[0] < 0.6)) and ((VRHip[0] < 0.6) and (VLHip[0] <0.6))):
                    badan = "tidak ada"
                    feedback = "badan tidak ada"
                    state = 1
               
               #kaki
               if (((VRKnee[0] > 0.7) and (VLKnee[0] > 0.7)) and ((VRAnkle[0] > 0.7) and (VLAnkle[0] >0.7))):
                    kaki = "ada"
                    feedback = None
          
               if (((VRKnee[0] < 0.7) or (VLKnee[0] < 0.7)) or ((VRAnkle[0] < 0.7) or (VLAnkle[0] <0.7))):
                    kaki = "tidak ada"
                    feedback = "kaki tidak ada"
                    state = 1
               
               # kondisi visibility
               if ((wajah == "ada") and (badan == "ada") and (kaki == "ada")):
                    v_state2 = "ada"
                    
          
            if state == 3:
               if(buzzCount == 1):
                    buzzCount += 1
                    play_buzzer()
                    buzzState = True
          
               # q4
               if prev_time_q4_1 is not None :
                    if q4 > prev_time_q4_1:
                         feed_count = "increment" 
                         prev_time_q4_2 =prev_time_q4_1 
                         

                    elif q4 < prev_time_q4_1 :
                         feed_count = "decrement"
                         prev_time_q4_2 =prev_time_q4_1 
                         
                    else :
                         feed_count = " sama"
                         prev_time_q4_2 =prev_time_q4_1 
                          
               prev_time_q4_1 = q4

               # p1
               if prev_time_p1_1 is not None :
                    if p1 > prev_time_p1_1:
                         feed_count2 = "increment" 
                         prev_time_p1_2 =prev_time_p1_1 
                         

                    elif p1 < prev_time_p1_1 :
                         feed_count2 = "decrement"
                         prev_time_p1_2 =prev_time_p1_1 
                         
                    else :
                         feed_count2 = " sama"
                         prev_time_p1_2 =prev_time_p1_1 
                          
               prev_time_p1_1 = p1


               # q2    
               if prev_time_q2_1 is not None :
                    if q2 > prev_time_q2_1:
                         feed_count4 = "increment" 
                         prev_time_q2_2 =prev_time_q2_1 
                         

                    elif q2 < prev_time_q2_1 :
                         feed_count4 = "decrement"
                         prev_time_q2_2 =prev_time_q2_1 
                         
                    else :
                         feed_count4 = " sama"
                         prev_time_q2_2 =prev_time_q2_1 
                          
               prev_time_q2_1 = q2

               
               # XRHip
               if prev_time_XRHip_1 is not None :
                    if RHip[0] > prev_time_XRHip_1:
                         feed_count6 = "increment" 
                         prev_time_XRHip_2 =prev_time_XRHip_1 
                         

                    elif RHip[0] < prev_time_XRHip_1 :
                         feed_count6 = "decrement"
                         prev_time_XRHip_2 =prev_time_XRHip_1 
                         
                    else :
                         feed_count6 = " sama"
                         prev_time_XRHip_2 =prev_time_XRHip_1 
                          
               prev_time_XRHip_1 = RHip[0]
               
               
               # q3
               if prev_time_q3_1 is not None :
                    if q3 > prev_time_q3_1:
                         feed_count8 = "increment" 
                         prev_time_q3_2 =prev_time_q3_1 
                         

                    elif q3 < prev_time_q3_1 :
                         feed_count8 = "decrement"
                         prev_time_q3_2 =prev_time_q3_1 
                         
                    else :
                         feed_count8 = " sama"
                         prev_time_q3_2 =prev_time_q3_1 
                          
               prev_time_q3_1 = q3

               #LAnkle
               if prev_time_YLAnkle_1 is not None :
                    if  LAnkle[1] > prev_time_YLAnkle_1:
                         feed_count10 = "increment" 
                         prev_time_YLAnkle_2 =prev_time_YLAnkle_1 
                         

                    elif LAnkle[1] < prev_time_YLAnkle_1 :
                         feed_count10 = "decrement"
                         prev_time_YLAnkle_2 =prev_time_YLAnkle_1 
                         
                    else :
                         feed_count10 = "sama"
                         prev_time_YLAnkle_2 =prev_time_YLAnkle_1 
                          
               prev_time_YLAnkle_1 = LAnkle[1]
               
                
               # XLHip
               if prev_time_XLHip_1 is not None :
                    if LHip[0] > prev_time_XLHip_1:
                         feed_count12 = "increment" 
                         prev_time_XLHip_2 =prev_time_XLHip_1 
                         

                    elif LHip[0] < prev_time_XLHip_1 :
                         feed_count12 = "decrement"
                         prev_time_XLHip_2 =prev_time_XLHip_1 
                         
                    else :
                         feed_count12 = " sama"
                         prev_time_XLHip_2 =prev_time_XLHip_1 
                          
               prev_time_XLHip_1 = LHip[0]
               
               
               # p4
               if prev_time_p4_1 is not None :
                    if p4 > prev_time_p4_1:
                         feed_count14 = "increment" 
                         prev_time_p4_2 =prev_time_p4_1 
                         

                    elif p4 < prev_time_p4_1 :
                         feed_count14 = "decrement"
                         prev_time_p4_2 =prev_time_p4_1 
                         
                    else :
                         feed_count14 = " sama"
                         prev_time_p4_2 =prev_time_p4_1 
                          
               prev_time_p4_1 = p4
               
              
               # q5
               if prev_time_q5_1 is not None :
                    if q5 > prev_time_q5_1:
                         feed_count16 = "increment" 
                         prev_time_q5_2 =prev_time_q5_1 
                         

                    elif q5 < prev_time_q5_1 :
                         feed_count16 = "decrement"
                         prev_time_q5_2 =prev_time_q5_1 
                         
                    else :
                         feed_count16 = " sama"
                         prev_time_q5_2 =prev_time_q5_1 
                          
               prev_time_q5_1 = q5
               

               # YRAnkle
               if prev_time_YRAnkle_1 is not None :
                    if RAnkle[1] > prev_time_YRAnkle_1:
                         feed_count18 = "increment" 
                         prev_time_YRAnkle_2 =prev_time_YRAnkle_1 
                         

                    elif RAnkle[1] < prev_time_YRAnkle_1 :
                         feed_count18 = "decrement"
                         prev_time_YRAnkle_2 =prev_time_YRAnkle_1 
                         
                    else :
                         feed_count18 = " sama"
                         prev_time_YRAnkle_2 =prev_time_YRAnkle_1 
                          
               prev_time_YRAnkle_1 = RAnkle[1]


               time_state3 += 1
               good_time3 = (1 / fps) * time_state3

               # time
               if (good_time3 > 3)  :
                    feedback ="time's up"
                    time_state3 = 0
                    state = 1
               


               # tumpuan badan kiri 
               if posisi == "kiri" :
                    ## Tendangan depan 
                    # transisi badan 
                    if (q4 <=10  and q3 <= 20) and ((q7 <= 100)) and (((feed_count == "decrement" ) and (RHip[0]<=LHip[0]) and (feed_count6 == "increment" )) and ((feed_count8 == "decrement" ) and (RShoulder[0]<=LShoulder[0]))) or (((feed_count8 == "increment" ) and (RShoulder[0]>=LShoulder[0])) and ((feed_count == "increment" ) and (RHip[0]>=LHip[0]) and (feed_count6 == "increment" ))) :
                         tubuh_ber = "iyaa"

                    if (q4 >=10  and q3 >= 20) and ((feed_count == "increment" ) and (RHip[0] <= LHip[0])) and ((feed_count8 == "increment" ) and (RShoulder[0]<=LShoulder[0]))  and (tubuh_ber=="iyaa"):
                         feedback = "gaajadi nendang kesamping?"
                         time_state3 = 0
                         state =2 
                    
                    # pergerakan kaki 
                    if (feed_count2 == "decrement" ) and (RHip[0] <=RKnee[0]) and (tubuh_ber=="iyaa") : # tinggi lutut 
                         if feed_count4 == "decrement"   and q2 <=153: # sudut lutut 
                              ayunan_dec = "decrement"
                              kondisi = "depan kanan"
                              feedback = None
                              time_state3 = 0
                              state = 4
                         
                    
                    # tubuh tidak transisi tapi nendang
                    if ((RShoulder[0]<=LShoulder[0]) and (RHip[0]<=LHip[0])) and  ((q3 >= 30) and (q4>= 1)) :
                         if (p1 <=150) and (RAnkle[1]<=LKnee[1]) and (q2 >=130):
                              feedback = "badan harus transisi"
                              time_state3 = 0
                              state = 2

                    # badan menghadap samping tapi tidak melakukan tendangan 
                    if ((RShoulder[0]>=LShoulder[0]) and (RHip[0]>=LHip[0])) and  ((q3 >= 30) and (q4>= 10)) :
                         if (RAnkle[1]>=LKnee[1]+2) and  (not(((feed_count18 == "decrement")  and (RKnee[0] >=RHip[0])) )):
                              feedback = "gaa jadi nendang?"
                              time_state3 = 0
                              state = 1
                         
                    #Langkah u/ Tend T 
                    if ((RShoulder[0]<=LShoulder[0]) and (RHip[0]<=LHip[0])) and  ((q3 >= 20) and (q4>= 8)) :
                         if (q2 >= 150 and p1 >= 150 ):
                              if (RAnkle[0] >= LAnkle[0]):
                                   if (p1<=170):
                                        Langkah = "iyes"
                    #transisi / angkat 
                    if  Langkah == "iyes":
                         ngangkat = "heem"
                         if   (((feed_count10 == "decrement" ) and (RKnee[1] + 10 >= LAnkle[1])) and (q5<=140) and (q4 >=10)  ):
                                   kondisi = "T kiri"
                                   feedback = None
                                   time_state3 = 0
                                   ngangkat = None
                                   state = 4 
                    

               
               if posisi == "kanan":
                    
                    ## Tendangan depan 
                    # transisi badan 
                    if (q4 <=10  and q3 <= 20) and ((q7 <= 20)) and (((feed_count == "decrement" ) and (RHip[0]<=LHip[0]) and (feed_count12 == "decrement" )) and ((feed_count8 == "decrement" ) and (RShoulder[0]<=LShoulder[0]))) or (((feed_count8 == "increment" ) and (RShoulder[0]>=LShoulder[0])) and ((feed_count == "increment" ) and (RHip[0]>=LHip[0]) and (feed_count12 == "decrement" ))) :
                         tubuh_ber = "iyaa"

                    if (q4 >=10  and q3 >= 20) and ((feed_count == "increment" ) and (RHip[0] <= LHip[0])) and ((feed_count8 == "increment" ) and (RShoulder[0]<=LShoulder[0]))  and (tubuh_ber=="iyaa"):
                         feedback = "gaajadi nendang kesamping?"
                         time_state3 = 0
                         state =2 
                    
                    
                    # pergerakan kaki 
                    if (feed_count14 == "decrement" ) and (LHip[0] >=LKnee[0]) and (tubuh_ber=="iyaa") : # tinggi lutut 
                         if feed_count16 == "decrement" and q5 <=153: # sudut lutut 
                              ayunan_dec = "decrement"
                              kondisi = "depan kiri"
                              feedback = None
                              time_state3 = 0
                              state = 4
                         
                    # tubuh tidak transisi tapi nendang 
                    if ((RShoulder[0]<=LShoulder[0]) and (RHip[0]<=LHip[0])) and  ((q3 >= 30) and (q4>= 1)) :
                         if (p4 <=150) and LAnkle[1]<=RKnee[1]:
                              feedback = "badan harus transisi"
                              time_state3 = 0
                              state = 2

                    # badan menghadap samping tapi tidak melakukan tendangan 
                    if ((RShoulder[0]>=LShoulder[0]) and (RHip[0]>=LHip[0])) and  ((q3 >= 30) and (q4>= 10)) :
                         if (LAnkle[1]>=RKnee[1]+2) and  (not(((feed_count10 == "decrement") and (LKnee[0] <=LHip[0])) )):
                              feedback = "gaa jadi nendang?"
                              time_state3 = 0
                              state = 1
                       
                    #Langkah u/ Tend T 
                    if ((RShoulder[0]<=LShoulder[0]) and (RHip[0]<=LHip[0])) and  ((q3 >= 25) and (q4>= 10)) :
                         if (q5 >= 140 and p4 >= 140 ):
                              if (RAnkle[0] >= LAnkle[0]):
                                   if (p4<=170):
                                        Langkah = "iyes"
                    #transisi / angkat 
                    if  Langkah == "iyes":
                         ngangkat = "heem"
                         if   (( (feed_count18 == "decrement" ) and (LKnee[1] + 10 >= RAnkle[1])) and (q2<=140) and (q4 >=10)  ):
                                   kondisi = "T kanan"
                                   feedback = None
                                   time_state3 = 0
                                   ngangkat = None
                                   state = 4 
                    

               ## Visibility
               # wajah     
               if ((VNose[0] > 0.7) and (VREar[0] > 0.7) and (VLEar[0] > 0.7)):
                    wajah = "ada"
                    
               if ((VNose[0] < 0.7) and (VREar[0] < 0.7) and (VLEar[0] < 0.7)):
                    wajah = "tidak ada"
                    feedback = "wajah tidak ada"
                    time_state3 = 0
                    state = 1

               #badan 
               if (((VRShoulder[0] > 0.6) and (VLShoulder[0] > 0.6)) and ((VRHip[0] > 0.6) and (VLHip[0] >0.6))):
                    badan = "ada"
                    
               if (((VRShoulder[0] < 0.6) and (VLShoulder[0] < 0.6)) and ((VRHip[0] < 0.6) and (VLHip[0] <0.6))):
                    badan = "tidak ada"
                    feedback = "badan tidak ada"
                    time_state3 = 0
                    state = 1
               
               #kaki
               if (((VRKnee[0] > 0.7) and (VLKnee[0] > 0.7)) and ((VRAnkle[0] > 0.7) and (VLAnkle[0] >0.7))):
                    kaki = "ada"
          
               if (((VRKnee[0] < 0.7) or (VLKnee[0] < 0.7)) or ((VRAnkle[0] < 0.7) or (VLAnkle[0] <0.7))):
                    kaki = "tidak ada"
                    feedback = "kaki tidak ada"
                    time_state3 = 0
                    state = 1
               
               # kondisi visibility
               if ((wajah == "ada") and (badan == "ada") and (kaki == "ada")):
                    v_state2 = "ada"

            if state == 3:  # Kondisi kuda
               if buzzCount > 0 and buzzState:
                    if buzzer_thread is None or not buzzer_thread.is_alive():
                         buzzer_thread = threading.Thread(target=play_buzzer)
                         buzzer_thread.start()
                         buzzState = False  # Pastikan buzzState dimatikan setelah buzzer dimainkan agar tidak berulang               
               
            if state == 4 :
               v_state2 = None
               if kondisi == "depan kanan":
                    # p1
                    if prev_time_p1_1 is not None :
                         if p1 > prev_time_p1_1:
                              feed_count2 = "increment" 
                              prev_time_p1_2 =prev_time_p1_1 
                              

                         elif p1 < prev_time_p1_1 :
                              feed_count2 = "decrement"
                              prev_time_p1_2 =prev_time_p1_1 
                              
                         else :
                              feed_count2 = " sama"
                              prev_time_p1_2 =prev_time_p1_1 
                              
                    prev_time_p1_1 = p1

                   

                    if (q2 >= 145)  :
                         if (p1 <=90) and (RAnkle[0]>=RKnee[0]):
                              count_D_kanan +=1
                              kondisi = None
                              posisi = None
                              #winsound.MessageBeep(winsound.MB_ICONASTERISK)
                              if(buzzState == True):
                                   buzzState = False
                                   buzzCount = 0
                              state = 1   
                         if (p1 >=130) and ((((feed_count2 == "increment")) and (RHip[0]<=RKnee[0]) ) or (((feed_count2 == "decrement") ) and (RHip[0]>=RKnee[0]) )):
                              count_false_D_kanan +=1
                              feedback = "paha kurang diangkat"
                              kondisi = None
                              posisi = None
                              if(buzzState == True):
                                   buzzState = False
                                   buzzCount = 0
                              #winsound.MessageBeep(winsound.MB_ICONHAND)
                              state = 1   
                              #playsound("C:/Users/Asus/Downloads/mixkit-wrong-long-buzzer-954.wav")
               
            
               if kondisi == "T kiri":
                    if (  (q4 >= 10)  and (q5 >=155)):
                         if p4 <= 95 and (LAnkle[0]>=LKnee[0]) :
                              count_T_kiri +=1
                              kondisi = None
                              posisi = None
                              if(buzzState == True):
                                   buzzState = False
                                   buzzCount = 0
                              state = 1
                         if (p4 >=150) and (LAnkle[1] >= RKnee[1]+3):
                              count_false_T_kiri +=1
                              feedback = "paha kurang diangkat"
                              kondisi = None
                              posisi = None
                              if(buzzState == True):
                                   buzzState = False
                                   buzzCount = 0
                              state = 1

               if kondisi == "depan kiri":
                    if prev_time_p4_1 is not None :
                         if p4 > prev_time_p4_1:
                              feed_count14 = "increment" 
                              prev_time_p4_2 =prev_time_p4_1 
                              
                         elif p4 < prev_time_p4_1 :
                              feed_count14 = "decrement"
                              prev_time_p4_2 =prev_time_p4_1 
                              
                         else :
                              feed_count14 = " sama"
                              prev_time_p4_2 =prev_time_p4_1 
                          
                    prev_time_p4_1 = p4
                    
                   
                    if (q5 >= 145) :
                         if (p4 <=90) and (LAnkle[0]<=LKnee[0]):
                              count_D_kiri +=1
                              kondisi = None
                              posisi = None
                              if(buzzState == True):
                                   buzzState = False
                                   buzzCount = 0
                              state = 1   
                         if (p4 >=130) and ((((feed_count14 == "increment") ) and (LHip[0]>=LKnee[0]) ) or (((feed_count14 == "decrement") ) and (LHip[0]<=LKnee[0]) )):
                              count_false_D_kiri +=1
                              feedback = "paha kurang diangkat"
                              kondisi = None
                              posisi = None
                              if(buzzState == True):
                                   buzzState = False
                                   buzzCount = 0
                              state = 1   
               
            
               if kondisi == "T kanan":
                    if (  (q4 >= 10)  and (q2 >=155)):
                         if p1 <= 95 and (RAnkle[0]<=RKnee[0]) :
                              count_T_kanan +=1
                              kondisi = None
                              posisi = None
                              if(buzzState == True):
                                   buzzState = False
                                   buzzCount = 0
                              state = 1
                         if (p1 >=150) and (RAnkle[1] >= LKnee[1]+3):
                              count_false_T_kanan +=1
                              feedback = "paha kurang diangkat"
                              kondisi = None
                              posisi = None
                              if(buzzState == True):
                                   buzzState = False
                                   buzzCount = 0
                              state = 1

               if kondisi == "tendangan sabit kanan":
                    if prev_time_p1_1 is not None:
                         if p1 > prev_time_p1_1:
                              feed_count2 = "increment"
                              prev_time_p1_2 = prev_time_p1_1
                         
                         elif p1 < prev_time_p1_1:
                              feed_count2 = "decrement"
                              prev_time_p1_2 = prev_time_p1_1
                         
                         else:
                              feed_count2 = "sama"
                              prev_time_p1_2 = prev_time_p1_1
                              prev_time_p1_1 = p1

                    if (q2 >= 170):
                         if (p1 <= 90) and (RAnkle[0] >= RKnee[0]):
                              count_S_kanan += 1
                              kondisi = None
                              posisi = None
                              if(buzzState == True):
                                   buzzState = False
                                   buzzCount = 0
                              state = 1
            
                    if (p1 >= 130) and (((feed_count2 == "increment") and (RHip[0] <= RKnee[0])) or ((feed_count2 == "decrement") and (RHip[0] >= RKnee[0]))):
                         count_false_S_kanan += 1
                         feedback = "paha kurang diangkat"
                         kondisi = None
                         posisi = None
                         if(buzzState == True):
                              buzzState = False
                              buzzCount = 0
                         state = 1

                    if (p2 >= 70):# Kondisi tambahan untuk memastikan tubuh condong ke belakang untuk keseimbangan
                         if (q2 >= 170) and (RAnkle[0] >= RKnee[0]):
                              count_S_kanan += 1
                              kondisi = None
                              posisi = None
                         if(buzzState == True):
                              buzzState = False
                              buzzCount = 0
                         state = 1

               if kondisi == "tendangan sabit kiri":
                    if prev_time_p4_1 is not None:
                         if p4 > prev_time_p4_1:
                              feed_count14 = "increment"
                              prev_time_p4_2 = prev_time_p4_1
                    
                         elif p4 < prev_time_p4_1:
                              feed_count14 = "decrement"
                              prev_time_p4_2 = prev_time_p4_1
                         
                         else:
                              feed_count14 = "sama"
                              prev_time_p4_2 = prev_time_p4_1
                              prev_time_p4_1 = p4

                    if (q5 >= 170):
                         if (p4 <= 90) and (LAnkle[0] <= LKnee[0]):
                              count_S_kiri += 1
                              kondisi = None
                              posisi = None
                              if(buzzState == True):
                                   buzzState = False
                                   buzzCount = 0
                              state = 1
            
                    if (p4 >= 130) and (((feed_count14 == "increment") and (LHip[0] >= LKnee[0])) or ((feed_count14 == "decrement") and (LHip[0] <= LKnee[0]))):
                         count_false_S_kiri += 1
                         feedback = "paha kurang diangkat"
                         kondisi = None
                         posisi = None
                         if(buzzState == True):
                              buzzState = False
                              buzzCount = 0
                         state = 1

                    if (p3 >= 70): # Kondisi tambahan untuk memastikan tubuh condong ke belakang untuk keseimbangan
                         if (q5 >= 170) and (LAnkle[0] <= LKnee[0]):
                              count_S_kiri += 1
                              kondisi = None
                              posisi = None
                         if(buzzState == True):
                              buzzState = False
                              buzzCount = 0
                         state = 1


        except:
            pass

       
        #Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color = (245,117,66), thickness = 2,circle_radius = 2))
        #image = cv2.flip(image,1)
        #image = cv2.flip(image,1)

        
        
        cv2.rectangle(image,(0,0),(140,70),(255,255,16),-1)
        cv2.putText(image, f"Visibility" ,
                           (0,10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA
                                )
        cv2.putText(image, f"Wajah = {wajah}" ,
                           (0,25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA
                                )
        cv2.putText(image, f"Badan = {badan}" ,
                           (0,40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA
                                )
        cv2.putText(image, f"Kaki = {kaki}" ,
                           (0,55), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA
                                )


        cv2.rectangle(image,(0,130),(140,350),(255,255,16),-1)
        cv2.putText(image, f"Kondisi = {str(kondisi)}" ,
                           (0,145), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA
                                )
        cv2.putText(image, f"D_Kanan = {str(count_D_kanan)}",
                           (0,160), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA
                                )
        
        cv2.putText(image, f"D_Kiri = {str(count_D_kiri)}",
                          (0,175), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA
                                )
        cv2.putText(image, f"T_Kanan = {str(count_T_kanan)}",
                          (0,190), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA
                                )
        cv2.putText(image, f"T_Kir = {str(count_T_kiri)}",
                          (0,205), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA
                                )
        cv2.putText(image, f"S_Kanan = {str(count_S_kanan)}",
                          (0,220), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA
                                )
        cv2.putText(image, f"S_Kiri = {str(count_S_kiri)}",
                          (0,235), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA
                                )
        cv2.putText(image, f"FD_Kanan = {str(count_false_D_kanan)}",
                          (0,250), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA
                                )
        cv2.putText(image, f"FD_kiri = {str(count_false_D_kiri)}",
                          (0,265), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA
                                )
        cv2.putText(image, f"FT_Kanan = {str(count_false_T_kanan)}",
                          (0,280), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA
                                )
        cv2.putText(image, f"FT_kiri = {str(count_false_T_kiri)}",
                          (0,295), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA
                                )
        cv2.putText(image, f"FS_Kanan = {str(count_false_S_kanan)}",
                          (0,310), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA
                                )
        cv2.putText(image, f"FS_kiri = {str(count_false_S_kiri)}",
                          (0,325), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA
                                )
        cv2.putText(image, f"state = {str(state)}",
                          (0,340), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA
                                )

        cv2.rectangle(image,(150,0),(700,30),(255,255,16),-1)
        cv2.putText(image, f"posisi menghadap  = {str(pos_had)}",
                          (150,10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA
                                )
        cv2.putText(image, f"feedback = {str(feedback)}",
                          (150,25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA
                                )
       
        
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(1) & 0xFF == ord ('q'):
            break

cap.release()
cv2.destroyAllWindows()