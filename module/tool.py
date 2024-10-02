import numpy as np
from playsound import playsound

def Play_buzzer():
    playsound('D:\PENS\PROYEK AKHIR\CODING\\buzzer1.mp3')

def Angle(var1,var2,var3):
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
def Y_angle(b,c):
    
    B = np.array(b) #second
    C = np.array(c) #third
    
    ab = (((-B[1])*(C[1]-B[1])))

    a = B[1]
    b = np.sqrt(np.power((C[0]-B[0]),2)+ np.power((C[1]-B[1]),2)) 
    theta = np.arccos((ab)/ (a*b))

    angle = np.abs(theta*180.0/np.pi)

    
    return angle 

def Distance(var1,var2):
    A = np.array(var1) #first
    B = np.array(var2) #second
    panj = np.sqrt(np.power((A[0]-B[0]),2)+ np.power((A[1]-B[1]),2))
    return panj