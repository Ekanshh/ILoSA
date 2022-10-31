#!/usr/bin/env python3

from distutils import file_util
from ILoSA import ILoSA
import time

if __name__ == '__main__':
    
    ILoSA=ILoSA()
    ILoSA.connect_ROS()
    time.sleep(5)
 
    time.sleep(1)
    print("Clear training data")
    ILoSA.Clear_Training_Data()

    time.sleep(1)
    print("Load the data") 
    ILoSA.load(file='last_uc')    
 
    time.sleep(1)
    print("Load the models")
    ILoSA.load_models(data='_uc')

    time.sleep(1)    
    #  actual
    default_pos = ILoSA.training_traj[:, 0]
    # +y
    # default_pos = [0.7319494, 0.010189465809362652, 0.4532519]
    # -y 
    # default_pos = [0.7319494, -0.1421327499710095, 0.4532519]
    # +z
    # default_pos = [0.7319494, -0.06311494, 0.5282793313862713]
    # xyz
    # default_pos =  [0.5972264452791827, -0.1421327499710095, 0.5282793313862713]
    ##################
    # Goal 2
    # default_pos = [0.7319494, 0.044066569553982465, 0.38513381899700793]
    # Goal 3
    # default_pos=[0.7319494, -0.07011494, 0.38913381899700794]
    # Goal 4
    # default_pos=[0.7319494, -0.17986822627203982,  0.3838127689300551]
    # Goal 5
    default_pos=[0.7319494, 0.025080543921986533,  0.45832350734821834]

    print(f"Reset to the starting cartesian position: {default_pos}")
    ILoSA.go_to_3d(default_pos)

    time.sleep(1)
    ILoSA.Interactive_Control(verboose=False, testing=True, default_pos=default_pos)