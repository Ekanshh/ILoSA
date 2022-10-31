#!/usr/bin/env python3

from distutils import file_util
from ILoSA import ILoSA
import time

if __name__ == '__main__':
    
    ILoSA=ILoSA()
    ILoSA.connect_ROS()
    time.sleep(5)
 
    time.sleep(1)
    print("Clear the training data")
    ILoSA.Clear_Training_Data()

    time.sleep(1)
    print("Load the data") 
    ILoSA.load(file='last_uc')    
 
    time.sleep(1)
    print("Load the models")
    ILoSA.load_models(data='_uc')

    time.sleep(1)    
    # Goal 1
    # default_pos = ILoSA.training_traj[:, 0]
    # Goal 2
    default_pos = [0.7319494, 0.044066569553982465, 0.39813381899700793]

    print(f"Reset to the starting cartesian position: {default_pos}")
    ILoSA.go_to_3d(default_pos)

    # time.sleep(1)
    # ILoSA.Interactive_Control(verboose=False, testing=True)