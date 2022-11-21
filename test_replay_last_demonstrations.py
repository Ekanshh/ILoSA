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
    ILoSA.load(file='last_uc_new')    
 
    time.sleep(1)
    print("Load the models")
    ILoSA.load_models(data='_uc_new')

    time.sleep(1)    
    #  actual
    default_pos = ILoSA.training_traj[:, 0]
    print(f"Reset to the starting cartesian position: {default_pos}")
    ILoSA.go_to_3d(default_pos)

    time.sleep(1)
    ILoSA.Interactive_Control(verboose=False, testing=True, default_pos=default_pos)