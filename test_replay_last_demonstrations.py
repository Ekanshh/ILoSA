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
    ILoSA.load(file='last_with_user_correction_2')    
 
    time.sleep(1)
    print("Load the models")
    ILoSA.load_models(data='_uc_2')

    time.sleep(1)    
    print("Reset to the starting cartesian position")
    ILoSA.go_to_3d(ILoSA.training_traj[:, 0])

    time.sleep(1)
    ILoSA.Interactive_Control(verboose=False, testing=True)
    
    

    
     
    
    
