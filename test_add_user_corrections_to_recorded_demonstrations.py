#!/usr/bin/env python3

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
    ILoSA.load(file='last')    
 
    time.sleep(1)
    print("Load the models")
    ILoSA.load_models(data='')

    time.sleep(1)    
    print("Reset to the starting cartesian position")
    ILoSA.go_to_3d(ILoSA.training_traj[:, 0])

    time.sleep(1)
    ILoSA.Interactive_Control(verboose=False, default_pos=ILoSA.training_traj[:, 0])
    
    

    
     
    
    
