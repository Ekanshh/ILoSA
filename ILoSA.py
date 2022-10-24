"""
Authors: Giovanni Franzese & Anna Mészáros, May 2022
Email: g.franzese@tudelft.nl
Cognitive Robotics, TU Delft
This code is part of TERI (TEaching Robots Interactively) project
"""
#!/usr/bin/env python
import numpy as np
import pandas as pd
from gaussian_process import *
from panda import *
from utils import *
from data_prep import *
import pickle
# class for storing different data types into one variable
class Struct:
    pass

class ILoSA(Panda):
    def __init__(self):
        super().__init__()
        self.rec_freq = 10  # [Hz]
        self.control_freq=100 # [Hz]
        # stiffness parameters
        self.K_min = 0.0
        self.K_max = 2000.0
        self.K_mean = 600
        self.dK_min = 0.0
        self.dK_max = self.K_max-self.K_mean
        self.K_null=5
        # maximum attractor distance along each axis
        self.attractor_lim = 0.04
        self.scaling_factor=1
        self.scaling_factor_ns=1
        # user-provided teleoperation input
        self.feedback = [0, 0, 0]
        # uncertainty threshold at which new points are added
        self.theta = 0.4
        # uncertainty threshold at which stiffness is automatically reduced
        self.theta_stiffness = 0.99
        self.theta_nullspace= 0
        # training data initialisation
        self.training_traj = []
        self.training_delta = []
        self.training_dK = []
        self.nullspace_traj=[]
        self.nullspace_joints=[]
        
        # maximum force of the gradient
        self.max_grad_force = 10

        self.NullSpaceControl=None

    def Record_NullSpace(self):
        self.Kinesthetic_Demonstration()
        print('Recording ended.')
        save_demo = input("Do you want to keep this demonstration? [y/n] \n")

        if save_demo.lower()=='y':
            if len(self.nullspace_traj)==0:
                self.nullspace_traj=np.zeros((3,1))
                self.nullspace_joints=np.zeros((7,1))
                self.nullspace_traj=np.concatenate((self.nullspace_traj,self.recorded_traj ), axis=1)
                self.nullspace_joints=np.concatenate((self.nullspace_joints,self.recorded_joint ), axis=1)
                self.nullspace_traj=np.delete(self.nullspace_traj, 0,1)
                self.nullspace_joints=np.delete(self.nullspace_joints,0,1)
            else:
                self.nullspace_traj=np.concatenate((self.nullspace_traj,self.recorded_traj ), axis=1)
                self.nullspace_joints=np.concatenate((self.nullspace_joints,self.recorded_joint ), axis=1)
            print("Demo Saved")
        else:
            print("Demo Discarded")
  

    def Record_Demonstration(self):
        self.Kinesthetic_Demonstration()
        print('Recording ended.')
        save_demo = input("Do you want to keep this demonstration? [y/n] \n")
        if save_demo.lower()=='y':
            if len(self.training_traj)==0:
                # Initialize the components with zero vector
                self.training_traj=np.zeros((3,1))
                self.training_delta=np.zeros((3,1))
                self.training_dK=np.zeros((3,1))
                # Append the components with respective recorded components
                self.training_traj=np.concatenate((self.training_traj,self.recorded_traj ), axis=1)
                self.training_delta=np.concatenate((self.training_delta,resample(self.recorded_traj, step=2)), axis=1)
                self.training_dK=np.concatenate((self.training_dK,np.zeros(np.shape(self.recorded_traj))), axis=1)
                # Delete the initial zero-vector components from the final components
                self.training_traj=np.delete(self.training_traj, 0,1)
                self.training_delta=np.delete(self.training_delta,0,1)
                self.training_dK=np.delete(self.training_dK,0,1)
            else:
                self.training_traj=np.concatenate((self.training_traj,self.recorded_traj ), axis=1)
                self.training_delta=np.concatenate((self.training_delta,resample(self.recorded_traj, step=2)), axis=1)
                self.training_dK=np.concatenate((self.training_dK,np.zeros(np.shape(self.recorded_traj))), axis=1)
            print("Demo Saved")
        else:
            print("Demo Discarded")

    def Clear_Training_Data(self):
        self.training_traj = []
        self.training_delta = []
        self.training_dK = []
        self.goal_pos = []

    def save(self, data='last'):
        np.savez(str(pathlib.Path().resolve())+'/data/'+str(data)+'.npz', 
        nullspace_traj=self.nullspace_traj, 
        nullspace_joints=self.nullspace_joints, 
        training_traj=self.training_traj,
        training_delta=self.training_delta,
        training_dK=self.training_dK,
        goal_pos=self.goal_pos) 
        

    def load(self, file='last'):
        data =np.load(str(pathlib.Path().resolve())+'/data/'+str(file)+'.npz')

        self.nullspace_traj=data['nullspace_traj'], 
        self.nullspace_joints=data['nullspace_joints'], 
        self.training_traj=data['training_traj'],
        self.training_delta=data['training_delta'],
        self.training_dK=data['training_dK'] 
        self.goal_pos=data['goal_pos']
        self.nullspace_traj=self.nullspace_traj[0]
        self.nullspace_joints=  self.nullspace_joints[0]
        self.training_traj=self.training_traj[0]
        self.training_delta=self.training_delta[0]
        self.training_dK=self.training_dK
        self.goal_pos=self.goal_pos

    def Train_GPs(self):
        # Learning Delta with GPs
        if len(self.training_traj)>0 and len(self.training_delta)>0:
            print("Training of Delta")
            # Constant kernel C
            kernel = C(constant_value = 0.01, constant_value_bounds=[0.0005, self.attractor_lim]) * RBF(length_scale=[0.1, 0.1, 0.1], length_scale_bounds=[0.025, 0.1]) + WhiteKernel(0.00025, [0.0001, 0.0005]) 
            self.Delta=InteractiveGP(X=self.training_traj, Y=self.training_delta, y_lim=[-self.attractor_lim, self.attractor_lim], kernel=kernel, n_restarts_optimizer=20)
            self.Delta.fit()
            # Save the model
            with open('models/delta.pkl','wb') as delta:
                pickle.dump(self.Delta,delta)
        else:
            raise TypeError("There are no data for learning a trajectory dynamical system")

        # TODO: Why are we saving the model here?
        with open('models/delta.pkl','wb') as delta:
            pickle.dump(self.Delta,delta)
        
        # Training of Stiffness with GPs
        if len(self.training_traj)>0 and len(self.training_dK)>0:
            print("Training of Stiffness")
            self.Stiffness=InteractiveGP(X=self.training_traj, Y=self.training_dK, y_lim=[self.K_min, self.K_max], kernel=self.Delta.kernel_, n_restarts_optimizer=0) 
            self.Stiffness.fit()
            with open('models/stiffness.pkl','wb') as stiffness:
                pickle.dump(self.Stiffness,stiffness)
        else:
            raise TypeError("There are no data for learning a stiffness dynamical system")

        # TODO: Training of Nullspace
        if len(self.nullspace_traj)>0 and len(self.nullspace_joints)>0:
            print("Training of Nullspace")
            kernel = C(constant_value = 0.1, constant_value_bounds=[0.0005, self.attractor_lim]) * RBF(length_scale=[0.1, 0.1, 0.1], length_scale_bounds=[0.025, 0.1]) + WhiteKernel(0.00025, [0.0001, 0.0005]) 
            self.NullSpaceControl=InteractiveGP(X=self.nullspace_traj, Y=self.nullspace_joints, y_lim=[-self.attractor_lim, self.attractor_lim], kernel=kernel, n_restarts_optimizer=20)
            self.NullSpaceControl.fit()
            with open('models/nullspace.pkl','wb') as nullspace:
                pickle.dump(self.NullSpaceControl,nullspace)
        else: 
            print('No Null Space Control Policy Learned')

    def save_models(self, data=''):
        with open(f'models/delta{data}.pkl','wb') as delta:
            pickle.dump(self.Delta,delta)
        with open(f'models/stiffness{data}.pkl','wb') as stiffness:
            pickle.dump(self.Stiffness,stiffness)
        if self.NullSpaceControl:
            with open(f'models/nullspace{data}.pkl','wb') as nullspace:
                pickle.dump(self.NullSpaceControl,nullspace)

    def load_models(self, data=''):
        try:
            with open(f'models/delta{data}.pkl', 'rb') as delta:
                self.Delta = pickle.load(delta)
        except:
            print("No delta model saved")
        try:
            with open(f'models/stiffness{data}.pkl', 'rb') as stiffness:
                self.Stiffness = pickle.load(stiffness)
        except:
            print("No stiffness model saved")
        try:
            with open(f'models/nullspace{data}.pkl', 'rb') as nullspace:
                self.NullSpace = pickle.load(nullspace)
        except:
            print("No NullSpace model saved")
            
    def find_alpha(self):
        rospy.loginfo(f"[ILoSA][find_alpha] Finding alpha..")
        alpha=np.zeros(len(self.Delta.X))
        for i in range(len(self.Delta.X)):         
            pos= self.Delta.X[i,:]+self.Delta.length_scales 
            dSigma_dx, dSigma_dy, dSigma_dz = self.Delta.var_gradient(pos.reshape(1,-1))                                                                                                                                                                
            alpha[i]=self.max_grad_force/ np.sqrt(dSigma_dx**2+dSigma_dy**2+dSigma_dz**2)
            self.alpha=np.min(alpha)

    def Interactive_Control(self, verboose=True, testing=False):
        rospy.loginfo(f"[ILoSA][interactive_control] Starting interactive control..")
        # Initialize rospy rate
        r=rospy.Rate(self.control_freq)
        # Get alpha
        self.find_alpha()
        # Counter to repeate the interactive control loop
        counter = 1
        counter_threshold = 20
        start_timer_flag = True
        successful_runs = 0
        failed_runs = 0
        try:
            while True:       

                if testing: 

                    if start_timer_flag == True:
                        start_time = rospy.Time.now().to_sec()
                        start_timer_flag = False
                        rospy.loginfo(f"START TIME: {start_time}")

                    # Monitor goal position reached
                    is_goal_reached = self.check_goal_reached(threshold=0.01)
                                    
                    if is_goal_reached and counter <= counter_threshold:
                        rospy.loginfo("[ILoSA][interactive_control] Goal position reached. Restarting Interactive control demonstrations.")
                        rospy.sleep(2.0)
                        print("[ILoSA][interactive_control] Reset to the starting cartesian position.")
                        self.go_to_3d(self.training_traj[:, 0])
                        rospy.sleep(1.0)
                        counter += 1
                        successful_runs += 1
                    elif is_goal_reached and counter > counter_threshold :
                        rospy.loginfo(f"[ILoSA][interactive_control] Goal position reached and counter is at maximum!")
                        rospy.sleep(2.0)
                        print("[ILoSA][interactive_control] Reset to the starting cartesian position.")
                        self.go_to_3d(self.training_traj[:, 0])
                        rospy.loginfo(f"[ILoSA][interactive_control] Stopping interactive control demonstrations.")
                        rospy.sleep(1.0)
                        # successful_runs += 1
                        rospy.loginfo(f"[ILoSA][interactive_control] Goal reached evaluation metrics: Total Runs: {counter_threshold} Successful Reached Goal: {successful_runs} Failed to reach Goal: {failed_runs}")
                        break
                    elif not is_goal_reached and counter > counter_threshold:
                        rospy.loginfo(f"[ILoSA][interactive_control] Goal not reached and counter is at maximum!")
                        rospy.sleep(2.0)
                        print("[ILoSA][interactive_control] Reset to the starting cartesian position.")
                        self.go_to_3d(self.training_traj[:, 0])
                        rospy.loginfo(f"[ILoSA][interactive_control] Stopping interactive control demonstrations.")
                        rospy.sleep(1.0)
                        # failed_runs += 1
                        rospy.loginfo(f"[ILoSA][interactive_control] Goal reached evaluation metrics: Total Runs: {counter_threshold} Successful Reached Goal: {successful_runs} Failed to reach Goal: {failed_runs}")
                        break
                    else:
                        rospy.loginfo(f"[ILoSA][interactive_control] Trying to reach goal position .. ")
                        current_time = rospy.Time.now().to_sec()
                        if abs(start_time - current_time) > 15.0:
                            rospy.loginfo(f"[ILoSA][interactive_control] Reached maximum time limit.")
                            failed_runs += 1
                            rospy.sleep(2.0)
                            print("[ILoSA][interactive_control] Reset to the starting cartesian position.")
                            self.go_to_3d(self.training_traj[:, 0])
                            rospy.sleep(1.0)
                            counter += 1
                            start_timer_flag = True
                        else:
                            rospy.loginfo(f"Number of runs: {counter} Time elapsed: {abs(start_time - current_time)}")
                            pass
                else:
                    # Monitor goal position reached
                    is_goal_reached = self.check_goal_reached(threshold=0.01)

                    if is_goal_reached:
                        rospy.loginfo(f"[ILoSA][interactive_control] Goal reached")
                        rospy.sleep(2.0)
                        rospy.loginfo(f"[ILoSA][interactive_control] Saving new data")
                        self.save(data="last_with_user_correction_2")
                        rospy.sleep(1.0)
                        rospy.loginfo(f"[ILoSA][interactive_control] Saving new models")
                        self.save_models(data="_uc_2")
                        rospy.sleep(1.0)
                        print("[ILoSA][interactive_control] Reset to the starting cartesian position.")
                        self.go_to_3d(self.training_traj[:, 0])
                        rospy.loginfo(f"[ILoSA][interactive_control] Stopping interactive control demonstrations.")
                        rospy.sleep(1.0)
                        break
                    else:
                        rospy.loginfo(f"[ILoSA][interactive_control] Goal not reached. User corrections needed...")
                
                # read the actual position of the robot
                cart_pos=np.array(self.cart_pos).reshape(1,-1)
                
                print("starting predictions")
                # GP predictions Delta_x
                [self.delta, self.sigma]=self.Delta.predict(cart_pos)
                print("delta predict")

                # GP prediction K stiffness
                [self.dK, _]=self.Stiffness.predict(cart_pos, return_std=False)
                print("stiffness predict")
                self.delta = np.clip(self.delta[0], -self.attractor_lim, self.attractor_lim)

                self.dK = np.clip(self.dK[0], self.dK_min, self.dK_max)

                dSigma_dx, dSigma_dy, dSigma_dz = self.Delta.var_gradient(cart_pos)
                
                f_stable=-self.alpha*np.array([dSigma_dx, dSigma_dy, dSigma_dz])

                self.K_tot = np.clip(np.add(self.dK, self.K_mean), self.K_min, self.K_max)

                
                if any(abs(np.array(self.feedback)) > 0.05): # Check for joystick feedback 
                    
                    rospy.loginfo(f"[ILoSA][interactive_control] Received user feedback")
                    delta_inc, dK_inc = Interpret_3D(feedback=self.feedback, delta=self.delta, K=self.K_tot, delta_lim=self.attractor_lim, K_mean=self.K_mean)
                    is_uncertain=self.Delta.is_uncertain(theta=self.theta)
                    self.Delta.update_with_k(x=cart_pos, mu=self.delta, epsilon_mu=delta_inc, is_uncertain=is_uncertain)
                    self.Stiffness.update_with_k(x=cart_pos, mu=self.dK, epsilon_mu=dK_inc, is_uncertain=is_uncertain)
                            
                self.delta, self.K_tot = Force2Impedance(self.delta, self.K_tot, f_stable, self.attractor_lim)
                self.K_tot=[self.K_tot]
                self.scaling_factor = (1- self.sigma / self.Delta.max_var) / (1 - self.theta_stiffness)
                if self.sigma / self.Stiffness.max_var > self.theta_stiffness: 
                    self.K_tot=self.K_tot*self.scaling_factor
                rospy.loginfo(f"[ILoSA][interactive_corrections] Current pos: {cart_pos}")
                x_new = cart_pos[0][0] + self.delta[0]  
                y_new = cart_pos[0][1] + self.delta[1]  
                z_new = cart_pos[0][2] + self.delta[2]  
                quat_goal=self.initial_orientation    # TODO: Hard-coded orientation, might need to change for our use-case
                pos_goal=[x_new, y_new, z_new]
                rospy.loginfo(f"[ILoSA][interactive_corrections] Next pos: {pos_goal}")
                self.set_attractor(pos_goal,quat_goal)

                null_stiff = [0]

                if self.NullSpaceControl:
                    [self.equilibrium_configuration, self.sigma_null_space]=self.NullSpaceControl.predict(np.array(pos_goal).reshape(1,-1))
                    self.scaling_factor_ns = (1-self.sigma_null_space / self.NullSpaceControl.max_var) / (1 - self.theta_nullspace)
                    
                    if self.sigma_null_space / self.NullSpaceControl.max_var > self.theta_nullspace:
                        self.null_stiff=self.K_null*self.scaling_factor_ns
                    else:
                        self.null_stiff=self.K_null      
                    self.set_configuration(self.equilibrium_configuration[0])
                    null_stiff = [self.null_stiff]
                
                pos_stiff = [self.K_tot[0][0],self.K_tot[0][1],self.K_tot[0][2]]
                rot_stiff = [self.K_ori, self.K_ori, self.K_ori]
                self.set_stiffness(pos_stiff, rot_stiff, null_stiff)
                if verboose :
                    print("Delta")
                    print(self.delta)
                    print("Stabilization field")
                    print(f_stable)
                    print("Scaling_factor_cartesian:" + str(self.scaling_factor))
                    print("Scaling_factor_nullspace:" + str(self.scaling_factor_ns))   
                r.sleep()
        except KeyboardInterrupt:
            rospy.logwarn(f"Keyboard Interrupt!")
            rospy.logwarn(f"Stopping....")
        except Exception as e:
            rospy.logerr(f"Error {e}")
