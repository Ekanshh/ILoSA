#!/usr/bin/env python
import pickle

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import WrenchStamped


class RecordWrenchData:

    def __init__(self):
        self.e_start = None
        self.side = None
        self.wrench_data = []
        self.dump_data = []
        self.wrench_data1 = []
        self.dump_data1 = []
        rospy.Subscriber("/record_wrench_data/event_in", String, self.event_in_callback)
        rospy.Subscriber("/force_torque_ext", WrenchStamped, self.record_wrench_callback)
        rospy.Subscriber("/franka_state_controller/F_ext", WrenchStamped, self.record_wrench_callback_1)
        

    def event_in_callback(self, msg):
        rospy.loginfo("Waiting for event call")
        if msg.data == 'e_start_right':
            rospy.loginfo("e_start | right")
            self.e_start= 'start'
            self.side=msg.data[8:]
            
        elif msg.data == 'e_start_left':
            rospy.loginfo("e_start | left")
            self.e_start= 'start'
            self.side=msg.data[8:]

        elif msg.data == 'e_stop':
            rospy.loginfo("e_stop")
            self.e_start= 'stop'
            rospy.loginfo("Stopping data")
            self.dump_data.append((self.side, self.wrench_data))
            self.dump_data1.append((self.side, self.wrench_data1))
            self.side = None
            self.wrench_data = []
            self.wrench_data1 = []

        elif msg.data == 'e_save':
            with open('record_wrench_data_left.pickle', 'ab+') as f:
                pickle.dump(self.dump_data, f)
                self.dump_data = []
            with open('record_wrench_data1_left.pickle', 'ab+') as f:
                pickle.dump(self.dump_data1, f)
                self.dump_data1 = []
        else:
            rospy.loginfo("unknown")
            self.e_start=None
            self.side=None 

    def record_wrench_callback(self, data):
        
        if self.e_start == 'start':
            rospy.loginfo("Recording data")
            wrench = [data.wrench.force.x, data.wrench.force.y, data.wrench.force.z, data.wrench.torque.x, data.wrench.torque.y, data.wrench.torque.z]
            rospy.loginfo(wrench)
            self.wrench_data.append(wrench)

        else:
            pass
    
    def record_wrench_callback_1(self, data):
        
        if self.e_start == 'start':
            rospy.loginfo("Recording data 1")
            wrench1 = [data.wrench.force.x, data.wrench.force.y, data.wrench.force.z, data.wrench.torque.x, data.wrench.torque.y, data.wrench.torque.z]
            rospy.loginfo(wrench1)
            self.wrench_data1.append(wrench1)

        else:
            pass

if __name__ == '__main__':
    rospy.init_node('record_wrench_data')
    RecordWrenchData()
    rospy.spin()