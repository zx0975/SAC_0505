#!/usr/bin/env python
# encoding: utf-8   #要打中文時加這行
import rospy
import sys
import numpy as np
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
class Gripper:
    def __init__(self): 
        self.publisher_rate=100
        self.r_finger_joint1='/mobile_dual_arm/r_finger_joint1_position_controller/command'
        self.r_finger_joint2='/mobile_dual_arm/r_finger_joint2_position_controller/command'
        self.r_finger_joint3='/mobile_dual_arm/r_finger_joint3_position_controller/command'
        self.r_finger_joint4='/mobile_dual_arm/r_finger_joint4_position_controller/command'
        self.r_finger_joint5='/mobile_dual_arm/r_finger_joint5_position_controller/command'
        self.r_finger_joint6='/mobile_dual_arm/r_finger_joint6_position_controller/command'

        self.l_finger_joint1='/mobile_dual_arm/l_finger_joint1_position_controller/command'
        self.l_finger_joint2='/mobile_dual_arm/l_finger_joint2_position_controller/command'
        self.l_finger_joint3='/mobile_dual_arm/l_finger_joint3_position_controller/command'
        self.l_finger_joint4='/mobile_dual_arm/l_finger_joint4_position_controller/command'
        self.l_finger_joint5='/mobile_dual_arm/l_finger_joint5_position_controller/command'
        self.l_finger_joint6='/mobile_dual_arm/l_finger_joint6_position_controller/command'

        rospy.init_node('gripper_control',anonymous=False) #初始化node    anonymous=True  在node名稱後加入亂碼    避免相同名稱的node踢掉彼此
    
        
    # def command_convert(self):
    #     direction=float(sys.argv[1])
    #     return direction
    
        


    def get_gripper_state(self,joint):
        joint_now=rospy.wait_for_message('/mobile_dual_arm/joint_states',JointState)
        bb=joint
        return joint_now.position[bb]

    def get_ratating_state(self):
        joint_now=rospy.wait_for_message('/mobile_dual_arm/joint_states',JointState)
        return joint_now.position[12]



    
if __name__ == "__main__":
    
    try:
        a = Gripper()
        # a.send_finger_direction('r',1)
        # a.send_finger_direction('r',2)
        # a.send_finger_direction('l',1)
        # a.send_finger_direction('l',2)

        # a.send_finger_direction(2.0)
        rospy.loginfo('456')
        a = np.linalg.norm(np.array([0.25, 0.0, 0.8, 0.706825181105, 0.0, 0.0, 0.70738826916]) - np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) )
        print a
        rospy.sleep(0.05)
        # a.send_finger_direction(1.0)

        rospy.loginfo('123')
    except  rospy.ROSInterruptException:
        rospy.loginfo('end')
        pass


