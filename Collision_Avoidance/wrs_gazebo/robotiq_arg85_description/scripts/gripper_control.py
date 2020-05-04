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

    # def send_finger_direction(self,side,direction):
    def send_finger_direction(self,direction):
        # if side=='r':
        pub1=rospy.Publisher(self.r_finger_joint1,Float64,queue_size=10)
        pub2=rospy.Publisher(self.r_finger_joint2,Float64,queue_size=10)
        pub3=rospy.Publisher(self.r_finger_joint3,Float64,queue_size=10)
        pub4=rospy.Publisher(self.r_finger_joint4,Float64,queue_size=10)
        pub5=rospy.Publisher(self.r_finger_joint5,Float64,queue_size=10)
        pub6=rospy.Publisher(self.r_finger_joint6,Float64,queue_size=10)

        pub7=rospy.Publisher(self.l_finger_joint1,Float64,queue_size=10)
        pub8=rospy.Publisher(self.l_finger_joint2,Float64,queue_size=10)
        pub9=rospy.Publisher(self.l_finger_joint3,Float64,queue_size=10)
        pub10=rospy.Publisher(self.l_finger_joint4,Float64,queue_size=10)
        pub11=rospy.Publisher(self.l_finger_joint5,Float64,queue_size=10)
        pub12=rospy.Publisher(self.l_finger_joint6,Float64,queue_size=10)

        #pub:publish名稱   chatter:topic name  queue_size=緩衝區大小
        rate=rospy.Rate(self.publisher_rate)
        
        if direction==1.0:
            rospy.loginfo('The gripper is opening')
            joint1_ang=self.get_gripper_state(2)
            joint2_ang=self.get_gripper_state(10)
            joint3_ang=self.get_gripper_state(11)
            joint4_ang=self.get_gripper_state(12)
            joint5_ang=self.get_gripper_state(13)
            joint6_ang=self.get_gripper_state(14)

            joint7_ang=self.get_gripper_state(15)
            joint8_ang=self.get_gripper_state(23)
            joint9_ang=self.get_gripper_state(24)
            joint10_ang=self.get_gripper_state(25)
            joint11_ang=self.get_gripper_state(26)
            joint12_ang=self.get_gripper_state(27)
            rospy.loginfo('Joint angle now = [ %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f]' ,joint1_ang,joint2_ang,joint3_ang,joint4_ang,joint5_ang,joint6_ang,joint7_ang,joint8_ang,joint9_ang,joint10_ang,joint11_ang,joint12_ang)
            
            joint1_ang==0.725
            joint2_ang==0.725
            joint3_ang==0.725
            joint4_ang==0.725
            joint5_ang==0.725
            joint6_ang==0.725
            joint7_ang==0.725
            joint8_ang==0.725
            joint9_ang==0.725
            joint10_ang==0.725
            joint11_ang==0.725
            joint12_ang==0.725
            rospy.sleep(0.01)

            while (joint3_ang>0):
                
                pub1.publish(joint1_ang)
                pub2.publish(joint2_ang)
                pub3.publish(joint3_ang)
                pub4.publish(joint4_ang)
                pub5.publish(joint5_ang)
                pub6.publish(joint6_ang)
                pub7.publish(joint7_ang)
                pub8.publish(joint8_ang)
                pub9.publish(joint9_ang)
                pub10.publish(joint10_ang)
                pub11.publish(joint11_ang)
                pub12.publish(joint12_ang)
                rospy.sleep(0.1)
                joint1_ang=joint1_ang-0.01
                joint2_ang=joint2_ang-0.01
                joint3_ang=joint3_ang-0.01
                joint4_ang=joint4_ang-0.01
                joint5_ang=joint5_ang-0.01
                joint6_ang=joint6_ang-0.01
                joint7_ang=joint7_ang-0.01
                joint8_ang=joint8_ang-0.01
                joint9_ang=joint9_ang-0.01
                joint10_ang=joint10_ang-0.01
                joint11_ang=joint11_ang-0.01
                joint12_ang=joint12_ang-0.01

                # if (joint1_ang == 0.72):
                #     direction = direction+1
                #     break
            
        elif direction==2.0:
            rospy.loginfo('The gripper is closing')
            joint1_ang=self.get_gripper_state(2)
            joint2_ang=self.get_gripper_state(10)
            joint3_ang=self.get_gripper_state(11)
            joint4_ang=self.get_gripper_state(12)
            joint5_ang=self.get_gripper_state(13)
            joint6_ang=self.get_gripper_state(14)

            joint7_ang=self.get_gripper_state(15)
            joint8_ang=self.get_gripper_state(23)
            joint9_ang=self.get_gripper_state(24)
            joint10_ang=self.get_gripper_state(25)
            joint11_ang=self.get_gripper_state(26)
            joint12_ang=self.get_gripper_state(27)
            rospy.loginfo('Joint angle now = [ %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f]' ,joint1_ang,joint2_ang,joint3_ang,joint4_ang,joint5_ang,joint6_ang,joint7_ang,joint8_ang,joint9_ang,joint10_ang,joint11_ang,joint12_ang)
            
            joint1_ang==0.0
            joint2_ang==0.0
            joint3_ang==0.0
            joint4_ang==0.0
            joint5_ang==0.0
            joint6_ang==0.0
            joint7_ang==0.0
            joint8_ang==0.0
            joint9_ang==0.0
            joint10_ang==0.0
            joint11_ang==0.0
            joint12_ang==0.0
            rospy.sleep(0.01)

            while (joint3_ang<0.725):

                pub1.publish(joint1_ang)
                pub2.publish(joint2_ang)
                pub3.publish(joint3_ang)
                pub4.publish(joint4_ang)
                pub5.publish(joint5_ang)
                pub6.publish(joint6_ang)
                pub7.publish(joint7_ang)
                pub8.publish(joint8_ang)
                pub9.publish(joint9_ang)
                pub10.publish(joint10_ang)
                pub11.publish(joint11_ang)
                pub12.publish(joint12_ang)
                rospy.sleep(0.1)
                joint1_ang=joint1_ang+0.01
                joint2_ang=joint2_ang+0.01
                joint3_ang=joint3_ang+0.01
                joint4_ang=joint4_ang+0.01
                joint5_ang=joint5_ang+0.01
                joint6_ang=joint6_ang+0.01
                joint7_ang=joint7_ang+0.01
                joint8_ang=joint8_ang+0.01
                joint9_ang=joint9_ang+0.01
                joint10_ang=joint10_ang+0.01
                joint11_ang=joint11_ang+0.01
                joint12_ang=joint12_ang+0.01

                # if (joint1_ang == 0):
                #     direction = direction-1
                #     break

        # if side=='l':

        # pub7=rospy.Publisher(self.l_finger_joint1,Float64,queue_size=10)
        # pub8=rospy.Publisher(self.l_finger_joint2,Float64,queue_size=10)
        # pub9=rospy.Publisher(self.l_finger_joint3,Float64,queue_size=10)
        # pub10=rospy.Publisher(self.l_finger_joint4,Float64,queue_size=10)
        # pub11=rospy.Publisher(self.l_finger_joint5,Float64,queue_size=10)
        # pub12=rospy.Publisher(self.l_finger_joint6,Float64,queue_size=10)

        # #pub:publish名稱   chatter:topic name  queue_size=緩衝區大小
        # rate=rospy.Rate(self.publisher_rate)
        
        # if direction==1.0:
        #     rospy.loginfo('The gripper is opening')
        #     joint7_ang=self.get_gripper_state(6)
        #     joint8_ang=self.get_gripper_state(7)
        #     joint9_ang=self.get_gripper_state(8)
        #     joint10_ang=self.get_gripper_state(9)
        #     joint11_ang=self.get_gripper_state(10)
        #     joint12_ang=self.get_gripper_state(11)
        #     rospy.loginfo('Joint angle now = [ %.2f %.2f %.2f %.2f %.2f %.2f]' ,joint7_ang,joint8_ang,joint9_ang,joint10_ang,joint11_ang,joint12_ang)
        #     while (joint1_ang>0):

        #         pub1.publish(joint1_ang)
        #         pub2.publish(joint2_ang)
        #         pub3.publish(joint3_ang)
        #         pub4.publish(joint4_ang)
        #         pub5.publish(joint5_ang)
        #         pub6.publish(joint6_ang)
        #         rospy.sleep(0.03)
        #         joint1_ang=joint1_ang-0.01
        #         joint2_ang=joint1_ang-0.01
        #         joint3_ang=joint1_ang-0.01
        #         joint4_ang=joint1_ang-0.01
        #         joint5_ang=joint1_ang-0.01
        #         joint6_ang=joint1_ang-0.01
            
        # elif direction==2.0:
        #     rospy.loginfo('The gripper is closing')
        #     joint7_ang=self.get_gripper_state(6)
        #     joint8_ang=self.get_gripper_state(7)
        #     joint9_ang=self.get_gripper_state(8)
        #     joint10_ang=self.get_gripper_state(9)
        #     joint11_ang=self.get_gripper_state(10)
        #     joint12_ang=self.get_gripper_state(11)
        #     rospy.loginfo('Joint angle now = [ %.2f %.2f %.2f %.2f %.2f %.2f]' ,joint7_ang,joint8_ang,joint9_ang,joint10_ang,joint11_ang,joint12_ang)
            
        #     while (joint1_ang<0.72):
        #         pub1.publish(joint1_ang)
        #         pub2.publish(joint2_ang)
        #         pub3.publish(joint3_ang)
        #         pub4.publish(joint4_ang)
        #         pub5.publish(joint5_ang)
        #         pub6.publish(joint6_ang)
        #         rospy.sleep(0.03)
        #         joint1_ang=joint2_ang+0.01
        #         joint2_ang=joint2_ang+0.01
        #         joint3_ang=joint2_ang+0.01
        #         joint4_ang=joint2_ang+0.01
        #         joint5_ang=joint2_ang+0.01
        #         joint6_ang=joint2_ang+0.01
        

####################################################


    
if __name__ == "__main__":
    
    try:
        a = Gripper()
        # a.send_finger_direction('r',1)
        # a.send_finger_direction('r',2)
        # a.send_finger_direction('l',1)
        # a.send_finger_direction('l',2)
        a.send_finger_direction(2.0)
        rospy.sleep(0.05)
        a.send_finger_direction(1.0)

        rospy.loginfo('123')
    except  rospy.ROSInterruptException:
        rospy.loginfo('end')
        pass


