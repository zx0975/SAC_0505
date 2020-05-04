#!/usr/bin/env python
# encoding: utf-8   #要打中文時加這行
import rospy
import sys
import numpy as np
from geometry_msgs.msg import Vector3 ,Wrench ,WrenchStamped


class Sensor:
    # rospy.init_node("ft_sensor")
    # rospy.Subscriber("sensor_data", WrenchStamped, self.adjust)

    # def adjust(self, ft_in):
    #     ft_out = WrenchStamped()
    #     ft_out.header.stamp = rospy.Time.now()
    #     ft_out.header.frame_id = ft_in.header.frame_id
    #     ft_out.wrench.force.x = ft_in.wrench.force.x  - self.x_force_offset + self.adjustment.wrench.force.x
    #     ft_out.wrench.force.y = ft_in.wrench.force.y  - self.y_force_offset + self.adjustment.wrench.force.y
    #     ft_out.wrench.force.z = ft_in.wrench.force.z  - self.z_force_offset + self.adjustment.wrench.force.z
    #     ft_out.wrench.torque.x = ft_in.wrench.torque.x - self.x_torque_offset  - self.adjustment.wrench.torque.x
    #     ft_out.wrench.torque.y = ft_in.wrench.torque.y  - self.y_torque_offset - self.adjustment.wrench.torque.y
    #     ft_out.wrench.torque.z = ft_in.wrench.torque.z  - self.z_torque_offset - self.adjustment.wrench.torque.z

    #     p1.pub = rospy.Publisher('/r_force_sensor', Vector3, queue_size=10)
    def __init__(self): 
        # self.force = []
        # self.torque = []
        # self.sensor =[]
        # self.state = []
        self.sensor_array = []
        rospy.init_node('ft_sensor',anonymous=False)
        rospy.Subscriber('/r_force_sensor',WrenchStamped,self.force_data)
        rospy.sleep(1.0)
    def force_data(self,data):
        # rospy.loginfo(data)
        sensor_array = []
        self.sensor_array = np.array([data.wrench.force.x,\
                                      data.wrench.force.y,\
                                      data.wrench.force.z,\
                                      data.wrench.torque.x,\
                                      data.wrench.torque.y,\
                                      data.wrench.torque.z])
        print(self.sensor_array, len(self.sensor_array))
            
        

    # def normalize(self,ft_in)
    #     ft_out = WrenchStamped()
    #     ft_out.header.stamp = rospy.Time.now()
    #     ft_out.header.frame_id = ft_in.header.frame_id
    #     ft_out.wrench.force.x = ft_in.wrench.force.x/2
    #     ft_out.wrench.force.y = ft_in.wrench.force.y/2
    #     ft_out.wrench.force.z = ft_in.wrench.force.z/2
    #     ft_out.wrench.torque.x = ft_in.wrench.torque.x/2
    #     ft_out.wrench.torque.y = ft_in.wrench.torque.y/2
    #     ft_out.wrench.torque.z = ft_in.wrench.torque.z/2

    #     self.ft_out.publish(ft_out)

    # def send_force_data(self):
    # def get_force_state(self,data):
    #     force_data=rospy.wait_for_message('/r_force_sensor',WrenchStamped)
    #     data=force_data
    #     return force_data.force[data]

    # def send_force_data(self):
    #     pub1=rospy.Publisher(self.r_force_sensor,WrenchStamped,queue_size=10)   

    # def get_force_sensor(self,side):
    #     if side =='r':
    #         r_force=rospy.wait_for_message('/r_force_sensor',WrenchStamped)
    #         force = np.append(r_force.wrench.force.x,r_force.wrench.force.y)
    #         force = np.append(force,r_force.wrench.force.z)            
    #         return force
    #     elif side =='l':
    #         l_force = rospy.wait_for_message('/l_force_sensor',WrenchStamped)
    #         force = np.append(l_force.wrench.force.x,l_force.wrench.force.y)
    #         force = np.append(force,l_force.wrench.force.z)
    #         return force

    # def get_torque_sensor(self,side):
    #     if side =='r':
    #         r_torque=rospy.wait_for_message('/r_force_sensor',WrenchStamped)
    #         torque = np.append(r_torque.wrench.torque.x,r_torque.wrench.torque.y)
    #         torque = np.append(torque,r_torque.wrench.torque.z)            
    #         return torque
    #     elif side =='l':
    #         l_torque = rospy.wait_for_message('/l_force_sensor',WrenchStamped)
    #         torque = np.append(l_torque.wrench.torque.x,l_torque.wrench.torque.y)
    #         torque = np.append(torque,l_torque.wrench.torque.z)
    #         return torque

    # def get_force_sensor(self,side):
    #     if side == 'r':
    #         r_force=rospy.wait_for_message('/r_force_sensor',WrenchStamped)
    #         self.force = np.append(r_force.wrench.force.x,r_force.wrench.force.y)
    #         self.force = np.append(self.force,r_force.wrench.force.z) 
    #         self.torque = np.append(r_force.wrench.torque.x,r_force.wrench.torque.y)
    #         self.torque = np.append(self.torque,r_force.wrench.torque.z)            
    #         self.sensor = np.append(self.force,self.torque)
    #         return self.sensor
    #     elif side =='l':
    #         l_force=rospy.wait_for_message('/l_force_sensor',WrenchStamped)
    #         self.force = np.append(l_force.wrench.force.x,l_force.wrench.force.y)
    #         self.force = np.append(self.force,l_force.wrench.force.z) 
    #         self.torque = np.append(l_force.wrench.torque.x,l_force.wrench.torque.y)
    #         self.torque = np.append(self.torque,l_force.wrench.torque.z)            
    #         return self.sensor
    # def abc(self,side):
    #     if side == 'r':
    #         # print (self.get_force_sensor('r'))
    #         data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    #         # print data
    #     elif side == 'l':
    #         data = self.get_force_sensor('l')
    #         self.state = data
    #         # print self.state
        
    # def get_l_force_sensor(self,force):
    #     l_force=rospy.wait_for_message('/l_force_sensor',WrenchStamped)
    #     return l_force.wrench.force[l_force]
    
    # def get_r_torque_sensor(self,torque):
    #     r_torque=rospy.wait_for_message('/r_force_sensor',WrenchStamped)
    #     return r_torque.wrench.torque[r_torque]

    # def get_l_torque_sensor(self,torque):
    #     l_torque=rospy.wait_for_message('/l_force_sensor',WrenchStamped)
    #     return l_torque.wrench.torque[l_torque]

    # def get_force_side(self,side):
    #     if side =='r':
            
    #         rospy.loginfo('Right force now = [ %.2f %.2f %.2f ]' ,r_force)
    #         rospy.loginfo('Right torque now = [ %.2f %.2f %.2f ]' ,r_torque)

    #         rospy.sleep(0.01)

    #     elif side =='l':
            # l_force_x = self.get_l_force_sensor.x(force)
            # l_force_y = self.get_l_force_sensor.y(force)
            # l_force_z = self.get_l_force_sensor.z(force)
            # l_torque_x = self.get_r_torque_sensor.x(torque)
            # l_torque_y = self.get_r_torque_sensor.y(torque)
            # l_torque_z = self.get_r_torque_sensor.z(torque)

            # pub7=rospy.Publisher(r_force_x,Float64,queue_size=10)
            # pub8=rospy.Publisher(r_force_y,Float64,queue_size=10)
            # pub9=rospy.Publisher(r_force_z,Float64,queue_size=10)
            # pub10=rospy.Publisher(r_torque_x,Float64,queue_size=10)
            # pub11=rospy.Publisher(r_torque_y,Float64,queue_size=10)
            # pub12=rospy.Publisher(r_torque_z,Float64,queue_size=10)

            # rate=rospy.Rate(self.publisher_rate)

            # rospy.loginfo('Left force now = [ %.2f %.2f %.2f ]' ,l_force_x,l_force_y,l_force_z)
            # rospy.loginfo('Left torque now = [ %.2f %.2f %.2f ]' ,l_torque_x,l_torque_y,l_torque_z)
            # rospy.loginfo('Right force now = [ ]' ,l_force)
            # rospy.loginfo('Right torque now = [ ]' ,l_torque)

            # rospy.sleep(0.01)
    # def send_sensor_force(self,side):
    #     # pub1=rospy.Publisher(self.r_force_sensor,Vector3,queue_size=10)
    #     # rate=rospy.Rate(self.publisher_rate)
    #     if side == 'r':

    #         rospy.loginfo('Sensor Force now = [ %.2f ]' ,force)

    #     elif side == 'l':
        
    #         rospy.loginfo('Sensor Force now = [ %.2f ]' ,force)

    # def send_sensor_torque(self,side):
    #     # pub2=rospy.Publisher(self.l_force_sensor,Vector3,queue_size=10)
    #     # rate=rospy.Rate(self.publisher_rate)
    #     if side == 'r':

    #         rospy.loginfo('Sensor Torque now = [ %.2f ]' ,torque)

    #     elif side == 'l':
        
    #         rospy.loginfo('Sensor Torque now = [ %.2f ]' ,torque)
    

if __name__ == "__main__":
    
    try:
        a = Sensor()
        # a.get_sensor_force(0)
        # a.get_sensor_torque(0)
        # a.send_sensor_force('r')
        # a.send_sensor_torque('r')
        # a.get_force_sensor('l')
        # a.get_force_sensor('r')
        # a.get_force_side('l')
        rospy.loginfo('123')
        rospy.sleep(0.1)
        # a.get_torque_sensor('l')
        # a.get_torque_sensor('r')
        # a.abc('r')
        # s = a.get_force_sensor('r')
        # b = np.linalg.norm(s[0:2] - s[3:5])
        # print(b)
        # rospy.loginfo('123')
        rospy.sleep(0.1)
  
        
    except  rospy.ROSInterruptException:
        rospy.loginfo('end')
        pass


