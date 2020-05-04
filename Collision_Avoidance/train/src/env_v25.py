#!/usr/bin/env python3

"""classic Acrobot task"""
import numpy as np
from numpy import sin, cos, pi
from gym import core, spaces
from gym.utils import seeding
import rospy
import math
import time
from train.srv import get_state, move_cmd, set_goal, set_start
# from CheckCollision_v1 import CheckCollision
from gazebo_msgs.msg import ModelState
from std_msgs.msg import String
from geometry_msgs.msg import Vector3 ,Wrench ,WrenchStamped
from std_msgs.msg import String,Float32MultiArray
class Test(core.Env):
    ACTION_VEC_TRANS = 1/3000
    ACTION_ORI_TRANS = 1/1000
    ACTION_PHI_TRANS = 1/1000
    NAME = ['/right_', '/right_', '/right_']   
    def __init__(self, name, workers):
        self.__name = self.NAME[name%2]
        self.workers = 'arm'
        # if workers == 0:
        #     self.workers = 'arm'
        # else:
        #     self.workers = 'arm'
        #     self.workers = str(workers)

        high = np.array([1.,1.,1.,1.,1.,1.,1.,1.,   #8
                         1.,1.,1.,1.,1.,1.])        #6
                                                    #14
        low = -1*high 
                    # ox,oy,oz,oa,ob,oc,od,of,
                    # fx,fy,fz,mx,my,mz                  
        
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32) #?
        self.action_space = spaces.Discrete(3) #?
        self.act_dim = 8
        self.obs_dim = 14
        self.state = []
        self.action = []
        self.cmd = []
        self.goal = []
        self.goal_pos = []        
        self.old = []
        self.old_pos = []
        self.dis_pos = []
        self.dis_ori = []
        self.r_adjust = []
        self.l_adjust = []
        self.force = []
        self.torque = []
        self.sensor = []
        self.sensor_array = []
        self.max_force = 0
        self.max_torque = 0
        self.range_cnt = 0.0
        self.rpy_range = 0.0
        self.done = True
        self.s_cnt = 0
        # self.goal_err = 0.0001
        # self.ori_err = 0.001
        self.goal_err = 0.001
        self.ori_err = 0.005
        self.reset()
        self.arm_move = rospy.Publisher(
            self.__name + 'arm_move',
            Float32MultiArray,
            queue_size=1,
            latch=True
            )
        rospy.Subscriber('/r_force_sensor',WrenchStamped,self.force_data)    
    @property
    def is_success(self):
        return self.done

    @property
    def success_cnt(self):
        return self.s_cnt
        
    def get_state_client(self, name):
        service = name+self.workers+'/get_state'
        try:
            rospy.wait_for_service(service, timeout=1.)
        except rospy.ROSException as e:
            rospy.logwarn('wait_for_service timeout')
            self.get_state_client(name)
            
        client = rospy.ServiceProxy(
            service,
            get_state
        )
        # res = client(cmd)
        res = client.call()
        return res

    def move_cmd_client(self, cmd, name):
        service = name+self.workers+'/move_cmd'
        try:
            rospy.wait_for_service(service, timeout=1.)
        except rospy.ROSException as e:
            rospy.logwarn('wait_for_service timeout')
            self.move_cmd_client(cmd, name)
            
        client = rospy.ServiceProxy(
            service,
            move_cmd
        )
        # res = client(cmd)
        res = client.call(cmd)
        return res

    def set_start_client(self, cmd, rpy, name):
        service = name+self.workers+'/set_start'
        try:
            rospy.wait_for_service(service, timeout=1.)
        except rospy.ROSException as e:
            rospy.logwarn('wait_for_service timeout')
            self.set_start_client(cmd, rpy, name)
            
        client = rospy.ServiceProxy(
            service,
            set_start
        )
        # res = client(cmd)
        res = client(action=cmd, rpy=rpy)
        return res

    def set_goal_client(self, cmd, rpy, name):
        service = name+self.workers+'/set_goal'
        try:
            rospy.wait_for_service(service, timeout=1.)
        except rospy.ROSException as e:
            rospy.logwarn('wait_for_service timeout')
            self.set_goal_client(cmd, rpy, name)
            
        client = rospy.ServiceProxy(
            service,
            set_goal
        )
        # res = client(cmd)
        res = client(action=cmd, rpy=rpy)
        return res
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def force_data(self,data):
        # rospy.loginfo(data)
        self.sensor_array = np.array([data.wrench.force.x,\
                                      data.wrench.force.y,\
                                      data.wrench.force.z,\
                                      data.wrench.torque.x,\
                                      data.wrench.torque.y,\
                                      data.wrench.torque.z])
        # self.sensor_array = np.array([0,0,0,0,0,0])
        # print(self.sensor_array, len(self.sensor_array))

    def reset(self):
        self.old = self.set_old()
        self.goal = self.set_goal()
        self.state = np.append(self.old, self.sensor_array[:])
        self.dis_x = np.linalg.norm(self.goal[0] - self.old[0])
        self.dis_y = np.linalg.norm(self.goal[1] - self.old[1])
        self.dis_z = np.linalg.norm(self.goal[2] - self.old[2])
        self.delta_z = np.linalg.norm(self.old[2] + 0.555)
        self.dis_pos = np.linalg.norm(self.goal[:3] - self.old[:3])
        self.dis_ori = math.sqrt(np.linalg.norm(self.goal[3:7] - self.old[3:7]) + np.linalg.norm(-1*self.goal[3:7] - self.old[3:7]) - 2)
        self.max_force = np.linalg.norm(self.state[8:10])
        self.max_torque = np.linalg.norm(self.state[11:13]) 
        self.done = False
        self.success = False

        return self.state

    def set_goal(self):
        # self.goalpos = np.array([0.0, 55.36, -6.36, 0.0, 78.91, 56.61, -80.19, -14.50])*(np.pi/180)# [0.33, 0.0, -0.56, 0, 0, 0, 1]
        self.goalpos = np.array([0.0, 56.67, -6.87, 0.0, 80.83, 57.71, -81.27, -13.51])*(np.pi/180)# [0.33, 0.0, -0.58, 0, 0, 0, 1]
        # self.goalpos = np.array([0.0, 55.48, -6.40, 0.0, 79.09, 56.71, -80.29, -14.41])*(np.pi/180)# [0.33, 0.0, -0.59, 0, 0, 0, 1]
        rpy = np.array([0.0, 0.0, 0.0, 0.0])*(np.pi/180)
        self.goalpos = np.append(self.goalpos, self.range_cnt)
        res = self.set_goal_client(self.goalpos, rpy, self.__name)
        goal_pos = np.array(res.state)
        if not res.success:
            return self.set_goal()
        else:
            return goal_pos[:7]

    def set_old(self):
        # self.start = np.array([0.0, 63.02, -8.58, 0.0, 88.0, 63.42, -85.22, -9.45])*(np.pi/180)# [0.33, 0.0, -0.5, 0, 0, 0, 1]
        # self.start = np.array([0.0, 60.46, -8.02, 0.0, 85.47, 61.05, -83.85, -10.96])*(np.pi/180)# [0.33, 0.0, -0.55, 0, 0, 0, 1]
        self.start = np.array([0.0, 59.81, -7.85, 0.0, 84.75, 60.46, -83.46, -11.37])*(np.pi/180)# [0.33, 0.0, -0.555, 0, 0, 0, 1]
        # self.start = np.array([0.0, 59.16, -7.67, 0.0, 84.01, 59.88, -83.05, -11.79])*(np.pi/180)# [0.33, 0.0, -0.56, 0, 0, 0, 1]
        # self.start = np.array([0.0, 58.53, -7.48, 0.0, 83.25, 59.32, -82.62, -12.21])*(np.pi/180)# [0.33, 0.0, -0.565, 0, 0, 0, 1]
        rpy = np.array([0.0, 0.0, 0.0, 0.0])*(np.pi/180)
        self.start = np.append(self.start, self.range_cnt)
        res = self.set_start_client(self.start, rpy, self.__name)
        old_pos = np.array(res.state)      
        if not res.success:
            return self.set_old()
        else:
            return old_pos

    def step(self, a):      
        s = self.state
        action_vec = a[:3]*self.ACTION_VEC_TRANS
        action_ori = a[3:7]*self.ACTION_ORI_TRANS
        action_phi = a[7]*self.ACTION_PHI_TRANS
        self.action = np.append(action_vec, action_ori)
        self.action = np.append(self.action, action_phi)
        self.cmd = np.add(s[:8], self.action)
        self.cmd[3:7] /= np.linalg.norm(self.cmd[3:7])
        res = self.move_cmd_client(self.cmd, self.__name)
        aaaarmmm = Float32MultiArray()
        aaaarmmm.data =  self.cmd
        self.arm_move.publish(aaaarmmm)

        if res.success:
            self.old = np.array(res.state)
            s = np.append(self.old, self.sensor_array[:])
            self.dis_x = np.linalg.norm(self.goal[0] - s[0])
            self.dis_y = np.linalg.norm(self.goal[1] - s[1])
            self.dis_z = np.linalg.norm(self.goal[2] - s[2])
            self.delta_z = np.linalg.norm(s[2] + 0.555)
            self.dis_pos = np.linalg.norm(self.goal[:3] - s[:3])
            self.dis_ori = math.sqrt(np.linalg.norm(self.goal[3:7] - s[3:7]) + np.linalg.norm(-1*self.goal[3:7] - s[3:7]) - 2)
            self.max_force = np.linalg.norm(s[8:10])
            self.max_torque = np.linalg.norm(s[11:13])
        terminal = self._terminal(s, res.success)
        reward = self.get_reward(s, res.success, terminal, res.singularity)
        
        self.state = s
        fail = False
        # if self.max_force > 300 or self.max_torque > 30 or self.dis_x  > 1.0 or self.dis_y > 1.0:
        #     fail = True
        if not res.success or res.singularity:
            fail = True
        
        return self.state, reward, terminal, self.success, fail

    def _terminal(self, s, ik_success):
        if ik_success:
            if self.dis_pos < self.goal_err and self.dis_ori < self.ori_err:
                self.success = True
                if not self.done:
                    self.done = True
                    self.s_cnt += 1
                    self.goal_err = self.goal_err*0.993 if self.goal_err > 0.0001 else 0.0001
                    self.ori_err = self.ori_err*0.993 if self.ori_err > 0.0005 else 0.0005
                    # self.goal_err = self.goal_err*0.993 if self.goal_err > 0.00001 else 0.00001
                    # self.ori_err = self.ori_err*0.993 if self.ori_err > 0.0005 else 0.0005 
                return True
            else:
                self.success = False
                return False
        else:
            self.success = False
            return False

    # def get_reward(self, s, terminal, singularity, d_z, delta_z, f_max, m_max):
    def get_reward(self, s, ik_success, terminal, singularity):
        reward = 0.
        f_reward = 0.
        z_reward = 0.
 
        if not ik_success:
            return -1

        # reward -= self.dis_z
        # reward -= self.delta_z
        # print(self.max_force)
        # print(self.max_torque)
        # print(self.dis_z)
        # print(self.delta_z)
        
        if self.max_force > 80:
            if self.max_torque > 4:
                f_reward = -1.0
            elif self.max_torque > 3:
                f_reward = -1.0
            elif self.max_torque > 2:
                f_reward = -0.8
            elif self.max_torque > 1:
                f_reward = -0.6
            elif self.max_torque > 0:
                f_reward = -0.6
        elif self.max_force > 60:
            if self.max_torque > 4:
                f_reward = -1.0
            elif self.max_torque > 3:
                f_reward = -1.0
            elif self.max_torque > 2:
                f_reward = -0.6
            elif self.max_torque > 1:
                f_reward = -0.6
            elif self.max_torque > 0:
                f_reward = -0.6
        elif self.max_force > 40:
            if self.max_torque > 4:
                f_reward = -0.6
            elif self.max_torque > 3:
                f_reward = -0.6
            elif self.max_torque > 2:
                f_reward = -0.4
            elif self.max_torque > 1:
                f_reward = -0.2
            elif self.max_torque > 0:
                f_reward = -0.2
        elif self.max_force > 20:
            if self.max_torque > 4:
                f_reward = -0.4
            elif self.max_torque > 3:
                f_reward = -0.4
            elif self.max_torque > 2:
                f_reward = -0.4
            elif self.max_torque > 1:
                f_reward = -0.2
            elif self.max_torque > 0:
                f_reward = -0.2
        elif self.max_force > 0:
            if self.max_torque > 4:
                f_reward = -0.4
            elif self.max_torque > 3:
                f_reward = -0.4
            elif self.max_torque > 2:
                f_reward = -0.4
            elif self.max_torque > 1:
                f_reward = -0.2
            elif self.max_torque > 0:
                f_reward = -0.2

        if self.dis_z > 0.02:
            if self.delta_z < 0.001:
                z_reward = -1.0
            elif self.delta_z < 0.002:
                z_reward = -0.8
            elif self.delta_z < 0.003:
                z_reward = -0.6
            elif self.delta_z < 0.004:
                z_reward = -0.4
            elif self.delta_z < 0.005:
                z_reward = -0.4
        elif self.dis_z > 0.015:
            if self.delta_z < 0.001:
                z_reward = -1.0
            elif self.delta_z < 0.002:
                z_reward = -0.8
            elif self.delta_z < 0.003:
                z_reward = -0.4
            elif self.delta_z < 0.004:
                z_reward = -0.2
            elif self.delta_z < 0.005:
                z_reward = -0.2
        elif self.dis_z > 0.010:
            if self.delta_z < 0.001:
                z_reward = -0.6
            elif self.delta_z < 0.002:
                z_reward = -0.6
            elif self.delta_z < 0.003:
                z_reward = -0.4
            elif self.delta_z < 0.004:
                z_reward = -0.2
            elif self.delta_z < 0.005:
                z_reward = -0.2
        elif self.dis_z > 0.005:
            if self.delta_z < 0.001:
                z_reward = -0.4
            elif self.delta_z < 0.002:
                z_reward = -0.4
            elif self.delta_z < 0.003:
                z_reward = -0.4
            elif self.delta_z < 0.004:
                z_reward = -0.2
            elif self.delta_z < 0.005:
                z_reward = -0.2
        elif self.dis_z > 0.0:
            if self.delta_z < 0.001:
                z_reward = -0.4
            elif self.delta_z < 0.002:
                z_reward = -0.4
            elif self.delta_z < 0.003:
                z_reward = -0.4
            elif self.delta_z < 0.004:
                z_reward = -0.2
            elif self.delta_z < 0.005:
                z_reward = -0.2

        if f_reward == -1.0:
            if z_reward == -1.0:
                reward -= 1.0
            elif z_reward == -0.8:
                reward -= 1.0
            elif z_reward == -0.6:
                reward -= 0.8
            elif z_reward == -0.4:
                reward -= 0.6
            elif z_reward == -0.2:
                reward -= 0.6
        elif f_reward == -0.8:
            if z_reward == -1.0:
                reward -= 1.0
            elif z_reward == -0.8:
                reward -= 0.6
            elif z_reward == -0.6:
                reward -= 0.6
            elif z_reward == -0.4:
                reward -= 0.4
            elif z_reward == -0.2:
                reward -= 0.4
        elif f_reward == -0.6:
            if z_reward == -1.0:
                reward -= 0.6
            elif z_reward == -0.8:
                reward -= 0.6
            elif z_reward == -0.6:
                reward -= 0.4
            elif z_reward == -0.4:
                reward -= 0.4
            elif z_reward == -0.2:
                reward -= 0.2
        elif f_reward == -0.4:
            if z_reward == -1.0:
                reward -= 0.4
            elif z_reward == -0.8:
                reward -= 0.4
            elif z_reward == -0.6:
                reward -= 0.4
            elif z_reward == -0.4:
                reward -= 0.2
            elif z_reward == -0.2:
                reward -= 0.2
        elif f_reward == -0.2:
            if z_reward == -1.0:
                reward -= 0.4
            elif z_reward == -0.8:
                reward -= 0.2
            elif z_reward == -0.6:
                reward -= 0.2
            elif z_reward == -0.4:
                reward -= 0.2
            elif z_reward == -0.2:
                reward -= 0.2
        else:
            reward = -1.0
        
        reward -= 1
        # reward *= 10
        # reward += (f_reward*5)
        # reward += (z_reward*10)
        # reward -= self.delta_z
        # reward -= (self.dis_z*1000)

        if reward > 0:
            reward *= 2
        # elif reward < 0:
        #     reward *= 3  

        if singularity:
            reward -= 1
        # print (reward)
        return reward

        #==================================================================================


