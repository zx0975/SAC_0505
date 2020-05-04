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
from gazebo_msgs.msg import ModelState
from std_msgs.msg import String
from geometry_msgs.msg import Vector3 ,Wrench ,WrenchStamped
class Test(core.Env):
    ACTION_VEC_TRANS = 1/180
    ACTION_ORI_TRANS = 1/60
    ACTION_PHI_TRANS = 1/60

    # NAME = ['/right_', '/left_', '/right_']
    NAME = ['/right_', '/right_', '/right_']

    def __init__(self, name, workers):
        self.__name = self.NAME[name%2]
        self.__obname = self.NAME[name%2 + 1]
        if workers == 0:
            self.workers = 'arm'
        else:
            self.workers = str(workers)

        high = np.array([1.,1.,1.,1.,1.,1.,1.,1.,   #8
                         1.,1.,1.,1.,1.,1.])        #6
                                                    #14
        low = -1*high 
                    # ox,oy,oz,oa,ob,oc,od,of,
                    # fx,fy,fz,mx,my,mz                  
        
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32) #?
        self.action_space = spaces.Discrete(3) #?

        self.act_dim = 8
        # self.obs_dim = 14
        self.obs_dim = 8

        self.state = []
        self.action = []
        self.cmd = []

        self.goal = []
        self.goal_pos = []
        # self.goal_quat = []
        # self.goal_phi = 0
        # self.goal_rpy = []
        
        self.old = []
        self.old_pos = []
        # self.old_quat = []
        # self.old_phi = 0

        self.force = []
        self.torque = []
        self.sensor = []

        # self.joint_pos = []
        # self.joint_angle = []

        self.range_cnt = 0
        self.rpy_range = 0
        self.done = True
        self.s_cnt = 0
        self.goal_err = 0.01
        # self.dis_err = 0.08
        self.ori_err = 0.1

        # self.object_pub = 0
        # #random model pos
        # self.set_model_pub = rospy.Publisher(
        #     '/gazebo/set_model_state',
        #     ModelState,
        #     queue_size=1,
        #     latch=True
        # )
        self.set_mode_pub = rospy.Publisher(
            '/set_mode_msg',
            String,
            queue_size=1,
            latch=True
        )
    
        self.seed(345*(workers+1) + 467*(name+1))
        self.reset()
    
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

    # def set_object(self, name, pos, ori):
    #     msg = ModelState()
    #     msg.model_name = name+self.workers
    #     # msg.model_name = hole_2
    #     msg.pose.position.x = pos[0]
    #     msg.pose.position.y = pos[1]
    #     msg.pose.position.z = pos[2]
    #     msg.pose.orientation.w = ori[0]
    #     msg.pose.orientation.x = ori[1]
    #     msg.pose.orientation.y = ori[2]
    #     msg.pose.orientation.z = ori[3]
    #     msg.reference_frame = 'world'
    #     self.set_model_pub.publish(msg)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_force_sensor(self,side):
        if side == 'r':
            r_force=rospy.wait_for_message('/r_force_sensor',WrenchStamped)
            self.force = np.append(r_force.wrench.force.x,r_force.wrench.force.y)
            self.force = np.append(self.force,r_force.wrench.force.z) 
            self.torque = np.append(r_force.wrench.torque.x,r_force.wrench.torque.y)
            self.torque = np.append(self.torque,r_force.wrench.torque.z)            
            self.sensor = np.append(self.force,self.torque)
            return self.sensor
        elif side =='l':
            l_force=rospy.wait_for_message('/l_force_sensor',WrenchStamped)
            self.force = np.append(l_force.wrench.force.x,l_force.wrench.force.y)
            self.force = np.append(self.force,l_force.wrench.force.z) 
            self.torque = np.append(l_force.wrench.torque.x,l_force.wrench.torque.y)
            self.torque = np.append(self.torque,l_force.wrench.torque.z)            
            self.sensor = np.append(self.force,self.torque)
            return self.sensor

    def move(self, goal):
        self.goal = goal
        res = self.get_state_client(self.__name)
        self.old = np.array(res.state)
        # s = np.append(self.old, get_force_sensor('r'))
        s = self.old
        self.dis_pos = np.linalg.norm(self.goal[:3] - s[:3])
        self.state = s
        return s
 
    def reset(self):
        self.old = self.set_old()
        self.goal = self.set_goal()
        # self.state = np.append(self.old, self.get_force_sensor('r'))
        self.state = self.old
        self.dis_pos = np.linalg.norm(self.goal[:3] - self.old[:3])
        self.dis_ori = math.sqrt(np.linalg.norm(self.goal[3:7] - self.old[3:7]) + np.linalg.norm(-1*self.goal[3:7] - self.old[3:7]) - 2)
        self.done = False
        self.success = False

        return self.state

    def set_goal(self):
        # self.goal_start = np.array[0.0, 55.36, -6.36, 0.0, 78.91, 56.61, -80.19, -14.50]# [0.33, 0.0, -0.56, 0, 0, 0, 1] 
        self.goal = np.array([0.0, 56.67, -6.87, 0.0, 80.83, 57.71, -81.27, -13.51])*(np.pi/180)# [0.33, 0.0, -0.58, 0, 0, 0, 1]
        rpy = np.array([0.0, 0.0, 0.0, 0.0])*(np.pi/180)
        self.goal_start = np.append(self.goal, self.range_cnt)
        res = self.set_goal_client(self.goal, rpy, self.__name)
        goal_pos = np.array(res.state)
        if not res.success:
            return self.set_goal()
        else:
            return goal_pos[:7]

    def set_old(self):
        # self.start = np.array([0.0, 63.02, -8.58, 0.0, 88.0, 63.42, -85.22, -9.45])# [0.33, 0.0, -0.5, 0, 0, 0, 1]
        # self.start = np.array([0.0, 0.0, -120.0, 0.0, 150.0, 0.0, -30.0, 0.0])
        self.start = np.array([0.0, 60.46, -8.02, 0.0, 85.47, 61.05, -83.85, -10.96])*(np.pi/180)# [0.33, 0.0, -0.55, 0, 0, 0, 1]
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

        res = self.move_cmd_client(self.cmd, self.__name)# rpy?
        if res.success:
            self.old = np.array(res.state)
            self.limit = res.limit
            # s = np.append(self.old, self.get_force_sensor('r'))
            s = self.old
            # self.dis_z = np.linalg.norm(s[3] + 0.56) #abs?
            # self.theta_z = np.linalg.norm(self.goal[3] - s[3])
            self.dis_pos = np.linalg.norm(self.goal[:3] - s[:3])
            self.dis_ori = math.sqrt(np.linalg.norm(self.goal[3:7] - s[3:7]) + np.linalg.norm(-1*self.goal[3:7] - s[3:7]) - 2)

        terminal = self._terminal(s, res.success)
        reward = self.get_reward(s, res.success, terminal, res.singularity)

        self.state = s

        # if self.workers == 'arm':
        #     if self.object_pub == 0:
        #         self.set_object(self.__name, (0.25, 0.0, 0.8), (0.707388269167, 0.706825181105, 0.0, 0.0))
        #         self.object_pub = 1
        #     else:
        #         self.set_object(self.__name+'q', (0.25, 0.0, 0.8), (0.707388269167, 0.706825181105, 0.0, 0.0))
        #         self.object_pub = 0
        
        # if self.workers == 'arm':
        #     self.set_object(hole_2, (0.25, 0.0, 0.8), (0.707388269167, 0.706825181105, 0.0, 0.0)) #name?
        fail = False
        if not res.success or res.singularity:
            fail = True

        return self.state, reward, terminal, self.success, fail

    def _terminal(self, s, ik_success):
        if ik_success:
            # if self.dis_pos < self.dis_err and self.dis_ori < self.ori_err:
            if self.dis_pos < self.goal_err and self.dis_ori < self.ori_err:
                self.success = True
                if not self.done:
                    self.done = True
                    self.s_cnt += 1
                    self.range_cnt = self.range_cnt
                    self.rpy_range = self.rpy_range
                    self.goal_err = self.goal_err*0.993 if self.goal_err > 0.001 else 0.001
                    self.ori_err = self.ori_err*0.993 if self.ori_err > 0.05 else 0.05
                return True
            else:
                self.success = False
                return False
        else:
            self.success = False
            return False

    # def get_reward(self, s, terminal, singularity, d_z, theta_z, f_max, m_max):
    def get_reward(self, s, ik_success, terminal, singularity):
        reward = 0.
 
        if not ik_success:
            return -20
        
        reward -= self.dis_pos
        reward -= self.dis_ori
        reward += 0.4

        if reward > 0:
            reward *= 2
      
        if singularity:
            reward -= 10
        return reward

        #==================================================================================


