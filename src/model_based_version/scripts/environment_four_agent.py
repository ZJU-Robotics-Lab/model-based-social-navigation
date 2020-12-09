#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import os
import rospy
import numpy as np
import math
from math import pi
import random
import time

from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from gazebo_msgs.srv import SpawnModel, DeleteModel

from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState

import agent

goal_model_1_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], '..', '..', 'model_based_version', 'worlds', 'Target_1', 'model.sdf')
goal_model_2_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], '..', '..', 'model_based_version', 'worlds', 'Target_2', 'model.sdf')
goal_model_3_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], '..', '..', 'model_based_version', 'worlds', 'Target_3', 'model.sdf')
goal_model_4_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], '..', '..', 'model_based_version', 'worlds', 'Target_4', 'model.sdf')



class Env():
    def __init__(self, is_training):

        # Simulation 仿真设置
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)

        self.goal_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self.del_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)

        self.pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)

        threshold_goal = 0.0
        if is_training == True:
            threshold_goal = 0.5
            threshold_collision = 0.24
        else:
            threshold_goal = 0.55
            threshold_collision = 0.24 # 单个agent碰撞0.24，agent之间碰撞0.5，单个agent实际尺寸半径0.2，雷达最小范围0.21

        # Agent 智能体
        self.agent_1 = agent.Agent(num=1, threshold_goal=threshold_goal, threshold_collision=threshold_collision)
        self.agent_2 = agent.Agent(num=2, threshold_goal=threshold_goal, threshold_collision=threshold_collision)
        self.agent_3 = agent.Agent(num=3, threshold_goal=threshold_goal, threshold_collision=threshold_collision)
        self.agent_4 = agent.Agent(num=4, threshold_goal=threshold_goal, threshold_collision=threshold_collision)


        time.sleep(1.0)

        
    # 根据num获取agent状态
    def get_agent_state(self, num):

        done = False
        arrive = False
        if num == 1:
            state_all, done, arrive = self.agent_1.get_state()
        elif num == 2:
            state_all, done, arrive = self.agent_2.get_state()
        elif num == 3:
            state_all, done, arrive = self.agent_3.get_state()
        elif num == 4:
            state_all, done, arrive = self.agent_4.get_state()

        return state_all, done, arrive

    # 根据num获取agent reward
    def get_agent_reward(self, num):

        if num == 1:
            reward, flag_ego_safety, flag_social_safety = self.agent_1._compute_reward()
        elif num == 2:
            reward, flag_ego_safety, flag_social_safety = self.agent_2._compute_reward()
        elif num == 3:
            reward, flag_ego_safety, flag_social_safety = self.agent_3._compute_reward()
        elif num == 4:
            reward, flag_ego_safety, flag_social_safety = self.agent_4._compute_reward()

        return reward, flag_ego_safety, flag_social_safety

    # reset all agent
    def get_agent_reset(self):
        self.agent_1.reset()
        self.agent_2.reset()
        self.agent_3.reset()
        self.agent_4.reset()

    # env step
    def step(self, action_1, action_2, action_3, action_4):
        
        self.agent_1.set_action(action_1)
        self.agent_2.set_action(action_2)
        self.agent_3.set_action(action_3)
        self.agent_4.set_action(action_4)

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/unpause_physics service call failed")

        time.sleep(0.1)
        
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/pause_physics service call failed")
        
        time.sleep(0.1)

        t1 = time.time()
        state_all_1, done_1, arrive_1 = self.get_agent_state(1)
        state_all_2, done_2, arrive_2 = self.get_agent_state(2)
        state_all_3, done_3, arrive_3 = self.get_agent_state(3)
        state_all_4, done_4, arrive_4 = self.get_agent_state(4)
        t2 = time.time()
        # print("Step Time : {}  ms".format(round(1000*(t2-t1),2)))

        reward_1, flag_ego_safety_1, flag_social_safety_1 = self.get_agent_reward(1)
        reward_2, flag_ego_safety_2, flag_social_safety_2 = self.get_agent_reward(2)
        reward_3, flag_ego_safety_3, flag_social_safety_3 = self.get_agent_reward(3)
        reward_4, flag_ego_safety_4, flag_social_safety_4 = self.get_agent_reward(4)

        return state_all_1, reward_1, done_1 or arrive_1, state_all_2, reward_2, done_2 or arrive_2, state_all_3, reward_3, done_3 or arrive_3, state_all_4, reward_4, done_4 or arrive_4
        #, flag_ego_safety_1, flag_social_safety_1, flag_ego_safety_2, flag_social_safety_2, flag_ego_safety_3, flag_social_safety_3, flag_ego_safety_4, flag_social_safety_4
    
    # reset
    def reset(self):
        print("=============================>  Env Reset <=============================")
        # Reset the env

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/pause_physics service call failed")
        
        time.sleep(0.1)
        
        rospy.wait_for_service('/gazebo/delete_model')
        try:
            self.del_model('target_1')
            self.del_model('target_2')
            self.del_model('target_3')
            self.del_model('target_4')
        except (rospy.ServiceException) as e:
            print("gazebo/delete_model service call failed")

        
        time.sleep(0.1)

        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

        time.sleep(0.05)

        # Reset robot
        self.robot_ModelState = ModelState()
        self.robot_ModelState.model_name = 'mobile_base_1'
        self.robot_ModelState.reference_frame = 'map'

        self.robot_ModelState.pose.position.x = -3.5
        self.robot_ModelState.pose.position.y = (6. * random.random()) - 3.
        self.robot_ModelState.pose.position.z = 0.0

        yaw = 0 / 2.0
        # yaw = (6.28 * random.random()) - 3.14
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        self.robot_ModelState.pose.orientation.x = 0.0
        self.robot_ModelState.pose.orientation.y = 0.0
        self.robot_ModelState.pose.orientation.z = sin_yaw
        self.robot_ModelState.pose.orientation.w = cos_yaw

        self.robot_ModelState.twist.linear.x = 0.0
        self.robot_ModelState.twist.angular.z = 0.0

        self.pub.publish(self.robot_ModelState)
        time.sleep(0.05)

        self.robot_ModelState.model_name = 'mobile_base_2'
        self.robot_ModelState.reference_frame = 'map'

        self.robot_ModelState.pose.position.x = 3.5
        self.robot_ModelState.pose.position.y = (6. * random.random()) - 3.

        yaw = - pi / 2.0
        # yaw = (6.28 * random.random()) - 3.14
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        self.robot_ModelState.pose.orientation.x = 0.0
        self.robot_ModelState.pose.orientation.y = 0.0
        self.robot_ModelState.pose.orientation.z = sin_yaw
        self.robot_ModelState.pose.orientation.w = cos_yaw

        self.pub.publish(self.robot_ModelState)
        time.sleep(0.05)

        self.robot_ModelState.model_name = 'mobile_base_3'
        self.robot_ModelState.reference_frame = 'map'

        self.robot_ModelState.pose.position.x = (6. * random.random()) - 3.
        self.robot_ModelState.pose.position.y = 3.5


        yaw = - pi / 2.0 / 2.0
        # yaw = (6.28 * random.random()) - 3.14
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        self.robot_ModelState.pose.orientation.x = 0.0
        self.robot_ModelState.pose.orientation.y = 0.0
        self.robot_ModelState.pose.orientation.z = sin_yaw
        self.robot_ModelState.pose.orientation.w = cos_yaw

        self.pub.publish(self.robot_ModelState)
        time.sleep(0.05)

        self.robot_ModelState.model_name = 'mobile_base_4'
        self.robot_ModelState.reference_frame = 'map'

        self.robot_ModelState.pose.position.x = (6. * random.random()) - 3.
        self.robot_ModelState.pose.position.y = -3.5


        yaw = pi / 2.0 / 2.0
        # yaw = (6.28 * random.random()) - 3.14
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        self.robot_ModelState.pose.orientation.x = 0.0
        self.robot_ModelState.pose.orientation.y = 0.0
        self.robot_ModelState.pose.orientation.z = sin_yaw
        self.robot_ModelState.pose.orientation.w = cos_yaw

        self.pub.publish(self.robot_ModelState)
        
        time.sleep(0.05)

        self.get_agent_reset()

        # Build the targets
        # 1
        # Build the target
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        try:
            goal_urdf = open(goal_model_1_dir, "r").read()
            target_1 = SpawnModel
            target_1.model_name = 'target_1'  # the same with sdf name
            target_1.model_xml = goal_urdf
            goal_model_position = Pose()
            temp_x = 3.5
            temp_y = 0 + 6 * random.random() - 3
            goal_model_position.position.x, goal_model_position.position.y = temp_x, temp_y
            self.agent_1.update_target((temp_x, temp_y))
            self.goal_model(target_1.model_name, target_1.model_xml, 'namespace', goal_model_position, 'world')
            # print('=============    reset  Target reset for agent_1!')
        except (rospy.ServiceException) as e:
            print("/gazebo/failed to build the target_1")
        
        # 2
        # Build the target
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        try:
            goal_urdf = open(goal_model_2_dir, "r").read()
            target_2 = SpawnModel
            target_2.model_name = 'target_2'  # the same with sdf name
            target_2.model_xml = goal_urdf
            goal_model_position = Pose()
            temp_x = -3.5
            temp_y = 0 + 6 * random.random() - 3
            goal_model_position.position.x, goal_model_position.position.y = temp_x, temp_y
            self.agent_2.update_target((temp_x, temp_y))
            self.goal_model(target_2.model_name, target_2.model_xml, 'namespace', goal_model_position, 'world')
            # print('=============    reset  Target reset for agent_2!')
        except (rospy.ServiceException) as e:
            print("/gazebo/failed to build the target_2")
        
        # 3
        # Build the target
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        try:
            goal_urdf = open(goal_model_3_dir, "r").read()
            target_3 = SpawnModel
            target_3.model_name = 'target_3'  # the same with sdf name
            target_3.model_xml = goal_urdf
            goal_model_position = Pose()
            temp_x = 0 + 6 * random.random() - 3
            temp_y = -3.5
            goal_model_position.position.x, goal_model_position.position.y = temp_x, temp_y
            self.agent_3.update_target((temp_x, temp_y))
            self.goal_model(target_3.model_name, target_3.model_xml, 'namespace', goal_model_position, 'world')
            # print('=============    reset  Target reset for agent_3!')
        except (rospy.ServiceException) as e:
            print("/gazebo/failed to build the target_3")
       
        # 4
        # Build the target
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        try:
            goal_urdf = open(goal_model_4_dir, "r").read()
            target_4 = SpawnModel
            target_4.model_name = 'target_4'  # the same with sdf name
            target_4.model_xml = goal_urdf
            goal_model_position = Pose()
            temp_x = 0 + 6 * random.random() - 3
            temp_y = 3.5
            goal_model_position.position.x, goal_model_position.position.y = temp_x, temp_y
            self.agent_4.update_target((temp_x, temp_y))
            self.goal_model(target_4.model_name, target_4.model_xml, 'namespace', goal_model_position, 'world')
            # print('=============    reset  Target reset for agent_4!')
        except (rospy.ServiceException) as e:
            print("/gazebo/failed to build the target_4")

        time.sleep(0.5)

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/unpause_physics service call failed")

        time.sleep(0.2)
        state_all_1, done_1, arrive_1 = self.get_agent_state(1)
        state_all_2, done_2, arrive_2 = self.get_agent_state(2)
        state_all_3, done_3, arrive_3 = self.get_agent_state(3)
        state_all_4, done_4, arrive_4 = self.get_agent_state(4)

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/pause_physics service call failed")

        print("=============================>  Env Reset <=============================")
        return state_all_1, state_all_2, state_all_3, state_all_4
