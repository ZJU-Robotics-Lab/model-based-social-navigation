# -*- coding: utf-8 -*- 

# 单个机器人控制函数

import rospy
import math
import time
import cv2

from sensor_msgs.msg import LaserScan

from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState

from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist, Point, Pose

from nav_msgs.msg import Path
from nav_msgs.msg import Odometry

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

IMAGE_W = 160
IMAGE_H = 80
MAX_LASER_RANGE = 4.0

def Transformation(lidar_input, current_x, current_y, current_yaw, target_x, target_y, target_yaw):

    lidar_current = np.asarray(lidar_input)

    index_xy = np.linspace(0,360,360)
    x_current = lidar_current * np.sin(index_xy / 360.0 * math.pi).astype(np.float64)
    y_current = lidar_current * np.cos((1- index_xy / 360.0) * math.pi).astype(np.float64)
    z_current = np.zeros_like(x_current)
    ones = np.ones_like(x_current)
    coordinates_current = np.stack([x_current, y_current, z_current, ones], axis=0)

    current_reference_x = current_x
    current_reference_y = current_y
    current_reference_yaw = current_yaw
    target_reference_x = target_x
    target_reference_y = target_y
    target_reference_yaw = target_yaw

    # 坐标系转换：将当前坐标系下的坐标位置经由世界坐标系转到目标坐标系下
    T_target_relative_to_world = np.array([[math.cos(target_reference_yaw), -math.sin(target_reference_yaw), 0, target_reference_x],
                                            [math.sin(target_reference_yaw),  math.cos(target_reference_yaw), 0, target_reference_y],
                                            [                             0,                               0, 1,                  0],
                                            [                             0,                               0, 0,                  1]])

    T_current_relative_to_world = np.array([[math.cos(current_reference_yaw), -math.sin(current_reference_yaw), 0, current_reference_x],
                                            [math.sin(current_reference_yaw),  math.cos(current_reference_yaw), 0, current_reference_y],
                                            [                              0,                                 0, 1,                   0],
                                            [                              0,                                 0, 0,                   1]])

    T_world_relative_to_target = np.linalg.inv(T_target_relative_to_world)
    T_current_relative_to_target = T_world_relative_to_target.dot(T_current_relative_to_world)

    coordinates_target = T_current_relative_to_target.dot(coordinates_current)

    x_target = coordinates_target[0]
    y_target = coordinates_target[1]

    # 转到图像坐标系下
    image_x = np.floor(IMAGE_H - 1.0 - x_target / MAX_LASER_RANGE * IMAGE_H).astype(np.int)
    image_y = np.floor(IMAGE_H - 1.0 - y_target / MAX_LASER_RANGE * IMAGE_H).astype(np.int)

    image_x[(image_x < 0) | (image_x > (IMAGE_H - 1)) | (lidar_current > (MAX_LASER_RANGE - 0.2))] = 0
    image_y[(image_y < 0) | (image_y > (IMAGE_W - 1)) | (lidar_current > (MAX_LASER_RANGE - 0.2))] = 0

    # TEST
    # image_test = np.zeros((IMAGE_H, IMAGE_W))
    # image_test[image_x, image_y] = 1

    # image_test[0, :] = 0
    # image_test[:, 0] = 0

    # cv2.imshow('image_test', image_test)
    # cv2.waitKey(10)


    return image_x, image_y

def Transformation_List(lidar_input, current_x, current_y, current_yaw, target_x, target_y, target_yaw, scale=1):

    lidar_current = lidar_input
    current_x = current_x
    current_y = current_y
    current_yaw = current_yaw
    target_x = target_x
    target_y = target_y
    target_yaw = target_yaw

    index_xy = np.linspace(0, 360, 360)
    x_current = lidar_current * np.sin(index_xy / 360.0 * math.pi).astype(np.float64)
    y_current = lidar_current * np.cos((1 - index_xy / 360.0) * math.pi).astype(np.float64)
    z_current = np.zeros_like(x_current)
    ones = np.ones_like(x_current)
    coordinates_current = np.stack([x_current, y_current, z_current, ones], axis=0)

    current_reference_x = current_x
    current_reference_y = current_y
    current_reference_yaw = current_yaw
    target_reference_x = target_x
    target_reference_y = target_y
    target_reference_yaw = target_yaw

    T_target_relative_to_world = np.array(
        [[math.cos(target_reference_yaw), -math.sin(target_reference_yaw), 0, target_reference_x],
         [math.sin(target_reference_yaw), math.cos(target_reference_yaw), 0, target_reference_y],
         [0, 0, 1, 0],
         [0, 0, 0, 1]])

    T_current_relative_to_world = np.array(
        [[math.cos(current_reference_yaw), -math.sin(current_reference_yaw), 0, current_reference_x],
         [math.sin(current_reference_yaw), math.cos(current_reference_yaw), 0, current_reference_y],
         [0, 0, 1, 0],
         [0, 0, 0, 1]])
    T_world_relative_to_target = np.linalg.inv(T_target_relative_to_world)
    T_current_relative_to_target = T_world_relative_to_target.dot(T_current_relative_to_world)
    coordinates_target = T_current_relative_to_target.dot(coordinates_current)

    x_target = coordinates_target[0]
    y_target = coordinates_target[1]

    lidar_length = np.sqrt(x_target * x_target + y_target * y_target) # 0 ~ 4
    lidar_angle = np.arctan2(y_target, x_target) / math.pi * 180 # -pi ~ pi => -180 ~ 180
    
    flag_in_fov = (lidar_angle > -90) & (lidar_angle < 90) & (lidar_current < (MAX_LASER_RANGE - 0.1)) # -180 ~ 180 => -90 ~ 90

    lidar_length = lidar_length[flag_in_fov]
    lidar_angle = np.floor(lidar_angle[flag_in_fov] * 2 + 179).astype(np.int) # -90 ~ 90 => 0 ~ 359

    lidar_output = np.ones(360)
    lidar_output[lidar_angle] = lidar_length / MAX_LASER_RANGE

    lidar_output = lidar_output[::-1] # reverse

    return lidar_output



class Agent(object):
    def __init__(self, num, threshold_goal, threshold_collision):

        # Info 
        self.num = num
        self.robot_name = 'robot_' + str(num)

        # Goal position 目标位置
        self.goal = [0.0, 0.0]
        self.local_goal = [0.0, 0.0]
        self.delta_local_goal = [0.0, 0.0]
        self.final_local_goal = [0.0, 0.0] # 要考虑yaw角的
        self.agent_position_1 = [0.0, 0.0]
        self.agent_position_2 = [0.0, 0.0]
        self.agent_position_3 = [0.0, 0.0]
        self.agent_position_4 = [0.0, 0.0]

        # Global Goal
        self.goal_pose = PoseStamped() 
        self.goal_pose.header.frame_id = "map"
        self.goal_pose.pose.position.x = self.goal[0]
        self.goal_pose.pose.position.y = self.goal[1]
        self.goal_pose.pose.position.z = 0.0
        self.goal_pose.pose.orientation.x = 0
        self.goal_pose.pose.orientation.y = 0
        self.goal_pose.pose.orientation.z = 0
        self.goal_pose.pose.orientation.w = 1

        # Local Goal
        self.local_goal_pose = PoseStamped() 
        self.local_goal_pose.header.frame_id = self.robot_name + "/base_footprint"
        self.local_goal_pose.pose.position.x = self.local_goal[0]
        self.local_goal_pose.pose.position.y = self.local_goal[1]
        self.local_goal_pose.pose.position.z = 0.0
        self.local_goal_pose.pose.orientation.x = 0
        self.local_goal_pose.pose.orientation.y = 0
        self.local_goal_pose.pose.orientation.z = 0
        self.local_goal_pose.pose.orientation.w = 1

        # State agent状态
        self.odom = Odometry()
        self.global_yaw = 0 # 全局yaw角，用于计算局部目标点位置


        self.observe = [] # 最新lidar输入
        self.observe_image = [] # 40帧雷达图像

        self.reward_distance = 0.0

        final_observation_multi_array = np.zeros((10,363))
        self.final_observation_multi = final_observation_multi_array.tolist()


        # 任务状态
        self.done_collision = False
        self.done_reached_goal = False
        self.final_goal_distance = 20.0
        self.last_final_goal_distance = 20.0
        self.initial_final_goal_distance = 20.0


        self.cumulated_steps = 0.0
        self.cumulated_steps_goal = 0.0
        self.cumulated_reward = 0.0

        self.flag_of_new_goal = 1

        self.save_time = 0
        self.time_lidar = 0
        self.time_odom = 0

        # 任务参数
        self.min_laser_value = 0.21
        self.max_laser_value = MAX_LASER_RANGE
        self.min_range = threshold_collision
        self.reached_goal_threshold = threshold_goal

        
        # Publisher & Subscriber
        self.pub_cmd_vel = rospy.Publisher('/' + self.robot_name + '/cmd_vel', Twist, queue_size=10)
        self.pub_goal = rospy.Publisher('/' + self.robot_name + '/goal', PoseStamped, queue_size=1)
        self.pub_local_goal = rospy.Publisher('/' + self.robot_name + '/local_goal', PoseStamped, queue_size=10)
        
        rospy.Subscriber('/' + self.robot_name + '/kobuki/laser/scan', LaserScan, self.callback_laser, queue_size=1)

        rospy.Subscriber('/' + 'robot_' + str(1) + '/odom', Odometry, self.callback_odometry_1, queue_size=1)
        rospy.Subscriber('/' + 'robot_' + str(2) + '/odom', Odometry, self.callback_odometry_2, queue_size=1)
        rospy.Subscriber('/' + 'robot_' + str(3) + '/odom', Odometry, self.callback_odometry_3, queue_size=1)
        rospy.Subscriber('/' + 'robot_' + str(4) + '/odom', Odometry, self.callback_odometry_4, queue_size=1)


        self.max_state_1 = -9
        self.min_state_1 = 9
        self.max_state_2 = -9
        self.min_state_2 = 9
        self.max_state_3 = -9
        self.min_state_3 = 9
        self.max_state_4 = -9
        self.min_state_4 = 9



    # 重置机器人状态，主要是一些状态记录
    def reset(self):
        self.observe = []
        self.observe_2 = []
        self.observe_image = []
        self.reward_distance = 0.0

        final_observation_multi_array = np.zeros((10,363))
        self.final_observation_multi = final_observation_multi_array.tolist()


        self.goal = [0.0, 0.0]
        self.local_goal = [0.0, 0.0]
        self.delta_local_goal = [0.0, 0.0]
        self.final_local_goal = [0.0, 0.0]
        self.agent_position_1 = [0.0, 0.0]
        self.agent_position_2 = [0.0, 0.0]
        self.agent_position_3 = [0.0, 0.0]
        self.agent_position_4 = [0.0, 0.0]

        # Global Goal
        self.goal_pose = PoseStamped() 
        self.goal_pose.header.frame_id = "map"
        self.goal_pose.pose.position.x = self.goal[0]
        self.goal_pose.pose.position.y = self.goal[1]
        self.goal_pose.pose.position.z = 0.0
        self.goal_pose.pose.orientation.x = 0
        self.goal_pose.pose.orientation.y = 0
        self.goal_pose.pose.orientation.z = 0
        self.goal_pose.pose.orientation.w = 1

        # Local Goal
        self.local_goal_pose = PoseStamped() 
        self.local_goal_pose.header.frame_id = self.robot_name + "/base_footprint"
        self.local_goal_pose.pose.position.x = self.local_goal[0]
        self.local_goal_pose.pose.position.y = self.local_goal[1]
        self.local_goal_pose.pose.position.z = 0.0
        self.local_goal_pose.pose.orientation.x = 0
        self.local_goal_pose.pose.orientation.y = 0
        self.local_goal_pose.pose.orientation.z = 0
        self.local_goal_pose.pose.orientation.w = 1

        self.odom = Odometry()
        self.x = 0
        self.y = 0
        self.heading = 0
        self.x_2 = 0
        self.y_2 = 0
        self.heading_2 = 0

        self.done_collision = False
        self.done_reached_goal = False
        self.flag_of_new_goal = 1

        self.final_goal_distance = 20.0
        self.last_final_goal_distance = 20.0
        self.initial_final_goal_distance = 20.0


        self.cumulated_steps = 0.0
        self.cumulated_steps_goal = 0.0
        self.cumulated_reward = 0.0

        self.time_lidar = 0
        self.time_odom = 0


    # goal_pose 全局目标点保存
    def update_target(self, pose):
        # print("update_target  :   ", pose)
        self.goal = pose

        self.goal_pose.pose.position.x = pose[0]
        self.goal_pose.pose.position.y = pose[1]
        
        self.pub_goal.publish(self.goal_pose)

        self.done_reached_goal = False
        self.flag_of_new_goal = 1
    # laser callback 雷达回调函数: 获取雷达信息，得到 self.observe
    def callback_laser(self, data):
        
        t1 = time.time()
        self.observe = []
        self.reward_distance = self.max_laser_value * 1
        self.time_lidar += 1

        for i in range(len(data.ranges)):

            # 数据获取（降采样）
            # if i % 2 == 0:
            if data.ranges[i] == float('Inf'):
                self.observe.append(self.max_laser_value)
            elif np.isnan(data.ranges[i]):
                self.observe.append(self.min_laser_value)
            else:
                self.observe.append(data.ranges[i])

            if (self.reward_distance > data.ranges[i] > 0):
                self.reward_distance = data.ranges[i]

        t2 = time.time()
        # print("Lidar Time : {}  ms".format(round(1000*(t2-t1),2)))

    # odom callback 里程计回调函数
    def callback_odometry_1(self, odom):
        
        self.agent_position_1[0] = odom.pose.pose.position.x
        self.agent_position_1[1] = odom.pose.pose.position.y
        
        if self.num == 1:
            self.odom = odom
            orientation = odom.pose.pose.orientation
            temp_my_x, temp_my_y, temp_my_z, temp_my_w = orientation.x, orientation.y, orientation.z, orientation.w
            temp_atan2_y = 2.0 * (temp_my_w * temp_my_z + temp_my_x * temp_my_y)
            temp_atan2_x = 1.0 - 2.0 * (temp_my_y * temp_my_y + temp_my_z * temp_my_z)
            self.global_yaw = math.atan2(temp_atan2_y , temp_atan2_x)

    def callback_odometry_2(self, odom):

        self.agent_position_2[0] = odom.pose.pose.position.x
        self.agent_position_2[1] = odom.pose.pose.position.y

        if self.num == 2:
            self.odom = odom
            orientation = odom.pose.pose.orientation
            temp_my_x, temp_my_y, temp_my_z, temp_my_w = orientation.x, orientation.y, orientation.z, orientation.w
            temp_atan2_y = 2.0 * (temp_my_w * temp_my_z + temp_my_x * temp_my_y)
            temp_atan2_x = 1.0 - 2.0 * (temp_my_y * temp_my_y + temp_my_z * temp_my_z)
            self.global_yaw = math.atan2(temp_atan2_y , temp_atan2_x)  

    def callback_odometry_3(self, odom):

        self.agent_position_3[0] = odom.pose.pose.position.x
        self.agent_position_3[1] = odom.pose.pose.position.y

        if self.num == 3:
            self.odom = odom
            orientation = odom.pose.pose.orientation
            temp_my_x, temp_my_y, temp_my_z, temp_my_w = orientation.x, orientation.y, orientation.z, orientation.w
            temp_atan2_y = 2.0 * (temp_my_w * temp_my_z + temp_my_x * temp_my_y)
            temp_atan2_x = 1.0 - 2.0 * (temp_my_y * temp_my_y + temp_my_z * temp_my_z)
            self.global_yaw = math.atan2(temp_atan2_y , temp_atan2_x)

    def callback_odometry_4(self, odom):

        self.time_odom += 1

        self.agent_position_4[0] = odom.pose.pose.position.x
        self.agent_position_4[1] = odom.pose.pose.position.y

        if self.num == 4:
            self.odom = odom
            orientation = odom.pose.pose.orientation
            temp_my_x, temp_my_y, temp_my_z, temp_my_w = orientation.x, orientation.y, orientation.z, orientation.w
            temp_atan2_y = 2.0 * (temp_my_w * temp_my_z + temp_my_x * temp_my_y)
            temp_atan2_x = 1.0 - 2.0 * (temp_my_y * temp_my_y + temp_my_z * temp_my_z)
            self.global_yaw = math.atan2(temp_atan2_y , temp_atan2_x)

    # 获取当前状态： lidar_image & local_goal & current_vel
    def get_state(self):
        """
        观测获取
        """
        # 目标完成判断
        # state ~ final_goal_distance

        self.final_goal_distance = math.hypot(self.goal_pose.pose.position.x - self.odom.pose.pose.position.x, self.goal_pose.pose.position.y - self.odom.pose.pose.position.y)

        if self.final_goal_distance < self.reached_goal_threshold:
            self.done_reached_goal = True
            self.cumulated_steps_goal = 0

        if self.flag_of_new_goal == 1:
            self.flag_of_new_goal = 0
            self.last_final_goal_distance = self.final_goal_distance * 1
            self.initial_final_goal_distance = self.final_goal_distance * 1

        laser_scan = self.observe[:] # 当前雷达
        final_observation = [] # 完整state

        # 碰撞判断
        if self.odom.pose.pose.position.x > 7.78 or self.odom.pose.pose.position.x < -7.78 or self.odom.pose.pose.position.y > 7.78 or self.odom.pose.pose.position.y < -7.78:
            self.done_collision = True

        if (self.min_range > self.reward_distance > 0):
            self.done_collision = True

        # state ~ local_goal_direction
        self.local_goal = [self.goal_pose.pose.position.x, self.goal_pose.pose.position.y]
        self.local_goal_pose.pose.position.x = self.local_goal[0]
        self.local_goal_pose.pose.position.y = self.local_goal[1]

        self.delta_local_goal[0] = self.local_goal[0] - self.odom.pose.pose.position.x
        self.delta_local_goal[1] = self.local_goal[1] - self.odom.pose.pose.position.y
    
        distance_local_goal = math.sqrt((self.delta_local_goal[0])**2 + (self.delta_local_goal[1])**2) 
        theta_for_original_coordinate = math.atan2(self.delta_local_goal[1], self.delta_local_goal[0])
        
        theta_for_new_coordinate = theta_for_original_coordinate - self.global_yaw
        
        self.final_local_goal[0] = distance_local_goal * math.cos(theta_for_new_coordinate)
        self.final_local_goal[1] = distance_local_goal * math.sin(theta_for_new_coordinate)

        local_goal_normolize = math.sqrt((self.final_local_goal[0])**2 + (self.final_local_goal[1])**2)
        if local_goal_normolize >= 1:
            self.final_local_goal[0] = self.final_local_goal[0] / local_goal_normolize 
            self.final_local_goal[1] = self.final_local_goal[1] / local_goal_normolize 
        else:
            self.final_local_goal[0] = self.final_local_goal[0]
            self.final_local_goal[1] = self.final_local_goal[1]

        self.local_goal_pose.pose.position.x = self.final_local_goal[0]
        self.local_goal_pose.pose.position.y = self.final_local_goal[1]

        self.pub_local_goal.publish(self.local_goal_pose)
        self.pub_goal.publish(self.goal_pose)

        # ego
        final_observation = laser_scan[:]
        final_observation.append(self.odom.pose.pose.position.x)
        final_observation.append(self.odom.pose.pose.position.y)
        final_observation.append(self.global_yaw)

        if self.cumulated_steps == 0:
            for ii in range(10):
                self.final_observation_multi.pop(0)
                self.final_observation_multi.append(final_observation[:])
        
        self.final_observation_multi.pop(0)
        self.final_observation_multi.append(final_observation[:])

        final_observation_all = np.asarray(self.final_observation_multi[:])

        observation_output = final_observation_all.ravel().tolist()
        linear_velocity_denoise = round(self.odom.twist.twist.linear.x, 3)
        if linear_velocity_denoise < 0:
            linear_velocity_denoise = 0.0
        if linear_velocity_denoise > 1.0:
            linear_velocity_denoise = 1.0
        observation_output.append(linear_velocity_denoise)                                                                        # 0~1 

        linear_angular_denoise = round(((self.odom.twist.twist.angular.z / 1.5)+1.)/2, 3)
        if linear_angular_denoise < 0:
            linear_angular_denoise = 0.0
        if linear_angular_denoise > 1.0:
            linear_angular_denoise = 1.0
        observation_output.append(linear_angular_denoise)                                                                         # -1.5 ~ 1.5 => 0~1

        goal_distance = round(self.final_goal_distance / 20.0, 3)
        if goal_distance < 0:
            goal_distance = 0.0
        if goal_distance > 1.0:
            goal_distance = 1.0
        observation_output.append(goal_distance)                                                                                  # 0 ~ 20 => 0~1

        goal_direction = round(((math.atan2(self.final_local_goal[1], self.final_local_goal[0]) / math.pi)+1.)/2, 3)
        if goal_direction < 0:
            goal_direction = 0.0
        if goal_direction > 1.0:
            goal_direction = 1.0
        observation_output.append(goal_direction)                                                                                 # -pi ~ pi => 0~1

        # Cost Map show
        # Disentangle
        lidar_10 = final_observation_all[:, 0:360]
        x_10 = final_observation_all[:, 360]
        y_10 = final_observation_all[:, 361]
        yaw_10 = final_observation_all[:, 362]

        disentangle_x = np.zeros((10,360))
        disentangle_y = np.zeros((10,360))
        for i in range(10):
            temp_image_x, temp_image_y = Transformation(lidar_10[i], x_10[i], y_10[i], yaw_10[i], x_10[-1], y_10[-1], yaw_10[-1])
            disentangle_x[i] = temp_image_x
            disentangle_y[i] = temp_image_y

        lidar_ego_image = np.zeros((IMAGE_H, IMAGE_W))

        for i in range(10):
            lidar_ego_image[disentangle_x[i].astype(np.int), disentangle_y[i].astype(np.int)] = 0.1 * i
            
        lidar_ego_image = cv2.dilate(lidar_ego_image, np.ones((3,3), np.uint8), iterations=1)


        lidar_ego_image[0, :] = 0
        lidar_ego_image[:, 0] = 0

        cv2.imshow('lidar_ego_image_' + str(self.num), lidar_ego_image)
        cv2.waitKey(10)

        # lidar_ego_image = np.ones((10, 360))

        # for i in range(10):
        #     temp_lidar = Transformation_List(lidar_10[i], x_10[i], y_10[i], yaw_10[i], x_10[-1], y_10[-1], yaw_10[-1])
        #     lidar_ego_image[i] = temp_lidar[:]

        cv2.imshow('lidar_ego_image_' + str(self.num), lidar_ego_image)
        cv2.waitKey(10)



        return observation_output, self.done_collision, self.done_reached_goal

    # 执行action
    def set_action(self, action):
        
        command_velocity = (action[0] + 1.) / 2.     # -1 ~ 1 => 0~1
        command_angular_velocity = action[1] * 1.5   # -1 ~ 1 => -1.5 ~ 1.5

        delta_linear_speed = command_velocity - self.odom.twist.twist.linear.x

        if delta_linear_speed > 0.1:
            linear_speed = self.odom.twist.twist.linear.x + 0.1
        elif delta_linear_speed < -0.1:
            linear_speed = self.odom.twist.twist.linear.x - 0.1
        else:
            linear_speed = command_velocity

        delta_angular_speed = command_angular_velocity - self.odom.twist.twist.angular.z

        if delta_angular_speed > 0.1:
            angular_speed = self.odom.twist.twist.angular.z + 0.1
        elif delta_angular_speed < -0.1:
            angular_speed = self.odom.twist.twist.angular.z - 0.1
        else:
            angular_speed = command_angular_velocity

        if linear_speed > 1.0:
            linear_speed = 1.0
        if linear_speed < 0.05:
            linear_speed = 0.05

        if angular_speed > 1.5:
            angular_speed = 1.5
        if angular_speed < -1.5:
            angular_speed = -1.5

        if (self.done_reached_goal == True) or (self.done_collision == True):
            linear_speed = 0.0
            angular_speed = 0.0

        # 执行决策指令
        cmd_vel_value = Twist()
        cmd_vel_value.linear.x = linear_speed
        cmd_vel_value.angular.z = angular_speed

        self.pub_cmd_vel.publish(cmd_vel_value)

    def _compute_reward(self):
        
        reward = 0
        reward_goal = 0
        reward_goal_distance = 0
        reward_collision = 0
        reward_social = 0
        reward_rotation = 0
        reward_velocity = 0

        flag_ego_safety = 1
        flag_social_safety = 1

        if not (self.done_collision or self.done_reached_goal):
            
            # GOAL REWARD
            reward_goal = 9.0 * (self.last_final_goal_distance - self.final_goal_distance) 
            # reward_goal_distance = 0.1 * (1 - self.final_goal_distance / self.initial_final_goal_distance)
            self.last_final_goal_distance = self.final_goal_distance * 1

            # COLLISION REWARD
            if self.reward_distance < 1.25:
                reward_collision = - 0.4 * (1 - (self.reward_distance - 0.25) / 1.0)
                if self.reward_distance < 0.7:
                    flag_ego_safety = 0

            # SOCIAL REWARD
            if self.num == 1:
                reward_social += 0
            else:
                distance = math.hypot(self.agent_position_1[0] - self.odom.pose.pose.position.x, self.agent_position_1[1] - self.odom.pose.pose.position.y)
                angle = math.atan2(self.agent_position_1[1] - self.odom.pose.pose.position.y, self.agent_position_1[0] - self.odom.pose.pose.position.x)
                delta_angle = abs(angle - self.global_yaw) / math.pi * 180
                if delta_angle > 180:
                    delta_angle = 360 - delta_angle
                if (distance < 1.25) and (delta_angle < 90):
                    reward_social += - 0.25 * (1 - (distance - 0.25) / 1.)
                    if distance < 0.7:
                        flag_social_safety = 0

            if self.num == 2:
                reward_social += 0
            else:
                distance = math.hypot(self.agent_position_2[0] - self.odom.pose.pose.position.x, self.agent_position_2[1] - self.odom.pose.pose.position.y)
                angle = math.atan2(self.agent_position_2[1] - self.odom.pose.pose.position.y, self.agent_position_2[0] - self.odom.pose.pose.position.x)
                delta_angle = abs(angle - self.global_yaw) / math.pi * 180
                if delta_angle > 180:
                    delta_angle = 360 - delta_angle
                if (distance < 1.25) and (delta_angle < 90):
                    reward_social += - 0.25 * (1 - (distance - 0.25) / 1.)
                    if distance < 0.7:
                        flag_social_safety = 0
            
            if self.num == 3:
                reward_social += 0
            else:
                distance = math.hypot(self.agent_position_3[0] - self.odom.pose.pose.position.x, self.agent_position_3[1] - self.odom.pose.pose.position.y)
                angle = math.atan2(self.agent_position_3[1] - self.odom.pose.pose.position.y, self.agent_position_3[0] - self.odom.pose.pose.position.x)
                delta_angle = abs(angle - self.global_yaw) / math.pi * 180
                if delta_angle > 180:
                    delta_angle = 360 - delta_angle
                if (distance < 1.25) and (delta_angle < 90):
                    reward_social += - 0.25 * (1 - (distance - 0.25) / 1.)
                    if distance < 0.7:
                        flag_social_safety = 0

            if self.num == 4:
                reward_social += 0
            else:
                distance = math.hypot(self.agent_position_4[0] - self.odom.pose.pose.position.x, self.agent_position_4[1] - self.odom.pose.pose.position.y)
                angle = math.atan2(self.agent_position_4[1] - self.odom.pose.pose.position.y, self.agent_position_4[0] - self.odom.pose.pose.position.x)
                delta_angle = abs(angle - self.global_yaw) / math.pi * 180
                if delta_angle > 180:
                    delta_angle = 360 - delta_angle
                if (distance < 1.25) and (delta_angle < 90):
                    reward_social += - 0.25 * (1 - (distance - 0.25) / 1.)
                    if distance < 0.7:
                        flag_social_safety = 0

            # ROTATION REWARD
            if abs(self.odom.twist.twist.angular.z) > 0.5:
                reward_rotation = -0.1 * abs(self.odom.twist.twist.angular.z)

            # VELOCITY REWARD
            if abs(self.odom.twist.twist.linear.x) < 0.5:
                reward_velocity = -0.1 * (0.5 - abs(self.odom.twist.twist.linear.x)) / 0.3
            

            reward = reward_goal + reward_collision + reward_social # + reward_goal_distance # + reward_rotation + reward_velocity

        # DONE REWARD
        if self.done_collision:
            rospy.logerr("TurtleBot2 collision !!! ==> Agent " + str(self.num))
            reward = -20.0 
        
        if self.done_reached_goal:
            rospy.logwarn("TurtleBot2 reached GOAL !!! ==> Agent " + str(self.num))
            reward = 20.0

        #　episode reward
        self.cumulated_reward += reward
        # 总步数
        self.cumulated_steps += 1
        if self.num == 1:
            rospy.logwarn("Cumulated_steps=" + str(self.cumulated_steps))
        # 新目标点步数
        self.cumulated_steps_goal += 1

        return reward, flag_ego_safety, flag_social_safety