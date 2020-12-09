# -*- coding: utf-8 -*- 
import copy
import numpy as np
import torch
import math
import time
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from torchsummary import summary
from PIL import Image
import cv2
import kornia

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


IMAGE_W = 160
IMAGE_H = 80
MAX_LASER_RANGE = 4.0


def weights_init(m):
    if isinstance(m, nn.Conv3d):
        nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
        try:
            nn.init.constant_(m.bias, 0.001)
        except:
            pass
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(m.bias, 0.001)


# numpy
def Transformation(lidar_input, current_x, current_y, current_yaw, target_x, target_y, target_yaw):

    lidar_current = lidar_input
    current_x = current_x
    current_y = current_y
    current_yaw = current_yaw
    target_x = target_x
    target_y = target_y
    target_yaw = target_yaw

    t1 = time.time()
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

    image_x = np.rint(IMAGE_H - 1.0 - x_target / MAX_LASER_RANGE * IMAGE_H).astype(np.int)
    image_y = np.rint(IMAGE_H - 1.0 - y_target / MAX_LASER_RANGE * IMAGE_H).astype(np.int)

    image_x[(image_x < 0) | (image_x > (IMAGE_H - 1)) | (lidar_current > (MAX_LASER_RANGE - 0.2))] = 0
    image_y[(image_y < 0) | (image_y > (IMAGE_W - 1)) | (lidar_current > (MAX_LASER_RANGE - 0.2))] = 0

    t2 = time.time()
    # print("Transformation Time : {}  ms".format(round(1000*(t2-t1),2)))

    return image_x, image_y

        
lidar_length = 360
# 10 * 80 * 160  (范围8m)
# 连续输入10帧lidar数据
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.group1 = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))) # 1 * 10 * 80 * 160 => 4 * 5 * 40 * 80
        
        self.group2 = nn.Sequential(
            nn.Conv3d(4, 8, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))) # 4 * 5 * 40 * 80 => 8 * 2 * 20 * 40

        self.group3 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))) # 8 * 2 * 20 * 40 => 16 * 1 * 10 * 20

        self.group4 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))) # 16 * 2 * 10 * 20 => 32 * 1 * 5 * 10
        

        self.l_vel_ang = nn.Linear(2, 32)
        self.l_goal = nn.Linear(2, 32)
        self.l_lidar_state = nn.Linear(32 * 1 * 5 * 10, 256)

        self.l0 = nn.Linear(256, 256)
        self.l1 = nn.Linear(256 + 32 + 32, 256)
        self.l2 = nn.Linear(256, 64)
        self.l3 = nn.Linear(64, action_dim)
        self.max_action = max_action
        self.save_time = 0

    def forward(self, lidar_ego_image, vel_ang, goal):

        lidar_state = lidar_ego_image

        lidar_state = F.leaky_relu(self.group1(lidar_state))
        lidar_state = F.leaky_relu(self.group2(lidar_state))
        lidar_state = F.leaky_relu(self.group3(lidar_state))
        lidar_state = F.leaky_relu(self.group4(lidar_state))

        lidar_state = lidar_state.view(-1, 32 * 1 * 5 * 10)
        lidar_state = F.leaky_relu(self.l_lidar_state(lidar_state)) # 32 * 1 * 5 * 10 => 256
        lidar_state = F.leaky_relu(self.l0(lidar_state)) # 256 => 256

        vel_ang_state = F.leaky_relu(self.l_vel_ang(vel_ang)) # 2 => 32

        goal_state = F.leaky_relu(self.l_goal(goal)) # 2 => 32
        

        a = torch.cat([lidar_state, vel_ang_state, goal_state], dim=1)

        a = F.leaky_relu(self.l1(a)) # 256 + 32 + 32 => 256
        a = F.leaky_relu(self.l2(a)) # 256 => 64

        return self.max_action * torch.tanh(self.l3(a)) # 64 => 2 action



class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture

        self.group1_1 = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))) # 1 * 10 * 80 * 160 => 4 * 5 * 40 * 80
        
        self.group2_1 = nn.Sequential(
            nn.Conv3d(4, 8, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))) # 4 * 5 * 40 * 80 => 8 * 2 * 20 * 40

        self.group3_1 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))) # 8 * 2 * 20 * 40 => 16 * 1 * 10 * 20

        self.group4_1 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))) # 16 * 2 * 10 * 20 => 32 * 1 * 5 * 10

        self.l_vel_ang_1 = nn.Linear(2, 32)
        self.l_goal_1 = nn.Linear(2, 32)
        self.l_lidar_state_1 = nn.Linear(32 * 1 * 5 * 10, 256)
        self.l_action_1 = nn.Linear(action_dim, 32)

        self.l1_1 = nn.Linear(256, 256)
        self.l2_1 = nn.Linear(256 + 32 + 32 + 32,  256)
        self.l3_1 = nn.Linear(256, 64)
        self.l4_1 = nn.Linear(64, 1)


        # Q2 architecture

        self.group1_2 = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))) # 1 * 10 * 80 * 160 => 4 * 5 * 40 * 80
        
        self.group2_2 = nn.Sequential(
            nn.Conv3d(4, 8, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))) # 4 * 5 * 40 * 80 => 8 * 2 * 20 * 40

        self.group3_2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))) # 8 * 2 * 20 * 40 => 16 * 1 * 10 * 20

        self.group4_2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))) # 16 * 2 * 10 * 20 => 32 * 1 * 5 * 10

        self.l_vel_ang_2 = nn.Linear(2, 32)
        self.l_goal_2 = nn.Linear(2, 32)
        self.l_lidar_state_2 = nn.Linear(32 * 1 * 5 * 10, 256)
        self.l_action_2 = nn.Linear(action_dim, 32)

        self.l1_2 = nn.Linear(256, 256)
        self.l2_2 = nn.Linear(256 + 32 + 32 + 32,  256)
        self.l3_2 = nn.Linear(256, 64)
        self.l4_2 = nn.Linear(64, 1)


    def forward(self, lidar_ego_image, vel_ang, goal, action):

        lidar_state = lidar_ego_image

        # Q1 
        lidar_state_1 = F.leaky_relu(self.group1_1(lidar_state))
        lidar_state_1 = F.leaky_relu(self.group2_1(lidar_state_1))
        lidar_state_1 = F.leaky_relu(self.group3_1(lidar_state_1))
        lidar_state_1 = F.leaky_relu(self.group4_1(lidar_state_1))

        lidar_state_1 = lidar_state_1.view(-1, 32 * 1 * 5 * 10)
        lidar_state_1 = F.leaky_relu(self.l_lidar_state_1(lidar_state_1)) # 16 * 1 * 5 * 10 => 256
        lidar_state_1 = F.leaky_relu(self.l1_1(lidar_state_1)) # 256 => 256

        vel_ang_state_1 = F.leaky_relu(self.l_vel_ang_1(vel_ang)) # 2 => 32

        goal_state_1 = F.leaky_relu(self.l_goal_1(goal)) # 2 => 32

        action_1 = F.leaky_relu(self.l_action_1(action)) # 2 => 32

        sa1 = torch.cat([lidar_state_1, vel_ang_state_1, goal_state_1, action_1], dim=1)
        q1 = F.leaky_relu(self.l2_1(sa1))
        q1 = F.leaky_relu(self.l3_1(q1))
        q1 = self.l4_1(q1)

        # Q2 
        lidar_state_2 = F.leaky_relu(self.group1_2(lidar_state))
        lidar_state_2 = F.leaky_relu(self.group2_2(lidar_state_2))
        lidar_state_2 = F.leaky_relu(self.group3_2(lidar_state_2))
        lidar_state_2 = F.leaky_relu(self.group4_2(lidar_state_2))

        lidar_state_2 = lidar_state_2.view(-1, 32 * 1 * 5 * 10)
        lidar_state_2 = F.leaky_relu(self.l_lidar_state_2(lidar_state_2)) # 16 * 1 * 5 * 10 => 256
        lidar_state_2 = F.leaky_relu(self.l1_2(lidar_state_2)) # 256 => 256

        vel_ang_state_2 = F.leaky_relu(self.l_vel_ang_2(vel_ang)) # 2 => 32

        goal_state_2 = F.leaky_relu(self.l_goal_2(goal)) # 2 => 32

        action_2 = F.leaky_relu(self.l_action_2(action)) # 2 => 32

        sa2 = torch.cat([lidar_state_2, vel_ang_state_2, goal_state_2, action_2], dim=1)
        q2 = F.leaky_relu(self.l2_2(sa2))
        q2 = F.leaky_relu(self.l3_2(q2))
        q2 = self.l4_2(q2)

        return q1, q2

    def Q1(self, lidar_ego_image, vel_ang, goal, action):

        lidar_state = lidar_ego_image

        # Q1 
        lidar_state_1 = F.leaky_relu(self.group1_1(lidar_state))
        lidar_state_1 = F.leaky_relu(self.group2_1(lidar_state_1))
        lidar_state_1 = F.leaky_relu(self.group3_1(lidar_state_1))
        lidar_state_1 = F.leaky_relu(self.group4_1(lidar_state_1))

        lidar_state_1 = lidar_state_1.view(-1, 32 * 1 * 5 * 10)
        lidar_state_1 = F.leaky_relu(self.l_lidar_state_1(lidar_state_1)) # 16 * 1 * 5 * 10 => 256
        lidar_state_1 = F.leaky_relu(self.l1_1(lidar_state_1)) # 256 => 256

        vel_ang_state_1 = F.leaky_relu(self.l_vel_ang_1(vel_ang)) # 2 => 32

        goal_state_1 = F.leaky_relu(self.l_goal_1(goal)) # 2 => 32

        action_1 = F.leaky_relu(self.l_action_1(action)) # 2 => 32

        sa1 = torch.cat([lidar_state_1, vel_ang_state_1, goal_state_1, action_1], dim=1)
        q1 = F.leaky_relu(self.l2_1(sa1))
        q1 = F.leaky_relu(self.l3_1(q1))
        q1 = self.l4_1(q1)

        return q1

class TD3(object):
    def __init__(
        self,
        state_dim=3634,
        action_dim=2,
        max_action=1.0,
        discount=0.99,
        tau=0.005,
        policy_noise=0.1,
        noise_clip=0.15,
        policy_freq=2
    ):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor.apply(weights_init)
        # nn.init.xavier_normal_(self.actor.l3.weight)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic.apply(weights_init)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.scheduler_actor = lr_scheduler.StepLR(self.actor_optimizer, 50, 0.99)
        self.scheduler_critic = lr_scheduler.StepLR(self.critic_optimizer, 50, 0.99)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.critic_loss_viz = 0
        self.actor_loss_viz = 0

        self.total_it = 0
        self.save_time = 0

    def select_action(self, state, images, transformation, num_pre):

        # state preperation
        final_observation_all = state[0:3630]
        vel_ang = state[3630:3632]
        goal = state[3632:3634]

        final_observation_all = final_observation_all.reshape(-1, 10, 363)
        lidar_10 = final_observation_all[0, :, 0:360]
        x_10 = final_observation_all[0, :, 360]
        y_10 = final_observation_all[0, :, 361]
        yaw_10 = final_observation_all[0, :, 362]

        disentangle_x = np.zeros((1, 10, 360))
        disentangle_y = np.zeros((1, 10, 360))

        lidar_ego_image = np.zeros((1, 10, 80, 160))

        for i in range(10):
            temp_image_x, temp_image_y = Transformation(lidar_10[i], x_10[i], y_10[i], yaw_10[i], x_10[-1], y_10[-1], yaw_10[-1])
            disentangle_x[0][i] = temp_image_x
            disentangle_y[0][i] = temp_image_y
            lidar_ego_image[0, i, disentangle_x[0, i].astype(np.int), disentangle_y[0, i].astype(np.int)] = 1
            lidar_ego_image[0, i, 0, :] = 0
            lidar_ego_image[0, i, :, 0] = 0
            lidar_ego_image[0, i] = cv2.dilate(lidar_ego_image[0, i], np.ones((3,3), np.uint8), iterations=1)
            # lidar_ego_image[0, i] = cv2.GaussianBlur(lidar_ego_image[0, i], (5, 5), 0)

        # combine with prediction
        lidar_ego_image = torch.from_numpy(lidar_ego_image).float()
        for n in range(int(num_pre[0])):
            init_delta_x = transformation[0][n][0]
            init_delta_y = transformation[0][n][1]
            init_delta_theta = transformation[0][n][2]

            init_translation_matrix = torch.zeros((10, 2, 3))
            init_translation_matrix[:, 0, 0] = 1
            init_translation_matrix[:, 1, 1] = 1
            init_translation_matrix[:, 0, 2] = init_delta_y
            init_translation_matrix[:, 1, 2] = init_delta_x

            lidar_ego_image[0] = kornia.warp_affine(lidar_ego_image[0].unsqueeze(1),
                                                            init_translation_matrix, dsize=(80, 160),
                                                            flags='bilinear', padding_mode='zeros',
                                                            align_corners=False).squeeze(1)

            init_scale = torch.ones(10)
            init_center = torch.ones(10, 2)
            init_angle = torch.ones(10) * init_delta_theta
            init_center[:, 0] = IMAGE_H
            init_center[:, 1] = IMAGE_H

            init_rotation_matrix = kornia.get_rotation_matrix2d(init_center, init_angle, init_scale)

            lidar_ego_image[0] = kornia.warp_affine(lidar_ego_image[0].unsqueeze(1),
                                                            init_rotation_matrix, dsize=(80, 160),
                                                            flags='bilinear', padding_mode='zeros',
                                                            align_corners=False).squeeze(1)

            # new_image = cv2.erode(image[0][n], np.ones((3,3), np.uint8), iterations=1)
            # new_image = cv2.dilate(new_image, np.ones((3,3), np.uint8), iterations=1)
            new_image = torch.from_numpy(image[0][n]).unsqueeze(0).float()
            lidar_ego_image[0] = torch.cat([lidar_ego_image[0][1:10], new_image], dim=0)

        lidar_ego_image = lidar_ego_image.unsqueeze(1).to(device)
        vel_ang = torch.from_numpy(vel_ang).unsqueeze(0).float().to(device)
        goal = torch.from_numpy(goal).unsqueeze(0).float().to(device)

        self.actor.eval()
        
        return self.actor(lidar_ego_image, vel_ang, goal).cpu().data.numpy().flatten()

    def select_action_batch(self, state, images, transformation, num_pre):

        final_observation_all = state[:, 0:3630]
        vel_ang = state[:, 3630:3632]
        goal = state[:, 3632:3634]

        final_observation_all = final_observation_all.reshape(-1, 10, 363)
        lidar_10 = final_observation_all[:, :, 0:360]
        x_10 = final_observation_all[:, :, 360]
        y_10 = final_observation_all[:, :, 361]
        yaw_10 = final_observation_all[:, :, 362]

        disentangle_x = np.zeros((state.shape[0], 10,360))
        disentangle_y = np.zeros((state.shape[0], 10,360))

        lidar_ego_image = np.zeros((state.shape[0], 10, 80, 160))

        for m in range(state.shape[0]):
            for i in range(10):
                temp_image_x, temp_image_y = Transformation(lidar_10[m][i], x_10[m][i], y_10[m][i], yaw_10[m][i], x_10[m][-1], y_10[m][-1], yaw_10[m][-1])
                disentangle_x[m][i] = temp_image_x
                disentangle_y[m][i] = temp_image_y
                lidar_ego_image[m, i, disentangle_x[m, i].astype(np.int), disentangle_y[m, i].astype(np.int)] = 1
                lidar_ego_image[m, i, 0, :] = 0
                lidar_ego_image[m, i, :, 0] = 0
                lidar_ego_image[m, i] = cv2.dilate(lidar_ego_image[m, i], np.ones((3,3), np.uint8), iterations=1)
                # lidar_ego_image[m, i] = cv2.GaussianBlur(lidar_ego_image[m, i], (5, 5), 0)
        
        # combine with prediction
        lidar_ego_image = torch.from_numpy(lidar_ego_image).float()
        for m in range(state.shape[0]):
            for n in range(int(num_pre[m])):
                init_delta_x = transformation[m][n][0]
                init_delta_y = transformation[m][n][1]
                init_delta_theta = transformation[m][n][2]

                init_translation_matrix = torch.zeros((10, 2, 3))
                init_translation_matrix[:, 0, 0] = 1
                init_translation_matrix[:, 1, 1] = 1
                init_translation_matrix[:, 0, 2] = init_delta_y
                init_translation_matrix[:, 1, 2] = init_delta_x

                lidar_ego_image[m] = kornia.warp_affine(lidar_ego_image[m].unsqueeze(1),
                                                                init_translation_matrix, dsize=(80, 160),
                                                                flags='bilinear', padding_mode='zeros',
                                                                align_corners=False).squeeze(1)

                init_scale = torch.ones(10)
                init_center = torch.ones(10, 2)
                init_angle = torch.ones(10) * init_delta_theta
                init_center[:, 0] = IMAGE_H
                init_center[:, 1] = IMAGE_H

                init_rotation_matrix = kornia.get_rotation_matrix2d(init_center, init_angle, init_scale)

                lidar_ego_image[m] = kornia.warp_affine(lidar_ego_image[m].unsqueeze(1),
                                                                init_rotation_matrix, dsize=(80, 160),
                                                                flags='bilinear', padding_mode='zeros',
                                                                align_corners=False).squeeze(1)
                # new_image = cv2.erode(image[m][n], np.ones((3,3), np.uint8), iterations=1)
                # new_image = cv2.dilate(new_image, np.ones((3,3), np.uint8), iterations=1)
                new_image = torch.from_numpy(image[m][n]).unsqueeze(0).float()
                lidar_ego_image[m] = torch.cat([lidar_ego_image[m][1:10], new_image], dim=0)

        with torch.no_grad():
            lidar_ego_image = lidar_ego_image.unsqueeze(1).to(device)
            vel_ang = torch.from_numpy(vel_ang).float().to(device)
            goal = torch.from_numpy(goal).float().to(device)
            action_output= self.actor(lidar_ego_image, vel_ang, goal).cpu().data.numpy()

        return action_output


    def train(self, all_input):
        self.actor.train()
        self.critic.train()
        self.total_it += 1

        # Sample replay buffer 
        state, image, transformation, num_pre, next_state, next_image, next_transformation, next_num_pre, action, reward, not_done = all_input

        # =====================================>>>  state preperation  <<<=====================================
        final_observation_all = state[:, 0:3630]
        vel_ang = state[:, 3630:3632]
        goal = state[:, 3632:3634]

        final_observation_all = final_observation_all.reshape(-1, 10, 363)
        lidar_10 = final_observation_all[:, :, 0:360]
        x_10 = final_observation_all[:, :, 360]
        y_10 = final_observation_all[:, :, 361]
        yaw_10 = final_observation_all[:, :, 362]

        disentangle_x = np.zeros((state.shape[0], 10,360))
        disentangle_y = np.zeros((state.shape[0], 10,360))

        lidar_ego_image = np.zeros((state.shape[0], 10, 80, 160))

        for m in range(state.shape[0]):
            for i in range(10):
                temp_image_x, temp_image_y = Transformation(lidar_10[m][i], x_10[m][i], y_10[m][i], yaw_10[m][i], x_10[m][-1], y_10[m][-1], yaw_10[m][-1])
                disentangle_x[m][i] = temp_image_x
                disentangle_y[m][i] = temp_image_y
                lidar_ego_image[m, i, disentangle_x[m, i].astype(np.int), disentangle_y[m, i].astype(np.int)] = 1
                lidar_ego_image[m, i, 0, :] = 0
                lidar_ego_image[m, i, :, 0] = 0
                lidar_ego_image[m, i] = cv2.dilate(lidar_ego_image[m, i], np.ones((3,3), np.uint8), iterations=1)
                # lidar_ego_image[m, i] = cv2.GaussianBlur(lidar_ego_image[m, i], (5, 5), 0)
            
        # combine with prediction
        lidar_ego_image = torch.from_numpy(lidar_ego_image).float()
        for m in range(state.shape[0]):
            for n in range(int(num_pre[m])):
                init_delta_x = transformation[m][n][0]
                init_delta_y = transformation[m][n][1]
                init_delta_theta = transformation[m][n][2]

                init_translation_matrix = torch.zeros((10, 2, 3))
                init_translation_matrix[:, 0, 0] = 1
                init_translation_matrix[:, 1, 1] = 1
                init_translation_matrix[:, 0, 2] = init_delta_y
                init_translation_matrix[:, 1, 2] = init_delta_x

                lidar_ego_image[m] = kornia.warp_affine(lidar_ego_image[m].unsqueeze(1),
                                                                init_translation_matrix, dsize=(80, 160),
                                                                flags='bilinear', padding_mode='zeros',
                                                                align_corners=False).squeeze(1)

                init_scale = torch.ones(10)
                init_center = torch.ones(10, 2)
                init_angle = torch.ones(10) * init_delta_theta
                init_center[:, 0] = IMAGE_H
                init_center[:, 1] = IMAGE_H

                init_rotation_matrix = kornia.get_rotation_matrix2d(init_center, init_angle, init_scale)

                lidar_ego_image[m] = kornia.warp_affine(lidar_ego_image[m].unsqueeze(1),
                                                                init_rotation_matrix, dsize=(80, 160),
                                                                flags='bilinear', padding_mode='zeros',
                                                                align_corners=False).squeeze(1)
                
                # new_image = cv2.erode(image[m][n], np.ones((3,3), np.uint8), iterations=1)
                # new_image = cv2.dilate(new_image, np.ones((3,3), np.uint8), iterations=1)
                new_image = torch.from_numpy(image[m][n]).unsqueeze(0).float()
                lidar_ego_image[m] = torch.cat([lidar_ego_image[m][1:10], new_image], dim=0)

        # =====================================>>>  next state preperation  <<<=====================================
        next_final_observation_all = next_state[:, 0:3630]
        next_vel_ang = next_state[:, 3630:3632]
        next_goal = next_state[:, 3632:3634]

        next_final_observation_all = next_final_observation_all.reshape(-1, 10, 363)
        next_lidar_10 = next_final_observation_all[:, :, 0:360]
        next_x_10 = next_final_observation_all[:, :, 360]
        next_y_10 = next_final_observation_all[:, :, 361]
        next_yaw_10 = next_final_observation_all[:, :, 362]

        next_disentangle_x = np.zeros((state.shape[0], 10,360))
        next_disentangle_y = np.zeros((state.shape[0], 10,360))

        next_lidar_ego_image = np.zeros((state.shape[0], 10, 80, 160))


        for m in range(state.shape[0]):
            for i in range(10):
                next_temp_image_x, next_temp_image_y = Transformation(next_lidar_10[m][i], next_x_10[m][i], next_y_10[m][i], next_yaw_10[m][i], next_x_10[m][-1], next_y_10[m][-1], next_yaw_10[m][-1])
                next_disentangle_x[m][i] = next_temp_image_x
                next_disentangle_y[m][i] = next_temp_image_y
                next_lidar_ego_image[m, i, next_disentangle_x[m, i].astype(np.int), next_disentangle_y[m, i].astype(np.int)] = 1
                next_lidar_ego_image[m, i, 0, :] = 0
                next_lidar_ego_image[m, i, :, 0] = 0
                next_lidar_ego_image[m, i] = cv2.dilate(next_lidar_ego_image[m, i], np.ones((3,3), np.uint8), iterations=1)
                # next_lidar_ego_image[m, i] = cv2.GaussianBlur(next_lidar_ego_image[m, i], (5, 5), 0)
        
        # combine with prediction
        next_lidar_ego_image = torch.from_numpy(next_lidar_ego_image).float()
        for m in range(state.shape[0]):
            for n in range(int(next_num_pre[m])):
                init_delta_x = next_transformation[m][n][0]
                init_delta_y = next_transformation[m][n][1]
                init_delta_theta = next_transformation[m][n][2]

                init_translation_matrix = torch.zeros((10, 2, 3))
                init_translation_matrix[:, 0, 0] = 1
                init_translation_matrix[:, 1, 1] = 1
                init_translation_matrix[:, 0, 2] = init_delta_y
                init_translation_matrix[:, 1, 2] = init_delta_x

                next_lidar_ego_image[m] = kornia.warp_affine(next_lidar_ego_image[m].unsqueeze(1),
                                                                init_translation_matrix, dsize=(80, 160),
                                                                flags='bilinear', padding_mode='zeros',
                                                                align_corners=False).squeeze(1)

                init_scale = torch.ones(10)
                init_center = torch.ones(10, 2)
                init_angle = torch.ones(10) * init_delta_theta
                init_center[:, 0] = IMAGE_H
                init_center[:, 1] = IMAGE_H

                init_rotation_matrix = kornia.get_rotation_matrix2d(init_center, init_angle, init_scale)

                next_lidar_ego_image[m] = kornia.warp_affine(next_lidar_ego_image[m].unsqueeze(1),
                                                                init_rotation_matrix, dsize=(80, 160),
                                                                flags='bilinear', padding_mode='zeros',
                                                                align_corners=False).squeeze(1)

                # print(next_transformation[m])
                # print(state[m][-4:])
                # print(next_state[m][-4:])
                # print(next_state - state)

                # cv2.imshow('new_image', next_image[m][n])
                # cv2.imshow('difference', np.clip(next_image[m][n] + next_lidar_ego_image[m][9].numpy() * 0.5, 0, 1))
                # cv2.imshow('difference', next_image[m][n] - next_lidar_ego_image[m][9].numpy())
                # cv2.waitKey(0)

                new_image = torch.from_numpy(next_image[m][n]).unsqueeze(0).float()
                next_lidar_ego_image[m] = torch.cat([next_lidar_ego_image[m][1:10], new_image], dim=0)


        lidar_ego_image = torch.from_numpy(lidar_ego_image.unsqueeze(1).numpy()).float().to(device)
        vel_ang = torch.from_numpy(vel_ang).float().to(device)
        goal = torch.from_numpy(goal).float().to(device)

        next_lidar_ego_image = torch.from_numpy(next_lidar_ego_image.unsqueeze(1).numpy()).float().to(device)
        next_vel_ang = torch.from_numpy(next_vel_ang).float().to(device)
        next_goal = torch.from_numpy(next_goal).float().to(device)

        action = torch.from_numpy(action).float().to(device)
        reward = torch.from_numpy(reward).unsqueeze(1).float().to(device)
        not_done = torch.from_numpy(not_done).unsqueeze(1).float().to(device)

        # print("reward ", reward)


        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                self.actor_target(next_lidar_ego_image, next_vel_ang, next_goal) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_lidar_ego_image, next_vel_ang, next_goal, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(lidar_ego_image, vel_ang, goal, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_loss_viz = critic_loss.item()

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10, norm_type=2)
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor losse
            actor_loss = -self.critic.Q1(lidar_ego_image, vel_ang, goal, self.actor(lidar_ego_image, vel_ang, goal)).mean()
            self.actor_loss_viz = actor_loss.item()
            
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10, norm_type=2)
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return self.actor_loss_viz, self.critic_loss_viz



    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)

