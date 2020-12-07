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
# import kornia

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


IMAGE_W = 160
IMAGE_H = 80
MAX_LASER_RANGE = 3.0


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

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight,  mode='fan_in', nonlinearity='relu')
        nn.init.constant_(m.bias, 0.001)


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

lidar_length = 360
# 10 * 80 * 160  (范围8m)
# 连续输入10帧lidar数据
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        # self.robot_state_emb = nn.Linear(4, 480)
        self.vel_ang_emb = nn.Linear(2, 64)
        self.goal_emb = nn.Linear(2, 64)
        self.ego_motion_emb = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), stride=(3,3)),
            nn.LeakyReLU(),
            # nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,3), stride=(3,3)),
            nn.LeakyReLU(),
            # nn.BatchNorm2d(num_features=64),
            nn.Flatten(),
            nn.Linear(1280, 128)
            )
        # actor layers
        self.lin1 = nn.Linear(256, 128)
        self.lin2 = nn.Linear(128, 2)
        self.max_action = max_action

    def forward(self, lidar_ego_image, vel_ang, goal):
        # observation-robot state embedding
        # robot_state = torch.cat([vel_ang, goal], dim=1)
        # emb1 = F.leaky_relu(self.robot_state_emb(robot_state))
        emb1 = F.leaky_relu(self.vel_ang_emb(vel_ang))
        emb2 = F.leaky_relu(self.goal_emb(goal))
        emb3 = F.leaky_relu(self.ego_motion_emb(lidar_ego_image))
        # print(emb1.shape, emb2.shape)
        emb = torch.cat([emb1, emb2, emb3], dim=1)
        # layer norm
        # out = nn.LayerNorm(emb.size()[1:])(emb)
        out = emb
        out = self.lin1(out)
        out = F.leaky_relu(out)
        out = torch.tanh(self.lin2(out))

        return self.max_action * out


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 architecture
        # self.robot_state_emb_1 = nn.Linear(4, 480)
        self.vel_ang_emb_1 = nn.Linear(2, 64)
        self.goal_emb_1 = nn.Linear(2, 64)
        self.ego_motion_emb_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), stride=(3,3)),
            nn.LeakyReLU(),
            # nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,3), stride=(3,3)),
            nn.LeakyReLU(),
            # nn.BatchNorm2d(num_features=64),
            nn.Flatten(),
            nn.Linear(1280, 128)
            )
        self.lin1_1 = nn.Linear(2, 64)
        self.lin2_1 = nn.Linear(256, 128)
        self.lin3_1 = nn.Linear(192, 128)
        self.lin4_1 = nn.Linear(128, 64)
        self.lin5_1 = nn.Linear(64, 32)
        self.lin6_1 = nn.Linear(32, 1)
        # Q2 architecture
        # self.robot_state_emb_2 = nn.Linear(4, 480)
        self.vel_ang_emb_2 = nn.Linear(2, 64)
        self.goal_emb_2 = nn.Linear(2, 64)
        self.ego_motion_emb_2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), stride=(3,3)),
            nn.LeakyReLU(),
            # nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,3), stride=(3,3)),
            nn.LeakyReLU(),
            # nn.BatchNorm2d(num_features=64),
            nn.Flatten(),
            nn.Linear(1280, 128)
            )
        self.lin1_2 = nn.Linear(2, 64)
        self.lin2_2 = nn.Linear(256, 128)
        self.lin3_2 = nn.Linear(192, 128)
        self.lin4_2 = nn.Linear(128, 64)
        self.lin5_2 = nn.Linear(64, 32)
        self.lin6_2 = nn.Linear(32, 1)

    def forward(self, lidar_ego_image, vel_ang, goal, action):
        # Q1
        # action embedding
        # action_1 = nn.LayerNorm(action.size()[1:])(action)
        action_1 = F.leaky_relu(self.lin1_1(action))
        # observation-robot state embedding
        # robot_state_1 = torch.cat([vel_ang, goal], dim=1)
        # emb1 = F.leaky_relu(self.robot_state_emb_1(robot_state_1))
        emb1_1 = F.leaky_relu(self.vel_ang_emb_1(vel_ang))
        emb2_1 = F.leaky_relu(self.goal_emb_1(goal))
        emb3_1 = F.leaky_relu(self.ego_motion_emb_1(lidar_ego_image))
        robot_state_1 = torch.cat([emb1_1, emb2_1, emb3_1], dim=1)
        # robot_state_1 = nn.LayerNorm(robot_state_1.size()[1:])(robot_state_1)
        robot_state_1 = F.leaky_relu(self.lin2_1(robot_state_1))
        # merge
        all_1 = torch.cat([action_1, robot_state_1], dim=1)
        # q1 = nn.LayerNorm(q1.size()[1:])(q1)
        q1 = F.leaky_relu(self.lin3_1(all_1))
        q1 = F.leaky_relu(self.lin4_1(q1))
        q1 = F.leaky_relu(self.lin5_1(q1))
        q1 = self.lin6_1(q1)

        # Q2 
        # action embedding
        # action_2 = nn.LayerNorm(action.size()[1:])(action)
        action_2 = F.leaky_relu(self.lin1_2(action))
        # observation-robot state embedding
        # robot_state_2 = torch.cat([vel_ang, goal], dim=1)
        # emb1_2 = F.leaky_relu(self.robot_state_emb_2(robot_state_2))
        emb1_2 = F.leaky_relu(self.vel_ang_emb_2(vel_ang))
        emb2_2 = F.leaky_relu(self.goal_emb_2(goal))
        emb3_2 = F.leaky_relu(self.ego_motion_emb_2(lidar_ego_image))
        robot_state_2 = torch.cat([emb1_2, emb2_2, emb3_2], dim=1)
        # robot_state_2 = nn.LayerNorm(robot_state_2.size()[1:])(robot_state_2)
        robot_state_2 = F.leaky_relu(self.lin2_2(robot_state_2))
        # merge
        all_2 = torch.cat([action_2, robot_state_2], dim=1)
        # q2 = nn.LayerNorm(q2.size()[1:])(q2)
        q2 = F.leaky_relu(self.lin3_2(all_2))
        q2 = F.leaky_relu(self.lin4_2(q2))
        q2 = F.leaky_relu(self.lin5_2(q2))
        q2 = self.lin6_2(q2)
        return q1, q2

    def Q1(self, lidar_ego_image, vel_ang, goal, action):
        # Q1
        # action embedding
        # action_1 = nn.LayerNorm(action.size()[1:])(action)
        action_1 = F.leaky_relu(self.lin1_1(action))
        # observation-robot state embedding
        # robot_state_1 = torch.cat([vel_ang, goal], dim=1)
        # emb1 = F.leaky_relu(self.robot_state_emb_1(robot_state_1))
        emb1_1 = F.leaky_relu(self.vel_ang_emb_1(vel_ang))
        emb2_1 = F.leaky_relu(self.goal_emb_1(goal))
        emb3_1 = F.leaky_relu(self.ego_motion_emb_1(lidar_ego_image))
        robot_state_1 = torch.cat([emb1_1, emb2_1, emb3_1], dim=1)
        # robot_state_1 = nn.LayerNorm(robot_state_1.size()[1:])(robot_state_1)
        robot_state_1 = F.leaky_relu(self.lin2_1(robot_state_1))
        # merge
        all_1 = torch.cat([action_1, robot_state_1], dim=1)
        # q1 = nn.LayerNorm(q1.size()[1:])(q1)
        q1 = F.leaky_relu(self.lin3_1(all_1))
        q1 = F.leaky_relu(self.lin4_1(q1))
        q1 = F.leaky_relu(self.lin5_1(q1))
        q1 = self.lin6_1(q1)

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

    def select_action(self, state):

        # state preperation
        final_observation_all = state[0:3630]
        vel_ang = state[3630:3632]
        goal = state[3632:3634]

        final_observation_all = final_observation_all.reshape(-1, 10, 363)
        lidar_10 = final_observation_all[0, :, 0:360]
        x_10 = final_observation_all[0, :, 360]
        y_10 = final_observation_all[0, :, 361]
        yaw_10 = final_observation_all[0, :, 362]

        lidar_ego_image = np.ones((10, 360))

        for i in range(10):
            temp_lidar = Transformation_List(lidar_10[i], x_10[i], y_10[i], yaw_10[i], x_10[-1], y_10[-1], yaw_10[-1])
            lidar_ego_image[i] = temp_lidar[:]

        # combine with prediction
        lidar_ego_image = torch.from_numpy(lidar_ego_image).float()

        lidar_ego_image = lidar_ego_image.unsqueeze(0).unsqueeze(1).to(device)
        vel_ang = torch.from_numpy(vel_ang).unsqueeze(0).float().to(device)
        goal = torch.from_numpy(goal).unsqueeze(0).float().to(device)

        self.actor.eval()
        
        return self.actor(lidar_ego_image, vel_ang, goal).cpu().data.numpy().flatten()



    def train(self, replay_buffer, batch_size=100):
        t1_all = time.time()
        self.actor.train()
        self.critic.train()
        self.total_it += 1

        # Sample replay buffer 
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # =====================================>>>  state preperation  <<<=====================================
        final_observation_all = state[:, 0:3630]
        vel_ang = state[:, 3630:3632]
        goal = state[:, 3632:3634]

        final_observation_all = final_observation_all.reshape(-1, 10, 363)
        lidar_10 = final_observation_all[:, :, 0:360]
        x_10 = final_observation_all[:, :, 360]
        y_10 = final_observation_all[:, :, 361]
        yaw_10 = final_observation_all[:, :, 362]

        lidar_ego_image = np.ones((state.shape[0], 10, 360))

        for m in range(state.shape[0]):
            for i in range(10):
                temp_lidar = Transformation_List(lidar_10[m][i], x_10[m][i], y_10[m][i], yaw_10[m][i], x_10[m][-1], y_10[m][-1], yaw_10[m][-1])
                lidar_ego_image[m][i] = temp_lidar[:]

        # cv2.imshow('lidar_ego_image', lidar_ego_image[0])
        # cv2.waitKey(10)
            
        lidar_ego_image = torch.from_numpy(lidar_ego_image).float()

        # =====================================>>>  next state preperation  <<<=====================================
        next_final_observation_all = next_state[:, 0:3630]
        next_vel_ang = next_state[:, 3630:3632]
        next_goal = next_state[:, 3632:3634]

        next_final_observation_all = next_final_observation_all.reshape(-1, 10, 363)
        next_lidar_10 = next_final_observation_all[:, :, 0:360]
        next_x_10 = next_final_observation_all[:, :, 360]
        next_y_10 = next_final_observation_all[:, :, 361]
        next_yaw_10 = next_final_observation_all[:, :, 362]

        next_lidar_ego_image = np.ones((state.shape[0], 10, 360))

        for m in range(state.shape[0]):
            for i in range(10):
                next_temp_lidar = Transformation_List(next_lidar_10[m][i], next_x_10[m][i], next_y_10[m][i], next_yaw_10[m][i], next_x_10[m][-1], next_y_10[m][-1], next_yaw_10[m][-1])
                next_lidar_ego_image[m][i] = next_temp_lidar[:]
        
        next_lidar_ego_image = torch.from_numpy(next_lidar_ego_image).float()


        lidar_ego_image = torch.from_numpy(lidar_ego_image.unsqueeze(1).numpy()).float().to(device)
        vel_ang = torch.from_numpy(vel_ang).float().to(device)
        goal = torch.from_numpy(goal).float().to(device)
        # print('shape', lidar_ego_image.shape, vel_ang.shape, goal.shape)

        next_lidar_ego_image = torch.from_numpy(next_lidar_ego_image.unsqueeze(1).numpy()).float().to(device)
        next_vel_ang = torch.from_numpy(next_vel_ang).float().to(device)
        next_goal = torch.from_numpy(next_goal).float().to(device)
        # print('next shape', next_lidar_ego_image.shape, next_vel_ang.shape, next_goal.shape)

        action = torch.from_numpy(action).float().to(device)
        reward = torch.from_numpy(reward).float().to(device)
        not_done = torch.from_numpy(not_done).float().to(device)



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

