# -*- coding: utf-8 -*- 
import copy
import numpy as np
import torch
import math
import time
import torch.nn as nn
import torch.nn.functional as F
# import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from torchsummary import summary
from PIL import Image
import cv2
# import kornia

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


MAX_LASER_RANGE = 3.0


def weights_init(m):
    if isinstance(m, nn.Conv1d):
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
        nn.init.kaiming_uniform_(m.weight,  mode='fan_in', nonlinearity='relu')
        nn.init.constant_(m.bias, 0.001)

        
lidar_length = 360
# 10 * 80 * 160  (范围8m)
# 连续输入10帧lidar数据
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=3, stride=5)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=5)
        self.lin1 = nn.Linear(448, 128)
        self.lin2 = nn.Linear(256, 128)
        self.lin3 = nn.Linear(128, 2)
        self.vel_ang_emb = nn.Linear(2, 64)
        self.goal_emb = nn.Linear(2, 64)
        
        self.max_action = max_action

    def forward(self, lidar_ego_image, vel_ang, goal):
        lidar = F.relu(self.conv1(lidar_ego_image))
        lidar = F.relu(self.conv2(lidar))
        lidar = torch.flatten(lidar, start_dim=1)
        # print(lidar.shape)
        lidar = F.relu(self.lin1(lidar))
        # robot_state = torch.cat([vel_ang, goal], dim=1)
        # robot_state = F.relu(self.robot_state_emb(robot_state))
        emb1 = F.relu(self.vel_ang_emb(vel_ang))
        emb2 = F.relu(self.goal_emb(goal))
        out = torch.cat([lidar, emb1, emb2], dim=1)
        out = F.relu(self.lin2(out))
        out = torch.tanh(self.lin3(out))

        return self.max_action * out 


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 architecture
        self.conv1_1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=3, stride=5)
        self.conv2_1 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=5)
        self.lin1_1 = nn.Linear(448, 128)
        self.lin2_1 = nn.Linear(320, 128)
        self.lin3_1 = nn.Linear(128, 1)
        self.vel_ang_emb_1 = nn.Linear(2, 64)
        self.goal_emb_1 = nn.Linear(2, 64)
        self.action_emb_1 = nn.Linear(2, 64)

        # Q2 architecture
        self.conv1_2 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=3, stride=5)
        self.conv2_2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=5)
        self.lin1_2 = nn.Linear(448, 128)
        self.lin2_2 = nn.Linear(320, 128)
        self.lin3_2 = nn.Linear(128, 1)
        self.vel_ang_emb_2 = nn.Linear(2, 64)
        self.goal_emb_2 = nn.Linear(2, 64)
        self.action_emb_2 = nn.Linear(2, 64)

    def forward(self, lidar_ego_image, vel_ang, goal, action):
        # Q1 
        lidar_1 = F.relu(self.conv1_1(lidar_ego_image))
        lidar_1 = F.relu(self.conv2_1(lidar_1))
        lidar_1 = torch.flatten(lidar_1, start_dim=1)
        # print(lidar_1.shape)
        lidar_1 = F.relu(self.lin1_1(lidar_1))
        # robot_state_1 = torch.cat([vel_ang, goal], dim=1)
        # robot_state_1 = F.relu(self.robot_state_emb_1(robot_state_1))
        emb1_1 = F.relu(self.vel_ang_emb_1(vel_ang))
        emb2_1 = F.relu(self.goal_emb_1(goal))
        action_state_1 = F.relu(self.action_emb_1(action))
        q1 = torch.cat([lidar_1, emb1_1, emb2_1, action_state_1], dim=1)
        q1 = F.relu(self.lin2_1(q1))
        q1 = self.lin3_1(q1)

        # Q2 
        lidar_2 = F.relu(self.conv1_2(lidar_ego_image))
        lidar_2 = F.relu(self.conv2_2(lidar_2))
        lidar_2 = torch.flatten(lidar_2, start_dim=1)
        # print(lidar_2.shape)
        lidar_2 = F.relu(self.lin1_2(lidar_2))
        # robot_state_2 = torch.cat([vel_ang, goal], dim=1)
        # robot_state_2 = F.relu(self.robot_state_emb_2(robot_state_2))
        emb1_2 = F.relu(self.vel_ang_emb_2(vel_ang))
        emb2_2 = F.relu(self.goal_emb_2(goal))
        action_state_2 = F.relu(self.action_emb_2(action))
        q2 = torch.cat([lidar_2, emb1_2, emb2_2, action_state_2], dim=1)
        q2 = F.relu(self.lin2_2(q2))
        q2 = self.lin3_2(q2)
        return q1, q2

    def Q1(self, lidar_ego_image, vel_ang, goal, action):
        # Q1 
        lidar_1 = F.relu(self.conv1_1(lidar_ego_image))
        lidar_1 = F.relu(self.conv2_1(lidar_1))
        lidar_1 = torch.flatten(lidar_1, start_dim=1)
        # print(lidar_1.shape)
        lidar_1 = F.relu(self.lin1_1(lidar_1))
        # robot_state_1 = torch.cat([vel_ang, goal], dim=1)
        # robot_state_1 = F.relu(self.robot_state_emb_1(robot_state_1))
        emb1_1 = F.relu(self.vel_ang_emb_1(vel_ang))
        emb2_1 = F.relu(self.goal_emb_1(goal))
        action_state_1 = F.relu(self.action_emb_1(action))
        q1 = torch.cat([lidar_1, emb1_1, emb2_1, action_state_1], dim=1)
        q1 = F.relu(self.lin2_1(q1))
        q1 = self.lin3_1(q1)

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

        lidar_ego_image = lidar_10[-3:, :] / MAX_LASER_RANGE
        # print('lidar_ego_image', lidar_ego_image.shape)
        # print(lidar_ego_image)

        # combine with prediction
        lidar_ego_image = torch.from_numpy(lidar_ego_image).float()

        lidar_ego_image = lidar_ego_image.unsqueeze(0).to(device)
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

        lidar_ego_image = lidar_10[:, -3:, :] / MAX_LASER_RANGE
        print('lidar_ego_image', lidar_ego_image.shape)
        # print(lidar_ego_image)

        # =====================================>>>  next state preperation  <<<=====================================
        next_final_observation_all = next_state[:, 0:3630]
        next_vel_ang = next_state[:, 3630:3632]
        next_goal = next_state[:, 3632:3634]

        next_final_observation_all = next_final_observation_all.reshape(-1, 10, 363)
        next_lidar_10 = next_final_observation_all[:, :, 0:360]

        next_lidar_ego_image = next_lidar_10[:, -3:, :] / MAX_LASER_RANGE
        print('next_lidar_ego_image', next_lidar_ego_image.shape)


        lidar_ego_image = torch.from_numpy(lidar_ego_image).float().to(device)
        vel_ang = torch.from_numpy(vel_ang).float().to(device)
        goal = torch.from_numpy(goal).float().to(device)

        next_lidar_ego_image = torch.from_numpy(next_lidar_ego_image).float().to(device)
        next_vel_ang = torch.from_numpy(next_vel_ang).float().to(device)
        next_goal = torch.from_numpy(next_goal).float().to(device)

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

