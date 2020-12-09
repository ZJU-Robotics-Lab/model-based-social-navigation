#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

# Env 包装接口，返回 cur_state, action, next_state, reward, terminal, info，等同于env.step


import gym
import numpy as np
import random

class EnvSampler():
    def __init__(self, env, max_path_length=400, start_timesteps=5000):
        self.env = env

        self.path_length = 0

        self.current_state_1 = None
        self.current_state_2 = None
        self.current_state_3 = None
        self.current_state_4 = None

        self.total_step = 0
        self.episode_num = 0
        self.sum_reward = 0
        
        self.noise_decay_episode = 100
        self.net_action_noise = 0.05 # smaller in mbpo

        self.max_path_length = max_path_length
        self.start_timesteps = start_timesteps

    def sample(self, policy, flag_test=0):

        max_action = 1.0
        action_1 = [0.0, 0.0]
        action_2 = [0.0, 0.0]
        action_3 = [0.0, 0.0]
        action_4 = [0.0, 0.0]

        temp_image = np.zeros((1, 2, 80, 160))
        temp_transformation = np.zeros((1, 2, 3))
        temp_num_pre = np.zeros((1, 1), dtype=np.int)

        if self.current_state_1 is None:
            self.current_state_1, self.current_state_2, self.current_state_3, self.current_state_4 = self.env.reset()

        cur_state_1 = self.current_state_1
        cur_state_2 = self.current_state_2
        cur_state_3 = self.current_state_3
        cur_state_4 = self.current_state_4

        # agent 1
        if self.total_step < self.start_timesteps:
            action_1[0] = random.uniform(-max_action, max_action)
            action_1[1] = random.uniform(-max_action, max_action)
        else:
            action_1 = policy.select_action(np.array(self.current_state_1), temp_image, temp_transformation, temp_num_pre)
            print(action_1)
            action_1[0] = (
                action_1[0]
                + np.random.normal(0, max_action * self.net_action_noise, size=1)
            ).clip(-max_action, max_action)
            action_1[1] = (
                action_1[1]
                + np.random.normal(0, max_action * self.net_action_noise, size=1)
            ).clip(-max_action, max_action)


        # agent 2
        if self.total_step < self.start_timesteps:
            action_2[0] = random.uniform(-max_action, max_action)
            action_2[1] = random.uniform(-max_action, max_action)
        else:
            action_2 = policy.select_action(np.array(self.current_state_2), temp_image, temp_transformation, temp_num_pre)
            
            action_2[0] = (
                action_2[0]
                + np.random.normal(0, max_action * self.net_action_noise, size=1)
            ).clip(-max_action, max_action)
            action_2[1] = (
                action_2[1]
                + np.random.normal(0, max_action * self.net_action_noise, size=1)
            ).clip(-max_action, max_action)


        # agent 3
        if self.total_step < self.start_timesteps:
            action_3[0] = random.uniform(-max_action, max_action)
            action_3[1] = random.uniform(-max_action, max_action)
        else:
            action_3 = policy.select_action(np.array(self.current_state_3), temp_image, temp_transformation, temp_num_pre)
            
            action_3[0] = (
                action_3[0]
                + np.random.normal(0, max_action * self.net_action_noise, size=1)
            ).clip(-max_action, max_action)
            action_3[1] = (
                action_3[1]
                + np.random.normal(0, max_action * self.net_action_noise, size=1)
            ).clip(-max_action, max_action)


        # agent 4
        if self.total_step < self.start_timesteps:
            action_4[0] = random.uniform(-max_action, max_action)
            action_4[1] = random.uniform(-max_action, max_action)
        else:
            action_4 = policy.select_action(np.array(self.current_state_4), temp_image, temp_transformation, temp_num_pre)
            
            action_4[0] = (
                action_4[0]
                + np.random.normal(0, max_action * self.net_action_noise, size=1)
            ).clip(-max_action, max_action)
            action_4[1] = (
                action_4[1]
                + np.random.normal(0, max_action * self.net_action_noise, size=1)
            ).clip(-max_action, max_action)
            

        next_state_1, reward_1, terminal_1, next_state_2, reward_2, terminal_2, \
            next_state_3, reward_3, terminal_3, next_state_4, reward_4, terminal_4 \
                = self.env.step(action_1, action_2, action_3, action_4)

        self.total_step += 1
        self.path_length += 1
        self.sum_reward += reward_1

        if self.path_length >= self.max_path_length:
            terminal_1 = True
            terminal_2 = True
            terminal_3 = True
            terminal_4 = True

        if terminal_1 or self.path_length >= self.max_path_length:
            self.current_state_1 = None
            self.episode_num += 1
            self.path_length = 0
            self.sum_reward = 0
            # noise 衰减
            if self.episode_num > self.noise_decay_episode:
                self.net_action_noise = self.net_action_noise * 0.99
        else:
            self.current_state_1 = next_state_1
        
        self.current_state_2 = next_state_2
        self.current_state_3 = next_state_3
        self.current_state_4 = next_state_4

        # print("================")
        # print(terminal)
        return cur_state_1, next_state_1, action_1, reward_1, terminal_1, cur_state_2, next_state_2, action_2, reward_2, terminal_2, \
            cur_state_3, next_state_3, action_3, reward_3, terminal_3, cur_state_4, next_state_4, action_4, reward_4, terminal_4
