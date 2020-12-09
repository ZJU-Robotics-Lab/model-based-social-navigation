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
        self.current_state = None
        self.max_path_length = max_path_length
        self.sum_reward = 0

        self.total_step = 0
        self.episode_num = 0
        self.noise_decay_episode = 30
        self.net_action_noise = 0.1
        self.start_timesteps = start_timesteps

    def sample(self, agent, flag_test=0):

        max_action = 1.0
        action = [0.0, 0.0]

        temp_image = np.zeros((1, 2, 80, 160))
        temp_transformation = np.zeros((1, 2, 3)) 
        temp_num_pre = np.zeros((1, 1), dtype=np.int)

        if self.current_state is None:
            self.current_state = self.env.reset()

        cur_state = self.current_state

        if self.total_step < self.start_timesteps:
            action[0] = random.uniform(-max_action, max_action)
            action[1] = random.uniform(-max_action, max_action)
        else:
            action = agent.select_action(np.array(self.current_state), temp_image, temp_transformation, temp_num_pre)
            print("action : ", action)
            
            action[0] = (
                action[0]
                + np.random.normal(0, max_action * self.net_action_noise, size=1)
            ).clip(-max_action, max_action)
            action[1] = (
                action[1]
                + np.random.normal(0, max_action * self.net_action_noise, size=1)
            ).clip(-max_action, max_action)
        
        
        next_state, reward, terminal = self.env.step(action)

        self.total_step += 1
        self.path_length += 1
        self.sum_reward += reward

        if self.path_length >= self.max_path_length:
            terminal = True

        if terminal or self.path_length >= self.max_path_length:
            self.current_state = None
            self.path_length = 0
            self.sum_reward = 0
            self.episode_num += 1
            # noise 衰减
            if self.episode_num > self.noise_decay_episode:
                self.net_action_noise = self.net_action_noise * 0.99
        else:
            self.current_state = next_state

        return cur_state, next_state, action, reward, terminal
