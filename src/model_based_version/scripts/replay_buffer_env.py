# -*- coding: utf-8 -*- 
import random
import numpy as np
import torch
from operator import itemgetter

class ReplayBuffer_Env:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.save_time = 0
        
    
    # 加入单帧数据
    def push(self, state, next_state, action, reward, done, flag_save=False):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, next_state, action, reward, done)
        self.position = int((self.position + 1) % self.capacity)
        # print("position              ", self.position)
        if flag_save == True:
            temp_state = state[:]
            temp_state_array = np.asarray(temp_state)
            np.save('/home/cyx/my_gym_ros_rl/src/model_based_policy_gradient/data_pre/state/'  + str(self.save_time).zfill(7), temp_state_array)
            # action
            temp_action = action[:]
            temp_action_array = np.asarray(temp_action)
            np.save('/home/cyx/my_gym_ros_rl/src/model_based_policy_gradient/data_pre/action/' + str(self.save_time).zfill(7), temp_action_array)


            # next state
            temp_next_state = next_state[:]
            temp_next_state_array = np.asarray(temp_next_state)
            np.save('/home/cyx/my_gym_ros_rl/src/model_based_policy_gradient/data_pre/next_state/'  + str(self.save_time).zfill(7), temp_next_state_array)


            # reward
            temp_reward = reward
            temp_reward_array = np.asarray(temp_reward)
            np.save('/home/cyx/my_gym_ros_rl/src/model_based_policy_gradient/data_pre/reward/'  + str(self.save_time).zfill(7), temp_reward_array)

            # not_done
            temp_done = done
            temp_done_array = np.asarray(temp_done)
            np.save('/home/cyx/my_gym_ros_rl/src/model_based_policy_gradient/data_pre/done/'  + str(self.save_time).zfill(7), temp_done_array)

            self.save_time += 1

    # 整批加入
    def push_batch(self, batch):
        if len(self.buffer) < self.capacity:
            append_len = min(self.capacity - len(self.buffer), len(batch))
            self.buffer.extend([None] * append_len)

        if self.position + len(batch) < self.capacity:
            self.buffer[self.position : self.position + len(batch)] = batch
            self.position += len(batch)
        else:
            self.buffer[self.position : len(self.buffer)] = batch[:len(self.buffer) - self.position]
            self.buffer[:len(batch) - len(self.buffer) + self.position] = batch[len(self.buffer) - self.position:]
            self.position = len(batch) - len(self.buffer) + self.position

    # 采样一个小batch作为一次学习的输入，不重复
    def sample(self, batch_size):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        batch = random.sample(self.buffer, int(batch_size))
        
        state, next_state, action, reward, done = map(np.stack, zip(*batch))
        return state, next_state, action, reward, done

    
    # 采样一批数据组成新的数据集，可以重复
    def sample_all_batch(self, batch_size):
        
        idxes = np.random.randint(0, len(self.buffer), int(batch_size))
        batch = list(itemgetter(*idxes)(self.buffer))
        state, next_state, action, reward, done = map(np.stack, zip(*batch))

        return state, next_state, action, reward, done

    def return_all(self):
        return self.buffer

    def __len__(self):
        return len(self.buffer)
