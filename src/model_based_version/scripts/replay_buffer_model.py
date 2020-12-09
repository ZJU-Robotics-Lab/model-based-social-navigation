# -*- coding: utf-8 -*- 
import random
import numpy as np
import torch
from operator import itemgetter

class ReplayBuffer_Model:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    
    # 加入单帧数据
    def push(self, state, image, transformation, num_pre, next_state, next_image, next_transformation, next_num_pre, action, reward, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, image, transformation, num_pre, next_state, next_image, next_transformation, next_num_pre, action, reward, done)
        self.position = int((self.position + 1) % self.capacity)


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
        state, image, transformation, num_pre, next_state, next_image, next_transformation, next_num_pre, action, reward, done = map(np.stack, zip(*batch))

        return state, image, transformation, num_pre, next_state, next_image, next_transformation, next_num_pre, action, reward, done

    
    # 采样一批数据组成新的数据集，可以重复
    def sample_all_batch(self, batch_size):
        
        idxes = np.random.randint(0, len(self.buffer), int(batch_size))
        batch = list(itemgetter(*idxes)(self.buffer))
        state, image, transformation, num_pre, next_state, next_image, next_transformation, next_num_pre, action, reward, done = map(np.stack, zip(*batch))

        return state, image, transformation, num_pre, next_state, next_image, next_transformation, next_num_pre, action, reward, done

    def return_all(self):
        return self.buffer

    def __len__(self):
        return len(self.buffer)
