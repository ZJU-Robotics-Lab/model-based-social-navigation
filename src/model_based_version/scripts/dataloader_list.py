# -*- coding: utf-8 -*- 
import torch
import os, glob
import random, csv
# import visdom
import time
import cv2
import math


import numpy as np

from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms as transforms

root = '/home/cyx/model_based_social_navigation/src/model_based_version/dataset'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


class Transition_Model_Dataset(Dataset):
    def __init__(self, root, mode):
        super(Transition_Model_Dataset, self).__init__()

        self.root = root
        
        self.indexes= self.load_csv('state_data.csv')

        max_length = len(self.indexes)
        if mode == 'training':
            self.indexes = self.indexes[:int(0.7 * max_length)]

        elif mode == 'validation':
            self.indexes= self.indexes[int(0.7 * max_length):int(0.8 * max_length)]

        else:
            self.indexes = self.indexes[int(0.8 * max_length):]



    def load_csv(self, filename):
        # 创建或者读取csvstart_timesteps
        # print(self.root + '/lidar')
        if not os.path.exists(os.path.join(self.root, filename)):
            indexes = os.listdir(self.root + '/state')  
            random.shuffle(indexes)
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)

                max_all = -9
                min_all = 9

                max_reward = -9
                min_reward = 9

                print('length dataset: ', len(indexes))
                for index in indexes:
                    # 找到当前图片的timestamp
                    num = index.split(os.sep)[-1]
                    num = index.split('.')[-2]
                    index = num + '.npy'

                    a = np.load(self.root + '/state/' + index)
                    temp_max = np.max(a[3630:3634])
                    temp_min = np.min(a[3630:3634])

                    if temp_max > max_all:
                        max_all = temp_max
                    
                    if temp_min < min_all:
                        min_all = temp_min

                    reward = np.load(self.root + '/reward/' + index)

                    if reward > max_reward:
                        max_reward = reward
                    
                    if reward < min_reward:
                        min_reward = reward

                    writer.writerow([index])
                print('writen into csv file: ', filename)
                print("max_value : ", max_all)
                print("min_value : ", min_all)
                print("max_reward : ", max_reward)
                print("min_reward : ", min_reward)


        # read from csv file
        indexes = []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                lidar = row
                indexes.append(lidar)

        return indexes

    def __len__(self):
        return len(self.indexes)



    def __getitem__(self, index):
        """[获取训练数据]

        Args:
            index ([type]): [编号]

        Returns:
            return [type]: [state, action, next_state, reward]

            net input:
                state
                action

            net output label:
                next_state
                reward

        """

        index_item = self.indexes[index]
        current_state = np.load(self.root + '/state/' + index_item[0])
        current_state = torch.tensor(current_state, dtype=torch.float32)

        next_state = np.load(self.root + '/next_state/' + index_item[0])
        next_state = torch.tensor(next_state, dtype=torch.float32)
        
        action = np.load(self.root + '/action/' + index_item[0]) # -1 ~ 1
        action = torch.tensor(action, dtype=torch.float32)
        # action = (action + 1.) / 2 # -1 ~ 1 => 0 ~ 1

        reward = np.load(self.root + '/reward/' + index_item[0]) # -1 ~ 1
        reward = torch.tensor(reward, dtype=torch.float32)
        # reward = (reward + 1.) / 2 # -1 ~ 1 => 0 ~ 1

        # print("==========")
        # print("==========")
        # print((action[0] + 1.) / 2)
        # print((action[0] + 1.) / 2 - current_state[3630] * 1.2)
        # print(next_state[3630] * 1.2 - current_state[3630] * 1.2)
        # print(reward)

        return current_state, action, next_state, reward

if __name__ == "__main__":
    
    train_db = Transition_Model_Dataset(root, mode='training')
    print(train_db.root)



