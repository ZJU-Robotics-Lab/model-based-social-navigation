# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import math
import time
import numpy as np
import cv2

import gzip
from transition_model import Transition_Model
from dataloader import Transition_Model_Dataset
from torch.utils.data import Dataset, DataLoader

from torch.utils.tensorboard import SummaryWriter

# torch.cuda.set_device(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')

IMAGE_W = 160
IMAGE_H = 80
MAX_LASER_RANGE = 3.0


# numpy
def Transformation(lidar_input, current_x, current_y, current_yaw, target_x, target_y, target_yaw, is_numpy=True):
    if is_numpy:
        lidar_current = lidar_input
        current_x = current_x
        current_y = current_y
        current_yaw = current_yaw
        target_x = target_x
        target_y = target_y
        target_yaw = target_yaw
    else:
        lidar_current = lidar_input.cpu().numpy()
        current_x = current_x.cpu().numpy()
        current_y = current_y.cpu().numpy()
        current_yaw = current_yaw.cpu().numpy()
        target_x = target_x.cpu().numpy()
        target_y = target_y.cpu().numpy()
        target_yaw = target_yaw.cpu().numpy()

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

    image_x = np.floor(IMAGE_H - 1.0 - x_target / MAX_LASER_RANGE * IMAGE_H).astype(np.int)
    image_y = np.floor(IMAGE_H - 1.0 - y_target / MAX_LASER_RANGE * IMAGE_H).astype(np.int)

    image_x[(image_x < 0) | (image_x > (IMAGE_H - 1)) | (lidar_current > (MAX_LASER_RANGE - 0.2))] = 0
    image_y[(image_y < 0) | (image_y > (IMAGE_W - 1)) | (lidar_current > (MAX_LASER_RANGE - 0.2))] = 0

    return image_x, image_y


class Ensemble_Model():

    def __init__(self, network_size, elite_size, state_size=3634, action_size=2, reward_size=1):
        self.network_size = network_size
        self.elite_size = elite_size

        self.state_size = state_size
        self.state_size_unit = 4  # vel & ang & goal
        self.action_size = action_size
        self.reward_size = reward_size

        self.elite_model_idxes = []

        self.model_list = []
        for i in range(network_size):
            self.model_list.append(Transition_Model(flag_only_image=False))


    def train(self, inputs, actions, state_labels, reward_labels, batch_size=16):

        losses_G = []

        correct_vel = 0
        correct_ang = 0
        correct_goal = 0
        correct_goal_distance = 0
        correct_reward = 0
        batch_num = 0

        for start_pos in range(0, inputs.shape[0], batch_size):

            batch_num += 1
            # print("==============>  Train  <==============")

            state_input = inputs[start_pos: start_pos + batch_size]
            state_label = state_labels[start_pos: start_pos + batch_size]
            action = actions[start_pos: start_pos + batch_size]
            reward_label = reward_labels[start_pos: start_pos + batch_size]
            images_pre = np.zeros((batch_size, 2, 80, 160))
            transformations_pre = np.zeros((batch_size, 2, 3))
            num_pre = np.zeros((batch_size, 1))

            losses_indexes = 0
            for model in self.model_list:
                model.generator.train()

                out_before, out, transformation_out, vel_ang_out, goal_out, reward = \
                    model.generator(state_input, images_pre, transformations_pre, num_pre, action)

                temp_out_before = out_before[0].detach().cpu().numpy()
                temp_out_img = out[0].detach().cpu().numpy()

                # =========================  vel_ang & goal & reward  =========================
                temp_vel_ang = vel_ang_out.detach().cpu().numpy()
                temp_goal = goal_out.detach().cpu().numpy()
                temp_reward = reward.detach().cpu().numpy()

                # 只计算第一个模型的准确率观察训练状况
                if losses_indexes == 0:
                    for i in range(state_input.shape[0]):
                        # reward
                        if abs(temp_reward[i] - reward_label[i]) < 0.1:
                            correct_reward += 1
                        # vel
                        if abs(temp_vel_ang[i][0] - state_label[i][-4]) < 0.05:
                            correct_vel += 1
                        # ang
                        if abs(temp_vel_ang[i][1] - state_label[i][-3]) < 0.05:
                            correct_ang += 1
                        # goal angle
                        goal_angle = (state_label[i][-2] * 2. - 1.) * math.pi
                        goal_angle_prediction = (temp_goal[i][0] * 2. - 1.) * math.pi

                        angle_goal = math.degrees(goal_angle)
                        angle_goal_prediction = math.degrees(goal_angle_prediction)
                        delta_angle = angle_goal_prediction - angle_goal

                        if abs(delta_angle) > 180:
                            delta_angle = abs(delta_angle) - 180

                        if abs(delta_angle) < 3:
                            correct_goal += 1

                        # goal distance
                        if abs(temp_goal[i][1] - state_label[i][-1]) < 0.03:
                            correct_goal_distance += 1

                # =========================          costmap          =========================
                # only label in new reference
                index_show = 0
                final_observation = state_label[index_show, 0:3630]
                final_observation = final_observation.reshape(10, 363)
                lidar_10 = final_observation[:, 0:360]
                x_10 = final_observation[:, 360]
                y_10 = final_observation[:, 361]
                yaw_10 = final_observation[:, 362]

                disentangle_x = np.zeros((10, 360))
                disentangle_y = np.zeros((10, 360))
                for i in range(10):
                    temp_image_x, temp_image_y = Transformation(lidar_10[i], x_10[i], y_10[i], yaw_10[i], x_10[-1], y_10[-1], yaw_10[-1], is_numpy=True)
                    disentangle_x[i] = temp_image_x
                    disentangle_y[i] = temp_image_y

                lidar_ego_image = np.zeros((80, 160))
                for i in range(10):
                    if i == 9:
                        lidar_ego_image[disentangle_x[i].astype(np.int), disentangle_y[i].astype(np.int)] = 1
                    else:
                        lidar_ego_image[disentangle_x[i].astype(np.int), disentangle_y[i].astype(np.int)] = 0.5
                    lidar_ego_image[:, 0] = 0
                    lidar_ego_image[0, :] = 0

                lidar_ego_image = cv2.dilate(lidar_ego_image, np.ones((3, 3), np.uint8), iterations=1)
                temp_out_img_label = np.expand_dims(lidar_ego_image, axis=0)

                # content with prediction
                content_with_prediction = np.zeros((80, 160))
                content_with_prediction[disentangle_x[-1].astype(np.int), disentangle_y[-1].astype(np.int)] = 0.5

                content_with_prediction = cv2.dilate(content_with_prediction, np.ones((3, 3), np.uint8), iterations=1)
                temp_content_with_prediction = np.clip((np.expand_dims(content_with_prediction, axis=0) + temp_out_before), 0, 1)

                # label with prediction
                lidar_ego_image_with_prediction = np.zeros((80, 160))
                lidar_ego_image_with_prediction[disentangle_x[-1].astype(np.int), disentangle_y[-1].astype(np.int)] = 0.5

                lidar_ego_image_with_prediction = cv2.dilate(lidar_ego_image_with_prediction, np.ones((3, 3), np.uint8), iterations=1)
                temp_label_with_prediction = np.clip((np.expand_dims(lidar_ego_image_with_prediction, axis=0) + temp_out_img), 0, 1)

                loss_G = model.train_all(out, vel_ang_out, goal_out, reward, state_label, reward_label)
                
                if start_pos == 0:
                    losses_G.append(loss_G.item())
                else:
                    losses_G[losses_indexes] += loss_G.item()

                losses_indexes += 1

        sorted_loss_idx = np.argsort(losses_G)

        # 按照step中累计loss的排序，找出前几名
        self.elite_model_idxes = sorted_loss_idx[:self.elite_size].tolist()

        sum_loss_G = 0
        for i in range(len(losses_G)):
            sum_loss_G += losses_G[i]

        mean_loss_G = sum_loss_G / len(losses_G) / batch_num

        return mean_loss_G, temp_content_with_prediction, temp_out_img, temp_out_img_label, temp_label_with_prediction, correct_vel / inputs.shape[0], correct_ang / inputs.shape[0], correct_goal / inputs.shape[0], correct_goal_distance / inputs.shape[0],correct_reward / inputs.shape[0]


    def predict(self, inputs, images_pres, transformations_pres, num_pres, actions, batch_size=32):

        with torch.no_grad():
            ensemble_out = np.zeros((self.network_size, inputs.shape[0], 80, 160))
            ensemble_out_before = np.zeros((self.network_size, inputs.shape[0], 80, 160))
            ensemble_transformation_out = np.zeros((self.network_size, inputs.shape[0], 3))
            ensemble_vel_ang_out = np.zeros((self.network_size, inputs.shape[0], 2))
            ensemble_goal_out = np.zeros((self.network_size, inputs.shape[0], 2))
            ensemble_reward = np.zeros((self.network_size, inputs.shape[0], 1))

            for i in range(0, inputs.shape[0], batch_size):
                state_input = inputs[i:min(i + batch_size, inputs.shape[0])]
                images_pre = images_pres[i:min(i + batch_size, inputs.shape[0])]
                transformations_pre = transformations_pres[i:min(i + batch_size, inputs.shape[0])]
                num_pre = num_pres[i:min(i + batch_size, inputs.shape[0])]
                action = actions[i:min(i + batch_size, inputs.shape[0])]

                for idx in range(self.network_size):
                    out_before, out, transformation_out, vel_ang_out, goal_out, reward = \
                        self.model_list[idx].generator(state_input, images_pre, transformations_pre, num_pre, action)
                    ensemble_out[idx, i:min(i + batch_size, inputs.shape[0]), :, :] = out.squeeze(1).detach().cpu().numpy()
                    ensemble_out_before[idx, i:min(i + batch_size, inputs.shape[0]), :, :] = out_before.squeeze(1).detach().cpu().numpy()
                    ensemble_transformation_out[idx, i:min(i + batch_size, inputs.shape[0]), :] = transformation_out.detach().cpu().numpy()
                    ensemble_vel_ang_out[idx, i:min(i + batch_size, inputs.shape[0]), :] = vel_ang_out.detach().cpu().numpy()
                    ensemble_goal_out[idx, i:min(i + batch_size, inputs.shape[0]), :] = goal_out.detach().cpu().numpy()
                    ensemble_reward[idx, i:min(i + batch_size, inputs.shape[0]), :] = reward.detach().cpu().numpy()

        return ensemble_out_before, ensemble_out, ensemble_transformation_out, ensemble_vel_ang_out, ensemble_goal_out, ensemble_reward


    def save(self, filename):
        for idx in range(self.network_size):
            torch.save(self.model_list[idx].generator.state_dict(), filename + "_generator")
            torch.save(self.model_list[idx].optimizer_G.state_dict(), filename + "_generator_optimizer")


    def load(self, filename):
        for idx in range(self.network_size):
            self.model_list[idx].generator.load_state_dict(torch.load(filename + "_generator", map_location='cuda:0'))
            self.model_list[idx].optimizer_G.load_state_dict(torch.load(filename + "_generator_optimizer", map_location='cuda:0'))


    def evaluate(self, inputs, actions, state_labels, reward_labels, batch_size=16):
        print("====================================>  Evaluate  <====================================")

        correct_vel = 0
        correct_ang = 0
        correct_goal = 0
        correct_goal_distance = 0
        correct_reward = 0

        total = inputs.shape[0]
        print("evaluate total : ", total)

        for start_pos in range(0, inputs.shape[0], batch_size):

            current_state = inputs[start_pos: start_pos + batch_size]
            next_state = state_labels[start_pos: start_pos + batch_size]
            action = actions[start_pos: start_pos + batch_size]
            reward = reward_labels[start_pos: start_pos + batch_size]
            images = np.zeros((batch_size, 2, 80, 160))
            transformations = np.zeros((batch_size, 2, 3))
            num_pre = np.zeros((batch_size, 1))

            for net in self.model_list:
                net.generator.eval()

            with torch.no_grad():

                ensemble_out_before, ensemble_out, ensemble_transformation_out, ensemble_vel_ang_out, ensemble_goal_out, ensemble_reward = self.predict(current_state, images, transformations, num_pre, action, batch_size=64)

                num_models, batch_size, _, _ = ensemble_out.shape

                model_idxes = 0
                batch_idxes = np.arange(0, batch_size)

                # ========================= vel_ang & goal & reward =========================
                model_vel_ang = ensemble_vel_ang_out[model_idxes, batch_idxes]
                model_goal = ensemble_goal_out[model_idxes, batch_idxes]
                model_reward = ensemble_reward[model_idxes, batch_idxes]

                for i in range(batch_size):
                    # reward
                    if abs(model_reward[i] - reward[i]) < 0.1:
                        correct_reward += 1
                    # vel
                    if abs(model_vel_ang[i][0] - next_state[i][-4]) < 0.05:
                        correct_vel += 1
                    # ang
                    if abs(model_vel_ang[i][1] - next_state[i][-3]) < 0.05:
                        correct_ang += 1
                    # goal angle
                    goal_angle = (next_state[i][-2] * 2. - 1.) * math.pi
                    goal_angle_prediction = (model_goal[i][0] * 2. - 1.) * math.pi

                    angle_goal = math.degrees(goal_angle)
                    angle_goal_prediction = math.degrees(goal_angle_prediction)
                    delta_angle = angle_goal_prediction - angle_goal

                    if abs(delta_angle) > 180:
                        delta_angle = abs(delta_angle) - 180

                    if abs(delta_angle) < 3:
                        correct_goal += 1
                    # goal distance
                    if abs(model_goal[i][1] - next_state[i][-1]) < 0.03:
                        correct_goal_distance += 1

        return correct_vel / total, correct_ang / total, correct_goal / total, correct_goal_distance / total, correct_reward / total



def evaluate(model, loader):
    total = len(loader.dataset)
    print("====================================>  Evaluate  <====================================")

    correct_vel = 0
    correct_ang = 0
    correct_goal = 0
    correct_goal_distance = 0
    correct_reward = 0
    total = len(loader.dataset)

    for data in loader:
        length_data = len(data[0])
        current_state, action, next_state, reward = data

        for net in model.model_list:
            net.generator.eval()

        with torch.no_grad():

            current_state = current_state.numpy()
            action = action.numpy()
            next_state = next_state.numpy()
            reward = reward.numpy()

            images = np.zeros((length_data, 2, 80, 160))
            transformations = np.zeros((length_data, 2, 3))
            num_pre = np.zeros((length_data, 1))

            ensemble_out_before, ensemble_out, ensemble_transformation_out, ensemble_vel_ang_out, ensemble_goal_out, ensemble_reward = model.predict(current_state, images, transformations, num_pre, action, batch_size=64)

            num_models, batch_size, _, _ = ensemble_out.shape

            model_idxes = np.random.choice(model.elite_model_idxes, size=batch_size)
            batch_idxes = np.arange(0, batch_size)

            # ========================= vel_ang & goal & reward =========================
            model_vel_ang = ensemble_vel_ang_out[model_idxes, batch_idxes]
            model_goal = ensemble_goal_out[model_idxes, batch_idxes]
            model_reward = ensemble_reward[model_idxes, batch_idxes]

            for i in range(length_data):
                # reward
                if abs(model_reward[i] - reward[i]) < 0.1:
                    correct_reward += 1
                # vel
                if abs(model_vel_ang[i][0] - next_state[i][-4]) < 0.05:
                    correct_vel += 1
                # ang
                if abs(model_vel_ang[i][1] - next_state[i][-3]) < 0.05:
                    correct_ang += 1
                # goal angle
                goal_angle = (next_state[i][-2] * 2. - 1.) * math.pi
                goal_angle_prediction = (model_goal[i][0] * 2. - 1.) * math.pi

                angle_goal = math.degrees(goal_angle)
                angle_goal_prediction = math.degrees(goal_angle_prediction)
                delta_angle = angle_goal_prediction - angle_goal

                if abs(delta_angle) > 180:
                    delta_angle = abs(delta_angle) - 180

                if abs(delta_angle) < 5:
                    correct_goal += 1
                # goal distance
                if abs(model_goal[i][1] - next_state[i][-1]) < 0.03:
                    correct_goal_distance += 1

            # =========================         costmap         =========================
            model_out = ensemble_out[model_idxes, batch_idxes]
            model_out_before = ensemble_out_before[model_idxes, batch_idxes]
            temp_out_img_before_affine  = np.expand_dims(model_out_before[0], axis=0)

            # only label in new reference
            index_show = 0
            final_observation = next_state[index_show, 0:3630]
            final_observation = final_observation.reshape(10, 363)
            lidar_10 = final_observation[:, 0:360]
            x_10 = final_observation[:, 360]
            y_10 = final_observation[:, 361]
            yaw_10 = final_observation[:, 362]

            disentangle_x = np.zeros((10, 360))
            disentangle_y = np.zeros((10, 360))
            for i in range(10):
                temp_image_x, temp_image_y = Transformation(lidar_10[i], x_10[i], y_10[i], yaw_10[i], x_10[-1], y_10[-1], yaw_10[-1], is_numpy=True)
                disentangle_x[i] = temp_image_x
                disentangle_y[i] = temp_image_y

            lidar_ego_image = np.zeros((80, 160))
            for i in range(10):
                if i == 9:
                    lidar_ego_image[disentangle_x[i].astype(np.int), disentangle_y[i].astype(np.int)] = 1
                else:
                    lidar_ego_image[disentangle_x[i].astype(np.int), disentangle_y[i].astype(np.int)] = 0.5

                lidar_ego_image[:, 0] = 0
                lidar_ego_image[0, :] = 0

            lidar_ego_image = cv2.dilate(lidar_ego_image, np.ones((3, 3), np.uint8), iterations=1)
            temp_out_img_label = np.expand_dims(lidar_ego_image, axis=0)
            temp_out_img = np.expand_dims(model_out[0], axis=0)

            # content with prediction
            content_with_prediction = np.zeros((80, 160))
            content_with_prediction[disentangle_x[-1].astype(np.int), disentangle_y[-1].astype(np.int)] = 0.5

            content_with_prediction = cv2.dilate(content_with_prediction, np.ones((3, 3), np.uint8), iterations=1)
            temp_content_with_prediction = np.clip((np.expand_dims(content_with_prediction, axis=0) + temp_out_img_before_affine), 0, 1)

            # label with prediction
            lidar_ego_image_with_prediction = np.zeros((80, 160))
            lidar_ego_image_with_prediction[disentangle_x[-1].astype(np.int), disentangle_y[-1].astype(np.int)] = 0.5

            lidar_ego_image_with_prediction = cv2.dilate(lidar_ego_image_with_prediction, np.ones((3, 3), np.uint8), iterations=1)
            temp_label_with_prediction = np.clip((np.expand_dims(lidar_ego_image_with_prediction, axis=0) + temp_out_img), 0, 1)


    return temp_content_with_prediction, temp_out_img, temp_out_img_label, temp_label_with_prediction, correct_vel / total, \
        correct_ang / total, correct_goal / total, correct_goal_distance / total, correct_reward / total


def main():

    EPOCH = 10000
    pre_epoch = 0
    BATCH_SIZE = 32

    torch.manual_seed(999)
    np.random.seed(999)
    torch.cuda.manual_seed(999)
    
    root = '/home/cyx/model-based-social-navigation/src/model_based_version/dataset'
    writer = SummaryWriter("/home/cyx/model-based-social-navigation/src/model_based_version/dataset/log")

    train_db = Transition_Model_Dataset(root, mode='training')
    val_db = Transition_Model_Dataset(root, mode='validation')
    test_db = Transition_Model_Dataset(root, mode='testing')

    train_loader = DataLoader(train_db, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_db, batch_size=BATCH_SIZE, num_workers=2)
    test_loader = DataLoader(test_db, batch_size=BATCH_SIZE, num_workers=2)

    net = Ensemble_Model(1, 1)
    net.load("/home/cyx/model-based-social-navigation/src/model_based_version/World_Models/TEST/world_model")
    global_step = 0

    print("Start Training, Transition_Model!")  # 定义遍历数据集的次数

    for epoch in range(pre_epoch, EPOCH):
        print('\nEpoch: %d' % (epoch + 1))

        if epoch > 0:
            for model in net.model_list:
                model.scheduler_G.step()

        for i, data in enumerate(train_loader):
            # Load Data
            current_state, action, next_state, reward = data
            current_state = current_state.numpy()
            action = action.numpy()
            next_state = next_state.numpy()
            reward = reward.numpy()

            # Step
            loss_G, content_with_prediction, out, all_label, label_with_prediction, acc_vel, acc_ang, \
                acc_goal_direction, acc_goal_distance, acc_reward = net.train(current_state, action, next_state, reward, batch_size=64)

            # print(loss_G)

            writer.add_scalar('loss_G', loss_G, global_step)

            writer.add_image('Train/cost_map content_with_prediction', content_with_prediction, global_step)
            writer.add_image('Train/cost_map', out, global_step)
            writer.add_image('Train/cost_map label_with_prediction', label_with_prediction, global_step)
            writer.add_image('Train/cost_map all label', all_label, global_step)

            writer.add_scalar('Train/acc_vel', acc_vel, global_step)
            writer.add_scalar('Train/acc_ang', acc_ang, global_step)
            writer.add_scalar('Train/acc_goal_direction', acc_goal_direction, global_step)
            writer.add_scalar('Train/acc_goal_distance', acc_goal_distance, global_step)
            writer.add_scalar('Train/acc_reward', acc_reward, global_step)

            writer.add_scalar('Learning Rate', net.model_list[0].optimizer_G.param_groups[0]['lr'], global_step)

            global_step += 1


        # if epoch == 0:
        if epoch % 2 == 0:
            content_with_prediction, out, all_label, label_with_prediction, acc_vel, acc_ang, acc_goal_direction, acc_goal_distance, acc_reward = \
                evaluate(net, val_loader)

            writer.add_image('Evaluate/cost_map content_with_prediction', content_with_prediction, epoch / 2)
            writer.add_image('Evaluate/cost_map', out, epoch / 2)
            writer.add_image('Evaluate/cost_map label_with_prediction', label_with_prediction, epoch / 2)
            writer.add_image('Evaluate/cost_map all label', all_label, epoch / 2)

            writer.add_scalar('Evaluate/acc_vel', acc_vel, epoch / 2)
            writer.add_scalar('Evaluate/acc_ang', acc_ang, epoch / 2)
            writer.add_scalar('Evaluate/acc_goal_direction', acc_goal_direction, epoch / 2)
            writer.add_scalar('Evaluate/acc_goal_distance', acc_goal_distance, epoch / 2)
            writer.add_scalar('Evaluate/acc_reward', acc_reward, epoch / 2)

            net.save(root + '/World_models')


if __name__ == '__main__':
    main()
