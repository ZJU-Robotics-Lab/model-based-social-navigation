# -*- coding: utf-8 -*-
# transition model of cost map in the same reference, to test whether the model can predict motion
#

import math
import time
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torchsummary import summary
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
# from torchviz import make_dot
import cv2
import kornia
from math import floor

import matplotlib.pyplot as plt

# torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


IMAGE_W = 160
IMAGE_H = 80
MAX_LASER_RANGE = 4.0

using_GAN = False


####################################
#              Utils
####################################


def Transformation(lidar_input, current_x, current_y, current_yaw, target_x, target_y, target_yaw, scale=1):
    # lidar_current = lidar_input.cpu().numpy()
    # current_x = current_x.cpu().numpy()
    # current_y = current_y.cpu().numpy()
    # current_yaw = current_yaw.cpu().numpy()
    # target_x = target_x.cpu().numpy()
    # target_y = target_y.cpu().numpy()
    # target_yaw = target_yaw.cpu().numpy()
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
    t1 = time.time()
    T_world_relative_to_target = np.linalg.inv(T_target_relative_to_world)
    T_current_relative_to_target = T_world_relative_to_target.dot(T_current_relative_to_world)
    coordinates_target = T_current_relative_to_target.dot(coordinates_current)

    x_target = coordinates_target[0]
    y_target = coordinates_target[1]

    image_x = np.floor(IMAGE_H * scale - 1.0 - x_target / MAX_LASER_RANGE * IMAGE_H * scale).astype(np.int)
    image_y = np.floor(IMAGE_H * scale - 1.0 - y_target / MAX_LASER_RANGE * IMAGE_H * scale).astype(np.int)

    image_x[(image_x < 0) | (image_x > (IMAGE_H * scale - 1)) | (lidar_current > (MAX_LASER_RANGE - 0.2))] = 0
    image_y[(image_y < 0) | (image_y > (IMAGE_W * scale - 1)) | (lidar_current > (MAX_LASER_RANGE - 0.2))] = 0

    t2 = time.time()
    # print("Time : {}  ms".format(round(1000*(t2-t1),2)))

    return image_x, image_y


def Unpooling(x):
    x = x.permute(0, 2, 3, 1)
    out = torch.cat((x, Variable(torch.zeros(x.size())).to(device)), dim=3)
    out = torch.cat((out, Variable(torch.zeros(out.size())).to(device)), dim=2)

    shape = x.size()
    B, H, W, C = int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3])
    H *= 2
    W *= 2

    out = out.view(B, H, W, C).permute(0, 3, 1, 2)  # BHWC => BCHW
    out = out.contiguous(memory_format=torch.contiguous_format)
    return out


def Vel_Ang_Prediction(in_vel_ang, action):

    command_velocity = (action[:, 0] + 1.) / 2. # -1~1 => 0~1
    command_angular_velocity = action[:, 1] * 1.5 # -1~1 => -1.5~1.5
    current_velocity = in_vel_ang[:, 0] # 0~1
    current_angular_velocity = (in_vel_ang[:, 1] * 2. - 1.) * 1.5 # 0~1 => -1.5~1.5

    delta_vel = command_velocity - current_velocity
    delta_ang = command_angular_velocity - current_angular_velocity

    out_vel = np.zeros_like(current_velocity)
    out_ang = np.zeros_like(current_angular_velocity)

    for m in range(in_vel_ang.shape[0]):
        if delta_vel[m] > 0.05:
            out_vel[m] = current_velocity[m] + 0.05
        elif delta_vel[m] < -0.05:
            out_vel[m] = current_velocity[m] - 0.05
        else:
            out_vel[m] = command_velocity[m]

        if out_vel[m] < 0.05:
            out_vel[m] = 0.05
        elif out_vel[m] > 1.0:
            out_vel[m] = 1.0
        
        if delta_ang[m] > 0.1:
            out_ang[m] = current_angular_velocity[m] + 0.1
        elif delta_ang[m] < -0.1:
            out_ang[m] = current_angular_velocity[m] - 0.1
        else:
            out_ang[m] = command_angular_velocity[m]

        if out_ang[m] < -1.5:
            out_ang[m] = -1.5
        elif out_ang[m] > 1.5:
            out_ang[m] = 1.5

    out_vel_ang = np.concatenate((np.expand_dims(out_vel, axis=1), (np.expand_dims(out_ang, axis=1) / 1.5 + 1.) / 2.), axis=1)

    return out_vel_ang # 0~1


def Goal_Prediction(goal, cur_vel_ang, pre_vel_ang):

    current_velocity = cur_vel_ang[:, 0]
    current_angular_velocity = (cur_vel_ang[:, 1] * 2. - 1.) * 1.5
    out_vel = pre_vel_ang[:, 0]
    out_ang = (pre_vel_ang[:, 1] * 2. - 1.) * 1.5

    average_vel = current_velocity * 0.5 + out_vel * 0.5
    average_ang = current_angular_velocity * 0.5 + out_ang * 0.5

    for m in range(goal.shape[0]): # in case that average_ang == 0
        if average_ang[m] == 0:
            average_ang[m] += 1e-5

    d_x = average_vel / average_ang * np.sin(average_ang * 0.1) 
    d_y = average_vel / average_ang * (1 - np.cos(average_ang * 0.1)) 
    d_theta = average_ang * 0.1

    current_goal_x = goal[:, 0] * 25. * np.cos((goal[:, 1] * 2 - 1) * np.pi)
    current_goal_y = goal[:, 0] * 25. * np.sin((goal[:, 1] * 2 - 1) * np.pi)
    out_goal_x = current_goal_x - d_x
    out_goal_y = current_goal_y - d_y
    out_goal_distance = np.sqrt(out_goal_x * out_goal_x + out_goal_y * out_goal_y) / 25.

    # 角度用sin cos联合表示，避免-pi和pi之间跳变
    # angle -pi ~ 0 => sin() 0 ~ -1 ~ 0  &  cos() -1 ~ 0 ~  1
    # angle  0 ~ pi => sin() 0 ~  1 ~ 0  &  cos()  1 ~ 0 ~ -1

    in_goal_direction = (goal[:, 1] * 2 - 1) * np.pi
    out_goal_direction = (goal[:, 1] * 2 - 1) * np.pi - d_theta

    in_goal_sin = np.sin(in_goal_direction)
    in_goal_cos = np.cos(in_goal_direction)
    out_goal_sin = np.sin(out_goal_direction)
    out_goal_cos = np.cos(out_goal_direction)


    for m in range(goal.shape[0]): # in case that direction angle out of range
        if out_goal_direction[m] > np.pi:
            out_goal_direction[m] = out_goal_direction[m] - 2 * np.pi
        if out_goal_direction[m] < -np.pi:
            out_goal_direction[m] = out_goal_direction[m] + 2 * np.pi
    out_goal_direction = (out_goal_direction / np.pi + 1.) / 2.


    in_goal = np.concatenate((np.expand_dims(goal[:, 0], axis=1), np.expand_dims(in_goal_sin, axis=1), np.expand_dims(in_goal_cos, axis=1)), axis=1)
    out_goal = np.concatenate((np.expand_dims(out_goal_distance, axis=1), np.expand_dims(out_goal_sin, axis=1), np.expand_dims(out_goal_cos, axis=1)), axis=1)
    final_goal = np.concatenate((np.expand_dims(out_goal_distance, axis=1), np.expand_dims(out_goal_direction, axis=1)), axis=1)

    return in_goal, out_goal, final_goal, d_x, d_y, d_theta


####################################
#             Classes
####################################


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x


class Motion_Encoder(nn.Module):
    def __init__(self, G_dim):
        super(Motion_Encoder, self).__init__()

        self.motion_mask_conv = nn.Sequential(
            nn.Conv2d(1, G_dim, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(G_dim, G_dim, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(G_dim, 1, 3, padding=1),
            nn.Sigmoid())

        self.motion_conv_1 = nn.Sequential(
            nn.Conv2d(1, G_dim, 3, padding=1),
            nn.LeakyReLU(inplace=True))

        self.motion_conv_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(G_dim, G_dim * 2, 3, padding=1),
            nn.LeakyReLU(inplace=True))

        self.motion_conv_3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(G_dim * 2, G_dim * 4, 3, padding=1),
            nn.LeakyReLU(inplace=True))

        self.pool_out = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, input_diff):
        """
        input_diff:                  [batch_size, 1, h, w]
        res_in: a list of 3 tensors, [batch_size, G_dim,     h,   w],
                                     [batch_size, G_dim*2, h/2, w/2],
                                     [batch_size, G_dim*4, h/4, w/4]
        output:                      [batch_size, G_dim*4, h/8, w/8]
        """
        motion_mask = self.motion_mask_conv(input_diff)
        masked_input = motion_mask * input_diff
        res_out_1 = self.motion_conv_1(masked_input)
        res_out_2 = self.motion_conv_2(res_out_1)
        res_out_3 = self.motion_conv_3(res_out_2)

        output = self.pool_out(res_out_3)
        res_out = [res_out_1, res_out_2, res_out_3]

        return output, res_out


class ConvLstmCell(nn.Module):
    def __init__(self, feature_size, num_features, forget_bias=1, bias=True):
        super(ConvLstmCell, self).__init__()

        self.feature_size = feature_size
        self.num_features = num_features
        self.forget_bias = forget_bias

        self.conv = nn.Conv2d(num_features * 2, num_features * 4, feature_size, padding=1, bias=bias)
        self.conv_mask = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(num_features, num_features, 3, padding=1),
            nn.Sigmoid())

    def forward(self, input, state):
        c, h = torch.chunk(state, 2, dim=1)

        masked_input = self.conv_mask(input) * input
        conv_input = torch.cat((masked_input, h), dim=1)

        conv_output = self.conv(conv_input)

        (i, j, f, o) = torch.chunk(conv_output, 4, dim=1)
        new_c = c * torch.sigmoid(f + self.forget_bias) + torch.sigmoid(i) * torch.tanh(j)
        new_h = torch.tanh(new_c) * torch.sigmoid(o)
        new_state = torch.cat((new_c, new_h), dim=1)
        return new_h, new_state


class Content_Encoder(nn.Module):
    def __init__(self, G_dim):
        super(Content_Encoder, self).__init__()

        self.content_conv_1 = nn.Sequential(
            nn.Conv2d(1, G_dim, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(G_dim, G_dim, 3, padding=1),
            nn.LeakyReLU(inplace=True)
        )

        self.content_conv_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(G_dim, G_dim * 2, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(G_dim * 2, G_dim * 2, 3, padding=1),
            nn.LeakyReLU(inplace=True)
        )

        self.content_conv_3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(G_dim * 2, G_dim * 4, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(G_dim * 4, G_dim * 4, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(G_dim * 4, G_dim * 4, 3, padding=1),
            nn.LeakyReLU(inplace=True)
        )

        self.pool_out = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, content):
        """
        input_diff:                  [batch_size, c_dim, h, w]
        res_in: a list of 3 tensors, [batch_size, G_dim  ,   h,   w],
                                     [batch_size, G_dim*2, h/2, w/2],
                                     [batch_size, G_dim*4, h/4, w/4]
        output:                      [batch_size, G_dim*4, h/8, w/8]
        """
        res_out_1 = self.content_conv_1(content)
        res_out_2 = self.content_conv_2(res_out_1)
        res_out_3 = self.content_conv_3(res_out_2)

        output = self.pool_out(res_out_3)
        res_out = [res_out_1, res_out_2, res_out_3]

        return output, res_out


class Combine_Encoder(nn.Module):
    def __init__(self, G_dim):
        super(Combine_Encoder, self).__init__()

        self.combine_conv = nn.Sequential(
            nn.Conv2d(G_dim * 8, G_dim * 4, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(G_dim * 4, G_dim * 2, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(G_dim * 2, G_dim * 4, 3, padding=1),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, motion, content):
        input = torch.cat((motion, content), dim=1)
        output = self.combine_conv(input)

        return output


class Residual_Encoder(nn.Module):
    def __init__(self, out_dim):
        super(Residual_Encoder, self).__init__()

        self.residual_conv = nn.Sequential(
            nn.Conv2d(out_dim * 2, out_dim, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, 3, padding=1)
        )

    def forward(self, motion, content):
        input = torch.cat((motion, content), dim=1)
        output = self.residual_conv(input)
        return output


class Decoder(nn.Module):
    def __init__(self, G_dim):
        super(Decoder, self).__init__()

        self.decoder_unit3 = nn.Sequential(
            nn.ConvTranspose2d(G_dim * 4, G_dim * 4, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(G_dim * 4, G_dim * 4, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(G_dim * 4, G_dim * 2, 3, padding=1),
            nn.LeakyReLU(inplace=True)
        )

        self.decoder_unit2 = nn.Sequential(
            nn.ConvTranspose2d(G_dim * 2, G_dim * 2, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(G_dim * 2, G_dim, 3, padding=1),
            nn.LeakyReLU(inplace=True)
        )

        self.decoder_unit1 = nn.Sequential(
            nn.ConvTranspose2d(G_dim, G_dim, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(G_dim, 1, 3, padding=1),
            nn.Tanh()
        )


        self.upsample_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.upsample_2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.upsample_3 = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, res_1, res_2, res_3, combine):
        up_combine = Unpooling(combine)
        # up_combine = self.upsample_1(combine)
        input3 = up_combine + res_3
        dec3_out = self.decoder_unit3(input3)

        up_dec3_out = Unpooling(dec3_out)
        # up_dec3_out = self.upsample_2(dec3_out)
        input2 = up_dec3_out + res_2
        dec2_out = self.decoder_unit2(input2)

        up_dec2_out = Unpooling(dec2_out)
        # up_dec2_out = self.upsample_3(dec2_out)
        input1 = up_dec2_out + res_1
        dec1_out = self.decoder_unit1(input1)

        return dec1_out


class Reward_Prediction(nn.Module):
    def __init__(self, G_dim):
        super(Reward_Prediction, self).__init__()

        h, w = IMAGE_H, IMAGE_W

        self.conv = nn.Sequential(
            nn.Conv2d(1, 2, 4, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(2, 4, 4, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(4, 8, 4, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(8, 16, 4, stride=2, padding=1),
            nn.LeakyReLU(inplace=True)
        )

        h = floor((floor(
            (floor((floor((h + 2 * 1 - 4) / 2 + 1) + 2 * 1 - 4) / 2 + 1) + 2 * 1 - 4) / 2 + 1) + 2 * 1 - 4) / 2 + 1)
        w = floor((floor(
            (floor((floor((w + 2 * 1 - 4) / 2 + 1) + 2 * 1 - 4) / 2 + 1) + 2 * 1 - 4) / 2 + 1) + 2 * 1 - 4) / 2 + 1)

        in_features = int(h * w * 16)
        self.fc_features = int(G_dim * 3)

        self.l_costmap_out = nn.Linear(in_features, self.fc_features * 2)

        self.l_goal_distance = nn.Linear(1, int(self.fc_features))
        self.l_goal_direction = nn.Linear(4, int(self.fc_features))

        self.l_out_0 = nn.Linear(self.fc_features * 4, self.fc_features * 2)
        self.l_out_1 = nn.Linear(self.fc_features * 2, self.fc_features)
        self.l_out = nn.Linear(self.fc_features, 1)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, costmap, goal_before, goal_after):
        """
        input:                  costmap          : [batch_size, 1, IMAGE_H, IMAGE_W]
        input:                  goal             : [batch_size, 3]  0 ~ 1
                                goal_after       : [batch_size, 3]  0 ~ 1
        output:                 prediction       : [batch_size, 1] -1 ~ 1
        """
        costmap_out = self.conv(costmap)
        costmap_out = costmap_out.view(costmap.shape[0], -1)
        costmap_out = F.leaky_relu(self.l_costmap_out(costmap_out))

        delta_goal_distance = (goal_after[:,0].unsqueeze(1) - goal_before[:,0].unsqueeze(1)) * 20
        goal_direction = torch.cat([goal_before[:, 1:], goal_after[:, 1:]], dim=1)

        goal_distance_out = F.leaky_relu(self.l_goal_distance(delta_goal_distance))
        goal_direction_out = F.leaky_relu(self.l_goal_direction(goal_direction))

        all_input = torch.cat([costmap_out, goal_distance_out, goal_direction_out], dim=1)

        output = self.dropout(F.leaky_relu(self.l_out_0(all_input)))
        output = self.dropout(F.leaky_relu(self.l_out_1(output)))
        output = self.l_out(output)

        return output

# pixel gradients, to know the gradient layout different
class GDL(nn.Module):
    def __init__(self):
        super(GDL, self).__init__()
        self.loss = nn.L1Loss()
        a = np.array([[-1, 1]])
        b = np.array([[1], [-1]])
        self.filter_w = np.zeros([1, 2])
        self.filter_h = np.zeros([2, 1])

        self.filter_w = np.zeros([1, 1, 1, 2])
        self.filter_h = np.zeros([1, 1, 2, 1])

        self.filter_w[0, 0, :, :] = a
        self.filter_h[0, 0, :, :] = b

    def __call__(self, output, target):
        filter_w = Variable(torch.from_numpy(self.filter_w).float().to(device))
        filter_h = Variable(torch.from_numpy(self.filter_h).float().to(device))

        output_w = F.conv2d(output, filter_w, padding=(0, 1))
        output_h = F.conv2d(output, filter_h, padding=(1, 0))
        target_w = F.conv2d(target, filter_w, padding=(0, 1))
        target_h = F.conv2d(target, filter_h, padding=(1, 0))
        return self.loss(output_w, target_w) + self.loss(output_h, target_h)

class Motion_VAE(nn.Module):
    def __init__(self, G_dim, latent_dim=512):
        super(Motion_VAE, self).__init__()
        self.G_dim = G_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(G_dim * 4, G_dim * 4, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(G_dim * 4, G_dim * 4, 3, padding=1),
            nn.LeakyReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(G_dim * 4, G_dim * 4, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(G_dim * 4, G_dim * 4, 3, padding=1),
            nn.LeakyReLU(inplace=True)
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(G_dim * 4, G_dim * 4, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(G_dim * 4, G_dim * 4, 3, padding=1),
            nn.LeakyReLU(inplace=True)
        )

        self.fc_mu = nn.Linear(int(G_dim * 4 * IMAGE_H / 8 * IMAGE_W / 8), latent_dim)
        self.fc_var = nn.Linear(int(G_dim * 4 * IMAGE_H / 8 * IMAGE_W / 8), latent_dim)

        self.fc_decoder_input = nn.Linear(latent_dim, int(G_dim * 4 * IMAGE_H / 8 * IMAGE_W / 8))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        exp = torch.rand_like(std)

        return exp * std + mu

    def loss_kld(self, mu, logvar):
        return torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)


    def forward(self, input):
        """
        input:                       [batch_size, G_dim*4, h/8, w/8]
        output:                      [batch_size, G_dim*4, h/8, w/8]
        """
        x = self.encoder(input)
        x = torch.flatten(x, start_dim=1)

        mu = self.fc_mu(x)
        logvar = self.fc_var(x)

        z = self.reparameterize(mu, logvar)
        z = self.fc_decoder_input(z)
        z = z.view(-1, self.G_dim * 4, int(IMAGE_H / 8), int(IMAGE_W / 8))

        out = self.decoder(z)
        out = self.final_conv(out)

        loss_kld = self.loss_kld(mu, logvar)

        return out, loss_kld


####################################
#       LSTM U-net Generator
####################################
class Generator(nn.Module):

    def __init__(self, G_dim, feature_size, flag_only_image=True):
        """init 初始化
        """
        super(Generator, self).__init__()

        self.motion_encoder = Motion_Encoder(G_dim)  # [B, 1, h, w] => [B, G_dim*4, h/8, w/8]
        self.content_encoder = Content_Encoder(G_dim)  # [B, 1, h, w] => [B, G_dim*4, h/8, w/8]
        self.combine_encoder = Combine_Encoder(G_dim)  # [B, G_dim*4*2, h/8, w/8] => [B, G_dim*4, h/8, w/8]
        self.conv_lstm_cell = ConvLstmCell(feature_size, 4 * G_dim)
        self.motion_vae = Motion_VAE(G_dim)

        self.reward_prediction = Reward_Prediction(G_dim)

        self.residual_1 = Residual_Encoder(G_dim * 1)  # [B, G_dim*1*2, h, w] => [B, G_dim, h, w]
        self.residual_2 = Residual_Encoder(G_dim * 2)  # [B, G_dim*2*2, h, w] => [B, G_dim*2, h/2, w/2]
        self.residual_3 = Residual_Encoder(G_dim * 4)  # [B, G_dim*4*2, h, w] => [B, G_dim*4, h/4, w/4]

        self.decoder = Decoder(G_dim)


    def forward(self, state, images, transformation, num_pre, action):

        # ================================> Data Preparation <================================
        final_observation_all = state[:, 0:3630]
        vel_ang = state[:, 3630:3632]
        goal = state[:, 3632:3634]

        final_observation_all = final_observation_all.reshape(-1, 10, 363)
        lidar_10 = final_observation_all[:, :, 0:360]
        x_10 = final_observation_all[:, :, 360]
        y_10 = final_observation_all[:, :, 361]
        yaw_10 = final_observation_all[:, :, 362]

        disentangle_x = np.zeros((state.shape[0], 10, 360))
        disentangle_y = np.zeros((state.shape[0], 10, 360))

        lidar_ego_image_motion = np.zeros((state.shape[0], 10, 80, 160))
        lidar_ego_image_content = np.zeros((state.shape[0], 1, 80, 160))
        content = np.zeros((state.shape[0], 80, 160))

        # *** motion ===============================================================
        # disentangle
        for m in range(state.shape[0]):
            for i in range(10):
                temp_image_x, temp_image_y = Transformation(lidar_10[m][i], x_10[m][i], y_10[m][i], yaw_10[m][i],
                                                            x_10[m][-1], y_10[m][-1], yaw_10[m][-1])
                disentangle_x[m][i] = temp_image_x
                disentangle_y[m][i] = temp_image_y
                lidar_ego_image_motion[m, i, disentangle_x[m, i].astype(np.int), disentangle_y[m, i].astype(np.int)] = 1
                lidar_ego_image_motion[m, i, 0, :] = 0
                lidar_ego_image_motion[m, i, :, 0] = 0  # BTHW

        # combine with prediction
        lidar_ego_image_motion = torch.from_numpy(lidar_ego_image_motion)
        init_scale = torch.ones(10)
        init_center = torch.ones(10, 2)
        init_center[:, 0] = IMAGE_H
        init_center[:, 1] = IMAGE_H
        
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

                lidar_ego_image_motion[m] = kornia.warp_affine(lidar_ego_image_motion[m].unsqueeze(1),
                                                                init_translation_matrix, dsize=(80, 160),
                                                                flags='bilinear', padding_mode='zeros',
                                                                align_corners=False).squeeze(1)

                init_angle = torch.ones(10) * init_delta_theta
                init_rotation_matrix = kornia.get_rotation_matrix2d(init_center, init_angle, init_scale)

                lidar_ego_image_motion[m] = kornia.warp_affine(lidar_ego_image_motion[m].unsqueeze(1),
                                                                init_rotation_matrix, dsize=(80, 160),
                                                                flags='bilinear', padding_mode='zeros',
                                                                align_corners=False).squeeze(1)

                new_image = cv2.erode(images[m][n].astype(np.float), np.ones((3, 3), np.uint8), iterations=1)
                new_image = torch.from_numpy(new_image).unsqueeze(0)
                lidar_ego_image_motion[m] = torch.cat([lidar_ego_image_motion[m][1:10], new_image], dim=0)

        lidar_ego_image_motion = lidar_ego_image_motion.numpy()

        motion_input = np.zeros((state.shape[0], 1, 9, 80, 160))
        motion_output = np.zeros((state.shape[0], 1, 80, 160))
        # difference
        for m in range(state.shape[0]):
            for i in range(9):
                if i == 0:
                    before_index = i
                else:
                    before_index = i - 1
                after_index = i + 1
                before = lidar_ego_image_motion[m][before_index].astype(np.uint8)
                after = lidar_ego_image_motion[m][after_index].astype(np.uint8)

                before_dilate = cv2.dilate(before, np.ones((3, 3), np.uint8), iterations=1)
                after_dilate = cv2.dilate(after, np.ones((3, 3), np.uint8), iterations=1)

                delta_old_dilate = cv2.dilate(
                    np.clip(-(after_dilate.astype(np.float) - before.astype(np.float)), 0, 1),
                    np.ones((5, 5), np.uint8), iterations=1)
                delta_new_dilate = cv2.dilate(
                    np.clip(-(before_dilate.astype(np.float) - after.astype(np.float)), 0, 1),
                    np.ones((7, 7), np.uint8), iterations=1)  # slightly bigger

                add_dilate = delta_new_dilate - delta_old_dilate  # -1~1
                add_dilate = cv2.GaussianBlur(add_dilate, (5, 5), 0)

                motion_input[m, 0, i] = add_dilate  # -1~1

        motion_output = motion_input[:, :, -1]

        motion_input = torch.from_numpy(motion_input).float().to(device)
        motion_output = (torch.from_numpy(motion_output).float().to(device) + 1.) / 2

        # *** content ===============================================================
        for m in range(state.shape[0]):
            temp_image_x, temp_image_y = Transformation(lidar_10[m][-1], x_10[m][-1], y_10[m][-1], yaw_10[m][-1],
                                                        x_10[m][-1], y_10[m][-1], yaw_10[m][-1], 1)
            lidar_ego_image_content[m, -1, temp_image_x.astype(np.int), temp_image_y.astype(np.int)] = 1
            lidar_ego_image_content[m, -1, 0, :] = 0
            lidar_ego_image_content[m, -1, :, 0] = 0  # BTHW

            content[m] = cv2.dilate(lidar_ego_image_content[m, -1, :, :], np.ones((3, 3), np.uint8), iterations=1)

        for m in range(state.shape[0]):
            if num_pre[m] > 0:
                content[m] = images[m][int(num_pre[m] - 1)]

        content_input = torch.from_numpy(content).unsqueeze(1) * 2 - 1  # 0~1 => -1~1
        # content_output = torch.from_numpy(content).unsqueeze(1)
        content_input = content_input.float().to(device)

        # ================================> Network <================================
        # 　=======================>   costmap prediction in original reference
        h_state = Variable(torch.zeros((state.shape[0], 128, 10, 20)), requires_grad=False).to(device)

        for t in range(10 - 1):
            code_motion, residual_motion = self.motion_encoder.forward(motion_input[:, :, t, :, :])
            h_motion, h_state = self.conv_lstm_cell.forward(code_motion, h_state)

        # TRY: VAE
        # h_motion, kld_loss = self.motion_vae(h_motion)
        # kld_loss = 0

        # One step prediction
        h_content, residual_content = self.content_encoder.forward(content_input)
        h_combine = self.combine_encoder.forward(h_motion, h_content)
        res_1 = self.residual_1.forward(residual_motion[0], residual_content[0])
        res_2 = self.residual_2.forward(residual_motion[1], residual_content[1])
        res_3 = self.residual_3.forward(residual_motion[2], residual_content[2])
        x_hat = self.decoder.forward(res_1, res_2, res_3, h_combine)  # -1~1

        out = (x_hat.view(state.shape[0], 1, IMAGE_H, IMAGE_W) + 1.) / 2  # -1~1 => 0~1

        out_before = out
        #  =======================>   coordinate prediction
        out_vel_ang = Vel_Ang_Prediction(vel_ang, action)
        input_goal, output_goal, final_goal, d_x, d_y, d_theta = Goal_Prediction(goal, vel_ang, out_vel_ang)

        #  =======================>   costmap prediction in new reference
        delta_x = d_x / MAX_LASER_RANGE * IMAGE_H
        delta_y = d_y / MAX_LASER_RANGE * IMAGE_H
        delta_theta = - d_theta / math.pi * 180

        delta_x = torch.from_numpy(delta_x).float().to(device)
        delta_y = torch.from_numpy(delta_y).float().to(device)
        delta_theta = torch.from_numpy(delta_theta).float().to(device)

        transformation_output = torch.cat([delta_x.unsqueeze(1), delta_y.unsqueeze(1), delta_theta.unsqueeze(1)], dim=1) # [B, 3]

        translation_matrix = torch.zeros((state.shape[0], 2, 3)).to(device)
        translation_matrix[:, 0, 0] = 1
        translation_matrix[:, 1, 1] = 1
        translation_matrix[:, 0, 2] = delta_y
        translation_matrix[:, 1, 2] = delta_x # 注意车子运动坐标系的xy和图像坐标系的xy不同
        out = kornia.warp_affine(out, translation_matrix, dsize=(IMAGE_H, IMAGE_W), flags='bilinear',
                                    padding_mode='zeros', align_corners=False)

        scale = torch.ones(state.shape[0]).to(device)
        center = torch.ones(state.shape[0], 2).to(device)
        center[:, 0] = IMAGE_H
        center[:, 1] = IMAGE_H
        rotation_matrix = kornia.get_rotation_matrix2d(center, delta_theta, scale).to(device)
        out = kornia.warp_affine(out, rotation_matrix, dsize=(IMAGE_H, IMAGE_W), flags='bilinear',
                                    padding_mode='zeros', align_corners=False)

        vel_ang = torch.from_numpy(vel_ang).float().to(device)
        out_vel_ang = torch.from_numpy(out_vel_ang).float().to(device)
        input_goal = torch.from_numpy(input_goal).float().to(device)
        output_goal = torch.from_numpy(output_goal).float().to(device)
        out_goal = torch.from_numpy(final_goal).float().to(device)
        out_reward = self.reward_prediction(out, input_goal, output_goal)

        return out_before, out, transformation_output, out_vel_ang, out_goal, out_reward


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
        try:
            nn.init.constant_(m.bias, 0.001)
        except:
            pass
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.constant_(m.bias, 0.001)


class Transition_Model(object):

    def __init__(self, flag_only_image=True):

        # Models
        self.generator = Generator(16, 3, flag_only_image).apply(weights_init).to(device)
        # self.gdl = GDL()

        # Learning rate
        self.lr_G = 3e-4

        self.update_G = True

        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), weight_decay=1e-5, lr=self.lr_G)

        self.scheduler_G = lr_scheduler.StepLR(self.optimizer_G, 20, 0.5)


    def train_all(self, lidar_img_pre, vel_ang_pre, goal_pre, reward_pre, state_labels, reward_labels):

        # ===========>  Real: image prediction label
        final_observation_all = state_labels[:, 0:3630]
        final_observation_all = final_observation_all.reshape(-1, 10, 363)
        lidar_input = final_observation_all[:, -1, 0:360]

        vel_ang_input = state_labels[:, 3630:3632]
        goal_input = state_labels[:, 3632:3634]

        # ===========>  other labels
        vel_ang_labels = vel_ang_input
        goal_labels = goal_input
        reward_labels = np.expand_dims(reward_labels, axis=1)
        reward_labels = torch.from_numpy(reward_labels).float().to(device)

        # ===========>  Real: labels sequence
        labels_final_observation_all = state_labels[:, 0:3630]
        labels_final_observation_all = labels_final_observation_all.reshape(-1, 10, 363)
        labels_lidar = labels_final_observation_all[:, :, 0:360]
        labels_x = labels_final_observation_all[:, :, 360]
        labels_y = labels_final_observation_all[:, :, 361]
        labels_yaw = labels_final_observation_all[:, :, 362]

        labels_disentangle_x = np.zeros((state_labels.shape[0], 1, 360))
        labels_disentangle_y = np.zeros((state_labels.shape[0], 1, 360))

        labels_lidar_ego_image = np.zeros((state_labels.shape[0], 1, 80, 160))

        for m in range(state_labels.shape[0]):
            labels_temp_image_x, labels_temp_image_y = Transformation(labels_lidar[m][-1], \
                labels_x[m][-1], labels_y[m][-1], labels_yaw[m][-1], \
                    labels_x[m][-1], labels_y[m][-1], labels_yaw[m][-1])
            labels_disentangle_x[m][0] = labels_temp_image_x
            labels_disentangle_y[m][0] = labels_temp_image_y
            labels_lidar_ego_image[m, 0, labels_disentangle_x[m, 0].astype(np.int), labels_disentangle_y[m, 0].astype(np.int)] = 1
            labels_lidar_ego_image[m, 0, 0, :] = 0
            labels_lidar_ego_image[m, 0, :, 0] = 0
            labels_lidar_ego_image[m, 0] = cv2.dilate(labels_lidar_ego_image[m, 0], np.ones((3, 3), np.uint8), iterations=1)

        lidar_img_labels = torch.from_numpy(labels_lidar_ego_image[:, 0]).unsqueeze(1).float().to(device)


        # ====================   Generator Loss   ====================
        mse_loss = nn.MSELoss()
        bce_loss = nn.BCELoss()
        smooth_l1_loss = nn.SmoothL1Loss()
        # Pixel-wise Loss
        mse_loss_costmap = smooth_l1_loss(input=lidar_img_pre, target=lidar_img_labels)
        # GDL Loss
        # gdl_loss_costmap = self.gdl(lidar_img_pre, lidar_img_labels)

        # Reward Loss
        mse_loss_reward = mse_loss(input=reward_pre, target=reward_labels)

        if using_GAN:
            print("using_GAN")
        else:
            loss_G = 30 * mse_loss_costmap + 5 * mse_loss_reward
            self.optimizer_G.zero_grad()
            loss_G.backward()
            torch.nn.utils.clip_grad_value_(self.generator.parameters(), clip_value=10)
            self.optimizer_G.step()
            # print("pixel loss : ", 10 * mse_loss_costmap)
            # print("Reward loss : ", 5 * mse_loss_reward)

        return loss_G
