#!/usr/bin/env python3
# -*- coding: utf-8 -*- 


import gym
import numpy 
import time
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import random

import rospy
import rospkg

from gym import wrappers
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from torch.utils.tensorboard import SummaryWriter

from policy import TD3
import utils

from environment_four import Env


S_DIM = 3634
A_DIM = 2
action_linear_max = 1.0  # m/s  
delta_action_linear_max = 0.1 # m^2/s
action_angular_max = 1.5  # rad/s
max_action = 1.0
is_training = True

MAX_STEPS_TRAINING = 400
start_timesteps = 3000
noise_decay_episode = 30
batch_size = 32
init_net_action_noise = 0.2

model_name = 'ours'
if is_training == False:
    model_name = model_name + '/test'

rospack = rospkg.RosPack()
pkg_path = rospack.get_path('model_free_version')
weight_outdir = pkg_path + '/Models/' + model_name
writer = SummaryWriter(pkg_path + "/log/")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Set seeds
torch.manual_seed(666)
np.random.seed(666)
torch.cuda.manual_seed(666)
random.seed(666)


def main():
    rospy.init_node('model_free_version', log_level=rospy.WARN)
    env = Env(is_training)

    policy = TD3(S_DIM, A_DIM)
    print()
    policy.load(pkg_path + '/Models/TEST/test')

    replay_buffer = utils.ReplayBuffer(S_DIM, A_DIM)

    print('State Dimensions: ' + str(S_DIM))
    print('Action Dimensions: ' + str(A_DIM))
    print('Action Max: ' + str(action_linear_max) + ' m/s and ' + str(action_angular_max) + ' rad/s')

    total_step = 0
    save_time = 0
    episode_num = 1
    success_num = 0

    actor_loss_episode = 0
    critic_loss_episode = 0

    net_action_noise = init_net_action_noise
    action_1 = [0.0, 0.0]
    action_2 = [0.0, 0.0]
    action_3 = [0.0, 0.0]
    action_4 = [0.0, 0.0]

    episode_reward_all = 0.0

    social_safe_step_1 = 0
    social_safe_step_2 = 0
    social_safe_step_3 = 0
    social_safe_step_4 = 0

    ego_safe_step_1 = 0
    ego_safe_step_2 = 0
    ego_safe_step_3 = 0
    ego_safe_step_4 = 0

    total_step_1 = 0
    total_step_2 = 0
    total_step_3 = 0
    total_step_4 = 0

    if is_training:
        print('Training mode')

        while True:
            state_all_1, state_all_2, state_all_3, state_all_4 = env.reset()
            one_round_step = 0

            flag_stop_1 = 0
            flag_stop_2 = 0
            flag_stop_3 = 0
            flag_stop_4 = 0

            print("Training Episode ===================================>  " + str(episode_num))
            
            while True:

                if not len(state_all_1) == len(state_all_2) == len(state_all_3) == len(state_all_4) == S_DIM:
                    print(len(state_all_1))
                    print("Something Wrong with the simulator !!!")
                    break

                
                # Set action
                if total_step < start_timesteps:
                    action_1[0] = random.uniform(-max_action, max_action)
                    action_1[1] = random.uniform(-max_action, max_action)

                else:
                    net_action_1 = policy.select_action(np.array(state_all_1))
                    
                    action_1[0] = (
                        net_action_1[0]
                        + np.random.normal(0, max_action * net_action_noise, size=1)
                    ).clip(-max_action, max_action)
                    action_1[1] = (
                        net_action_1[1]
                        + np.random.normal(0, max_action * net_action_noise, size=1)
                    ).clip(-max_action, max_action)

                if total_step < start_timesteps:
                    action_2[0] = random.uniform(-max_action, max_action)
                    action_2[1] = random.uniform(-max_action, max_action)

                else:
                    net_action_2 = policy.select_action(np.array(state_all_2))
                    
                    action_2[0] = (
                        net_action_2[0]
                        + np.random.normal(0, max_action * net_action_noise, size=1)
                    ).clip(-max_action, max_action)
                    action_2[1] = (
                        net_action_2[1]
                        + np.random.normal(0, max_action * net_action_noise, size=1)
                    ).clip(-max_action, max_action)

                if total_step < start_timesteps:
                    action_3[0] = random.uniform(-max_action, max_action)
                    action_3[1] = random.uniform(-max_action, max_action)

                else:
                    net_action_3 = policy.select_action(np.array(state_all_3))
                    
                    action_3[0] = (
                        net_action_3[0]
                        + np.random.normal(0, max_action * net_action_noise, size=1)
                    ).clip(-max_action, max_action)
                    action_3[1] = (
                        net_action_3[1]
                        + np.random.normal(0, max_action * net_action_noise, size=1)
                    ).clip(-max_action, max_action)

                if total_step < start_timesteps:
                    action_4[0] = random.uniform(-max_action, max_action)
                    action_4[1] = random.uniform(-max_action, max_action)

                else:
                    net_action_4 = policy.select_action(np.array(state_all_4))
                    
                    action_4[0] = (
                        net_action_4[0]
                        + np.random.normal(0, max_action * net_action_noise, size=1)
                    ).clip(-max_action, max_action)
                    action_4[1] = (
                        net_action_4[1]
                        + np.random.normal(0, max_action * net_action_noise, size=1)
                    ).clip(-max_action, max_action)

                action_1 = np.around(action_1, decimals=5)
                action_2 = np.around(action_2, decimals=5)
                action_3 = np.around(action_3, decimals=5)
                action_4 = np.around(action_4, decimals=5)
                print(action_1)
                print(action_2)
                print(action_3)
                print(action_4)

                action = [action_1[0], action_1[1], action_2[0], action_2[1], action_3[0], action_3[1], action_4[0], action_4[1]]
                
                next_state_all_1, reward_1, done_1, arrive_1, next_state_all_2, reward_2, done_2, arrive_2, next_state_all_3, reward_3, done_3, arrive_3, next_state_all_4, reward_4, done_4, arrive_4, flag_ego_safety_1, flag_social_safety_1, flag_ego_safety_2, flag_social_safety_2, flag_ego_safety_3, flag_social_safety_3, flag_ego_safety_4, flag_social_safety_4 = env.step(action)


                writer.add_scalar('Reward/reward_1',reward_1,total_step)
                writer.add_scalar('Reward/reward_2',reward_2,total_step)
                writer.add_scalar('Reward/reward_3',reward_3,total_step)
                writer.add_scalar('Reward/reward_4',reward_4,total_step)


                t1 = time.time()
                if flag_stop_1 == 0:
                    replay_buffer.add(state_all_1, action_1, next_state_all_1, reward_1, done_1 or arrive_1, 1)
                    # replay_buffer.save_file(state_all_1, action_1, next_state_all_1, reward_1, done_1 or arrive_1, 1)
                    episode_reward_all += reward_1
                    social_safe_step_1 += flag_social_safety_1
                    ego_safe_step_1 += flag_ego_safety_1
                    total_step_1 += 1

                if flag_stop_2 == 0:
                    replay_buffer.add(state_all_2, action_2, next_state_all_2, reward_2, done_2 or arrive_2, 2)
                    # replay_buffer.save_file(state_all_2, action_2, next_state_all_2, reward_2, done_2 or arrive_2, 2)
                    episode_reward_all += reward_2
                    social_safe_step_2 += flag_social_safety_2
                    ego_safe_step_2 += flag_ego_safety_2
                    total_step_2 += 1

                if flag_stop_3 == 0:
                    replay_buffer.add(state_all_3, action_3, next_state_all_3, reward_3, done_3 or arrive_3, 3)
                    # replay_buffer.save_file(state_all_3, action_3, next_state_all_3, reward_3, done_3 or arrive_3, 3)
                    episode_reward_all += reward_3
                    social_safe_step_3 += flag_social_safety_3
                    ego_safe_step_3 += flag_ego_safety_3
                    total_step_3 += 1

                if flag_stop_4 == 0:
                    replay_buffer.add(state_all_4, action_4, next_state_all_4, reward_4, done_4 or arrive_4, 4)
                    # replay_buffer.save_file(state_all_4, action_4, next_state_all_4, reward_4, done_4 or arrive_4, 4)
                    episode_reward_all += reward_4
                    social_safe_step_4 += flag_social_safety_4
                    ego_safe_step_4 += flag_ego_safety_4
                    total_step_4 += 1
                t2 = time.time()
                # print("Save Time : {}  ms".format(round(1000*(t2-t1),2)))


                state_all_1 = next_state_all_1[:]
                state_all_2 = next_state_all_2[:]
                state_all_3 = next_state_all_3[:]
                state_all_4 = next_state_all_4[:]


                if total_step >= start_timesteps:
                    print("TRAIN   step   :", total_step - start_timesteps)
                    t1 = time.time()
                    actor_loss_episode, critic_loss_episode = policy.train(replay_buffer, batch_size)
                    t2 = time.time()
                    print("TRAIN Time : {}  ms".format(round(1000*(t2-t1),2)))

                    writer.add_scalar('Training/actor_loss_episode', actor_loss_episode, total_step - start_timesteps)
                    writer.add_scalar('Training/critic_loss_episode', critic_loss_episode, total_step - start_timesteps)


                one_round_step += 1
                total_step += 1


                if arrive_1:
                    result_1 = 'Success'
                else:
                    result_1 = 'Fail'

                if arrive_2:
                    result_2 = 'Success'
                else:
                    result_2 = 'Fail'

                if arrive_3:
                    result_3 = 'Success'
                else:
                    result_3 = 'Fail'

                if arrive_4:
                    result_4 = 'Success'
                else:
                    result_4 = 'Fail'

            

                if (arrive_1 or done_1) and flag_stop_1 == 0:
                    print('Agent 1  : Step: %4i' % one_round_step,  '| Time step: %i' % total_step, '|', result_1)

                    flag_stop_1 = 1
                    if arrive_1:
                        success_num += 1
                
                if (arrive_2 or done_2) and flag_stop_2 == 0:
                    print('Agent 2  : Step: %4i' % one_round_step,  '| Time step: %i' % total_step, '|', result_2)

                    flag_stop_2 = 1
                    if arrive_2:
                        success_num += 1

                if (arrive_3 or done_3) and flag_stop_3 == 0:
                    print('Agent 3  : Step: %4i' % one_round_step,  '| Time step: %i' % total_step, '|', result_3)

                    flag_stop_3 = 1
                    if arrive_3:
                        success_num += 1

                if (arrive_4 or done_4) and flag_stop_4 == 0:
                    print('Agent 4  : Step: %4i' % one_round_step,  '| Time step: %i' % total_step, '|', result_4)

                    flag_stop_4 = 1
                    if arrive_4:
                        success_num += 1
                    
             

                if ((arrive_1 or done_1) and (arrive_2 or done_2) and (arrive_3 or done_3) and (arrive_4 or done_4)) or one_round_step >= MAX_STEPS_TRAINING:
                    writer.add_scalar('Criteria/episode_reward_all', episode_reward_all, episode_num)
                    writer.add_scalar('Criteria/success_rate', success_num / episode_num / 4, episode_num)
                    writer.add_scalar('Criteria/average_step', total_step / episode_num, episode_num)

                    writer.add_scalar('Social_score/1', social_safe_step_1 / total_step_1, episode_num)
                    writer.add_scalar('Social_score/2', social_safe_step_2 / total_step_2, episode_num)
                    writer.add_scalar('Social_score/3', social_safe_step_3 / total_step_3, episode_num)
                    writer.add_scalar('Social_score/4', social_safe_step_4 / total_step_4, episode_num)

                    writer.add_scalar('Ego_score/1', ego_safe_step_1 / total_step_1, episode_num)
                    writer.add_scalar('Ego_score/2', ego_safe_step_2 / total_step_2, episode_num)
                    writer.add_scalar('Ego_score/3', ego_safe_step_3 / total_step_3, episode_num)
                    writer.add_scalar('Ego_score/4', ego_safe_step_4 / total_step_4, episode_num)


                    print('All Agents DONE !!! : Step: %4i' % one_round_step,  '| Time step: %i' % total_step, '|')
                    print('net_action_noise: %4f' % net_action_noise)

                    if episode_num % 15 == 0:
                        policy.save(weight_outdir + '/' + str(save_time))
                        save_time += 1

                    if total_step > start_timesteps:
                        policy.scheduler_actor.step()
                        policy.scheduler_critic.step()
                    
                    writer.add_scalar('Learning Rate', policy.actor_optimizer.param_groups[0]['lr'], episode_num)

                    episode_reward_all = 0.0
                    episode_num += 1

                    # noise 衰减
                    if episode_num > noise_decay_episode:
                        net_action_noise = net_action_noise * 0.999
                    break

    else:
        print('Testing mode')
        while True:
            state_all_1, state_all_2, state_all_3, state_all_4 = env.reset()
            one_round_step = 0

            flag_stop_1 = 0
            flag_stop_2 = 0
            flag_stop_3 = 0
            flag_stop_4 = 0

            print("Training Episode ===================================>  " + str(episode_num))
            
            while True:

                if not len(state_all_1) == len(state_all_2) == len(state_all_3) == len(state_all_4) == S_DIM:
                    print(len(state_all_1))
                    print("Something Wrong with the simulator !!!")
                    break

            
                # Set action
                net_action_1 = policy.select_action(np.array(state_all_1))
                net_action_2 = policy.select_action(np.array(state_all_2))
                net_action_3 = policy.select_action(np.array(state_all_3))
                net_action_4 = policy.select_action(np.array(state_all_4))

                action = [net_action_1[0], net_action_1[1], net_action_2[0], net_action_2[1], net_action_3[0], net_action_3[1], net_action_4[0], net_action_4[1]]
                
                next_state_all_1, reward_1, done_1, arrive_1, next_state_all_2, reward_2, done_2, arrive_2, next_state_all_3, reward_3, done_3, arrive_3, next_state_all_4, reward_4, done_4, arrive_4, flag_ego_safety_1, flag_social_safety_1, flag_ego_safety_2, flag_social_safety_2, flag_ego_safety_3, flag_social_safety_3, flag_ego_safety_4, flag_social_safety_4 = env.step(action)

                writer.add_scalar('Reward/reward_1',reward_1,total_step)
                writer.add_scalar('Reward/reward_2',reward_2,total_step)
                writer.add_scalar('Reward/reward_3',reward_3,total_step)
                writer.add_scalar('Reward/reward_4',reward_4,total_step)

                t1 = time.time()
                if flag_stop_1 == 0:
                    # replay_buffer.save_file(state_all_1, net_action_1, next_state_all_1, reward_1, done_1 or arrive_1, 1)
                    episode_reward_all += reward_1
                    social_safe_step_1 += flag_social_safety_1
                    ego_safe_step_1 += flag_ego_safety_1
                    total_step_1 += 1

                if flag_stop_2 == 0:
                    # replay_buffer.save_file(state_all_2, net_action_2, next_state_all_2, reward_2, done_2 or arrive_2, 2)
                    episode_reward_all += reward_2
                    social_safe_step_2 += flag_social_safety_2
                    ego_safe_step_2 += flag_ego_safety_2
                    total_step_2 += 1

                if flag_stop_3 == 0:
                    # replay_buffer.save_file(state_all_3, net_action_3, next_state_all_3, reward_3, done_3 or arrive_3, 3)
                    episode_reward_all += reward_3
                    social_safe_step_3 += flag_social_safety_3
                    ego_safe_step_3 += flag_ego_safety_3
                    total_step_3 += 1

                if flag_stop_4 == 0:
                    # replay_buffer.save_file(state_all_4, net_action_4, next_state_all_4, reward_4, done_4 or arrive_4, 4)
                    episode_reward_all += reward_4
                    social_safe_step_4 += flag_social_safety_4
                    ego_safe_step_4 += flag_ego_safety_4
                    total_step_4 += 1
                t2 = time.time()
                # print("Save Time : {}  ms".format(round(1000*(t2-t1),2)))

                state_all_1 = next_state_all_1[:]
                state_all_2 = next_state_all_2[:]
                state_all_3 = next_state_all_3[:]
                state_all_4 = next_state_all_4[:]

                one_round_step += 1
                total_step += 1

                if arrive_1:
                    result_1 = 'Success'
                else:
                    result_1 = 'Fail'

                if arrive_2:
                    result_2 = 'Success'
                else:
                    result_2 = 'Fail'

                if arrive_3:
                    result_3 = 'Success'
                else:
                    result_3 = 'Fail'

                if arrive_4:
                    result_4 = 'Success'
                else:
                    result_4 = 'Fail'

                if (arrive_1 or done_1) and flag_stop_1 == 0:
                    print('Agent 1  : Step: %4i' % one_round_step,  '| Time step: %i' % total_step, '|', result_1)

                    flag_stop_1 = 1
                    if arrive_1:
                        success_num += 1
                
                if (arrive_2 or done_2) and flag_stop_2 == 0:
                    print('Agent 2  : Step: %4i' % one_round_step,  '| Time step: %i' % total_step, '|', result_2)

                    flag_stop_2 = 1
                    if arrive_2:
                        success_num += 1

                if (arrive_3 or done_3) and flag_stop_3 == 0:
                    print('Agent 3  : Step: %4i' % one_round_step,  '| Time step: %i' % total_step, '|', result_3)

                    flag_stop_3 = 1
                    if arrive_3:
                        success_num += 1

                if (arrive_4 or done_4) and flag_stop_4 == 0:
                    print('Agent 4  : Step: %4i' % one_round_step,  '| Time step: %i' % total_step, '|', result_4)

                    flag_stop_4 = 1
                    if arrive_4:
                        success_num += 1
                    

                if (arrive_1 or done_1) and (arrive_2 or done_2) and (arrive_3 or done_3) and (arrive_4 or done_4) or one_round_step >= MAX_STEPS_TRAINING:
                    writer.add_scalar('Criteria/episode_reward_all', episode_reward_all, episode_num)
                    writer.add_scalar('Criteria/success_rate', success_num / episode_num / 4, episode_num)
                    writer.add_scalar('Criteria/average_step', total_step / episode_num, episode_num)

                    writer.add_scalar('Social_score/1', social_safe_step_1 / total_step_1, episode_num)
                    writer.add_scalar('Social_score/2', social_safe_step_2 / total_step_2, episode_num)
                    writer.add_scalar('Social_score/3', social_safe_step_3 / total_step_3, episode_num)
                    writer.add_scalar('Social_score/4', social_safe_step_4 / total_step_4, episode_num)

                    writer.add_scalar('Ego_score/1', ego_safe_step_1 / total_step_1, episode_num)
                    writer.add_scalar('Ego_score/2', ego_safe_step_2 / total_step_2, episode_num)
                    writer.add_scalar('Ego_score/3', ego_safe_step_3 / total_step_3, episode_num)
                    writer.add_scalar('Ego_score/4', ego_safe_step_4 / total_step_4, episode_num)
                    print('All Agents DONE !!! : Step: %4i' % one_round_step,  '| Time step: %i' % total_step, '|')

                    episode_reward_all = 0.0
                    episode_num += 1

                    break

if __name__ == '__main__':
     main()