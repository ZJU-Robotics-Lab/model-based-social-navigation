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

from environment_one_agent import Env
from env_sample import EnvSampler

from replay_buffer_env import ReplayBuffer_Env



S_DIM = 3634
A_DIM = 2

max_action = 1.0
is_training = True

REPLAY_SIZE = 1e5
MAX_STEPS_TRAINING = 400

# POLICY UPDATE ITERATION
# NUM_TRAIN_REPEAT = 20 
# NUM_TRAIN_REPEAT = 10
NUM_TRAIN_REPEAT = 1

start_timesteps = 1000
noise_decay_episode = 30
batch_size = 32
init_net_action_noise = 0.2


rospack = rospkg.RosPack()
pkg_path = rospack.get_path('model_based_version')
weight_outdir = pkg_path + '/Models'
writer = SummaryWriter(pkg_path + "/log")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Set seeds
torch.manual_seed(666) # 666
np.random.seed(666)
torch.cuda.manual_seed(666)
random.seed(666)


def main():
    rospy.init_node('model_based_version', log_level=rospy.WARN)
    env = Env(is_training)
    env_sampler = EnvSampler(env, max_path_length=MAX_STEPS_TRAINING, start_timesteps=start_timesteps)
    policy = TD3(S_DIM, A_DIM)
    # policy.load(pkg_path + '/Models/TEST/test')
    env_pool = ReplayBuffer_Env(REPLAY_SIZE)


    total_step = 0
    train_step = 0
    save_time = 0
    episode_num = 1
    success_num = 0

    actor_loss_episode = 0
    critic_loss_episode = 0

    net_action_noise = init_net_action_noise
    action_1 = [0.0, 0.0]

    episode_reward_all = 0.0
    sum_reward = 0.0

    if is_training:
        print('Training mode')

        while True:

            one_round_step = 0
            flag_stop_1 = 0

            print("Training Episode ===================================>  " + str(episode_num))
            
            while True:
                state_all_1, next_state_all_1, action_1, reward_1, done_1 = env_sampler.sample(policy)
                sum_reward += reward_1
                writer.add_scalar('Reward/reward', reward_1, total_step)
                writer.add_scalar('Reward/average_reward', sum_reward / (total_step + 1), total_step)

                t1 = time.time()
                if flag_stop_1 == 0:
                    env_pool.push(state_all_1, next_state_all_1, action_1, reward_1, done_1)
                    episode_reward_all += reward_1
                t2 = time.time()

                if total_step >= start_timesteps:
                    print("TRAIN   step   :", total_step - start_timesteps)
                    t1 = time.time()

                    for i in range(NUM_TRAIN_REPEAT):
                        env_state, env_next_state, env_action, env_reward, env_done = env_pool.sample(int(batch_size))
                        env_done = (~env_done).astype(int) # done => not_done
                        env_image = np.zeros((batch_size, 2, 80, 160))
                        env_transformation = np.zeros((batch_size, 2, 3))
                        env_num_pre = np.zeros((batch_size, 1))
                        env_next_image = np.zeros((batch_size, 2, 80, 160))
                        env_next_transformation = np.zeros((batch_size, 2, 3))
                        env_next_num_pre = np.zeros((batch_size, 1))
                        actor_loss_episode, critic_loss_episode = policy.train(( \
                            env_state, env_image, env_transformation, env_num_pre, \
                            env_next_state, env_next_image, env_next_transformation, env_next_num_pre, \
                                env_action, env_reward, env_done))
                        train_step += 1
                        writer.add_scalar('Policy/actor_loss', actor_loss_episode, train_step)
                        writer.add_scalar('Policy/critic_loss', critic_loss_episode, train_step)

                    t2 = time.time()
                    print("TRAIN Time : {}  ms".format(round(1000*(t2-t1),2)))


                one_round_step += 1
                total_step += 1

                if done_1 and flag_stop_1 == 0:
                    print('Agent 1  : Step: %4i' % one_round_step,  '| Time step: %i' % total_step)
                    if reward_1 > 10:
                        print('Success !!!')
                        success_num += 1
                    else:
                        print('Collision !!!')

                    flag_stop_1 = 1

                if done_1 or one_round_step >= MAX_STEPS_TRAINING:
                    writer.add_scalar('Criteria/episode_reward', episode_reward_all, episode_num)
                    # writer.add_scalar('Criteria/success_rate', success_num / episode_num, episode_num)
                    writer.add_scalar('Criteria/average_step', total_step / episode_num, episode_num)

                    print('All Agents DONE !!! : Step: %4i' % one_round_step,  '| Time step: %i' % total_step, '|')
                    print('net_action_noise: %4f' % net_action_noise)

                    if episode_num % 30 == 0:
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
            state_all_1 = env.reset()
            one_round_step = 0

            flag_stop_1 = 0

            print("Training Episode ===================================>  " + str(episode_num))
            
            while True:

                # Set action
                net_action_1 = policy.select_action(np.array(state_all_1))

                action = [net_action_1[0], net_action_1[1]]
                
                next_state_all_1, reward_1, done_1 = env.step(action)

                writer.add_scalar('Reward/reward',reward_1,total_step)

                if flag_stop_1 == 0:
                    episode_reward_all += reward_1

                state_all_1 = next_state_all_1[:]

                one_round_step += 1
                total_step += 1

                if done_1 and flag_stop_1 == 0:
                    print('Agent 1  : Step: %4i' % one_round_step,  '| Time step: %i' % total_step, '|', result_1)

                    flag_stop_1 = 1
                    if arrive_1:
                        success_num += 1
                    
                if done_1 or one_round_step >= MAX_STEPS_TRAINING:
                    writer.add_scalar('Criteria/episode_reward', episode_reward_all, episode_num)
                    writer.add_scalar('Criteria/success_rate', success_num / episode_num, episode_num)
                    writer.add_scalar('Criteria/average_step', total_step / episode_num, episode_num)

                    print('All Agents DONE !!! : Step: %4i' % one_round_step,  '| Time step: %i' % total_step, '|')

                    episode_reward_all = 0.0
                    episode_num += 1

                    break

if __name__ == '__main__':
     main()