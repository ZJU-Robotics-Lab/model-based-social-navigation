#!/usr/bin/env python3
# -*- coding: utf-8 -*- 


# MBPO_Multi_Agent_Social_Navigation

import time
import gym
import torch
import numpy as np
from itertools import count
import os, glob

import rospy
import rospkg
import random

from torch.utils.tensorboard import SummaryWriter

from policy import TD3
from replay_buffer_env import ReplayBuffer_Env
from replay_buffer_model import ReplayBuffer_Model

from ensemble_model import Ensemble_Model
from env_predict import PredictEnv
from env_sample import EnvSampler

from environment_one_agent import Env

env_name = "Multi_Agent_Social_Navigation_Gazebo"

# training parameters
NUM_EPOCH = 1000 # 总训练epoch数
EPOCH_LENGTH = 1000 # epoch步长
TEST_FREQUENCY = 1e5 # 每多少步policy更新，测试一次

BATCH_SIZE_POLICY = 32
BATCH_SIZE_MODEL = 16
BATCH_SIZE_ROLLOUT = 256

# REAL_RATIO = 1 # model free
REAL_RATIO = 0.1
REPLAY_SIZE = 2e4

INIT_EXPLORATION_STEPS = 1000 # 初始探索步数
MIN_POOL_SIZE = INIT_EXPLORATION_STEPS
ROLLOUT_ROUND_SIZE = 1000 # 单次rollout数目

MODEL_TRAIN_FREQUENCY = 250 # 每多少步policy更新，训练一次模型
NUM_TRAIN_REPEAT = 20  # 一个step中policy训练次数

# bootstrap ensemble
NUM_NET = 1
NUM_ELITES = 1
MODEL_RETAIN_EPOCH = 1 # 模型保留周期

ROLL_OUT_MIN_LENGTH = 1
ROLL_OUT_MAX_LENGTH = 2 # 展开长度
ROLL_OUT_MIN_EPOCH = 1 
ROLL_OUT_MAX_EPOCH = 50 # 展开变化区间

rospack = rospkg.RosPack()
pkg_path = rospack.get_path('model_based_version')
weight_outdir = pkg_path + '/Models/'
world_models_weight_outdir = pkg_path + '/World_Models'
writer = SummaryWriter(pkg_path + "/log/")

model_step = 0
eval_step = 0
policy_step = 0
reward_sum = 0
env_episode_num = 0
total_env_step = 0
episode_reward = 0

flag_stop_1 = 0

def train(env_sampler, predict_env, agent, env_pool, model_pool):
    global reward_sum
    global env_episode_num
    global total_env_step
    global episode_reward

    global flag_stop_1

    total_step = 0
    test_num = 0
    rollout_length = 1
    save_time = 0

    exploration_before_start(env_sampler, env_pool, agent)
    train_predict_model(env_pool, predict_env, True)

    for epoch_step in range(NUM_EPOCH):
        print("====================================>  EPOCH  <====================================   ", epoch_step)
        start_step = total_step

        for i in count():
            cur_step = total_step - start_step
            print("cur_step", cur_step)

            # 一个epoch最大步长
            if cur_step > EPOCH_LENGTH and len(env_pool) > MIN_POOL_SIZE:
                break
            
            # 每隔 MODEL_TRAIN_FREQUENCY 训练模型
            if cur_step > 0 and cur_step % int(MODEL_TRAIN_FREQUENCY * 1) == 0:
            # if cur_step % MODEL_TRAIN_FREQUENCY == 0:
                train_predict_model(env_pool, predict_env) #　训练模型只需要真实env数据

                # 调整rollout长度
                new_rollout_length = set_rollout_length(epoch_step)
                if rollout_length != new_rollout_length:
                    rollout_length = new_rollout_length
                
                rollout_model(predict_env, agent, model_pool, env_pool, rollout_length)

            #　真实环境step 填充真实环境数据集
            cur_state_1, next_state_1, action_1, reward_1, done_1 = env_sampler.sample(agent)

            if flag_stop_1 == 0:
                env_pool.push(cur_state_1, next_state_1, action_1, reward_1, done_1)

            total_env_step += 1
            reward_sum += reward_1
            episode_reward += reward_1

            writer.add_scalar('Reward/reward', reward_1, total_env_step)
            writer.add_scalar('Reward/average_reward', reward_sum / total_env_step, total_env_step)

            if done_1 and flag_stop_1 == 0:
                flag_stop_1 = 1

            if done_1:
                env_episode_num += 1
                writer.add_scalar('Criteria/episode_reward', episode_reward, env_episode_num)
                writer.add_scalar('Criteria/average_step', total_env_step / env_episode_num, env_episode_num)
                episode_reward = 0
                flag_stop_1 = 0
                    
            # 在合并数据集上，重复训练policy NUM_TRAIN_REPEAT 次
            if len(env_pool) > MIN_POOL_SIZE:
                train_policy_repeats(env_pool, model_pool, agent)

            total_step += 1


# 初始探索 √
def exploration_before_start(env_sampler, env_pool, agent):
    print("======================>  Exploration  <======================")
    global reward_sum
    global env_episode_num
    global episode_reward
    global total_env_step

    global flag_stop_1
    
    for i in range(INIT_EXPLORATION_STEPS):
        #　真实环境step 填充真实环境数据集
        cur_state_1, next_state_1, action_1, reward_1, done_1 = env_sampler.sample(agent)

        if flag_stop_1 == 0:
            env_pool.push(cur_state_1, next_state_1, action_1, reward_1, done_1)

        total_env_step += 1
        reward_sum += reward_1
        episode_reward += reward_1

        writer.add_scalar('Reward/reward', reward_1, total_env_step)
        writer.add_scalar('Reward/average_reward', reward_sum / total_env_step, total_env_step)

        if done_1 and flag_stop_1 == 0:
            flag_stop_1 = 1

        if done_1:
            env_episode_num += 1
            writer.add_scalar('Criteria/episode_reward', episode_reward, env_episode_num)
            writer.add_scalar('Criteria/average_step', total_env_step / env_episode_num, env_episode_num)
            episode_reward = 0

            flag_stop_1 = 0
            
    
    print("======================>  Exploration Done  <======================")


# 根据当前的epoch数设定rollout长度，随着训练进行，逐渐增长 √
def set_rollout_length(epoch_step):
    rollout_length = (min(max(ROLL_OUT_MIN_LENGTH + (epoch_step - ROLL_OUT_MIN_EPOCH)
        / (ROLL_OUT_MAX_EPOCH - ROLL_OUT_MIN_EPOCH) * (ROLL_OUT_MAX_LENGTH - ROLL_OUT_MIN_LENGTH),
        ROLL_OUT_MIN_LENGTH), ROLL_OUT_MAX_EPOCH))
    return int(rollout_length)


# 训练环境预测模型 √
def train_predict_model(env_pool, predict_env, flag=False):
    print("========>  Model Train  <========  ")
    global model_step
    global eval_step

    if flag==True:
        model_train_num = 3
    else:
        model_train_num = 1

    for i in range(model_train_num):
        
        t1 = time.time()
        state, next_state, action, reward, done = env_pool.sample(len(env_pool))

        # 筛除done reward，否则不收敛
        done_state = (reward < 2) & (reward > -2)
        state = state[done_state]
        next_state = next_state[done_state]
        action = action[done_state]
        reward = reward[done_state]
        done = done[done_state]

        loss_G, content_with_prediction, out, all_label, label_with_prediction, \
            acc_vel, acc_ang, acc_goal_direction, acc_goal_distance, acc_reward = \
            predict_env.model.train(state, action, next_state, reward, batch_size=BATCH_SIZE_MODEL)

        t2 = time.time()
        print("TRAIN Time : {}  ms".format(round(1000*(t2-t1),2)))
        print("========>  Model Train Done  <========  ", acc_reward)
        print("========>  env_pool_size  <========  ", len(env_pool))

        # 针对于当前真实数据上的准确率
        writer.add_scalar('Model/loss_G', loss_G, model_step)
        writer.add_image('cost_map content_with_prediction', content_with_prediction, model_step)
        writer.add_image('cost_map', out, model_step)
        writer.add_image('cost_map label_with_prediction', label_with_prediction, model_step)
        writer.add_image('cost_map all label', all_label, model_step)

        writer.add_scalar('Train/acc_vel', acc_vel, model_step)
        writer.add_scalar('Train/acc_ang', acc_ang, model_step)
        writer.add_scalar('Train/acc_goal_direction', acc_goal_direction, model_step)
        writer.add_scalar('Train/acc_goal_distance', acc_goal_distance, model_step)
        writer.add_scalar('Train/acc_reward', acc_reward, model_step)
        model_step += 1
    
        
    # evaluate
    state, next_state, action, reward, done = env_pool.sample(len(env_pool))
    # 筛除done reward
    done_state = (reward < 2) & (reward > -2)
    state = state[done_state]
    next_state = next_state[done_state]
    action = action[done_state]
    reward = reward[done_state]


    acc_vel, acc_ang, acc_goal_direction, acc_goal_distance, acc_reward = predict_env.model.evaluate(state, action, next_state, reward, batch_size=BATCH_SIZE_MODEL)
    eval_step += 1
    writer.add_scalar('Evaluate/acc_vel', acc_vel, eval_step)
    writer.add_scalar('Evaluate/acc_ang', acc_ang, eval_step)
    writer.add_scalar('Evaluate/acc_goal_direction', acc_goal_direction, eval_step)
    writer.add_scalar('Evaluate/acc_goal_distance', acc_goal_distance, eval_step)
    writer.add_scalar('Evaluate/acc_reward', acc_reward, eval_step)


# 根据model进行rollout填充模型数据集 √
def rollout_model(predict_env, agent, model_pool, env_pool, rollout_length):
    print("========>  Rollout Model  <========")
    # state_all, next_state_all, action_all, reward_all, done_all = env_pool.sample_all_batch(ROLLOUT_ROUND_SIZE)
    state_all, next_state_all, action_all, reward_all, done_all = env_pool.sample(len(env_pool))

    #Get a batch of actions
    for m in range(0, state_all.shape[0], BATCH_SIZE_ROLLOUT):

        state = state_all[m:min(m + BATCH_SIZE_ROLLOUT, state_all.shape[0])]
        action = action_all[m:min(m + BATCH_SIZE_ROLLOUT, state_all.shape[0])]

        images = np.zeros((state.shape[0], 2, 80, 160))
        transformations = np.zeros((state.shape[0], 2, 3))
        num_pre = np.zeros((state.shape[0], 1), dtype=np.int)
        
        for i in range(rollout_length):

            action = agent.select_action_batch(state, images, transformations, num_pre)
            next_state, next_images, next_transformations, next_num_pre, rewards, terminals, info = predict_env.step(state, images, transformations, num_pre, action)

            #Push a batch of samples
            model_pool.push_batch([(state[j], images[j], transformations[j], num_pre[j], \
                next_state[j], next_images[j], next_transformations[j], next_num_pre[j], \
                    action[j], rewards[j], terminals[j]) for j in range(state.shape[0])])

            nonterm_mask = ~terminals
            if nonterm_mask.sum() == 0:
                break
            state = next_state[nonterm_mask]
            images = next_images[nonterm_mask]
            transformations = next_transformations[nonterm_mask]
            num_pre = next_num_pre[nonterm_mask]
        
        print("rollout length : ", m)

    print("========>  Rollout Model Done  <========")

# 连续训练policy多步 √
def train_policy_repeats(env_pool, model_pool, agent):
    print("========>  Policy Train  <========")
    global policy_step
    t1 = time.time()
    for i in range(NUM_TRAIN_REPEAT):
        
        env_batch_size = int(BATCH_SIZE_POLICY * REAL_RATIO) # policy训练数据batch中真实数据数量
        model_batch_size = BATCH_SIZE_POLICY - env_batch_size # policy训练数据batch中模型生成数据数量

        if model_batch_size > 0 and len(model_pool) > 0:
            env_state, env_next_state, env_action, env_reward, env_done = env_pool.sample(int(env_batch_size)) # 采集指定数目数据

            model_state, model_image, model_transformation, model_num_pre, \
                model_next_state, model_next_image, model_next_transformation, model_next_num_pre, \
                    model_action, model_reward, model_done = model_pool.sample(int(model_batch_size)) 

            # 拼接数据
            batch_state, batch_image, batch_transformation, batch_num_pre, \
                batch_next_state, batch_next_image, batch_next_transformation, batch_next_num_pre, \
                    batch_action, batch_reward, batch_done = \
                np.concatenate((env_state, model_state), axis=0), \
                np.concatenate((np.zeros((env_batch_size, 2, 80, 160)), model_image), axis=0), \
                np.concatenate((np.zeros((env_batch_size, 2, 3)), model_transformation), axis=0), \
                np.concatenate((np.zeros((env_batch_size, 1), dtype=np.int), model_num_pre), axis=0), \
                np.concatenate((env_next_state, model_next_state), axis=0), \
                np.concatenate((np.zeros((env_batch_size, 2, 80, 160)), model_next_image), axis=0), \
                np.concatenate((np.zeros((env_batch_size, 2, 3)), model_next_transformation), axis=0), \
                np.concatenate((np.zeros((env_batch_size, 1), dtype=np.int), model_next_num_pre), axis=0), \
                np.concatenate((env_action, model_action), axis=0), \
                np.concatenate((env_reward, model_reward), axis=0), \
                np.concatenate((env_done, model_done), axis=0)
        else:
            env_state, env_next_state, env_action, env_reward, env_done = env_pool.sample(BATCH_SIZE_POLICY) # 采集指定数目数据

            batch_state, batch_image, batch_transformation, batch_num_pre, \
                batch_next_state, batch_next_image, batch_next_transformation, batch_next_num_pre, \
                    batch_action, batch_reward, batch_done = \
                env_state, \
                np.zeros((BATCH_SIZE_POLICY, 2, 80, 160)), \
                np.zeros((BATCH_SIZE_POLICY, 2, 3)), \
                np.zeros((BATCH_SIZE_POLICY, 1), dtype=np.int), \
                env_next_state, \
                np.zeros((BATCH_SIZE_POLICY, 2, 80, 160)), \
                np.zeros((BATCH_SIZE_POLICY, 2, 3)), \
                np.zeros((BATCH_SIZE_POLICY, 1), dtype=np.int), \
                env_action, \
                env_reward, \
                env_done

        batch_done = (~batch_done).astype(int) # done => not_done

        actor_loss_viz, critic_loss_viz = agent.train((\
            batch_state, batch_image, batch_transformation, batch_num_pre, \
                batch_next_state, batch_next_image, batch_next_transformation, batch_next_num_pre, \
                    batch_action, batch_reward, batch_done))

        writer.add_scalar('Policy/actor_loss',actor_loss_viz,policy_step)
        writer.add_scalar('Policy/critic_loss',critic_loss_viz,policy_step)
        policy_step += 1

    t2 = time.time()
    print("TRAIN Time : {}  ms".format(round(1000*(t2-t1),2)))
    print("========>  Policy Train Done <========")




S_DIM = 3634
A_DIM = 2

max_action = 1.0
is_training = True


def main():

    # Initial environment
    rospy.init_node('model_based_policy_gradient', log_level=rospy.WARN)
    env = Env(is_training)

    # Set random seed
    torch.manual_seed(666)
    np.random.seed(666)
    torch.cuda.manual_seed(666)
    random.seed(666)

    # Intial policy
    policy = TD3(S_DIM, A_DIM)

    # Initial ensemble model
    env_model = Ensemble_Model(NUM_NET, NUM_ELITES, S_DIM, A_DIM)
    env_model.load("/home/cyx/model-based-social-navigation/src/model_based_version/World_Models/TEST/world_model")

    predict_env = PredictEnv(env_model)

    env_pool = ReplayBuffer_Env(REPLAY_SIZE)
    model_pool = ReplayBuffer_Model(REPLAY_SIZE)

    # Sampler of environment
    env_sampler = EnvSampler(env, start_timesteps=INIT_EXPLORATION_STEPS)

    train(env_sampler, predict_env, policy, env_pool, model_pool)


if __name__ == '__main__':
    main()
