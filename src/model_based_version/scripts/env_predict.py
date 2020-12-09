# -*- coding: utf-8 -*- 
import numpy as np
from ensemble_model import Ensemble_Model
import cv2

COLLISION_RANGE = 0.3
COLLISION_NUM = 10
ARRIVAL_RANGE = 0.48

class PredictEnv:
    def __init__(self, model):
        self.model = model

    def _termination_fn(self, image, goal):

        collision = False
        arrival = False

        num_in_range = np.sum(image[:, 75:80, 75:85], axis=(1,2))
        collision = num_in_range > COLLISION_NUM
        # print("collision : ", collision.shape)

        goal_distance = goal[:, 0] * 25
        arrival = goal_distance < ARRIVAL_RANGE
        # print("arrival : ", arrival.shape)

        return collision, arrival, collision | arrival

    def step(self, states, images, transformations, num_pre, action, deterministic=False):

        # Predict 预测
        ensemble_out_before, ensemble_out, ensemble_transformation_out, ensemble_vel_ang_out, ensemble_goal_out, ensemble_reward = \
            self.model.predict(states, images, transformations, num_pre, action, batch_size=32)

        num_models, batch_size, _, _ = ensemble_out.shape

        # model_idxes = np.random.choice(self.model.elite_model_idxes, size=batch_size)
        model_idxes = 0
        batch_idxes = np.arange(0, batch_size)

        # =========================         costmap         =========================
        model_out = ensemble_out[model_idxes, batch_idxes]
        model_out_before = ensemble_out_before[model_idxes, batch_idxes]

        # ========================= vel_ang & goal & reward =========================
        model_transformation = ensemble_transformation_out[model_idxes, batch_idxes]
        model_vel_ang = ensemble_vel_ang_out[model_idxes, batch_idxes]
        model_goal = ensemble_goal_out[model_idxes, batch_idxes]
        model_reward = ensemble_reward[model_idxes, batch_idxes]

        next_states = states.copy()

        for i in range(states.shape[0]):
            next_states[i, 3630:3632] = model_vel_ang[i][:]
            next_states[i, 3632:3634] = model_goal[i][:]

        images_pre = images.copy()
        transformations_pre = transformations.copy()
        for i in range(states.shape[0]):

            images_pre[i, num_pre[i,0], :, :] = model_out[i][:]
            big = images_pre[i, num_pre[i,0]] > 0.36
            small = images_pre[i, num_pre[i,0]] < 0.36
            images_pre[i, num_pre[i,0]][big] = 1
            images_pre[i, num_pre[i,0]][small] = 0
            images_pre[i, num_pre[i,0]] = cv2.erode(images_pre[i, num_pre[i,0]], np.ones((3,3), np.uint8), iterations=1)
            images_pre[i, num_pre[i,0]] = cv2.dilate(images_pre[i, num_pre[i,0]], np.ones((3,3), np.uint8), iterations=1)
            # cv2.imshow('pre_after', images_pre[i, num_pre[i,0]])
            # cv2.waitKey(0)
            
            transformations_pre[i, num_pre[i,0], :] = model_transformation[i][:]

            
        num_pre = num_pre + 1
        rewards = model_reward

        # 判断 terminal 修正 reward
        collision, arrival, terminals = self._termination_fn(model_out, model_goal)

        rewards[collision] = -20
        rewards[arrival] = 20

        info = {}

        return next_states, images_pre, transformations_pre, num_pre, rewards, terminals, info


def main():
    print("try")
    models = Ensemble_Model(3,2)
    env_pre = PredictEnv(models)
    
    test_input = np.random.randn(2, 3635)
    test_actions = np.random.randn(2, 2)
    test_images = np.random.randn(2, 2, 80, 160)
    test_numpre = np.array([[1], [0]], dtype=np.int)
    test_state_labels = np.random.randn(2, 3635)
    test_reward_labels = np.random.randn(2, 1)

    models.train(test_input, test_images, test_numpre, test_actions, test_state_labels, test_reward_labels, batch_size=1)
    env_pre.step(test_input, test_images, test_numpre, test_actions)



if __name__ == '__main__':
    main()


