import numpy as np
import torch

from gym import wrappers
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.autograd import Variable
import torchvision.transforms as transforms
import random


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(150000)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0
		self.save_time = 0

		self.state = np.zeros((max_size, state_dim))
		# self.state = np.zeros((max_size, state_dim[0], state_dim[1]))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		# self.next_state = np.zeros((max_size, state_dim[0], state_dim[1]))

		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done, num):


		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done
		

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

		if self.size % 500 == 0 and self.size != self.max_size:
			print("=========================>  replay buffer size :  {}".format(self.size))
			

	def save_file(self, state, action, next_state, reward, done, num):

		# ==========================>  save data  <==========================
		not_done = 1. - done
		if not_done == 1:
			# state
			temp_state = state[:]
			temp_state_array = np.asarray(temp_state)
			np.save('../dataset/state/'  + str(self.save_time).zfill(7), temp_state_array)
			

			# action
			temp_action = action[:]
			temp_action_array = np.asarray(temp_action)
			np.save('../dataset/action/' + str(self.save_time).zfill(7), temp_action_array)


			# next state
			temp_next_state = next_state[:]
			temp_next_state_array = np.asarray(temp_next_state)
			np.save('../dataset/next_state/'  + str(self.save_time).zfill(7), temp_next_state_array)


			# reward
			temp_reward = reward
			temp_reward_array = np.asarray(temp_reward)
			np.save('../dataset/reward/'  + str(self.save_time).zfill(7), temp_reward_array)

		# not_done
		# temp_not_done = done
		# temp_not_done_array = np.asarray(temp_not_done)
		# np.save('../dataset/not_done/'  + str(self.save_time).zfill(7), temp_not_done_array)

		self.save_time += 1


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		# return (
		# 	torch.FloatTensor(self.state[ind]).to(self.device),
		# 	torch.FloatTensor(self.action[ind]).to(self.device),
		# 	torch.FloatTensor(self.next_state[ind]).to(self.device),
		# 	torch.FloatTensor(self.reward[ind]).to(self.device),
		# 	torch.FloatTensor(self.not_done[ind]).to(self.device)
		# )

		return (
			self.state[ind][:],
			self.action[ind][:],
			self.next_state[ind][:],
			self.reward[ind][:],
			self.not_done[ind][:]
		)