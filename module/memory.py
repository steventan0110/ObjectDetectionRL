import random
import numpy as np
import torch


class ReplayMemory(object):
	import random
	# TODO implement me!

	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.position = 0

	def push(self, current_state, action, next_state, reward):
		if len(self.memory) < self.capacity:
			self.memory.append(None)
		self.memory[self.position] = (current_state, action, next_state, reward)
		self.position = (self.position + 1) % self.capacity

	def sample(self, batch_size):
		# memory = np.array(self.memory)
		# idx = np.random.choice(len(self.memory), size=batch_size)
		# return memory[idx]

		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)