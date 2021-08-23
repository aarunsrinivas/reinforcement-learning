import numpy as np


class ReplayBuffer:

	def __init__(self, input_dims, batch_size, mem_size=10000):
		self.mem_size = mem_size
		self.mem_ctr = 0
		self.batch_size = batch_size
		self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
		self.next_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
		self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
		self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
		self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

	def store_transition(self, state, action, reward, next_state, done):
		index = self.mem_ctr % self.mem_size
		self.state_memory[index] = state
		self.next_state_memory[index] = next_state
		self.action_memory[index] = action
		self.reward_memory[index] = reward
		self.terminal_memory[index] = done
		self.mem_ctr += 1

	def __len__(self):
		return min(self.mem_ctr, len(self.state_memory))

	def batchable(self):
		return True if self.mem_ctr > self.batch_size else False

	def batch(self):
		max_mem = min(self.mem_ctr, self.mem_size)
		batch = np.random.choice(max_mem, self.batch_size, replace=False)
		states = self.state_memory[batch]
		next_states = self.next_state_memory[batch]
		actions = self.action_memory[batch]
		rewards = self.reward_memory[batch]
		dones = self.terminal_memory[batch]
		return states, actions, rewards, next_states, dones
