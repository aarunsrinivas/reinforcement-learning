import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np


class DQN(nn.Module):

	def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
		super(DQN, self).__init__()
		self.fc1 = nn.Linear(*input_dims, fc1_dims)
		self.fc2 = nn.Linear(fc1_dims, fc2_dims)
		self.fc3 = nn.Linear(fc2_dims, n_actions)
		self.optimizer = optim.Adam(self.parameters(), lr=lr)
		self.loss = nn.MSELoss()

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x


class DQNAgent:

	def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
	             eps_min=0.01, eps_dec=5e-4):
		self.gamma = gamma
		self.epsilon = epsilon
		self.eps_min = eps_min
		self.eps_dec = eps_dec
		self.action_space = [i for i in range(n_actions)]
		self.batch_size = batch_size
		self.mem_size = 100000
		self.mem_ctr = 0
		self.model = DQN(lr=lr, input_dims=input_dims,
		                 fc1_dims=256, fc2_dims=256, n_actions=n_actions)
		self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
		self.next_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
		self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
		self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
		self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

	def store_transition(self, state, action, reward, next_state, done):
		index = self.mem_ctr % self.mem_size
		self.state_memory[index] = state
		self.next_state_memory[index] = next_state
		self.action_memory[index] = action
		self.reward_memory[index] = reward
		self.terminal_memory[index] = done
		self.mem_ctr += 1

	def get_action(self, state):
		if np.random.random() < self.epsilon:
			action = np.random.choice(self.action_space)
		else:
			state = torch.tensor(state)
			actions = self.model.forward(state)
			action = torch.argmax(actions).item()
		return action

	def replay(self):
		if self.mem_ctr < self.batch_size:
			return
		self.model.optimizer.zero_grad()
		max_mem = min(self.mem_ctr, self.mem_size)
		batch = np.random.choice(max_mem, self.batch_size, replace=False)
		batch_index = np.arange(self.batch_size, dtype=np.int32)
		state_batch = torch.tensor(self.state_memory[batch])
		next_state_batch = torch.tensor(self.next_state_memory[batch])
		reward_batch = torch.tensor(self.reward_memory[batch])
		terminal_batch = torch.tensor(self.terminal_memory[batch])
		action_batch = self.action_memory[batch]
		q_eval = self.model.forward(state_batch)[batch_index, action_batch]
		q_next = self.model.forward(next_state_batch)
		q_next[terminal_batch] = 0
		q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]
		loss = self.model.loss(q_target, q_eval)
		loss.backward()
		self.model.optimizer.step()
		self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
