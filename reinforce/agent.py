import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np


class REINFORCE(nn.Module):

	def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions, lr=1e-3):
		super(REINFORCE, self).__init__()
		self.fc1 = nn.Linear(*input_dims, fc1_dims)
		self.fc2 = nn.Linear(fc1_dims, fc2_dims)
		self.fc3 = nn.Linear(fc2_dims, n_actions)
		self.optimizer = optim.Adam(self.parameters(), lr)
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.to(self.device)

	def loss(self, y_pred, y_true, utility):
		y_pred = torch.clip(y_pred, 1e-8, 1 - 1e-8)
		log_prob = y_true * torch.log(y_pred)
		return torch.sum(-utility * log_prob)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.softmax(self.fc3(x), dim=1)
		return x


class REINFORCEAgent:
	def __init__(self, gamma, input_dims, fc1_dims, fc2_dims, n_actions, lr=1e-3):
		self.gamma = gamma
		self.n_actions = n_actions
		self.model = REINFORCE(input_dims, fc1_dims, fc2_dims, n_actions, lr)
		self.state_memory = []
		self.action_memory = []
		self.reward_memory = []
		self.next_state_memory = []
		self.terminal_memory = []

	def store_transition(self, state, action, reward, next_state, done):
		self.state_memory.append(state)
		self.action_memory.append(action)
		self.reward_memory.append(reward)
		self.next_state_memory.append(next_state)
		self.terminal_memory.append(done)

	def clear_memory(self):
		self.state_memory = []
		self.action_memory = []
		self.reward_memory = []
		self.next_state_memory = []
		self.terminal_memory = []

	def choose_action(self, state):
		state = state.astype(np.float32)
		state = torch.tensor(state).unsqueeze(0).to(self.model.device)
		prob_vec = self.model(state).ravel().cpu().detach().numpy()
		return np.random.choice(self.n_actions, p=prob_vec)

	def learn(self):
		# use batch gradient descent to improve stability
		states = np.array(self.state_memory).astype(np.float32)
		states = torch.tensor(states).to(self.model.device)
		prob_vecs = self.model(states)
		action_vecs = torch.zeros_like(prob_vecs)
		indices = np.arange(len(prob_vecs))
		actions = np.array(self.action_memory)
		action_vecs[indices, actions] = 1
		utilities = []
		for k in range(len(self.reward_memory)):
			utility = 0
			for t in range(k, len(self.reward_memory)):
				utility += np.power(self.gamma, t - k) * self.reward_memory[t]
			utilities.append(utility)
		utilities = torch.tensor(utilities).unsqueeze(1).to(self.model.device)
		# normalize utilities to improve stability
		mean = utilities.mean()
		std = utilities.std()
		utilities = (utilities - mean) / (std if std > 0 else 1)
		self.model.optimizer.zero_grad()
		loss = self.model.loss(prob_vecs, action_vecs, utilities)
		loss.backward()
		self.model.optimizer.step()
