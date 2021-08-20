import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from util.replay_buffer import ReplayBuffer


class Actor(nn.Module):
	def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions, lr=1e-3):
		super(Actor, self).__init__()
		self.fc1 = nn.Linear(*input_dims, fc1_dims)
		self.fc2 = nn.Linear(fc1_dims, fc2_dims)
		self.fc3 = nn.Linear(fc2_dims, n_actions)
		self.optimizer = optim.Adam(self.parameters(), lr)
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.to(self.device)

	def loss(self, y_pred, y_true, advantage):
		y_pred = torch.clip(y_pred, 1e-8, 1 - 1e-8)
		log_prob = y_true * torch.log(y_pred)
		return torch.sum(-advantage * log_prob)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.softmax(self.fc3(x), dim=1)
		return x


class Critic(nn.Module):
	def __init__(self, input_dims, fc1_dims, fc2_dims, lr=1e-3):
		super(Critic, self).__init__()
		self.fc1 = nn.Linear(*input_dims, fc1_dims)
		self.fc2 = nn.Linear(fc1_dims, fc2_dims)
		self.fc3 = nn.Linear(fc2_dims, 1)
		self.loss = nn.MSELoss()
		self.optimizer = optim.Adam(self.parameters(), lr)
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.to(self.device)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x


class A2CAgent:
	def __init__(self, gamma, input_dims, fc1_dims, fc2_dims, n_actions, batch_size=32, lr=1e-3):
		self.gamma = gamma
		self.n_actions = n_actions
		self.batch_size = batch_size
		self.buffer = ReplayBuffer(input_dims, batch_size)
		self.actor = Actor(input_dims, fc1_dims, fc2_dims, n_actions, lr)
		self.critic = Critic(input_dims, fc1_dims, fc2_dims, lr)
		self.state_memory = []
		self.next_state_memory = []
		self.action_memory = []
		self.reward_memory = []
		self.terminal_memory = []

	def store_transition(self, state, action, reward, next_state, done):
		self.state_memory.append(state)
		self.action_memory.append(action)
		self.reward_memory.append(reward)
		self.next_state_memory.append(next_state)
		self.terminal_memory.append(done)
		self.buffer.store_transition(state, action, reward, next_state, done)

	def clear_memory(self):
		self.state_memory = []
		self.next_state_memory = []
		self.action_memory = []
		self.reward_memory = []
		self.terminal_memory = []

	def choose_action(self, state):
		state = state.astype(np.float32)
		state = torch.tensor(state).unsqueeze(0).to(self.actor.device)
		prob_vec = self.actor(state).ravel().cpu().detach().numpy()
		return np.random.choice(self.n_actions, p=prob_vec)

	def learn(self):
		# train actor
		# use batch gradient descent to improve stability
		states = np.array(self.state_memory).astype(np.float32)
		states = torch.tensor(states).to(self.actor.device)
		prob_vecs = self.actor(states)
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
		utilities = torch.tensor(utilities).unsqueeze(1).to(self.actor.device)
		# normalize utilities to improve stability
		mean = utilities.mean()
		std = utilities.std()
		utilities = (utilities - mean) / (std if std > 0 else 1)
		values = self.critic(states).to(self.actor.device)
		advantages = utilities - values

		self.actor.optimizer.zero_grad()
		loss = self.actor.loss(prob_vecs, action_vecs, advantages)
		loss.backward()
		self.actor.optimizer.step()

		# train critic
		if not self.buffer.batchable():
			return

		states, actions, rewards, next_states, dones = self.buffer.batch()
		states = torch.tensor(states).to(self.critic.device)
		next_states = torch.tensor(next_states).to(self.critic.device)
		rewards = torch.tensor(rewards).to(self.critic.device)
		dones = torch.tensor(dones).to(self.critic.device)
		current_values = self.critic(states)
		next_values = self.critic(next_states)
		next_values[dones] = 0
		target_values = rewards + self.gamma * next_values

		self.critic.optimizer.zero_grad()
		loss = self.critic.loss(current_values, next_values)
		loss.backward()
		self.critic.optimizer.step()
