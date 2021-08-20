import gym
import numpy as np
from agent import DQNAgent

scores = []
EPISODES = 500
env = gym.make('CartPole-v0')
agent = DQNAgent(gamma=0.99, epsilon=1, batch_size=64, n_actions=2, eps_min=0.01, input_dims=[4], lr=0.003)

for episode in range(EPISODES):
	score = 0
	done = False
	state = env.reset()
	while not done:
		action = agent.get_action(state)
		next_state, reward, done, _ = env.step(action)
		score += reward
		agent.store_transition(state, action, reward, next_state, done)
		agent.replay()
		state = next_state
	scores.append(score)
	avg_score = np.mean(scores[-100:])
	print(f'Episode: {episode}, Score: {score}, Avg Score: {avg_score}')
