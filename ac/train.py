import gym
import numpy as np
from agent import ActorCriticAgent

scores = []
EPISODES = 10_000
env = gym.make('CartPole-v0')
agent = ActorCriticAgent(0.95, [4], 32, 32, 2, lr=1e-3)

for episode in range(EPISODES):
	score = 0
	done = False
	state = env.reset()
	while not done:
		action = agent.choose_action(state)
		next_state, reward, done, _ = env.step(action)
		agent.store_transition(state, action, reward, next_state, done)
		state = next_state
		score += reward
	agent.learn()
	agent.clear_memory()
	scores.append(score)
	print(f'Episode: {episode}, Score: {score}, Avg Score: {np.mean(scores[-100:])}')
