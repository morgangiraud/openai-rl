import gym, os, time
import tensorflow as tf
import numpy as np

from agents import TabularQAgent, DeepQAgent

dir = os.path.dirname(os.path.realpath(__file__))

env = gym.make('CartPole-v0') # obs: [x, vx, theta, vtheta]

# nb_state = self.n = 16 * 2
# def phi(self, obs, done=False):
#     phi =  [
#         0 if obs[0] < 0 else 1,
#         0 if obs[1] < 0 else 1,
#         0 if obs[2] < 0 else 1,
#         0 if obs[3] < 0 else 1,
#     ]
#     return phi[0] + 2 * phi[1] + 4 * phi[2] + 8 * phi[3] + 16 * (1 if done else 0)
# agent = TabularQAgent(phi.n, env.action_space, phi.compute, {'render': False})

agent = DeepQAgent(env.observation_space, env.action_space, {'render': False})

for episode_id in range(1, 100001):
    agent.learnFromEpisode(env, episode_id)
