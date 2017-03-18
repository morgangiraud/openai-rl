import gym, os, time
import tensorflow as tf
import numpy as np

from agents import TabularQAgent, TabularQplusAgent, TabularFixedQplusAgent, BackwardTabularFixedQplusAgent, DeepQAgent

dir = os.path.dirname(os.path.realpath(__file__))

env = gym.make('CartPole-v0') # obs: [x, vx, theta, vtheta]

# nb_state = 2**4 + 1
# def phi(obs, done=False):
#     if done:
#         return 2**4

#     phi =  [
#         1 if obs[0] < 0 else 0,
#         1 if obs[1] < 0 else 0,
#         1 if obs[2] < 0 else 0,
#         1 if obs[3] < 0 else 0,
#     ]
#     return 2**0 * phi[0] + 2**1 * phi[1] + 2**2 * phi[2] + 2**3 * phi[3]

# nb_state = 2**8 + 1
# def phi(obs, done=False):
#     if done:
#         return 2**8

#     phi =  [
#         1 if obs[0] < 0 else 0,
#         1 if abs(obs[0]) > 1.8 else 0, # 3/4th of the field, danger zone
#         1 if obs[1] < 0 else 0,
#         1 if abs(obs[1]) > 0.5 else 0,
#         1 if obs[2] < 0 else 0,
#         1 if abs(obs[2]) > 31 else 0, # 3/4th of the rotation, danger zone
#         1 if obs[3] < 0 else 0,
#         1 if abs(obs[3]) > 0.5 else 0,
#     ]
#     return ( 
#         2**0 * phi[0] + 2**1 * phi[1] + 2**2 * phi[2] + 2**3 * phi[3]
#         + 2**4 * phi[4] + 2**5 * phi[5] + 2**6 * phi[6] + 2**7 * phi[7]
#     )
# agent = TabularQAgent(nb_state, env.action_space, phi, {'render': False})
# agent = TabularQplusAgent(nb_state, env.action_space, phi, {'render': False})
# agent = TabularFixedQplusAgent(nb_state, env.action_space, phi, {'render': False})
# agent = BackwardTabularFixedQplusAgent(nb_state, env.action_space, phi, {'render': False})

agent = DeepQAgent(env.observation_space, env.action_space, {'render': False})

for episode_id in range(1, 2001):
    agent.learnFromEpisode(env, episode_id)

agent.config['render'] = True
agent.learnFromEpisode(env, 2001)

