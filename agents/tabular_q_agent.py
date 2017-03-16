import numpy as np
import tensorflow as tf

from gym.spaces import Discrete
from agents import BasicAgent

class TabularQAgent(BasicAgent):
    """
    Agent implementing tabular Q-learning.
    """

    def __init__(self, nb_state, action_space, phi, config):
        self.nb_state = nb_state
        if not isinstance(action_space, Discrete):
            raise Exception('Action space {} incompatible with {}. (Only supports Discrete action spaces.)'.format(action_space, self))
        self.action_space = action_space
        self.phi = phi
        self.config = config

        self.initModel()

    def initModel(self):
        super(TabularQAgent, self).initModel()
        
        self.N_0 = 100
        self.N_s = np.ones(self.nb_state)
        self.discount = 1
        self.av_score = 0
        self.Qs = np.zeros([self.nb_state, self.action_space.n])
        

    def act(self, obs, eps=None):
        state_id = self.phi(obs)
        eps = max(self.N_0 / (self.N_0 + self.N_s[state_id]), 1e-2)
        act = np.argmax(self.Qs[state_id]) if np.random.random() > eps else self.action_space.sample()
        
        return act, state_id

    def learnFromEpisode(self, env, episode_id):
        obs = env.reset()
        score = 0
        done = False
        while True:
            if self.config['render']:
                env.render()
            act, state_id = self.act(obs)
            obs2, reward, done, info = env.step(act)
            score += reward
            self.N_s[state_id] += 1

            # print(reward)

            next_state_id = self.phi(obs2, done)
            td_error = reward + self.discount * np.max(self.Qs[next_state_id]) - self.Qs[state_id, act]
            # print(reward, np.max(self.Qs[next_state_id]), self.discount * np.max(self.Qs[next_state_id]), self.Qs[state_id, act], td_error)
            alpha = max(10 / (10 + episode_id), 1e-2)
            self.Qs[state_id, act] += alpha * td_error
            obs = obs2

            if done:
                break

        self.av_score += max(1 / episode_id, 1e-1) * (score - self.av_score)
        
        if episode_id % 100 == 0:
            print('episode: %d, moving average score: %f' % (episode_id, self.av_score))



        return 


