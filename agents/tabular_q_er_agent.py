import numpy as np
import tensorflow as tf

from agents import TabularQAgent, capacities

class TabularQERAgent(TabularQAgent):
    """
    Agent implementing tabular Q-learning with experience replay.
    """
    def set_agent_props(self):
        super(TabularQERAgent, self).set_agent_props()

        self.er_batch_size = self.config['er_batch_size']
        self.er_rm_size = self.config['er_rm_size']

        self.replayMemoryDt = np.dtype([('states', 'int32'), ('actions', 'int32'), ('rewards', 'float32'), ('next_states', 'int32')])
        self.replayMemory = np.array([], dtype=self.replayMemoryDt)

    def get_best_config(self, env_name=""):
        return {
            'lr': 0.03
            , 'lr_decay_steps': 40000
            , 'discount': 0.999
            , 'N0': 75
            , 'min_eps': 0.005
            , 'initial_q_value': 0
            , 'er_batch_size': 783
            , 'er_rm_size': 36916
        }

    @staticmethod
    def get_random_config(fixed_params={}):
        get_lr = lambda: 1e-3 + (1 - 1e-3) * np.random.random(1)[0]
        get_lr_decay_steps = lambda: np.random.randint(1e3, 1e5)
        get_discount = lambda: 0.5 + (1 - 0.5) * np.random.random(1)[0]
        get_N0 = lambda: np.random.randint(1, 5e3)
        get_min_eps = lambda: 1e-4 + (1e-1 - 1e-4) * np.random.random(1)[0]
        get_initial_q_value = lambda: 0 # int(np.random.random(1)[0] * 200)
        get_er_batch_size = lambda: np.random.randint(16, 1024)
        get_er_rm_size = lambda: np.random.randint(1000, 50000)

        random_config = {
            'lr': get_lr()
            , 'lr_decay_steps': get_lr_decay_steps()
            , 'discount': get_discount()
            , 'N0': get_N0()
            , 'min_eps': get_min_eps()
            , 'initial_q_value': get_initial_q_value()
            , 'er_batch_size': get_er_batch_size()
            , 'er_rm_size': get_er_rm_size()
        }
        random_config.update(fixed_params)

        return random_config

    def learn_from_episode(self, env, render=False):
        score = 0
        av_loss = []
        done = False

        obs = env.reset()
        while not done:
            if render:
                env.render()

            act, state_id = self.act(obs)
            next_obs, reward, done, info = env.step(act)
            next_state_id = self.phi(next_obs, done)

            memory = np.array([(state_id, act, reward, next_state_id)], dtype=self.replayMemoryDt)
            if self.replayMemory.shape[0] >= self.er_rm_size:
                self.replayMemory = np.delete(self.replayMemory, 0)
            self.replayMemory = np.append(self.replayMemory, memory)

            memories = np.random.choice(self.replayMemory, self.er_batch_size)
            loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={
                self.inputs_plh: memories['states'],
                self.actions_t: memories['actions'],
                self.rewards_plh: memories['rewards'],
                self.next_states_plh: memories['next_states'],
            })

            av_loss.append(loss)
            score += reward
            obs = next_obs

        summary, _, episode_id = self.sess.run([self.all_summary_t, self.inc_ep_id_op, self.episode_id], feed_dict={
            self.score_plh: score,
            self.loss_plh: np.mean(av_loss)
        })
        self.sw.add_summary(summary, episode_id)

        return