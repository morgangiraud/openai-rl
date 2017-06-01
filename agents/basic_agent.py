import json, os
import tensorflow as tf
from gym.spaces import Discrete, Box

from utils import phis

class BasicAgent(object):
    def __init__(self, config, env):
        if not 'best' in config:
            config['best'] = False
        if not 'debug' in config:
            config['debug'] = False
        if not 'max_iter' in config:
            config['max_iter'] = 1
        if config['best']:
            config.update(self.get_best_config(config['env_name']))

        self.config = config
        
        if config['debug']:
            print('config', config)
            
        self.random_seed = config['random_seed']
        self.result_dir = config['result_dir']
        self.max_iter = config['max_iter']

        self.env = env
        if 'nb_state' in config:
            self.nb_state = config['nb_state']
            self.phi = config['phi']
        else:
            if not isinstance(env.observation_space, Box):
                raise Exception('Observation space {} incompatible with {}. (Only supports Box observation spaces.)'.format(action_space, self))
            self.observation_space = env.observation_space
        if not isinstance(env.action_space, Discrete):
            raise Exception('Action space {} incompatible with {}. (Only supports Discrete action spaces.)'.format(action_space, self))
        self.action_space = env.action_space

        # Set custom properties of the agent
        self.set_agent_props()

        # Play part
        self.play_counter = 0

        # Graph part
        self.graph = self.build_graph(tf.Graph())

        # Misc
        with self.graph.as_default():
            self.saver = tf.train.Saver(
                max_to_keep=50,
            )
            self.init_op = tf.global_variables_initializer()

        gpu_options = tf.GPUOptions(allow_growth=True)
        sessConfig = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=sessConfig, graph=self.graph)
        self.sw = tf.summary.FileWriter(self.result_dir, self.sess.graph)
        self.init()

    def set_agent_props(self):
        pass

    def get_best_config(self, env_name=""):
        return {}

    @staticmethod
    def get_random_config(fixed_params={}):
        raise Exception('The get_random_config function must be overrided by the agent')

    def build_graph(self, graph):
        raise Exception('The build_graph function must be overrided by the agent')

    def act(self, obs, eps=None):
        raise Exception('The act function must be overrided by the agent')

    def learn_from_episode(self, env):
        raise Exception('The learn_from_episode function must be overrided by the agent')

    def train(self, render=False, save_every=49):
        for episode_id in range(0, self.max_iter):
            self.learn_from_episode(self.env, render)

            if save_every > 0 and episode_id % save_every == 0:
                self.save()

    def save(self):
        global_step_t = tf.train.get_global_step(self.graph)
        global_step, episode_id = self.sess.run([global_step_t, self.episode_id])
        if self.config['debug']:
            print('Saving to %s with global_step %d' % (self.result_dir, global_step))
        self.saver.save(self.sess, self.result_dir + '/agent-ep_' + str(episode_id), global_step)

        if not os.path.isfile(self.result_dir + '/config.json'):
            config = self.config
            if 'phi' in config:
                del config['phi']
            with open(self.result_dir + '/config.json', 'w') as f:
                json.dump(self.config, f)


    def init(self):
        checkpoint = tf.train.get_checkpoint_state(self.result_dir)
        if checkpoint is None:
            self.sess.run(self.init_op)
        else:
            if self.config['debug']:
                print('Loading the model from folder: %s' % self.result_dir)
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)

    def play(self, env, render=True):
        obs = env.reset()
        score = 0
        done = False
        while True:
            if render:
                env.render()

            act, state = self.act(obs)
            next_obs, reward, done, info = env.step(act)

            score += reward
            obs = next_obs
            if done:
                break

        self.play_counter += 1
        pscore_sum = self.sess.run(self.pscore_sum_t, feed_dict={
            self.pscore_plh: score,
        })
        self.sw.add_summary(pscore_sum, self.play_counter)

        return score

class TabularBasicAgent(BasicAgent):
    """
    Agent implementing tabular Q-learning.
    """
    def __init__(self, config, env):
        if 'debug' in config:
            config.update(phis.getPhiConfig(config['env_name'], config['debug']))
        else:
            config.update(phis.getPhiConfig(config['env_name']))
        super(TabularBasicAgent, self).__init__(config, env)
