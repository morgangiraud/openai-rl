import json, os
import tensorflow as tf
from gym.spaces import Discrete, Box

class BasicAgent(object):
    def __init__(self, config, env):
        if config['best']:
            config.update(self.get_best_config())
        self.config = config

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

        self.lr = config['lr']
        self.discount = config['discount']

        self.set_agent_props()

        # Play part
        self.play_counter = 0

        # Graph part
        self.graph = self.buildGraph(tf.Graph())

        # For tabular use, no need for a lot of GPU
        gpu_options = tf.GPUOptions(allow_growth=True)
        sessConfig = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=sessConfig, graph=self.graph)
        self.sw = tf.summary.FileWriter(self.result_dir, self.sess.graph)
        self.init()

    def get_best_config(self):
        return {}

    def set_agent_props(self):
        pass

    def buildGraph(self, graph):
        raise Exception('The buildGraph function must be overrided by the agent')

    def act(self, obs, eps=None):
        pass

    def learnFromEpisode(self, env):
        pass

    def train(self, render=False):
        for episode_id in range(0, self.max_iter):
            self.learnFromEpisode(self.env, render)

    def save(self):
        global_step_t = tf.train.get_global_step(self.graph)
        global_step = tf.train.global_step(self.sess, global_step_t)
        if self.config['debug']:
            print('Saving to %s with global_step %d' % (self.result_dir, global_step))
        self.saver.save(self.sess, self.result_dir + '/agent', global_step)

        if not os.path.isfile(self.result_dir + '/config.json'):
            if self.config['debug']:
                print('Saving configuration')
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
