import numpy as np
import tensorflow as tf

from gym.spaces import Discrete, Box
from agents import BasicAgent

class DeepQAgent(BasicAgent):
    """
    Agent implementing tabular Q-learning.
    """

    def __init__(self, observation_space, action_space, config):
        if not isinstance(observation_space, Box):
            raise Exception('Observation space {} incompatible with {}. (Only supports Box observation spaces.)'.format(action_space, self))
        self.observation_space = observation_space
        if not isinstance(action_space, Discrete):
            raise Exception('Action space {} incompatible with {}. (Only supports Discrete action spaces.)'.format(action_space, self))
        self.action_space = action_space
        self.config = config

        self.initModel()

    def initModel(self):
        super(DeepQAgent, self).initModel()

        self.memories = []
        self.N_0 = 100
        self.N = 0
        self.discount = 1
        self.nb_hidden_units = 100
        self.lr = 1e-3
        self.er_lr = 1e-3
        self.replayMemoryDt = np.dtype([('states', 'float32', (5,)), ('actions', 'int32'), ('rewards', 'float32'), ('next_states', 'float32', (5,))])
        self.replayMemory = np.array([], dtype=self.replayMemoryDt)
        
        # Model
        self.inputs = tf.placeholder(tf.float32, shape=[None, self.observation_space.shape[0] + 1], name='inputs')

        W1 = tf.get_variable('W1', shape=[self.observation_space.shape[0] + 1, self.nb_hidden_units], initializer=tf.random_normal_initializer(stddev=1e-2))
        b1 = tf.get_variable('b1', shape=[self.nb_hidden_units], initializer=tf.zeros_initializer())
        a1 = tf.nn.relu(tf.matmul(self.inputs, W1) + b1)

        W2 = tf.get_variable('W2', shape=[self.nb_hidden_units, self.action_space.n], initializer=tf.random_normal_initializer(stddev=1e-2))
        b2 = tf.get_variable('b2', shape=[self.action_space.n], initializer=tf.zeros_initializer())
        out = tf.matmul(a1, W2) + b2

        self.q_preds = tf.squeeze(out)
        self.action_t = tf.cast(tf.argmax(self.q_preds, 0), tf.int32)
        self.q_t = self.q_preds[self.action_t]

        # Update fixed Qs
        f_W1 = tf.get_variable('f_W1', shape=[self.observation_space.shape[0] + 1, self.nb_hidden_units])
        f_b1 = tf.get_variable('f_b1', shape=[self.nb_hidden_units])
        f_W2 = tf.get_variable('f_W2', shape=[self.nb_hidden_units, self.action_space.n])
        f_b2 = tf.get_variable('f_b2', shape=[self.action_space.n])
        self.update_f_W1_op = tf.assign(f_W1, W1)
        self.update_f_b1_op = tf.assign(f_b1, b1)
        self.update_f_W2_op = tf.assign(f_W2, W2)
        self.update_f_b2_op = tf.assign(f_b2, b2)

        # Learning Part
        self.reward = tf.placeholder(tf.float32, shape=[], name="reward")
        self.next_state = tf.placeholder(tf.float32, shape=[1, self.observation_space.shape[0] + 1], name="nextState")
        next_a1 = tf.nn.relu(tf.matmul(self.next_state, W1) + b1)
        next_q_preds = tf.squeeze(tf.matmul(next_a1, W2) + b2)
        next_max_action_t = tf.cast(tf.argmax(next_q_preds, 0), tf.int32)
        tartget_q1 = tf.stop_gradient(self.reward + self.discount * next_q_preds[next_max_action_t])
        tartget_q2 = self.reward
        target_q = tf.cond(tf.equal(self.next_state[0, 4], 1), lambda: tartget_q2, lambda: tartget_q1)
        with tf.control_dependencies([target_q]):
            self.loss = 1/2 * tf.square(target_q - self.q_t)
            # tf.summary.scalar('loss', self.loss)
            adam = tf.train.AdamOptimizer(self.lr)
            self.train_op = adam.minimize(self.loss)

        # Experienced replay part
        self.er_inputs = tf.placeholder(tf.float32, shape=[None, self.observation_space.shape[0] + 1], name="ERInputs")
        self.er_actions = tf.placeholder(tf.int32, shape=[None], name="ERInputs")
        self.er_rewards = tf.placeholder(tf.float32, shape=[None], name="ERReward")
        self.er_next_states = tf.placeholder(tf.float32, shape=[None, self.observation_space.shape[0] + 1], name="ERNextState")

        er_a1 = tf.nn.relu(tf.matmul(self.er_inputs, W1) + b1)
        er_qs_preds = tf.matmul(er_a1, W2) + b2
        er_sa_pairs = tf.stack([tf.range(0, tf.shape(self.er_actions)[0]), self.er_actions], 1)
        er_qs = tf.gather_nd(er_qs_preds, er_sa_pairs)

        er_next_a1 = tf.nn.relu(tf.matmul(self.er_next_states, f_W1) + f_b1)
        er_next_qs_preds = tf.matmul(er_next_a1, f_W2) + f_b2
        er_next_max_action_t = tf.cast(tf.argmax(er_next_qs_preds, 1), tf.int32)
        er_next_sa_pairs = tf.stack([tf.range(0, tf.shape(self.er_next_states)[0]), er_next_max_action_t], 1)
        er_next_qs = tf.gather_nd(er_next_qs_preds, er_next_sa_pairs)

        er_target_qs1 = tf.stop_gradient(self.er_rewards + self.discount * er_next_qs)
        er_target_qs2 = self.er_rewards
        scan_res = tf.scan(
            lambda indice, next_state: (indice[0] + 1, tf.cond(tf.equal(next_state[4], 1), lambda: er_target_qs2[indice[0]], lambda: er_target_qs1[indice[0]]))
            , self.er_next_states
            , initializer=(tf.constant(0, tf.int32), tf.constant(0., tf.float32))
        )
        er_target_qs = scan_res[1]
        er_loss = 1/2 * tf.reduce_sum(tf.square(er_target_qs - er_qs))
        er_adam = tf.train.AdamOptimizer(self.er_lr)
        self.er_train_op = er_adam.minimize(er_loss)

        # Score
        self.score = tf.placeholder(tf.float32, shape=[])
        tf.summary.scalar('score', self.score)

        # Misc
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.sw = tf.summary.FileWriter(self.result_folder, self.sess.graph)
        self.summary_t = tf.summary.merge_all()

    def act(self, obs, eps=None):
        eps = max(self.N_0 / (self.N_0 + self.N), 1e-2)
        self.N += 1

        if np.random.random() > eps:
            act = self.sess.run(self.action_t, feed_dict={
                self.inputs: [ np.concatenate((obs, [0])) ]
            }) 
        else:
            act = self.action_space.sample()

        return act

    def learnFromEpisode(self, env, episode_id):
        obs = env.reset()
        score = 0
        done = False
        while True:
            if self.config['render']:
                env.render()

            act = self.act(obs)
            next_obs, reward, done, info = env.step(act)
            score += reward

            _ = self.sess.run(self.train_op, feed_dict={
                self.inputs: [ np.concatenate((obs, [0])) ],
                self.action_t: act,
                self.reward: reward,
                self.next_state: [ np.concatenate((next_obs, [1 if done else 0])) ]
            })

            memory = np.array([(np.concatenate((obs, [0])), act, reward, np.concatenate((next_obs, [1 if done else 0])))], dtype=self.replayMemoryDt)
            self.replayMemory = np.append(self.replayMemory, memory)
            if self.replayMemory.shape[0] > 20000:
                self.replayMemory = np.delete(self.replayMemory, 0)
            obs = next_obs
            if done:
                break

        summary = self.sess.run(self.summary_t, feed_dict={ 
            self.score: score
        })
        self.sw.add_summary(summary, episode_id)

        # Experiencing replays
        if episode_id % 20 == 0:
            self.sess.run([self.update_f_W1_op, self.update_f_b1_op, self.update_f_W2_op, self.update_f_b2_op])
            for i in range(200):
                memories = np.random.choice(self.replayMemory, 16)
                _ = self.sess.run(self.er_train_op, feed_dict={
                    self.er_inputs: memories['states'],
                    self.er_actions: memories['actions'],
                    self.er_rewards: memories['rewards'],
                    self.er_next_states: memories['next_states'],
                })

        return 


