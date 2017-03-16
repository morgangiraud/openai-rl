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

        self.N_0 = 100
        self.N = 0
        self.discount = 1
        self.nb_hidden_units = 20
        self.lr = 1e-2
        

        # Model
        self.inputs = tf.placeholder(tf.float32, shape=[1, self.observation_space.shape[0] + 1], name='input')
        a1 = tf.contrib.layers.fully_connected(
            self.inputs
            , num_outputs=self.nb_hidden_units
            , activation_fn=tf.nn.relu
            , weights_initializer=tf.random_normal_initializer(stddev=1e-2)
            , trainable=True
        )
        a2 = tf.contrib.layers.fully_connected(
            a1
            , num_outputs=self.nb_hidden_units
            , activation_fn=tf.nn.relu
            , weights_initializer=tf.random_normal_initializer(stddev=1e-2)
            , trainable=True
        )
        q_preds = tf.contrib.layers.fully_connected(
            a2
            , num_outputs=self.action_space.n
            , activation_fn=None
            , weights_initializer=tf.random_normal_initializer(stddev=1e-2)
            , trainable=True
        )
        self.q_preds = tf.squeeze(q_preds)
        self.argmax_t = tf.cast(tf.argmax(self.q_preds, 0), tf.int32)
        self.max_q_t = self.q_preds[self.argmax_t]

        self.target_q = tf.placeholder(tf.float32, shape=[], name='targetQ')
        self.loss = tf.square(self.target_q - self.max_q_t)
        self.loss_summary_t = tf.summary.scalar('loss', self.loss)

        adam = tf.train.AdamOptimizer(self.lr)
        self.train_op = adam.minimize(self.loss)

        self.score = tf.placeholder(tf.float32, shape=[])
        self.score_summary_t = tf.summary.scalar('score', self.score)
        
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.sw = tf.summary.FileWriter(self.result_folder, self.sess.graph)


    def act(self, obs, eps=None):
        qs = self.sess.run(self.q_preds, feed_dict={
            self.inputs: [ np.concatenate((obs, [0])) ] # Done is False when ask to act
        }) 
        eps = max(self.N_0 / (self.N_0 + self.N), 1e-2)        
        act = np.argmax(qs) if np.random.random() > eps else self.action_space.sample()
        
        return act, qs

    def learnFromEpisode(self, env, episode_id):
        obs = env.reset()
        i = 0
        score = 0
        episode_loss = 0
        done = False
        while True:
            i += 1

            if self.config['render']:
                env.render()
            act, qs = self.act(obs)
            obs2, reward, done, info = env.step(act)
            score += reward
            self.N += 1

            qs2, max_q2 = self.sess.run([self.q_preds, self.max_q_t], feed_dict={
                self.inputs: [ np.concatenate((obs2, [1 if done else 0])) ]
            })
            _, loss = self.sess.run([self.train_op, self.loss], feed_dict={
                self.inputs: [ np.concatenate((obs, [0])) ], 
                self.argmax_t: act, # We intercept the argmax to use the behaviour policy
                self.target_q: reward + self.discount * max_q2,
            })
            episode_loss += 1 / i * (loss - episode_loss)

            obs = obs2

            if done:
                break

        loss_sum, score_sum = self.sess.run([self.loss_summary_t, self.score_summary_t], feed_dict={
            self.loss: episode_loss, 
            self.score: score
        })
        self.sw.add_summary(loss_sum, episode_id)
        self.sw.add_summary(score_sum, episode_id)

        return score


