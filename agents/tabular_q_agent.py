import numpy as np
import tensorflow as tf

from gym.spaces import Discrete
from agents import BasicAgent

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
        self.lr = 1e-1

        # Model
        # Inferring part
        self.inputs = tf.placeholder(tf.int32, shape=[], name="inputs")
        self.Qs = tf.Variable(
            initial_value=np.zeros([self.nb_state, self.action_space.n])
            , name = "Qs"
            , dtype=tf.float32
        )
        self.q_preds = self.Qs[self.inputs]
        self.action_t = tf.cast(tf.argmax(self.q_preds, 0), tf.int32)
        self.q_t = self.q_preds[self.action_t]

        # Learning Part
        self.reward = tf.placeholder(tf.float32, shape=[], name="reward")
        self.next_state = tf.placeholder(tf.int32, shape=[], name="nextState")
        next_max_action_t = tf.cast(tf.argmax(self.Qs[self.next_state], 0), tf.int32)
        target_q = tf.stop_gradient(self.reward + self.discount * self.Qs[self.next_state, next_max_action_t])
        self.loss = 1/2 * tf.square(target_q - self.q_t)
        # tf.summary.scalar('loss', self.loss)
        adam = tf.train.AdamOptimizer(self.lr)
        self.train_op = adam.minimize(self.loss)
        
        self.score = tf.placeholder(tf.float32, shape=[])
        tf.summary.scalar('score', self.score)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.sw = tf.summary.FileWriter(self.result_folder, self.sess.graph)
        self.summary_t = tf.summary.merge_all()

    def act(self, obs, eps=None):
        state_id = self.phi(obs)
        
        eps = max(self.N_0 / (self.N_0 + self.N_s[state_id]), 1e-2)
        self.N_s[state_id] += 1

        if np.random.random() > eps:
            act = self.sess.run(self.action_t, feed_dict={
                self.inputs: state_id
            }) 
        else:
            act = self.action_space.sample()

        return act, state_id

    def learnFromEpisode(self, env, episode_id):
        obs = env.reset()
        score = 0
        done = False
        while True:
            if self.config['render']:
                env.render()

            act, state_id = self.act(obs)
            next_obs, reward, done, info = env.step(act)
            score += reward
            next_state_id = self.phi(next_obs, done)           
            _ = self.sess.run(self.train_op, feed_dict={
                self.inputs: state_id,
                self.action_t: act,
                self.reward: reward,
                self.next_state: next_state_id,
            })

            obs = next_obs
            if done:
                break

        summary = self.sess.run(self.summary_t, feed_dict={ 
            self.score: score
        })
        self.sw.add_summary(summary, episode_id)

        return

class TabularQplusAgent(BasicAgent):
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
        super(TabularQplusAgent, self).initModel()
        
        self.N_0 = 100
        self.N_s = np.ones(self.nb_state)
        self.discount = 1
        self.lr = 1e-1
        self.er_lr = 1e-1
        self.replayMemoryDt = np.dtype([('states', 'int32'), ('actions', 'int32'), ('rewards', 'float32'), ('next_states', 'int32')])
        self.replayMemory = np.array([], dtype=self.replayMemoryDt)

        # Model
        # Infering part
        self.inputs = tf.placeholder(tf.int32, shape=[], name="inputs")
        self.Qs = tf.Variable(
            initial_value=np.zeros([self.nb_state, self.action_space.n])
            , name = "Qs"
            , dtype=tf.float32
        )
        self.q_preds = self.Qs[self.inputs]
        self.action_t = tf.cast(tf.argmax(self.q_preds, 0), tf.int32)
        self.q_t = self.q_preds[self.action_t]

        # Learning Part
        self.reward = tf.placeholder(tf.float32, shape=[], name="reward")
        self.next_state = tf.placeholder(tf.int32, shape=[], name="nextState")
        next_max_action_t = tf.cast(tf.argmax(self.Qs[self.next_state], 0), tf.int32)
        target_q = tf.stop_gradient(self.reward + self.discount * self.Qs[self.next_state, next_max_action_t])
        self.loss = 1/2 * tf.square(target_q - self.q_t)
        # tf.summary.scalar('loss', self.loss)
        adam = tf.train.AdamOptimizer(self.lr)
        self.train_op = adam.minimize(self.loss)
        
        # Experienced replay part
        self.er_inputs = tf.placeholder(tf.int32, shape=[None], name="ERInputs")
        self.er_actions = tf.placeholder(tf.int32, shape=[None], name="ERInputs")
        self.er_rewards = tf.placeholder(tf.float32, shape=[None], name="ERReward")
        self.er_next_states = tf.placeholder(tf.int32, shape=[None], name="ERNextState")
        er_next_max_action_t = tf.cast(tf.argmax(tf.gather(self.Qs, self.er_next_states), 1), tf.int32)
        er_next_sa_pairs = tf.stack([self.er_next_states, er_next_max_action_t], 1)
        er_target_qs = tf.stop_gradient(self.er_rewards + self.discount * tf.gather_nd(self.Qs, er_next_sa_pairs))
        er_sa_pairs = tf.stack([self.er_inputs, self.er_actions], 1)
        er_qs = tf.gather_nd(self.Qs, er_sa_pairs)
        er_loss = 1/2 * tf.reduce_sum(tf.square(er_target_qs - er_qs))
        er_adam = tf.train.AdamOptimizer(self.er_lr)
        self.er_train_op = er_adam.minimize(er_loss)

        self.score = tf.placeholder(tf.float32, shape=[])
        tf.summary.scalar('score', self.score)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.sw = tf.summary.FileWriter(self.result_folder, self.sess.graph)
        self.summary_t = tf.summary.merge_all()

    def act(self, obs, eps=None):
        state_id = self.phi(obs)
        
        eps = max(self.N_0 / (self.N_0 + self.N_s[state_id]), 1e-2)
        self.N_s[state_id] += 1

        if np.random.random() > eps:
            act = self.sess.run(self.action_t, feed_dict={
                self.inputs: state_id
            }) 
        else:
            act = self.action_space.sample()

        return act, state_id

    def learnFromEpisode(self, env, episode_id):
        obs = env.reset()
        score = 0
        done = False
        while True:
            if self.config['render']:
                env.render()

            act, state_id = self.act(obs)
            next_obs, reward, done, info = env.step(act)
            score += reward
            next_state_id = self.phi(next_obs, done)           
            _ = self.sess.run(self.train_op, feed_dict={
                self.inputs: state_id,
                self.action_t: act,
                self.reward: reward,
                self.next_state: next_state_id,
            })

            memory = np.array([(state_id, act, reward, next_state_id)], dtype=self.replayMemoryDt)
            self.replayMemory = np.append(self.replayMemory, memory)
            obs = next_obs
            if done:
                break

        summary = self.sess.run(self.summary_t, feed_dict={ 
            self.score: score
        })
        self.sw.add_summary(summary, episode_id)

        # Experiencing replays
        if episode_id % 20 == 0:
            for i in range(100):
                memories = np.random.choice(self.replayMemory, 16)
                _ = self.sess.run(self.er_train_op, feed_dict={
                    self.er_inputs: memories['states'],
                    self.er_actions: memories['actions'],
                    self.er_rewards: memories['rewards'],
                    self.er_next_states: memories['next_states'],
                })

        return

class TabularFixedQplusAgent(BasicAgent):
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
        super(TabularFixedQplusAgent, self).initModel()
        
        self.N_0 = 100
        self.N_s = np.ones(self.nb_state)
        self.discount = 1
        self.lr = 1e-1
        self.er_lr = 1e-1
        self.replayMemoryDt = np.dtype([('states', 'int32'), ('actions', 'int32'), ('rewards', 'float32'), ('next_states', 'int32')])
        self.replayMemory = np.array([], dtype=self.replayMemoryDt)

        # Model
        # Infering part
        self.inputs = tf.placeholder(tf.int32, shape=[], name="inputs")
        self.Qs = tf.Variable(
            initial_value=np.zeros([self.nb_state, self.action_space.n])
            , name = "Qs"
            , dtype=tf.float32
        )
        self.q_preds = self.Qs[self.inputs]
        self.action_t = tf.cast(tf.argmax(self.q_preds, 0), tf.int32)
        self.q_t = self.q_preds[self.action_t]

        # Update fixed Qs
        self.fixed_Qs = tf.Variable(
            initial_value=np.zeros([self.nb_state, self.action_space.n])
            , name = "fixedQs"
            , trainable=False
            , dtype=tf.float32
        )
        self.update_fixed_Q_op = tf.assign(self.fixed_Qs, self.Qs)

        # Learning Part
        self.reward = tf.placeholder(tf.float32, shape=[], name="reward")
        self.next_state = tf.placeholder(tf.int32, shape=[], name="nextState")
        next_max_action_t = tf.cast(tf.argmax(self.Qs[self.next_state], 0), tf.int32)
        target_q = tf.stop_gradient(self.reward + self.discount * self.Qs[self.next_state, next_max_action_t])
        self.loss = 1/2 * tf.square(target_q - self.q_t)
        # tf.summary.scalar('loss', self.loss)
        adam = tf.train.AdamOptimizer(self.lr)
        self.train_op = adam.minimize(self.loss)
        
        # Experienced replay part
        self.er_inputs = tf.placeholder(tf.int32, shape=[None], name="ERInputs")
        self.er_actions = tf.placeholder(tf.int32, shape=[None], name="ERInputs")
        self.er_rewards = tf.placeholder(tf.float32, shape=[None], name="ERReward")
        self.er_next_states = tf.placeholder(tf.int32, shape=[None], name="ERNextState")
        er_next_max_action_t = tf.cast(tf.argmax(tf.gather(self.fixed_Qs, self.er_next_states), 1), tf.int32)
        er_next_sa_pairs = tf.stack([self.er_next_states, er_next_max_action_t], 1)
        er_target_qs = tf.stop_gradient(self.er_rewards + self.discount * tf.gather_nd(self.fixed_Qs, er_next_sa_pairs))
        er_sa_pairs = tf.stack([self.er_inputs, self.er_actions], 1)
        er_qs = tf.gather_nd(self.Qs, er_sa_pairs)
        er_loss = 1/2 * tf.reduce_sum(tf.square(er_target_qs - er_qs))
        er_adam = tf.train.AdamOptimizer(self.er_lr)
        self.er_train_op = er_adam.minimize(er_loss)

        self.score = tf.placeholder(tf.float32, shape=[])
        tf.summary.scalar('score', self.score)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.sw = tf.summary.FileWriter(self.result_folder, self.sess.graph)
        self.summary_t = tf.summary.merge_all()

    def act(self, obs, eps=None):
        state_id = self.phi(obs)
        
        eps = max(self.N_0 / (self.N_0 + self.N_s[state_id]), 1e-2)
        self.N_s[state_id] += 1

        if np.random.random() > eps:
            act = self.sess.run(self.action_t, feed_dict={
                self.inputs: state_id
            }) 
        else:
            act = self.action_space.sample()

        return act, state_id

    def learnFromEpisode(self, env, episode_id):
        obs = env.reset()
        score = 0
        done = False
        while True:
            if self.config['render']:
                env.render()

            act, state_id = self.act(obs)
            next_obs, reward, done, info = env.step(act)
            score += reward
            next_state_id = self.phi(next_obs, done)           
            _ = self.sess.run(self.train_op, feed_dict={
                self.inputs: state_id,
                self.action_t: act,
                self.reward: reward,
                self.next_state: next_state_id,
            })

            memory = np.array([(state_id, act, reward, next_state_id)], dtype=self.replayMemoryDt)
            self.replayMemory = np.append(self.replayMemory, memory)
            obs = next_obs
            if done:
                break

        summary = self.sess.run(self.summary_t, feed_dict={ 
            self.score: score
        })
        self.sw.add_summary(summary, episode_id)

        # Experiencing replays
        if episode_id % 20 == 0:
            self.sess.run(self.update_fixed_Q_op)
            for i in range(100):
                memories = np.random.choice(self.replayMemory, 16)
                _ = self.sess.run(self.er_train_op, feed_dict={
                    self.er_inputs: memories['states'],
                    self.er_actions: memories['actions'],
                    self.er_rewards: memories['rewards'],
                    self.er_next_states: memories['next_states'],
                })

        return

class BackwardTabularFixedQplusAgent(BasicAgent):
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
        super(BackwardTabularFixedQplusAgent, self).initModel()
        
        self.N_0 = 100
        self.N_s = np.ones(self.nb_state)
        self.discount = 1
        self.lr = 1e-1
        self.er_lr = 1e-1
        self.replayMemoryDt = np.dtype([('states', 'int32'), ('actions', 'int32'), ('rewards', 'float32'), ('next_states', 'int32')])
        self.replayMemory = np.array([], dtype=self.replayMemoryDt)

        # Model
        # Inferring part
        self.inputs = tf.placeholder(tf.int32, shape=[], name="inputs")
        self.Qs = tf.Variable(
            initial_value=np.zeros([self.nb_state, self.action_space.n])
            , name = "Qs"
            , dtype=tf.float32
        )
        self.q_preds = self.Qs[self.inputs]
        self.action_t = tf.cast(tf.argmax(self.q_preds, 0), tf.int32)
        self.q_t = self.q_preds[self.action_t]
        
        # Update fixed Qs
        self.fixed_Qs = tf.Variable(
            initial_value=np.zeros([self.nb_state, self.action_space.n])
            , name = "fixedQs"
            , trainable=False
            , dtype=tf.float32
        )
        self.update_fixed_Q_op = tf.assign(self.fixed_Qs, self.Qs)

        # Learning Part
        self.reward = tf.placeholder(tf.float32, shape=[], name="reward")
        self.next_state = tf.placeholder(tf.int32, shape=[], name="nextState")
        next_max_action_t = tf.cast(tf.argmax(self.Qs[self.next_state], 0), tf.int32)
        target_q = tf.stop_gradient(self.reward + self.discount * self.Qs[self.next_state, next_max_action_t])
        self.Et = tf.placeholder(tf.float32, shape=[self.nb_state, self.action_space.n], name="Et")
        self.loss = - tf.stop_gradient(target_q - self.q_t) * self.Et * self.Qs
        # tf.summary.scalar('loss', self.loss)
        adam = tf.train.AdamOptimizer(self.lr)
        self.train_op = adam.minimize(self.loss)

        # Experienced replay part
        self.er_inputs = tf.placeholder(tf.int32, shape=[None], name="ERInputs")
        self.er_actions = tf.placeholder(tf.int32, shape=[None], name="ERInputs")
        self.er_rewards = tf.placeholder(tf.float32, shape=[None], name="ERReward")
        self.er_next_states = tf.placeholder(tf.int32, shape=[None], name="ERNextState")
        er_sa_pairs = tf.stack([self.er_inputs, self.er_actions], 1)
        er_qs = tf.gather_nd(self.Qs, er_sa_pairs)
        er_next_max_action_t = tf.cast(tf.argmax(tf.gather(self.fixed_Qs, self.er_next_states), 1), tf.int32)
        er_next_sa_pairs = tf.stack([self.er_next_states, er_next_max_action_t], 1)
        er_target_qs = tf.stop_gradient(self.er_rewards + self.discount * tf.gather_nd(self.fixed_Qs, er_next_sa_pairs))
        er_loss = 1/2 * tf.reduce_sum(tf.square(er_target_qs - er_qs))
        er_adam = tf.train.AdamOptimizer(self.er_lr)
        self.er_train_op = er_adam.minimize(er_loss)

        
        self.score = tf.placeholder(tf.float32, shape=[])
        tf.summary.scalar('score', self.score)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.sw = tf.summary.FileWriter(self.result_folder, self.sess.graph)
        self.summary_t = tf.summary.merge_all()

    def act(self, obs, eps=None):
        state_id = self.phi(obs)
        
        eps = max(self.N_0 / (self.N_0 + self.N_s[state_id]), 1e-2)
        self.N_s[state_id] += 1

        if np.random.random() > eps:
            act = self.sess.run(self.action_t, feed_dict={
                self.inputs: state_id
            }) 
        else:
            act = self.action_space.sample()

        return act, state_id

    def learnFromEpisode(self, env, episode_id):
        obs = env.reset()
        score = 0
        done = False
        Et = np.zeros([self.nb_state, self.action_space.n])
        visited_sa = []
        lambda_val = 0.9
        while True:
            if self.config['render']:
                env.render()

            act, state_id = self.act(obs)
            for s, a in visited_sa:
                Et[s, a] *= self.discount * lambda_val
            Et[state_id, act] = 1
            visited_sa.append((state_id, act))

            next_obs, reward, done, info = env.step(act)
            score += reward


            next_state_id = self.phi(next_obs, done)           
            _ = self.sess.run(self.train_op, feed_dict={
                self.inputs: state_id,
                self.action_t: act,
                self.reward: reward,
                self.next_state: next_state_id,
                self.Et: Et,
            })

            memory = np.array([(state_id, act, reward, next_state_id)], dtype=self.replayMemoryDt)
            self.replayMemory = np.append(self.replayMemory, memory)
            obs = next_obs
            if done:
                break

        summary = self.sess.run(self.summary_t, feed_dict={ 
            self.score: score
        })
        self.sw.add_summary(summary, episode_id)

        # Experiencing replays
        if episode_id % 20 == 0:
            self.sess.run(self.update_fixed_Q_op)
            for i in range(100):
                memories = np.random.choice(self.replayMemory, 16)
                _ = self.sess.run(self.er_train_op, feed_dict={
                    self.er_inputs: memories['states'],
                    self.er_actions: memories['actions'],
                    self.er_rewards: memories['rewards'],
                    self.er_next_states: memories['next_states'],
                })

        return 



