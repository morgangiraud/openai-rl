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
        get_discount = lambda: 0.5 + (1 - 0.5) * np.random.random(1)[0]
        get_N0 = lambda: np.random.randint(1, 5e3)
        get_min_eps = lambda: 1e-4 + (1e-1 - 1e-4) * np.random.random(1)[0]
        get_initial_q_value = lambda: 0 # int(np.random.random(1)[0] * 200)
        get_er_batch_size = lambda: np.random.randint(16, 1024)
        get_er_rm_size = lambda: np.random.randint(1000, 50000)

        random_config = {
            'lr': get_lr()
            , 'discount': get_discount()
            , 'N0': get_N0()
            , 'min_eps': get_min_eps()
            , 'initial_q_value': get_initial_q_value()
            , 'er_batch_size': get_er_batch_size()
            , 'er_rm_size': get_er_rm_size()
        }
        random_config.update(fixed_params)

        return random_config

    def build_graph(self, graph):
        with graph.as_default():
            tf.set_random_seed(self.random_seed)

            self.inputs = tf.placeholder(tf.int32, shape=[], name="inputs")
            
            q_scope = tf.VariableScope(reuse=False, name='QValues')            
            with tf.variable_scope(q_scope):
                self.Qs = tf.get_variable('Qs'
                    , shape=[self.nb_state, self.action_space.n]
                    , initializer=tf.constant_initializer(self.initial_q_value)
                    , dtype=tf.float32
                )
                tf.summary.histogram('Qarray', self.Qs)
                self.q_preds = self.Qs[self.inputs]

            policy_scope = tf.VariableScope(reuse=False, name='EpsilonGreedyPolicy')
            with tf.variable_scope(policy_scope):
                self.action_t = capacities.eps_greedy(
                    self.inputs, self.q_preds, self.env.action_space.n, self.N0, self.min_eps, self.nb_state
                )

            # Experienced replay part
            with tf.variable_scope('ExperienceReplay'):
                self.er_inputs = tf.placeholder(tf.int32, shape=[None], name="ERInputs")
                self.er_actions = tf.placeholder(tf.int32, shape=[None], name="ERInputs")
                self.er_rewards = tf.placeholder(tf.float32, shape=[None], name="ERReward")
                self.er_next_states = tf.placeholder(tf.int32, shape=[None], name="ERNextState")

                er_next_max_action_t = tf.cast(tf.argmax(tf.gather(self.Qs, self.er_next_states), 1), tf.int32)
                er_next_sa_pairs = tf.stack([self.er_next_states, er_next_max_action_t], 1)
                er_target_qs = tf.stop_gradient(self.er_rewards + self.discount * tf.gather_nd(self.Qs, er_next_sa_pairs))
                er_sa_pairs = tf.stack([self.er_inputs, self.er_actions], 1)
                er_qs = tf.gather_nd(self.Qs, er_sa_pairs)
                self.er_loss = 1/2 * tf.reduce_sum(tf.square(er_target_qs - er_qs))

                er_adam = tf.train.AdamOptimizer(self.lr)
                self.er_global_step = tf.Variable(0, trainable=False, name="global_step", collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
                self.er_train_op = er_adam.minimize(self.er_loss, global_step=self.er_global_step)

            self.score_plh = tf.placeholder(tf.float32, shape=[])
            self.score_sum_t = tf.summary.scalar('score', self.score_plh)
            self.loss_plh = tf.placeholder(tf.float32, shape=[])
            self.loss_sum_t = tf.summary.scalar('loss', self.loss_plh)
            self.all_summary_t = tf.summary.merge_all()

            self.episode_id, self.inc_ep_id_op = capacities.counter("episode_id")

            # Playing part
            self.pscore_plh = tf.placeholder(tf.float32, shape=[])
            self.pscore_sum_t = tf.summary.scalar('play_score', self.pscore_plh)

        return graph

    def learn_from_episode(self, env, render=False):
        obs = env.reset()
        score = 0
        av_loss = []
        done = False

        while True:
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
            loss, _ = self.sess.run([self.er_loss, self.er_train_op], feed_dict={
                self.er_inputs: memories['states'],
                self.er_actions: memories['actions'],
                self.er_rewards: memories['rewards'],
                self.er_next_states: memories['next_states'],
            })

            av_loss.append(loss)
            score += reward
            obs = next_obs
            if done:
                break

        summary, _, episode_id = self.sess.run([self.all_summary_t, self.inc_ep_id_op, self.episode_id], feed_dict={
            self.score_plh: score,
            self.loss_plh: np.mean(av_loss)
        })
        self.sw.add_summary(summary, episode_id)

        return

class TabularFixedQERAgent(TabularQERAgent):
    """
    Agent implementing tabular Q-learning with experience replay and a second fixed network.
    """
    def __init__(self, config, env):
        super(TabularFixedQERAgent, self).__init__(config, env)

        self.er_every = config['er_every']

    def get_best_config(self, env_name=""):
        return {
            'lr': 0.03
            , 'discount': 0.999
            , 'N0': 75
            , 'min_eps': 0.005
            , 'initial_q_value': 0
            , 'er_batch_size': 783
            , 'er_rm_size': 36916
            , "er_every": 200
        }

    @staticmethod
    def get_random_config(fixed_params={}):
        get_lr = lambda: 1e-3 + (1 - 1e-3) * np.random.random(1)[0]
        get_discount = lambda: 0.5 + (1 - 0.5) * np.random.random(1)[0]
        get_N0 = lambda: np.random.randint(1, 5e3)
        get_min_eps = lambda: 1e-4 + (1e-1 - 1e-4) * np.random.random(1)[0]
        get_initial_q_value = lambda: 0 # int(np.random.random(1)[0] * 200)
        get_er_batch_size = lambda: np.random.randint(16, 1024)
        get_er_rm_size = lambda: np.random.randint(1000, 50000)
        get_er_every = lambda: np.random.randint(1, 1000)

        random_config = {
            'lr': get_lr()
            , 'discount': get_discount()
            , 'N0': get_N0()
            , 'min_eps': get_min_eps()
            , 'initial_q_value': get_initial_q_value()
            , 'er_batch_size': get_er_batch_size()
            , 'er_rm_size': get_er_rm_size()
            , 'er_every': get_er_every()
        }
        random_config.update(fixed_params)

        return random_config

    def build_graph(self, graph):
        with graph.as_default():
            tf.set_random_seed(self.random_seed)

            self.inputs = tf.placeholder(tf.int32, shape=[], name="inputs")
            
            q_scope = tf.VariableScope(reuse=False, name='QValues')            
            with tf.variable_scope(q_scope):
                self.Qs = tf.get_variable('Qs'
                    , shape=[self.nb_state, self.action_space.n]
                    , initializer=tf.constant_initializer(self.initial_q_value)
                    , dtype=tf.float32
                )
                tf.summary.histogram('Qarray', self.Qs)
                self.q_preds = self.Qs[self.inputs]

            fixed_q_scope = tf.VariableScope(reuse=False, name='FixedQValues')
            with tf.variable_scope(fixed_q_scope):
                self.update_fixed_vars_op = capacities.fix_scope(q_scope)

            policy_scope = tf.VariableScope(reuse=False, name='EpsilonGreedyPolicy')
            with tf.variable_scope(policy_scope):
                self.action_t = capacities.eps_greedy(
                    self.inputs, self.q_preds, self.env.action_space.n, self.N0, self.min_eps, self.nb_state
                )

            # Experienced replay part
            with tf.variable_scope('ExperienceReplay'):
                with tf.variable_scope(fixed_q_scope, reuse=True):
                    fixed_Qs = tf.get_variable('Qs')

                self.er_inputs = tf.placeholder(tf.int32, shape=[None], name="ERInputs")
                self.er_actions = tf.placeholder(tf.int32, shape=[None], name="ERInputs")
                self.er_rewards = tf.placeholder(tf.float32, shape=[None], name="ERReward")
                self.er_next_states = tf.placeholder(tf.int32, shape=[None], name="ERNextState")

                er_next_max_action_t = tf.cast(tf.argmax(tf.gather(fixed_Qs, self.er_next_states), 1), tf.int32)
                er_next_sa_pairs = tf.stack([self.er_next_states, er_next_max_action_t], 1)
                er_target_qs = tf.stop_gradient(self.er_rewards + self.discount * tf.gather_nd(fixed_Qs, er_next_sa_pairs))
                er_sa_pairs = tf.stack([self.er_inputs, self.er_actions], 1)
                er_qs = tf.gather_nd(self.Qs, er_sa_pairs)
                self.er_loss = 1/2 * tf.reduce_sum(tf.square(er_target_qs - er_qs))

                er_adam = tf.train.AdamOptimizer(self.lr)
                self.er_global_step = tf.Variable(0, trainable=False, name="global_step", collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
                self.er_train_op = er_adam.minimize(self.er_loss, global_step=self.er_global_step)

            self.score_plh = tf.placeholder(tf.float32, shape=[])
            self.score_sum_t = tf.summary.scalar('score', self.score_plh)
            self.loss_plh = tf.placeholder(tf.float32, shape=[])
            self.loss_sum_t = tf.summary.scalar('loss', self.loss_plh)
            self.all_summary_t = tf.summary.merge_all()

            self.episode_id, self.inc_ep_id_op = capacities.counter("episode_id")
            self.event_count, self.inc_event_count_op = capacities.counter("event_count")

            # Playing part
            self.pscore_plh = tf.placeholder(tf.float32, shape=[])
            self.pscore_sum_t = tf.summary.scalar('play_score', self.pscore_plh)

        return graph

    def learn_from_episode(self, env, render=False):
        obs = env.reset()
        score = 0
        av_loss = []
        done = False

        while True:
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
            loss, _, event_count, _ = self.sess.run([self.er_loss, self.inc_event_count_op, self.event_count, self.er_train_op], feed_dict={
                self.er_inputs: memories['states'],
                self.er_actions: memories['actions'],
                self.er_rewards: memories['rewards'],
                self.er_next_states: memories['next_states'],
            })
            if event_count % self.er_every == 0:
                self.sess.run(self.update_fixed_vars_op)

            av_loss.append(loss)
            score += reward
            obs = next_obs
            if done:
                break

        summary, _, episode_id = self.sess.run([self.all_summary_t, self.inc_ep_id_op, self.episode_id], feed_dict={
            self.score_plh: score,
            self.loss_plh: np.mean(av_loss)
        })
        self.sw.add_summary(summary, episode_id)

        return