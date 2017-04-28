import numpy as np
import tensorflow as tf

from agents import BasicAgent, capacities

class DeepTDAgent(BasicAgent):
    """
    Agent implementing 2-layer NN Q-learning, using experience TD 0
    """
    def set_agent_props(self):
        self.q_params = {
            'nb_inputs': self.observation_space.shape[0] + 1
            , 'nb_units': self.config['nb_units']
            , 'nb_outputs': self.action_space.n
            , 'initial_mean': self.config['initial_mean']
            , 'initial_stddev': self.config['initial_stddev']
        }

        self.N0 = self.config['N0']
        self.min_eps = self.config['min_eps']

    def get_best_config(self, env_name=""):
        return {
            "lr": 0.006,
            "initial_mean": 0,
            "discount": 0.999,
            "nb_units": 81,
            "min_eps": 0.001,
            "initial_stddev": 0.44732464914245396,
            "N0": 4868
          }

    @staticmethod
    def get_random_config(fixed_params={}):
        get_lr = lambda: 1e-4 + (1 - 1e-4) * np.random.random(1)[0]
        get_discount = lambda: 0.5 + (1 - 0.5) * np.random.random(1)[0]
        get_nb_units = lambda: np.random.randint(10, 100)
        get_N0 = lambda: np.random.randint(1, 5e3)
        get_min_eps = lambda: 1e-4 + (2e-1 - 1e-4) * np.random.random(1)[0]
        get_initial_mean = lambda: 0
        get_initial_stddev = lambda: 1e-4 + (5e-1 - 1e-4) * np.random.random(1)[0]

        random_config = {
            'lr': get_lr()
            , 'discount': get_discount()
            , 'nb_units': get_nb_units()
            , 'N0': get_N0()
            , 'min_eps': get_min_eps()
            , 'initial_mean': get_initial_mean()
            , 'initial_stddev': get_initial_stddev()
        }
        random_config.update(fixed_params)

        return random_config

    def build_graph(self, graph):
        with graph.as_default():
            tf.set_random_seed(self.random_seed)

            self.N0_t = tf.constant(self.N0, tf.float32, name='N_0')
            self.N = tf.Variable(0., dtype=tf.float32, name='N', trainable=False)
            self.min_eps_t = tf.constant(self.min_eps, tf.float32, name='min_eps')

            self.inputs = tf.placeholder(tf.float32, shape=[None, self.observation_space.shape[0] + 1], name='inputs')

            q_scope = tf.VariableScope(reuse=False, name='QValues')
            with tf.variable_scope(q_scope):
                self.q_values = tf.squeeze(capacities.value_f(self.q_params, self.inputs))

            self.action_t = capacities.epsGreedy(
                self.inputs, self.q_values, self.env.action_space.n, self.N0, self.min_eps
            )
            self.q_t = self.q_values[self.action_t]

            with tf.variable_scope('Training'):
                self.reward = tf.placeholder(tf.float32, shape=[], name="reward")
                self.next_state = tf.placeholder(tf.float32, shape=[1, self.observation_space.shape[0] + 1], name="nextState")
                self.next_action = tf.placeholder(tf.int32, shape=[], name="nextAction")
                with tf.variable_scope(q_scope, reuse=True):
                    next_q_values = tf.squeeze(capacities.value_f(self.q_params, self.next_state))
                target_q1 = tf.stop_gradient(self.reward + self.discount * next_q_values[self.next_action])
                target_q2 = self.reward
                is_done = tf.equal(self.next_state[0, 4], 1)
                target_q = tf.cond(is_done, lambda: target_q2, lambda: target_q1)
                with tf.control_dependencies([target_q]):
                    self.loss = 1/2 * tf.square(target_q - self.q_t)

                adam = tf.train.AdamOptimizer(self.lr)
                self.global_step = tf.Variable(0, trainable=False, name="global_step", collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
                self.train_op = adam.minimize(self.loss, global_step=self.global_step)


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

    def act(self, obs):
        state = [ np.concatenate((obs, [0])) ]
        act = self.sess.run(self.action_t, feed_dict={
            self.inputs: state
        })

        return (act, state)

    def learn_from_episode(self, env, render):
        obs = env.reset()
        act, _ = self.act(obs)

        av_loss = []
        score = 0
        done = False

        while True:
            if render:
                env.render()

            next_obs, reward, done, info = env.step(act)
            next_act, _ = self.act(next_obs)

            loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={
                self.inputs: [ np.concatenate((obs, [0])) ],
                self.action_t: act,
                self.reward: reward,
                self.next_state: [ np.concatenate((next_obs, [1 if done else 0])) ],
                self.next_action: next_act
            })
            av_loss.append(loss)

            score += reward
            obs = next_obs
            act = next_act
            if done:
                break

        summary, _, episode_id = self.sess.run([self.all_summary_t, self.inc_ep_id_op, self.episode_id], feed_dict={
            self.score_plh: score,
            self.loss_plh: np.mean(av_loss)
        })
        self.sw.add_summary(summary, episode_id)

        return

class DQNAgent(DeepTDAgent):
    """
    Agent implementing The DQN (Experience replay abd fixed Q-Net).
    """
    def set_agent_props(self):
        super(DQNAgent, self).set_agent_props()

        self.er_every = self.config['er_every']
        self.er_batch_size = self.config['er_batch_size']
        self.er_rm_size = self.config['er_rm_size']

        self.replayMemoryDt = np.dtype([
            ('states', 'float32', (self.observation_space.shape[0] + 1,))
            , ('actions', 'int32')
            , ('rewards', 'float32')
            , ('next_states', 'float32', (self.observation_space.shape[0] + 1,))
        ])
        self.replayMemory = np.array([], dtype=self.replayMemoryDt)

    def get_best_config(self, env_name=""):
        return {
            'lr': 9e-4
            , 'nb_units': 22
            , 'discount': 0.999
            , 'N0': 2200
            , 'min_eps': 0.001
            , 'initial_mean': 0.
            , 'initial_stddev': 0.2776861144026909
            , 'er_every': 201
            , 'er_batch_size': 208
            , 'er_rm_size': 40000
        }

    @staticmethod
    def get_random_config(fixed_params={}):
        get_lr = lambda: 1e-4 + (1 - 1e-4) * np.random.random(1)[0]
        get_discount = lambda: 0.5 + (1 - 0.5) * np.random.random(1)[0]
        get_nb_units = lambda: np.random.randint(10, 100)
        get_N0 = lambda: np.random.randint(1, 5e3)
        get_min_eps = lambda: 1e-4 + (2e-1 - 1e-4) * np.random.random(1)[0]
        get_initial_mean = lambda: 0
        get_initial_stddev = lambda: 1e-4 + (5e-1 - 1e-4) * np.random.random(1)[0]
        get_er_batch_size = lambda: np.random.randint(16, 1024)
        get_er_rm_size = lambda: np.random.randint(10000, 50000)
        get_er_every = lambda: np.random.randint(1, 1000)

        random_config = {
            'lr': get_lr()
            , 'discount': get_discount()
            , 'nb_units': get_nb_units()
            , 'N0': get_N0()
            , 'min_eps': get_min_eps()
            , 'initial_mean': get_initial_mean()
            , 'initial_stddev': get_initial_stddev()
            , 'er_batch_size': get_er_batch_size()
            , 'er_rm_size': get_er_rm_size()
            , 'er_every': get_er_every()
        }
        random_config.update(fixed_params)

        return random_config

    def build_graph(self, graph):
        with graph.as_default():
            tf.set_random_seed(self.random_seed)

            self.inputs = tf.placeholder(tf.float32, shape=[None, self.observation_space.shape[0] + 1], name='inputs')

            q_scope = tf.VariableScope(reuse=False, name='QValues')
            with tf.variable_scope(q_scope):
                self.q_values = tf.squeeze(capacities.value_f(self.q_params, self.inputs))

            self.action_t = capacities.epsGreedy(
                self.inputs, self.q_values, self.env.action_space.n, self.N0, self.min_eps
            )
            self.q_t = self.q_values[self.action_t]

            fixed_q_scope = tf.VariableScope(reuse=False, name='FixedQValues')
            with tf.variable_scope(fixed_q_scope):
                self.update_fixed_vars_op = capacities.fixScope(q_scope)

            with tf.variable_scope('ExperienceReplay'):
                self.er_inputs = tf.placeholder(tf.float32, shape=[None, self.observation_space.shape[0] + 1], name="ERInputs")
                self.er_actions = tf.placeholder(tf.int32, shape=[None], name="ERInputs")
                self.er_rewards = tf.placeholder(tf.float32, shape=[None], name="ERReward")
                self.er_next_states = tf.placeholder(tf.float32, shape=[None, self.observation_space.shape[0] + 1], name="ERNextState")

                with tf.variable_scope(q_scope, reuse=True):
                    er_q_values = capacities.value_f(self.q_params, self.er_inputs)
                er_stacked_actions = tf.stack([tf.range(0, tf.shape(self.er_actions)[0]), self.er_actions], 1)
                er_qs = tf.gather_nd(er_q_values, er_stacked_actions)

                with tf.variable_scope(fixed_q_scope, reuse=True):
                    er_next_q_values = capacities.value_f(self.q_params, self.er_next_states)
                er_next_max_action_t = tf.cast(tf.argmax(er_next_q_values, 1), tf.int32)
                er_next_stacked_actions = tf.stack([tf.range(0, tf.shape(self.er_next_states)[0]), er_next_max_action_t], 1)
                er_next_qs = tf.gather_nd(er_next_q_values, er_next_stacked_actions)

                er_target_qs1 = tf.stop_gradient(self.er_rewards + self.discount * er_next_qs)
                er_target_qs2 = self.er_rewards
                er_stacked_targets = tf.stack([er_target_qs1, er_target_qs2], 1)
                select_targets = tf.stack([tf.range(0, tf.shape(self.er_next_states)[0]), tf.cast(self.er_next_states[:, -1], tf.int32)], 1)
                er_target_qs = tf.gather_nd(er_stacked_targets, select_targets)

                self.er_loss = 1/2 * tf.reduce_sum(tf.square(er_target_qs - er_qs))
                er_adam = tf.train.AdamOptimizer(self.lr)
                self.global_step = tf.Variable(0, trainable=False, name="global_step", collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
                self.er_train_op = er_adam.minimize(self.er_loss, global_step=self.global_step)

            self.score_plh = tf.placeholder(tf.float32, shape=[])
            self.score_sum_t = tf.summary.scalar('score', self.score_plh)
            self.loss_plh = tf.placeholder(tf.float32, shape=[])
            self.loss_sum_t = tf.summary.scalar('loss', self.loss_plh)
            self.all_summary_t = tf.summary.merge_all()

            self.episode_id, self.inc_ep_id_op = capacities.counter("episode_id")
            self.timestep, self.inc_timestep_op = capacities.counter("timestep")

            # Playing part
            self.pscore_plh = tf.placeholder(tf.float32, shape=[])
            self.pscore_sum_t = tf.summary.scalar('play_score', self.pscore_plh)

        return graph

    def act(self, obs):
        state = np.concatenate( (obs, [0]) )
        act = self.sess.run(self.action_t, feed_dict={
            self.inputs: [ state ]
        })

        return (act, state)

    def learn_from_episode(self, env, render):
        obs = env.reset()
        score = 0
        av_loss = []
        done = False

        while True:
            if render:
                env.render()

            act, state = self.act(obs)
            next_obs, reward, done, info = env.step(act)
            next_state = np.concatenate( (next_obs, [1. if done else 0.]) )

            if self.replayMemory.shape[0] >= self.er_rm_size:
                self.replayMemory = np.delete(self.replayMemory, 0)
            memory = np.array([(state, act, reward, next_state)], dtype=self.replayMemoryDt)
            self.replayMemory = np.append(self.replayMemory, memory)

            memories = np.random.choice(self.replayMemory, self.er_batch_size)
            loss, _, timestep, _ = self.sess.run([self.er_loss, self.inc_timestep_op, self.timestep, self.er_train_op], feed_dict={
                self.er_inputs: memories['states'],
                self.er_actions: memories['actions'],
                self.er_rewards: memories['rewards'],
                self.er_next_states: memories['next_states'],
            })
            if timestep % self.er_every == 0:
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

class DDQNAgent(DQNAgent):
    """
    Agent implementing The DDQN
    """
    def get_best_config(self, env_name=""):
        return {
            'lr': 8e-4
            , 'nb_units': 83
            , 'discount': 0.999
            , 'N0': 2571
            , 'min_eps': 0.001
            , 'initial_mean': 0.
            , 'initial_stddev': 0.39560728810993573
            , 'er_every': 116
            , 'er_batch_size': 259
            , 'er_rm_size': 40000
        }

    def build_graph(self, graph):
        with graph.as_default():
            tf.set_random_seed(self.random_seed)

            self.inputs = tf.placeholder(tf.float32, shape=[None, self.observation_space.shape[0] + 1], name='inputs')

            q_scope = tf.VariableScope(reuse=False, name='QValues')
            with tf.variable_scope(q_scope):
                self.q_values = tf.squeeze(capacities.value_f(self.q_params, self.inputs))

            self.action_t = capacities.epsGreedy(
                self.inputs, self.q_values, self.env.action_space.n, self.N0, self.min_eps
            )
            self.q_t = self.q_values[self.action_t]

            fixed_q_scope = tf.VariableScope(reuse=False, name='FixedQValues')
            with tf.variable_scope(fixed_q_scope):
                self.update_fixed_vars_op = capacities.fixScope(q_scope)

            with tf.variable_scope('ExperienceReplay'):
                self.er_inputs = tf.placeholder(tf.float32, shape=[None, self.observation_space.shape[0] + 1], name="ERInputs")
                self.er_actions = tf.placeholder(tf.int32, shape=[None], name="ERInputs")
                self.er_rewards = tf.placeholder(tf.float32, shape=[None], name="ERReward")
                self.er_next_states = tf.placeholder(tf.float32, shape=[None, self.observation_space.shape[0] + 1], name="ERNextState")

                with tf.variable_scope(q_scope, reuse=True):
                    er_q_values = capacities.value_f(self.q_params, self.er_inputs)
                er_stacked_actions = tf.stack([tf.range(0, tf.shape(self.er_actions)[0]), self.er_actions], 1)
                er_qs = tf.gather_nd(er_q_values, er_stacked_actions)

                with tf.variable_scope(fixed_q_scope, reuse=True):
                    er_fixed_next_q_values = capacities.value_f(self.q_params, self.er_next_states)
                with tf.variable_scope(q_scope, reuse=True):
                    er_next_q_values = capacities.value_f(self.q_params, self.er_next_states)
                er_next_max_action_t = tf.cast(tf.argmax(er_next_q_values, 1), tf.int32)
                er_next_stacked_actions = tf.stack([tf.range(0, tf.shape(self.er_next_states)[0]), er_next_max_action_t], 1)
                er_next_qs = tf.gather_nd(er_fixed_next_q_values, er_next_stacked_actions)

                er_target_qs1 = tf.stop_gradient(self.er_rewards + self.discount * er_next_qs)
                er_target_qs2 = self.er_rewards
                er_stacked_targets = tf.stack([er_target_qs1, er_target_qs2], 1)
                select_targets = tf.stack([tf.range(0, tf.shape(self.er_next_states)[0]), tf.cast(self.er_next_states[:, -1], tf.int32)], 1)
                er_target_qs = tf.gather_nd(er_stacked_targets, select_targets)

                self.er_loss = 1/2 * tf.reduce_sum(tf.square(er_target_qs - er_qs))
                er_adam = tf.train.AdamOptimizer(self.lr)
                self.global_step = tf.Variable(0, trainable=False, name="global_step", collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
                self.er_train_op = er_adam.minimize(self.er_loss, global_step=self.global_step)

            self.score_plh = tf.placeholder(tf.float32, shape=[])
            self.score_sum_t = tf.summary.scalar('score', self.score_plh)
            self.loss_plh = tf.placeholder(tf.float32, shape=[])
            self.loss_sum_t = tf.summary.scalar('loss', self.loss_plh)
            self.all_summary_t = tf.summary.merge_all()

            self.episode_id, self.inc_ep_id_op = capacities.counter("episode_id")
            self.timestep, self.inc_timestep_op = capacities.counter("timestep")

            # Playing part
            self.pscore_plh = tf.placeholder(tf.float32, shape=[])
            self.pscore_sum_t = tf.summary.scalar('play_score', self.pscore_plh)

        return graph

class DeepFixedQOfflineERAgent(DQNAgent):
    """
    Agent implementing 2-layer NN Q-learning, using experience replay, fixed Q Network and TD(0).
    """
    def set_agent_props(self):
        super(DeepFixedQOfflineERAgent, self).set_agent_props()

        self.er_epoch_size = self.config['er_epoch_size']

    def get_best_config(self, env_name=""):
        return {
            "initial_mean": 0,
            "lr": 0.0001,
            "er_batch_size": 805,
            "er_epoch_size": 63,
            "er_every": 1,
            "initial_stddev": 0.2608188033316199,
            "min_eps": 0.001,
            "N0": 1393,
            "nb_units": 20,
            "discount": 0.999,
            "er_rm_size": 36196
          }

    @staticmethod
    def get_random_config(fixed_params={}):
        get_lr = lambda: 1e-4 + (1 - 1e-4) * np.random.random(1)[0]
        get_discount = lambda: 0.5 + (1 - 0.5) * np.random.random(1)[0]
        get_nb_units = lambda: np.random.randint(10, 100)
        get_N0 = lambda: np.random.randint(1, 5e3)
        get_min_eps = lambda: 1e-4 + (2e-1 - 1e-4) * np.random.random(1)[0]
        get_initial_mean = lambda: 0
        get_initial_stddev = lambda: 1e-4 + (5e-1 - 1e-4) * np.random.random(1)[0]
        get_er_batch_size = lambda: np.random.randint(16, 1024)
        get_er_rm_size = lambda: np.random.randint(10000, 50000)
        get_er_epoch_size = lambda: np.random.randint(1, 200)
        get_er_every = lambda: np.random.randint(1, 1000)

        random_config = {
            'lr': get_lr()
            , 'discount': get_discount()
            , 'nb_units': get_nb_units()
            , 'N0': get_N0()
            , 'min_eps': get_min_eps()
            , 'initial_mean': get_initial_mean()
            , 'initial_stddev': get_initial_stddev()
            , 'er_batch_size': get_er_batch_size()
            , 'er_rm_size': get_er_rm_size()
            , 'er_epoch_size': get_er_epoch_size()
            , 'er_every': get_er_every()
        }
        random_config.update(fixed_params)

        return random_config

    def build_graph(self, graph):
        with graph.as_default():
            tf.set_random_seed(self.random_seed)

            self.inputs = tf.placeholder(tf.float32, shape=[None, self.observation_space.shape[0] + 1], name='inputs')

            q_scope = tf.VariableScope(reuse=False, name='QValues')
            with tf.variable_scope(q_scope):
                self.q_values = tf.squeeze(capacities.value_f(self.q_params, self.inputs))

            self.action_t = capacities.epsGreedy(
                self.inputs, self.q_values, self.env.action_space.n, self.N0, self.min_eps
            )
            self.q_t = self.q_values[self.action_t]

            fixed_q_scope = tf.VariableScope(reuse=False, name='FixedQValues')
            with tf.variable_scope(fixed_q_scope):
                self.update_fixed_vars_op = capacities.fixScope(q_scope)

            with tf.variable_scope('Training'):
                self.reward = tf.placeholder(tf.float32, shape=[], name="reward")
                self.next_state = tf.placeholder(tf.float32, shape=[1, self.observation_space.shape[0] + 1], name="nextState")
                with tf.variable_scope(q_scope, reuse=True):
                    next_q_values = tf.squeeze(capacities.value_f(self.q_params, self.next_state))
                next_max_action_t = tf.cast(tf.argmax(next_q_values, 0), tf.int32)
                target_q1 = tf.stop_gradient(self.reward + self.discount * next_q_values[next_max_action_t])
                target_q2 = self.reward
                is_done = tf.equal(self.next_state[0, 4], 1)
                target_q = tf.cond(is_done, lambda: target_q2, lambda: target_q1)
                with tf.control_dependencies([target_q]):
                    self.loss = 1/2 * tf.square(target_q - self.q_t)

                adam = tf.train.AdamOptimizer(self.lr)
                self.global_step = tf.Variable(0, trainable=False, name="global_step", collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
                self.train_op = adam.minimize(self.loss, global_step=self.global_step)

            with tf.variable_scope('ExperienceReplay'):
                self.er_inputs = tf.placeholder(tf.float32, shape=[None, self.observation_space.shape[0] + 1], name="ERInputs")
                self.er_actions = tf.placeholder(tf.int32, shape=[None], name="ERInputs")
                self.er_rewards = tf.placeholder(tf.float32, shape=[None], name="ERReward")
                self.er_next_states = tf.placeholder(tf.float32, shape=[None, self.observation_space.shape[0] + 1], name="ERNextState")

                with tf.variable_scope(q_scope, reuse=True):
                    er_q_values = capacities.value_f(self.q_params, self.er_inputs)
                er_stacked_actions = tf.stack([tf.range(0, tf.shape(self.er_actions)[0]), self.er_actions], 1)
                er_qs = tf.gather_nd(er_q_values, er_stacked_actions)

                with tf.variable_scope(fixed_q_scope, reuse=True):
                    er_next_q_values = capacities.value_f(self.q_params, self.er_next_states)
                er_next_max_action_t = tf.cast(tf.argmax(er_next_q_values, 1), tf.int32)
                er_next_stacked_actions = tf.stack([tf.range(0, tf.shape(self.er_next_states)[0]), er_next_max_action_t], 1)
                er_next_qs = tf.gather_nd(er_next_q_values, er_next_stacked_actions)

                er_target_qs1 = tf.stop_gradient(self.er_rewards + self.discount * er_next_qs)
                er_target_qs2 = self.er_rewards
                er_stacked_targets = tf.stack([er_target_qs1, er_target_qs2], 1)
                select_targets = tf.stack([tf.range(0, tf.shape(self.er_next_states)[0]), tf.cast(self.er_next_states[:, -1], tf.int32)], 1)
                er_target_qs = tf.gather_nd(er_stacked_targets, select_targets)
                er_loss = 1/2 * tf.reduce_sum(tf.square(er_target_qs - er_qs))
                er_adam = tf.train.AdamOptimizer(self.lr)
                self.er_train_op = er_adam.minimize(er_loss)

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

    def learn_from_episode(self, env, render):
        obs = env.reset()
        score = 0
        av_loss = []
        done = False

        while True:
            if render:
                env.render()

            act, state = self.act(obs)
            next_obs, reward, done, info = env.step(act)
            next_state = np.concatenate( (next_obs, [1. if done else 0.]) )

            if self.replayMemory.shape[0] >= self.er_rm_size:
                self.replayMemory = np.delete(self.replayMemory, 0)
            memory = np.array([(state, act, reward, next_state)], dtype=self.replayMemoryDt)
            self.replayMemory = np.append(self.replayMemory, memory)

            loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={
                self.inputs: [ state ],
                self.action_t: act,
                self.reward: reward,
                self.next_state: [ next_state ]
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

        if episode_id % self.er_every == 0:
            self.sess.run(self.update_fixed_vars_op)

            for i in range(self.er_epoch_size):
                memories = np.random.choice(self.replayMemory, self.er_batch_size)
                _ = self.sess.run(self.er_train_op, feed_dict={
                    self.er_inputs: memories['states'],
                    self.er_actions: memories['actions'],
                    self.er_rewards: memories['rewards'],
                    self.er_next_states: memories['next_states'],
                })

        return
