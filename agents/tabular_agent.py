import numpy as np
import tensorflow as tf

from agents import BasicAgent, phis, capacities
from agents.capacities import get_expected_rewards, get_lambda_expected_rewards

class TabularMCAgent(BasicAgent):
    """
    Agent implementing tabular Q-learning.
    """
    def __init__(self, config, env):
        if 'debug' in config:
            config.update(phis.getPhiConfig(config['env_name'], config['debug']))
        else:
            config.update(phis.getPhiConfig(config['env_name']))
        super(TabularMCAgent, self).__init__(config, env)

    def set_agent_props(self):
        self.N0 = self.config['N0']
        self.min_eps = self.config['min_eps']
        self.initial_q_value = self.config['initial_q_value']

    def get_best_config(self, env_name=""):
        return {
            'discount': 0.999 # ->1[ improve
            , 'N0': 76 # -> ~ 75 improve
            , 'min_eps': 0.001 # ->0.001[ improve
            , 'initial_q_value': 0
        }

    @staticmethod
    def get_random_config(fixed_params={}):
        get_discount = lambda: 0.5 + (1 - 0.5) * np.random.random(1)[0]
        get_N0 = lambda: np.random.randint(1, 5e3)
        get_min_eps = lambda: 1e-4 + (1e-1 - 1e-4) * np.random.random(1)[0]
        get_initial_q_value = lambda: 0 # int(np.random.random(1)[0] * 200)

        random_config = {
            'discount': get_discount()
            , 'N0': get_N0()
            , 'min_eps': get_min_eps()
            , 'initial_q_value': get_initial_q_value()
        }
        random_config.update(fixed_params)

        return random_config

    def build_graph(self, graph):
        with graph.as_default():
            tf.set_random_seed(self.random_seed)

            self.inputs = tf.placeholder(tf.int32, shape=[None, 1], name="inputs")

            q_scope = tf.VariableScope(reuse=False, name='QValues')
            with tf.variable_scope(q_scope):
                self.Qs = tf.get_variable('Qs'
                    , shape=[self.nb_state, self.action_space.n]
                    , initializer=tf.constant_initializer(self.initial_q_value)
                    , dtype=tf.float32
                )
                tf.summary.histogram('Qarray', self.Qs)
                self.q_preds = tf.squeeze(tf.nn.embedding_lookup(self.Qs, self.inputs), 1)

            policy_scope = tf.VariableScope(reuse=False, name='EpsilonGreedyPolicy')
            with tf.variable_scope(policy_scope):
                self.actions_t = capacities.batch_eps_greedy(
                    self.inputs, self.q_preds, self.env.action_space.n, self.N0, self.min_eps, self.nb_state
                )
                self.action_t = self.actions_t[0]
            self.q_value_t = self.q_preds[0][self.action_t]

            learning_scope = tf.VariableScope(reuse=False, name='MCLearning')
            with tf.variable_scope(learning_scope):
                self.Gs_ph = tf.placeholder(tf.float32, shape=[None], name="reward")
                self.actions_ph = tf.placeholder(tf.int32, shape=[None, 1], name="actions_taken")
                Ns = tf.Variable(tf.zeros(shape=[self.nb_state, self.action_space.n]), name='N', trainable=False)
                global_step = tf.Variable(0, trainable=False, name="global_step", collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
                inc_global_step = global_step.assign_add(1)

                state_action_pairs = tf.squeeze(tf.stack([self.inputs, self.actions_ph], 1), -1)
                update_Ns = tf.scatter_nd_add(Ns, state_action_pairs, tf.ones(shape=[tf.shape(self.inputs)[0]]))
                with tf.control_dependencies([update_Ns, inc_global_step]):
                    err_estimates = (1 / tf.gather_nd(Ns, state_action_pairs)) * (self.Gs_ph - tf.gather_nd(self.Qs, state_action_pairs))
                    self.loss = tf.reduce_sum(err_estimates)
                    self.train_op = tf.scatter_nd_add(self.Qs, state_action_pairs, err_estimates)

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
        state_id = self.phi(obs)
        act = self.sess.run(self.action_t, feed_dict={
            self.inputs: [ [state_id] ]
        })

        return act, state_id

    def learn_from_episode(self, env, render=False):
        score = 0
        historyType = np.dtype([('states', 'int32', (1,)), ('actions', 'int32', (1,)), ('rewards', 'float32')])
        history = np.array([], dtype=historyType)
        done = False

        obs = env.reset()
        while not done:
            if render:
                env.render()

            act, state_id = self.act(obs)
            obs, reward, done, info = env.step(act)

            memory = np.array([(state_id, act, reward)], dtype=historyType)
            history = np.append(history, memory)

            score += reward

        _, loss = self.sess.run([self.train_op, self.loss], feed_dict={
            self.inputs: history['states'],
            self.actions_ph: history['actions'],
            self.Gs_ph: get_expected_rewards(history['rewards'], self.discount),
        })
        summary, _, episode_id = self.sess.run([self.all_summary_t, self.inc_ep_id_op, self.episode_id], feed_dict={
            self.score_plh: score,
            self.loss_plh: loss
        })
        self.sw.add_summary(summary, episode_id)
        
class TabularLambdaAgent(TabularMCAgent):
    """
    Agent implementing tabular lambda return
    """

    def set_agent_props(self):
        self.lambda_value = self.config['lambda']
        super(TabularLambdaAgent, self).set_agent_props()

    def get_best_config(self, env_name=""):
        return {
            'discount': 0.999 # ->1[ improve
            , 'N0': 76 # -> ~ 75 improve
            , 'min_eps': 0.001 # ->0.001[ improve
            , 'initial_q_value': 0
            , 'lambda': 0.9
        }

    @staticmethod
    def get_random_config(fixed_params={}):
        get_discount = lambda: 0.5 + (1 - 0.5) * np.random.random(1)[0]
        get_N0 = lambda: np.random.randint(1, 5e3)
        get_min_eps = lambda: 1e-4 + (1e-1 - 1e-4) * np.random.random(1)[0]
        get_initial_q_value = lambda: 0 # int(np.random.random(1)[0] * 200)
        get_lambda = lambda: np.random.rand() 

        random_config = {
            'discount': get_discount()
            , 'N0': get_N0()
            , 'min_eps': get_min_eps()
            , 'initial_q_value': get_initial_q_value()
            , 'lambda': get_lambda()
        }
        random_config.update(fixed_params)

        return random_config

    def learn_from_episode(self, env, render=False):
        score = 0        
        historyType = np.dtype([('states', 'int32', (1,)), ('actions', 'int32', (1,)), ('rewards', 'float32'), ('estimates', 'float32')])
        history = np.array([], dtype=historyType)
        done = False
        
        obs = env.reset()
        act, state_id = self.act(obs)
        while not done:
            if render:
                env.render()
            
            next_obs, reward, done, info = env.step(act)
            next_act, next_state_id = self.act(next_obs)
            next_estimate = self.sess.run(self.q_value_t, feed_dict={
                self.inputs: [ [next_state_id] ]
                , self.action_t: next_act
            })

            memory = np.array([(state_id, act, reward, next_estimate)], dtype=historyType)
            history = np.append(history, memory)

            score += reward
            act = next_act
            state_id = next_state_id

        _, loss = self.sess.run([self.train_op, self.loss], feed_dict={
            self.inputs: history['states'],
            self.actions_ph: history['actions'],
            self.Gs_ph: get_lambda_expected_rewards(history['rewards'], history['estimates'], self.discount, self.lambda_value),
        })
        summary, _, episode_id = self.sess.run([self.all_summary_t, self.inc_ep_id_op, self.episode_id], feed_dict={
            self.score_plh: score,
            self.loss_plh: loss
        })
        self.sw.add_summary(summary, episode_id)

class TabularQAgent(BasicAgent):
    """
    Agent implementing tabular Q-learning.
    """
    def __init__(self, config, env):
        config.update(phis.getPhiConfig(config['env_name'], config['debug']))
        super(TabularQAgent, self).__init__(config, env)

    def set_agent_props(self):
        self.N0 = self.config['N0']
        self.min_eps = self.config['min_eps']
        self.initial_q_value = self.config['initial_q_value']

    def get_best_config(self, env_name=""):
        carpolev0 = {
            'lr': 0.1
            , 'discount': 0.999 # ->1[ improve
            , 'N0': 76 # -> ~ 75 improve
            , 'min_eps': 0.001 # ->0.001[ improve
            , 'initial_q_value': 0
        }
        mountaincarv0 = {
            'lr': 0.1
            , 'discount': 0.999
            , 'N0': 75
            , 'min_eps': 0.001
            , 'initial_q_value': 0
        }
        return {
            'CartPole-v0': carpolev0
            , 'MountainCar-v0': mountaincarv0
        }.get(env_name, carpolev0)
        

    @staticmethod
    def get_random_config(fixed_params={}):
        get_lr = lambda: 1e-3 + (1 - 1e-3) * np.random.random(1)[0]
        get_discount = lambda: 0.5 + (1 - 0.5) * np.random.random(1)[0]
        get_N0 = lambda: np.random.randint(1, 5e3)
        get_min_eps = lambda: 1e-4 + (1e-1 - 1e-4) * np.random.random(1)[0]
        get_initial_q_value = lambda: 0 # int(np.random.random(1)[0] * 200)

        random_config = {
            'lr': get_lr()
            , 'discount': get_discount()
            , 'N0': get_N0()
            , 'min_eps': get_min_eps()
            , 'initial_q_value': get_initial_q_value()
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

            learning_scope = tf.VariableScope(reuse=False, name='MSEQLearning')
            with tf.variable_scope(learning_scope):
                self.reward = tf.placeholder(tf.float32, shape=[], name="reward")
                self.next_state = tf.placeholder(tf.int32, shape=[], name="nextState")

                self.loss = capacities.mse_tabular_q_learning(
                    self.Qs, self.reward, self.next_state, self.discount, self.q_preds, self.action_t
                )

            with tf.variable_scope('Training'):
                global_step = tf.Variable(0, trainable=False, name="global_step", collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
                optimizer = tf.train.AdamOptimizer(self.lr)
                self.train_op = optimizer.minimize(self.loss, global_step=global_step)

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
        state_id = self.phi(obs)
        act = self.sess.run(self.action_t, feed_dict={
            self.inputs: state_id
        })

        return act, state_id

    def learn_from_episode(self, env, render=False):
        obs = env.reset()
        score = 0
        av_loss = []
        done = False

        while not done:
            if render:
                env.render()

            act, state_id = self.act(obs)
            next_obs, reward, done, info = env.step(act)

            next_state_id = self.phi(next_obs, done)
            loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={
                self.inputs: state_id,
                self.action_t: act,
                self.reward: reward,
                self.next_state: next_state_id,
            })

            av_loss.append(loss)
            score += reward
            obs = next_obs

        summary, _, episode_id = self.sess.run([self.all_summary_t, self.inc_ep_id_op, self.episode_id], feed_dict={
            self.score_plh: score,
            self.loss_plh: np.mean(av_loss),
        })
        self.sw.add_summary(summary, episode_id)

        return

# class TabularNStepAgent(TabularQAgent):
#     """
#     Agent implementing tabular Q-learning.
#     """
#     def set_agent_props(self):
#         self.n_step = self.config['n_step']

#         super(TabularNStepAgent, self).__init__(config, env)

#     def get_best_config(self, env_name=""):
#         return {
#             'lr': 0.1
#             , 'discount': 0.999 # ->1[ improve
#             , 'N0': 76 # -> ~ 75 improve
#             , 'min_eps': 0.001 # ->0.001[ improve
#             , 'initial_q_value': 0
#             , 'n_step': 4
#         }
        

#     @staticmethod
#     def get_random_config(fixed_params={}):
#         get_lr = lambda: 1e-3 + (1 - 1e-3) * np.random.random(1)[0]
#         get_discount = lambda: 0.5 + (1 - 0.5) * np.random.random(1)[0]
#         get_N0 = lambda: np.random.randint(1, 5e3)
#         get_min_eps = lambda: 1e-4 + (1e-1 - 1e-4) * np.random.random(1)[0]
#         get_initial_q_value = lambda: 0 # int(np.random.random(1)[0] * 200)
#         get_n_step = lambda: np.random.randint(1, 200)

#         random_config = {
#             'lr': get_lr()
#             , 'discount': get_discount()
#             , 'N0': get_N0()
#             , 'min_eps': get_min_eps()
#             , 'initial_q_value': get_initial_q_value()
#             , 'n_step': get_n_step()
#         }
#         random_config.update(fixed_params)

#         return random_config

#     def build_graph(self, graph):
#         with graph.as_default():
#             tf.set_random_seed(self.random_seed)

#             self.inputs = tf.placeholder(tf.int32, shape=[], name="inputs")

#             q_scope = tf.VariableScope(reuse=False, name='QValues')
#             with tf.variable_scope(q_scope):
#                 self.Qs = tf.get_variable('Qs'
#                     , shape=[self.nb_state, self.action_space.n]
#                     , initializer=tf.constant_initializer(self.initial_q_value)
#                     , dtype=tf.float32
#                 )
#                 tf.summary.histogram('Qarray', self.Qs)
#                 self.q_preds = self.Qs[self.inputs]

#             policy_scope = tf.VariableScope(reuse=False, name='EpsilonGreedyPolicy')
#             with tf.variable_scope(policy_scope):
#                 self.action_t = capacities.eps_greedy(
#                     self.inputs, self.q_preds, self.env.action_space.n, self.N0, self.min_eps, self.nb_state
#                 )

#             learning_scope = tf.VariableScope(reuse=False, name='NStepLearning')
#             with tf.variable_scope(learning_scope):
#                 self.G_n = tf.placeholder(tf.float32, shape=[], name="G_n")
#                 self.n_step_state = tf.placeholder(tf.int32, shape=[], name="nextState")
#                 Ns = tf.Variable(tf.zeros(shape=[self.nb_state, self.action_space.n]), name='N', trainable=False)
#                 global_step = tf.Variable(0, trainable=False, name="global_step", collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
#                 inc_global_step = global_step.assign_add(1)

#                 update_Ns = tf.scatter_nd_add(Ns, tf.stack([self.inputs, self.actions_t], 1), tf.ones(shape=[tf.shape(self.inputs)[0]]))
#                 with tf.control_dependencies([update_Ns, inc_global_step]):
#                     err_estimate = (1 / Ns[self.inputs, self.actions_t]) * (self.G_n - self.Qs[self.inputs, self.actions_t]))
#                     self.loss = tf.reduce_sum(err_estimates)
#                     self.train_op = tf.scatter_nd_add(self.Qs, tf.stack([self.inputs, self.actions_t], 1), err_estimate)
            

#             self.score_plh = tf.placeholder(tf.float32, shape=[])
#             self.score_sum_t = tf.summary.scalar('score', self.score_plh)
#             self.loss_plh = tf.placeholder(tf.float32, shape=[])
#             self.loss_sum_t = tf.summary.scalar('loss', self.loss_plh)
#             self.all_summary_t = tf.summary.merge_all()

#             self.episode_id, self.inc_ep_id_op = capacities.counter("episode_id")

#             # Playing part
#             self.pscore_plh = tf.placeholder(tf.float32, shape=[])
#             self.pscore_sum_t = tf.summary.scalar('play_score', self.pscore_plh)

#         return graph

#     def learn_from_episode(self, env, render=False):
#         obs = env.reset()
#         score = 0
        
#         historyType = np.dtype([('states', 'int32', (1,)), ('actions', 'int32', (1,)), ('rewards', 'float32')])
#         history = np.array([], dtype=historyType)

#         done = False
#         timestep = 0

#         while not done:
#             if render:
#                 env.render()

#             act, state_id = self.act(obs)
#             next_obs, reward, done, info = env.step(act)

#             memory = np.array([(state_id, act, reward)], dtype=historyType)
#             history = np.append(history, memory)

#             if timestep >= self.n_step:
#                 n_step_state = self.phi(next_obs, done)
#                 loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={
#                     self.inputs: history['states'][timestep - n_step],
#                     self.action_t: history['actions'][timestep - n_step],
#                     self.G_n: get_expected_rewards(history['actions'][timestep - n_step:])[0],
#                     self.n_step_state: n_step_state,
#                 })

#             timestep += 1
#             score += reward
#             obs = next_obs

#         summary, _, episode_id = self.sess.run([self.all_summary_t, self.inc_ep_id_op, self.episode_id], feed_dict={
#             self.score_plh: score,
#             self.loss_plh: np.mean(av_loss),
#         })
#         self.sw.add_summary(summary, episode_id)

#         return

class BackwardTabularQAgent(TabularQAgent):
    """
    Agent implementing Backward TD(lambda) tabular Q-learning.
    """
    def set_agent_props(self):
        super(BackwardTabularQAgent, self).set_agent_props()

        self.lambda_value = self.config['lambda']

    def get_best_config(self, env_name=""):
        return {
            'lr': 0.0001
            , 'discount': 0.999
            , 'N0': 220
            , 'min_eps': 0.005
            , 'initial_q_value': 0
            , 'lambda': .99
        }

    @staticmethod
    def get_random_config(fixed_params={}):
        get_lr = lambda: 1e-3 + (1 - 1e-3) * np.random.random(1)[0]
        get_discount = lambda: 0.5 + (1 - 0.5) * np.random.random(1)[0]
        get_N0 = lambda: np.random.randint(1, 5e3)
        get_min_eps = lambda: 1e-4 + (1e-1 - 1e-4) * np.random.random(1)[0]
        get_initial_q_value = lambda: 0 # int(np.random.random(1)[0] * 200)
        get_lambda = lambda: np.random.random(1)[0]

        random_config = {
            'lr': get_lr()
            , 'discount': get_discount()
            , 'N0': get_N0()
            , 'min_eps': get_min_eps()
            , 'initial_q_value': get_initial_q_value()
            , 'lambda': get_lambda()
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
            self.q_t = self.q_preds[self.action_t]

            et, update_et_op, self.reset_et_op = capacities.eligibility_traces(self.inputs, self.action_t, [self.nb_state, self.action_space.n], self.discount, self.lambda_value)

            with tf.variable_scope('Training'):
                self.reward = tf.placeholder(tf.float32, shape=[], name="reward")
                self.next_state = tf.placeholder(tf.int32, shape=[], name="nextState")
                next_max_action_t = tf.cast(tf.argmax(self.Qs[self.next_state], 0), tf.int32)
                target_q = tf.stop_gradient(self.reward + self.discount * self.Qs[self.next_state, next_max_action_t])
                with tf.control_dependencies([update_et_op]):
                    self.loss = - tf.stop_gradient(target_q - self.q_t) * et * self.Qs

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

    def learn_from_episode(self, env, render=False):
        self.sess.run(self.reset_et_op)

        super(BackwardTabularQAgent, self).learn_from_episode(env, render)

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

class TabularQOfflineERAgent(TabularQAgent):
    """
    Agent implementing tabular Q-learning with offline experience replay.
    """
    def set_agent_props(self):
        super(TabularQOfflineERAgent, self).set_agent_props()

        self.er_every = self.config['er_every']
        self.er_batch_size = self.config['er_batch_size']
        self.er_epoch_size = self.config['er_epoch_size']
        self.er_rm_size = self.config['er_rm_size']

        self.replayMemoryDt = np.dtype([('states', 'int32'), ('actions', 'int32'), ('rewards', 'float32'), ('next_states', 'int32')])
        self.replayMemory = np.array([], dtype=self.replayMemoryDt)

    def get_best_config(self, env_name=""):
        return {
            "lr": 0.2
            , "N0": 75
            , "er_epoch_size": 74
            , "er_every": 671
            , "min_eps": 0.001
            , "discount": 0.999
            , "initial_q_value": 0
            , "er_batch_size": 865
            , "er_rm_size": 25567
        }

    @staticmethod
    def get_random_config(fixed_params={}):
        get_lr = lambda: 1e-3 + (1 - 1e-3) * np.random.random(1)[0]
        get_discount = lambda: 0.5 + (1 - 0.5) * np.random.random(1)[0]
        get_N0 = lambda: np.random.randint(1, 5e3)
        get_min_eps = lambda: 1e-4 + (1e-1 - 1e-4) * np.random.random(1)[0]
        get_initial_q_value = lambda: 0 # int(np.random.random(1)[0] * 200)
        get_er_epoch_size = lambda: np.random.randint(5, 100)
        get_er_batch_size = lambda: np.random.randint(16, 1024)
        get_er_rm_size = lambda: np.random.randint(1000, 50000)
        get_er_every = lambda: np.random.randint(1, 1000)

        random_config = {
            'lr': get_lr()
            , 'discount': get_discount()
            , 'N0': get_N0()
            , 'min_eps': get_min_eps()
            , 'initial_q_value': get_initial_q_value()
            , 'er_epoch_size': get_er_epoch_size()
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

            policy_scope = tf.VariableScope(reuse=False, name='EpsilonGreedyPolicy')
            with tf.variable_scope(policy_scope):
                self.action_t = capacities.eps_greedy(
                    self.inputs, self.q_preds, self.env.action_space.n, self.N0, self.min_eps, self.nb_state
                )

            learning_scope = tf.VariableScope(reuse=False, name='MSEQLearning')
            with tf.variable_scope(learning_scope):
                self.reward = tf.placeholder(tf.float32, shape=[], name="reward")
                self.next_state = tf.placeholder(tf.int32, shape=[], name="nextState")

                self.loss = capacities.mse_tabular_q_learning(
                    self.Qs, self.reward, self.next_state, self.discount, self.q_preds, self.action_t
                )

            with tf.variable_scope('Training'):
                global_step = tf.Variable(0, trainable=False, name="global_step", collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
                optimizer = tf.train.AdamOptimizer(self.lr)
                # if optimizer == None:
                #     learning_rate = tf.train.inverse_time_decay(1., global_step, 1, 0.001, staircase=False, name="decay_lr")
                #     optimizer = tf.train.GradientDescentOptimizer(learning_rate)
                self.train_op = optimizer.minimize(self.loss, global_step=global_step)

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
                er_loss = 1/2 * tf.reduce_sum(tf.square(er_target_qs - er_qs))

                er_adam = tf.train.AdamOptimizer(self.lr)
                self.er_global_step = tf.Variable(0, trainable=False)
                self.er_train_op = er_adam.minimize(er_loss, global_step=self.er_global_step)

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
            loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={
                self.inputs: state_id,
                self.action_t: act,
                self.reward: reward,
                self.next_state: next_state_id,
            })
            memory = np.array([(state_id, act, reward, next_state_id)], dtype=self.replayMemoryDt)
            if self.replayMemory.shape[0] >= self.er_rm_size:
                self.replayMemory = np.delete(self.replayMemory, 0)
            self.replayMemory = np.append(self.replayMemory, memory)

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

        # Experiencing replays
        if episode_id % self.er_every == 0:
            for i in range(self.er_epoch_size):
                memories = np.random.choice(self.replayMemory, self.er_batch_size)
                _ = self.sess.run(self.er_train_op, feed_dict={
                    self.er_inputs: memories['states'],
                    self.er_actions: memories['actions'],
                    self.er_rewards: memories['rewards'],
                    self.er_next_states: memories['next_states'],
                })

        return

class TabularFixedQOfflineERAgent(TabularQOfflineERAgent):
    """
    Agent implementing tabular Q-learning with offline experience replay and a second fixed network.
    """        
    def get_best_config(self, env_name=""):
        return {
            "lr": 0.03
            , "min_eps": 0.001
            , "discount": 0.999
            , "er_every": 20
            , "er_batch_size": 607
            , "initial_q_value": 0
            , "er_epoch_size": 35
            , "er_rm_size": 20000
            , "N0": 75
        }

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

            learning_scope = tf.VariableScope(reuse=False, name='MSEQLearning')
            with tf.variable_scope(learning_scope):
                self.reward = tf.placeholder(tf.float32, shape=[], name="reward")
                self.next_state = tf.placeholder(tf.int32, shape=[], name="nextState")

                self.loss = capacities.mse_tabular_q_learning(
                    self.Qs, self.reward, self.next_state, self.discount, self.q_preds, self.action_t
                )

            with tf.variable_scope('Training'):
                global_step = tf.Variable(0, trainable=False, name="global_step", collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
                optimizer = tf.train.AdamOptimizer(self.lr)
                # if optimizer == None:
                #     learning_rate = tf.train.inverse_time_decay(1., global_step, 1, 0.001, staircase=False, name="decay_lr")
                #     optimizer = tf.train.GradientDescentOptimizer(learning_rate)
                self.train_op = optimizer.minimize(self.loss, global_step=global_step)

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
                er_loss = 1/2 * tf.reduce_sum(tf.square(er_target_qs - er_qs))
                er_adam = tf.train.AdamOptimizer(self.lr)
                self.er_global_step = tf.Variable(0, trainable=False)
                self.er_train_op = er_adam.minimize(er_loss, global_step=self.er_global_step)

            self.score_plh = tf.placeholder(tf.float32, shape=[])
            self.score_sum_t = tf.summary.scalar('score', self.score_plh)
            self.loss_plh = tf.placeholder(tf.float32, shape=[])
            self.loss_sum_t = tf.summary.scalar('loss', self.loss_plh)
            self.qs_plh = tf.placeholder(tf.float32, shape=[self.nb_state, self.action_space.n])
            self.qs_sum_t = tf.summary.histogram('Qarray', self.qs_plh)
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
            loss, qs, _ = self.sess.run([self.loss, self.Qs, self.train_op], feed_dict={
                self.inputs: state_id,
                self.action_t: act,
                self.reward: reward,
                self.next_state: next_state_id,
            })
            memory = np.array([(state_id, act, reward, next_state_id)], dtype=self.replayMemoryDt)
            if self.replayMemory.shape[0] >= self.er_rm_size:
                self.replayMemory = np.delete(self.replayMemory, 0)
            self.replayMemory = np.append(self.replayMemory, memory)

            av_loss.append(loss)
            score += reward
            obs = next_obs
            if done:
                break

        summary, _, episode_id = self.sess.run([self.all_summary_t, self.inc_ep_id_op, self.episode_id], feed_dict={
            self.score_plh: score,
            self.loss_plh: np.mean(av_loss),
            self.qs_plh: qs
        })
        self.sw.add_summary(summary, episode_id)

        # Experiencing replays
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

class BackwardTabularFixedQOfflineERAgent(TabularQOfflineERAgent):
    """
    Agent implementing tabular Q-learning with offline experience replay and a second fixed network + eligibility traces.
    """
    def set_agent_props(self):
        super(BackwardTabularFixedQOfflineERAgent, self).set_agent_props()

        self.lambda_value = self.config['lambda']

    def get_best_config(self, env_name=""):
        return {
            "lr": 0.03
            , "discount": 0.999
            , "lambda": 0.03485481041759786
            , "N0": 75
            , "min_eps": 0.001
            , "er_rm_size": 4983
            , "initial_q_value": 0
            , "er_every": 701
            , "er_batch_size": 1000
        }

    @staticmethod
    def get_random_config(fixed_params={}):
        get_lr = lambda: 1e-3 + (1 - 1e-3) * np.random.random(1)[0]
        get_discount = lambda: 0.5 + (1 - 0.5) * np.random.random(1)[0]
        get_N0 = lambda: np.random.randint(1, 5e3)
        get_min_eps = lambda: 1e-4 + (1e-1 - 1e-4) * np.random.random(1)[0]
        get_initial_q_value = lambda: 0 # int(np.random.random(1)[0] * 200)
        get_lambda = lambda: np.random.random(1)[0]
        get_er_batch_size = lambda: np.random.randint(16, 1024)
        get_er_rm_size = lambda: np.random.randint(1000, 50000)
        get_er_every = lambda: np.random.randint(1, 1000)

        random_config = {
            'lr': get_lr()
            , 'discount': get_discount()
            , 'N0': get_N0()
            , 'min_eps': get_min_eps()
            , 'initial_q_value': get_initial_q_value()
            , 'lambda': get_lambda()
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
            self.q_t = self.q_preds[self.action_t]

            with tf.variable_scope('Training'):
                self.reward = tf.placeholder(tf.float32, shape=[], name="reward")
                self.next_state = tf.placeholder(tf.int32, shape=[], name="nextState")
                next_max_action_t = tf.cast(tf.argmax(self.Qs[self.next_state], 0), tf.int32)
                target_q = tf.stop_gradient(self.reward + self.discount * self.Qs[self.next_state, next_max_action_t])
                self.Et = tf.placeholder(tf.float32, shape=[self.nb_state, self.action_space.n], name="Et")
                self.loss = - tf.stop_gradient(target_q - self.q_t) * self.Et * self.Qs

                adam = tf.train.AdamOptimizer(self.lr)
                self.global_step = tf.Variable(0, trainable=False, name="global_step", collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
                self.train_op = adam.minimize(self.loss, global_step=self.global_step)

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
                er_loss = 1/2 * tf.reduce_sum(tf.square(er_target_qs - er_qs))
                er_adam = tf.train.AdamOptimizer(self.lr)
                self.er_global_step = tf.Variable(0, trainable=False)
                self.er_train_op = er_adam.minimize(er_loss, global_step=self.er_global_step)


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

        Et = np.zeros([self.nb_state, self.action_space.n])
        visited_sa = []
        lambda_val = 0.9

        while True:
            if render:
                env.render()

            act, state_id = self.act(obs)
            next_obs, reward, done, info = env.step(act)

            for s, a in visited_sa:
                Et[s, a] *= self.discount * lambda_val
            Et[state_id, act] = 1
            visited_sa.append((state_id, act))


            next_state_id = self.phi(next_obs, done)
            loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={
                self.inputs: state_id,
                self.action_t: act,
                self.reward: reward,
                self.next_state: next_state_id,
                self.Et: Et,
            })
            memory = np.array([(state_id, act, reward, next_state_id)], dtype=self.replayMemoryDt)
            if self.replayMemory.shape[0] >= self.er_rm_size:
                self.replayMemory = np.delete(self.replayMemory, 0)
            self.replayMemory = np.append(self.replayMemory, memory)

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

        # Experiencing replays
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
