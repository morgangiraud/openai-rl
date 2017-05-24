import numpy as np
import tensorflow as tf

from agents import BasicAgent, capacities
from agents.capacities import get_expected_rewards

class DeepMCPolicyAgent(BasicAgent):
    """
    Agent implementing Policy gradient using Monte-Carlo control
    """
    def set_agent_props(self):
        self.policy_params = {
            'nb_inputs': self.observation_space.shape[0] + 1
            , 'nb_units': self.config['nb_units']
            , 'nb_outputs': self.action_space.n
            , 'initial_mean': self.config['initial_mean']
            , 'initial_stddev': self.config['initial_stddev']
        }

    def get_best_config(self, env_name=""):
        return {
            'lr': 1e-3
            , 'discount': 0.8209896594244546
            , 'nb_units': 41
            , 'initial_mean': 0.
            , 'initial_stddev': 0.423321967449976
        }

    @staticmethod
    def get_random_config(fixed_params={}):
        get_lr = lambda: 1e-4 + (1e-1 - 1e-4) * np.random.random(1)[0]
        get_discount = lambda: 0.5 + (1 - 0.5) * np.random.random(1)[0]
        get_nb_units = lambda: np.random.randint(10, 100)
        get_initial_mean = lambda: 0
        get_initial_stddev = lambda: 5e-1 * np.random.random(1)[0]

        random_config = {
            'lr': get_lr()
            , 'discount': get_discount()
            , 'nb_units': get_nb_units()
            , 'initial_mean': get_initial_mean()
            , 'initial_stddev': get_initial_stddev()
        }
        random_config.update(fixed_params)

        return random_config

    def build_graph(self, graph):
        with graph.as_default():
            tf.set_random_seed(self.random_seed)

            self.inputs = tf.placeholder(tf.float32, shape=[None, self.observation_space.shape[0] + 1], name='inputs')

            policy_scope = tf.VariableScope(reuse=False, name='Policy')
            with tf.variable_scope(policy_scope):
                self.probs, self.actions = capacities.policy(self.policy_params, self.inputs)
            self.action_t = tf.squeeze(self.actions, 1)[0]

            with tf.variable_scope('Training'):
                self.rewards = tf.placeholder(tf.float32, shape=[None], name="reward")
                stacked_actions = tf.stack([tf.range(0, tf.shape(self.actions)[0]), tf.squeeze(self.actions, 1)], 1)
                log_probs = tf.log(tf.gather_nd(self.probs, stacked_actions))
                # log_probs = tf.Print(log_probs, data=[tf.shape(self.probs), tf.shape(self.actions), tf.shape(log_probs)], message="tf.shape(log_probs):")
                self.loss = - tf.reduce_sum(log_probs * self.rewards)

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
        score = 0
        historyType = np.dtype([('states', 'float32', (env.observation_space.shape[0] + 1,)), ('actions', 'int32', (1,)), ('rewards', 'float32')])
        history = np.array([], dtype=historyType)
        done = False

        while True:
            if render:
                env.render()

            act, state = self.act(obs)
            next_obs, reward, done, info = env.step(act)

            memory = np.array([(np.concatenate((obs, [0])), act, reward)], dtype=historyType)
            history = np.append(history, memory)

            score += reward
            obs = next_obs
            if done:
                break

        # Learning
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict={
            self.inputs: history['states'],
            self.actions: history['actions'],
            self.rewards: get_expected_rewards(history['rewards']),
        })
        summary, _, episode_id = self.sess.run([self.all_summary_t, self.inc_ep_id_op, self.episode_id], feed_dict={
            self.score_plh: score,
            self.loss_plh: loss
        })
        self.sw.add_summary(summary, episode_id)

        return

class MCActorCriticAgent(DeepMCPolicyAgent):
    """
    Agent implementing Policy gradient using Monte-Carlo control
    """
    def set_agent_props(self):
        super(MCActorCriticAgent, self).set_agent_props()

        self.q_params = {
            'nb_inputs': self.observation_space.shape[0] + 1
            , 'nb_units': self.config['nb_units']
            , 'nb_outputs': self.action_space.n
            , 'initial_mean': self.config['initial_mean']
            , 'initial_stddev': self.config['initial_stddev']
        }
        self.q_scale_lr = self.config['q_scale_lr']

    def get_best_config(self, env_name=""):
        return {
            'lr': 1e-3
            , 'discount': 0.7987841714685839
            , 'nb_units': 87
            , 'initial_mean': 0.
            , 'initial_stddev': 0.423321967449976
            , 'q_scale_lr': 1.
        }

    @staticmethod
    def get_random_config(fixed_params={}):
        get_lr = lambda: 1e-4 + (1e-1 - 1e-4) * np.random.random(1)[0]
        get_discount = lambda: 0.5 + (1 - 0.5) * np.random.random(1)[0]
        get_nb_units = lambda: np.random.randint(10, 100)
        get_initial_mean = lambda: 0
        get_initial_stddev = lambda: 5e-1 * np.random.random(1)[0]
        get_q_scale_lr = lambda: 10 * np.random.random(1)[0]

        random_config = {
            'lr': get_lr()
            , 'discount': get_discount()
            , 'nb_units': get_nb_units()
            , 'initial_mean': get_initial_mean()
            , 'initial_stddev': get_initial_stddev()
            , 'q_scale_lr': get_q_scale_lr()
        }
        random_config.update(fixed_params)

        return random_config

    def build_graph(self, graph):
        with graph.as_default():
            tf.set_random_seed(self.random_seed)

            self.inputs = tf.placeholder(tf.float32, shape=[None, self.observation_space.shape[0] + 1], name='inputs')

            policy_scope = tf.VariableScope(reuse=False, name='Policy')
            with tf.variable_scope(policy_scope):
                self.probs, self.actions = capacities.policy(self.policy_params, self.inputs)
            self.action_t = tf.squeeze(self.actions, 1)[0]

            q_scope = tf.VariableScope(reuse=False, name='QValues')
            with tf.variable_scope(q_scope):
                self.q_values = capacities.value_f(self.q_params, self.inputs)
            self.q = self.q_values[0, tf.stop_gradient(self.action_t)]

            with tf.variable_scope('Training'):
                stacked_actions = tf.stack([tf.range(0, tf.shape(self.actions)[0]), tf.squeeze(self.actions, 1)], 1)
                qs = tf.gather_nd(self.q_values, stacked_actions)      
                log_probs = tf.log(tf.gather_nd(self.probs, stacked_actions))
                self.policy_loss = - tf.reduce_sum(log_probs * tf.stop_gradient(qs))

                self.rewards = tf.placeholder(tf.float32, shape=[None], name="rewards")
                self.next_states = tf.placeholder(tf.float32, shape=[None, self.observation_space.shape[0] + 1], name="next_states")
                self.next_actions = tf.placeholder(tf.int32, shape=[None], name="next_actions")
                with tf.variable_scope(q_scope, reuse=True):
                    next_q_values = capacities.value_f(self.q_params, self.next_states)
                next_stacked_actions = tf.stack([tf.range(0, tf.shape(self.next_actions)[0]), self.next_actions], 1)
                next_qs = tf.gather_nd(next_q_values, next_stacked_actions)
                target_qs1 = tf.stop_gradient(self.rewards + self.discount * next_qs)
                target_qs2 = self.rewards
                stacked_targets = tf.stack([target_qs1, target_qs2], 1)
                select_targets = tf.stack([tf.range(0, tf.shape(self.next_states)[0]), tf.cast(self.next_states[:, -1], tf.int32)], 1)
                target_qs = tf.gather_nd(stacked_targets, select_targets)
                self.q_loss = 1/2 * tf.reduce_sum(tf.square(target_qs - qs))
                
                self.loss = self.policy_loss + self.q_scale_lr * self.q_loss 

                adam = tf.train.AdamOptimizer(self.lr)
                self.global_step = tf.Variable(0, trainable=False, name="global_step", collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
                self.train_op = adam.minimize(self.loss, global_step=self.global_step)

            self.score_plh = tf.placeholder(tf.float32, shape=[])
            self.score_sum_t = tf.summary.scalar('score', self.score_plh)
            self.policy_loss_plh = tf.placeholder(tf.float32, shape=[])
            self.policy_loss_sum_t = tf.summary.scalar('policy_loss', self.policy_loss_plh)
            self.q_loss_plh = tf.placeholder(tf.float32, shape=[])
            self.q_loss_sum_t = tf.summary.scalar('q_loss', self.q_loss_plh)
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
        act, _ = self.act(obs)

        score = 0
        historyType = np.dtype([
            ('states', 'float32', (env.observation_space.shape[0] + 1,)),
            ('actions', 'int32', (1,)),
            ('rewards', 'float32'),
            ('next_states', 'float32', (env.observation_space.shape[0] + 1,)),
            ('next_actions', 'int32'),
        ])
        history = np.array([], dtype=historyType)
        done = False

        while True:
            if render:
                env.render()

            next_obs, reward, done, info = env.step(act)
            next_act, _ = self.act(next_obs)

            memory = np.array([(
                np.concatenate((obs, [0])),
                [act],
                reward,
                np.concatenate((next_obs, [1 if done else 0])),
                next_act
            )], dtype=historyType)
            history = np.append(history, memory)

            score += reward
            obs = next_obs
            act = next_act
            if done:
                break

        # Learning
        _, policy_loss, q_loss, loss = self.sess.run([self.train_op, self.policy_loss, self.q_loss, self.loss], feed_dict={
            self.inputs: history['states'],
            self.actions: history['actions'],
            self.rewards: get_expected_rewards(history['rewards']),
            self.next_states: history['next_states'],
            self.next_actions: history['next_actions'],
        })
        summary, _, episode_id = self.sess.run([self.all_summary_t, self.inc_ep_id_op, self.episode_id], feed_dict={
            self.score_plh: score,
            self.policy_loss_plh: policy_loss,
            self.q_loss_plh: q_loss,
            self.loss_plh: loss,
        })
        self.sw.add_summary(summary, episode_id)

        return

class ActorCriticAgent(MCActorCriticAgent):
    """
    Agent implementing Actor critic using REINFORCE
    """
    def build_graph(self, graph):
        with graph.as_default():
            tf.set_random_seed(self.random_seed)

            self.inputs = tf.placeholder(tf.float32, shape=[None, self.observation_space.shape[0] + 1], name='inputs')

            policy_scope = tf.VariableScope(reuse=False, name='Policy')
            with tf.variable_scope(policy_scope):
                self.probs, self.actions = capacities.policy(self.policy_params, self.inputs)
            self.action_t = tf.squeeze(self.actions, 1)[0]
            # self.action_t = tf.Print(self.action_t, data=[self.probs, self.action_t], message="self.probs, self.action_t:")

            q_scope = tf.VariableScope(reuse=False, name='QValues')
            with tf.variable_scope(q_scope):
                self.q_values = capacities.value_f(self.q_params, self.inputs)
            self.q = self.q_values[0, tf.stop_gradient(self.action_t)]

            with tf.control_dependencies([self.probs, self.q]):
                with tf.variable_scope('Training'):
                    stacked_actions = tf.stack([tf.range(0, tf.shape(self.actions)[0]), tf.squeeze(self.actions, 1)], 1)
                    log_probs = tf.log(tf.gather_nd(self.probs, stacked_actions))
                    qs = tf.gather_nd(self.q_values, stacked_actions)

                    self.policy_loss = - tf.reduce_sum(log_probs * tf.stop_gradient(qs))
                    policy_adam = tf.train.AdamOptimizer(self.policy_lr)
                    self.policy_global_step = tf.Variable(0, trainable=False, name="policy_global_step", collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
                    self.policy_train_op = policy_adam.minimize(self.policy_loss, global_step=self.policy_global_step)

                    self.rewards = tf.placeholder(tf.float32, shape=[None], name="rewards")
                    self.next_states = tf.placeholder(tf.float32, shape=[None, self.observation_space.shape[0] + 1], name="next_states")
                    self.next_actions = tf.placeholder(tf.int32, shape=[None], name="next_actions")
                    with tf.variable_scope(q_scope, reuse=True):
                        next_q_values = capacities.value_f(self.q_params, self.next_states)
                    next_stacked_actions = tf.stack([tf.range(0, tf.shape(self.next_actions)[0]), self.next_actions], 1)
                    next_qs = tf.gather_nd(next_q_values, next_stacked_actions)
                    target_qs1 = tf.stop_gradient(self.rewards + self.discount * next_qs)
                    target_qs2 = self.rewards
                    stacked_targets = tf.stack([target_qs1, target_qs2], 1)
                    select_targets = tf.stack([tf.range(0, tf.shape(self.next_states)[0]), tf.cast(self.next_states[:, -1], tf.int32)], 1)
                    target_qs = tf.gather_nd(stacked_targets, select_targets)

                    self.q_loss = 1/2 * tf.reduce_sum(tf.square(target_qs - qs))
                    q_adam = tf.train.AdamOptimizer(self.q_lr)
                    self.q_global_step = tf.Variable(0, trainable=False, name="q_global_step")
                    self.q_train_op = q_adam.minimize(self.q_loss, global_step=self.q_global_step)

            self.score_plh = tf.placeholder(tf.float32, shape=[])
            self.score_sum_t = tf.summary.scalar('score', self.score_plh)
            self.policy_loss_plh = tf.placeholder(tf.float32, shape=[])
            self.policy_loss_sum_t = tf.summary.scalar('policy_loss', self.policy_loss_plh)
            self.q_loss_plh = tf.placeholder(tf.float32, shape=[])
            self.q_loss_sum_t = tf.summary.scalar('q_loss', self.q_loss_plh)
            self.all_summary_t = tf.summary.merge_all()

            self.episode_id, self.inc_ep_id_op = capacities.counter("episode_id")

            # Playing part
            self.pscore_plh = tf.placeholder(tf.float32, shape=[])
            self.pscore_sum_t = tf.summary.scalar('play_score', self.pscore_plh)

        return graph

    def learn_from_episode(self, env, render):
        obs = env.reset()
        act, _ = self.act(obs)

        av_policy_loss = []
        av_q_loss = []
        score = 0
        done = False

        while True:
            if render:
                env.render()

            next_obs, reward, done, info = env.step(act)
            next_act, _ = self.act(next_obs)

            # print([ np.concatenate((obs, [0])) ], act, reward, [ np.concatenate((next_obs, [1 if done else 0])) ], next_act)
            _, policy_loss, _, q_loss = self.sess.run([self.policy_train_op, self.policy_loss , self.q_train_op, self.q_loss], feed_dict={
                self.inputs: [ np.concatenate((obs, [0])) ],
                self.actions: [ [act] ],
                self.rewards: [ reward ],
                self.next_states: [ np.concatenate((next_obs, [1 if done else 0])) ],
                self.next_actions: [ next_act ]
            })
            av_policy_loss.append(policy_loss)
            av_q_loss.append(q_loss)

            score += reward
            obs = next_obs
            act = next_act
            if done:
                break


        summary, _, episode_id = self.sess.run([self.all_summary_t, self.inc_ep_id_op, self.episode_id], feed_dict={
            self.score_plh: score,
            self.policy_loss_plh: np.mean(av_policy_loss),
            self.q_loss_plh: np.mean(av_q_loss),
        })
        self.sw.add_summary(summary, episode_id)

        return

class A2CAgent(ActorCriticAgent):
    """
    Agent implementing Advantage Actor critic using REINFORCE
    """
    def set_agent_props(self):
        super(A2CAgent, self).set_agent_props()

        self.v_params = {
            'nb_inputs': self.observation_space.shape[0] + 1
            , 'nb_units': self.config['nb_units']
            , 'nb_outputs': 1
            , 'initial_mean': self.config['initial_mean']
            , 'initial_stddev': self.config['initial_stddev']
        }
        self.v_lr = self.lr

    def build_graph(self, graph):
        with graph.as_default():
            tf.set_random_seed(self.random_seed)

            self.inputs = tf.placeholder(tf.float32, shape=[None, self.observation_space.shape[0] + 1], name='inputs')

            policy_scope = tf.VariableScope(reuse=False, name='Policy')
            with tf.variable_scope(policy_scope):
                self.probs, self.actions = capacities.policy(self.policy_params, self.inputs)
            self.action_t = tf.squeeze(self.actions, 1)[0]
            # self.action_t = tf.Print(self.action_t, data=[self.probs, self.action_t], message="self.probs, self.action_t:")

            q_scope = tf.VariableScope(reuse=False, name='QValues')
            with tf.variable_scope(q_scope):
                self.q_values = capacities.value_f(self.q_params, self.inputs)

            v_scope = tf.VariableScope(reuse=False, name='VValues')
            with tf.variable_scope(v_scope):
                vs = capacities.value_f(self.v_params, self.inputs)

            with tf.control_dependencies([self.probs, self.q_values, vs]):
                with tf.variable_scope('Training'):
                    stacked_actions = tf.stack([tf.range(0, tf.shape(self.actions)[0]), tf.squeeze(self.actions, 1)], 1)
                    qs = tf.gather_nd(self.q_values, stacked_actions)

                    self.rewards = tf.placeholder(tf.float32, shape=[None], name="rewards")
                    self.next_states = tf.placeholder(tf.float32, shape=[None, self.observation_space.shape[0] + 1], name="next_states")
                    self.next_actions = tf.placeholder(tf.int32, shape=[None], name="next_actions")

                    with tf.variable_scope(v_scope, reuse=True):
                        next_vs = tf.squeeze(capacities.value_f(self.v_params, self.next_states), 1)

                    with tf.variable_scope('TargetVs'):
                        target_vs1 = tf.stop_gradient(self.rewards + self.discount * next_vs)
                        target_vs2 = self.rewards
                        stacked_targets = tf.stack([target_vs1, target_vs2], 1)
                        select_targets = tf.stack([tf.range(0, tf.shape(self.next_states)[0]), tf.cast(self.next_states[:, -1], tf.int32)], 1)
                        target_vs = tf.gather_nd(stacked_targets, select_targets)

                    with tf.variable_scope(q_scope, reuse=True):
                        next_q_values = capacities.value_f(self.q_params, self.next_states)

                    with tf.variable_scope('TargetQs'):
                        next_stacked_actions = tf.stack([tf.range(0, tf.shape(self.next_actions)[0]), self.next_actions], 1)
                        next_qs = tf.gather_nd(next_q_values, next_stacked_actions)
                        target_qs1 = tf.stop_gradient(self.rewards + self.discount * next_qs)
                        target_qs2 = self.rewards
                        stacked_targets = tf.stack([target_qs1, target_qs2], 1)
                        select_targets = tf.stack([tf.range(0, tf.shape(self.next_states)[0]), tf.cast(self.next_states[:, -1], tf.int32)], 1)
                        target_qs = tf.gather_nd(stacked_targets, select_targets)

                    log_probs = tf.log(tf.gather_nd(self.probs, stacked_actions))

                    with tf.control_dependencies([log_probs, target_qs, target_vs]):
                        self.v_loss = 1/2 * tf.reduce_sum(tf.square(target_vs - vs))
                        v_adam = tf.train.AdamOptimizer(self.v_lr)
                        self.v_global_step = tf.Variable(0, trainable=False, name="v_global_step")
                        self.v_train_op = v_adam.minimize(self.v_loss, global_step=self.v_global_step)

                        self.q_loss = 1/2 * tf.reduce_sum(tf.square(target_qs - qs))
                        q_adam = tf.train.AdamOptimizer(self.q_lr)
                        self.q_global_step = tf.Variable(0, trainable=False, name="q_global_step")
                        self.q_train_op = q_adam.minimize(self.q_loss, global_step=self.q_global_step)

                        advantages = qs - vs
                        self.policy_loss = - tf.reduce_sum(log_probs * tf.stop_gradient(advantages))
                        policy_adam = tf.train.AdamOptimizer(self.policy_lr)
                        self.policy_global_step = tf.Variable(0, trainable=False, name="policy_global_step", collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
                        self.policy_train_op = policy_adam.minimize(self.policy_loss, global_step=self.policy_global_step)

            self.score_plh = tf.placeholder(tf.float32, shape=[])
            self.score_sum_t = tf.summary.scalar('score', self.score_plh)
            self.policy_loss_plh = tf.placeholder(tf.float32, shape=[])
            self.policy_loss_sum_t = tf.summary.scalar('policy_loss', self.policy_loss_plh)
            self.q_loss_plh = tf.placeholder(tf.float32, shape=[])
            self.q_loss_sum_t = tf.summary.scalar('q_loss', self.q_loss_plh)
            self.v_loss_plh = tf.placeholder(tf.float32, shape=[])
            self.v_loss_sum_t = tf.summary.scalar('v_loss', self.v_loss_plh)
            self.all_summary_t = tf.summary.merge_all()

            self.episode_id, self.inc_ep_id_op = capacities.counter("episode_id")

            # Playing part
            self.pscore_plh = tf.placeholder(tf.float32, shape=[])
            self.pscore_sum_t = tf.summary.scalar('play_score', self.pscore_plh)

        return graph

    def learn_from_episode(self, env, render):
        obs = env.reset()
        act, _ = self.act(obs)

        av_policy_loss = []
        av_q_loss = []
        av_v_loss = []
        score = 0
        done = False

        while True:
            if render:
                env.render()

            next_obs, reward, done, info = env.step(act)
            next_act, _ = self.act(next_obs)

            # print([ np.concatenate((obs, [0])) ], act, reward, [ np.concatenate((next_obs, [1 if done else 0])) ], next_act)
            _, policy_loss, _, q_loss, _, v_loss = self.sess.run([self.policy_train_op, self.policy_loss , self.q_train_op, self.q_loss, self.v_train_op, self.v_loss], feed_dict={
                self.inputs: [ np.concatenate((obs, [0])) ],
                self.actions: [ [act] ],
                self.rewards: [ reward ],
                self.next_states: [ np.concatenate((next_obs, [1 if done else 0])) ],
                self.next_actions: [ next_act ]
            })
            av_policy_loss.append(policy_loss)
            av_q_loss.append(q_loss)
            av_v_loss.append(v_loss)

            score += reward
            obs = next_obs
            act = next_act
            if done:
                break

        summary, _, episode_id = self.sess.run([self.all_summary_t, self.inc_ep_id_op, self.episode_id], feed_dict={
            self.score_plh: score,
            self.policy_loss_plh: np.mean(av_policy_loss),
            self.q_loss_plh: np.mean(av_q_loss),
            self.v_loss_plh: np.mean(av_v_loss),
        })
        self.sw.add_summary(summary, episode_id)

        return

class TDACAgent(DeepMCPolicyAgent):
    """
    Agent implementing TD Actor critic using REINFORCE
    """
    def set_agent_props(self):
        super(TDACAgent, self).set_agent_props()

        self.v_params = {
            'nb_inputs': self.observation_space.shape[0] + 1
            , 'nb_units': self.config['nb_units']
            , 'nb_outputs': 1
            , 'initial_mean': self.config['initial_mean']
            , 'initial_stddev': self.config['initial_stddev']
        }
        self.policy_lr = self.lr
        self.v_lr = self.lr

    def build_graph(self, graph):
        with graph.as_default():
            tf.set_random_seed(self.random_seed)

            self.inputs = tf.placeholder(tf.float32, shape=[None, self.observation_space.shape[0] + 1], name='inputs')

            policy_scope = tf.VariableScope(reuse=False, name='Policy')
            with tf.variable_scope(policy_scope):
                self.probs, self.actions = capacities.policy(self.policy_params, self.inputs)
            self.action_t = tf.squeeze(self.actions, 1)[0]
            # self.action_t = tf.Print(self.action_t, data=[self.probs, self.action_t], message="self.probs, self.action_t:")

            v_scope = tf.VariableScope(reuse=False, name='VValues')
            with tf.variable_scope(v_scope):
                vs = capacities.value_f(self.v_params, self.inputs)

            with tf.control_dependencies([self.probs, vs]):
                with tf.variable_scope('Training'):
                    stacked_actions = tf.stack([tf.range(0, tf.shape(self.actions)[0]), tf.squeeze(self.actions, 1)], 1)

                    self.rewards = tf.placeholder(tf.float32, shape=[None], name="rewards")
                    self.next_states = tf.placeholder(tf.float32, shape=[None, self.observation_space.shape[0] + 1], name="next_states")
                    self.next_actions = tf.placeholder(tf.int32, shape=[None], name="next_actions")

                    with tf.variable_scope(v_scope, reuse=True):
                        next_vs = tf.squeeze(capacities.value_f(self.v_params, self.next_states), 1)

                    with tf.variable_scope('TargetVs'):
                        target_vs1 = tf.stop_gradient(self.rewards + self.discount * next_vs)
                        target_vs2 = self.rewards
                        stacked_targets = tf.stack([target_vs1, target_vs2], 1)
                        select_targets = tf.stack([tf.range(0, tf.shape(self.next_states)[0]), tf.cast(self.next_states[:, -1], tf.int32)], 1)
                        target_vs = tf.gather_nd(stacked_targets, select_targets)

                    log_probs = tf.log(tf.gather_nd(self.probs, stacked_actions))

                    with tf.control_dependencies([log_probs, target_vs]):
                        self.v_loss = 1/2 * tf.reduce_sum(tf.square(target_vs - vs))
                        v_adam = tf.train.AdamOptimizer(self.v_lr)
                        self.v_global_step = tf.Variable(0, trainable=False, name="v_global_step")
                        self.v_train_op = v_adam.minimize(self.v_loss, global_step=self.v_global_step)

                        td = target_vs - vs
                        self.policy_loss = - tf.reduce_sum(log_probs * tf.stop_gradient(td))
                        policy_adam = tf.train.AdamOptimizer(self.policy_lr)
                        self.policy_global_step = tf.Variable(0, trainable=False, name="policy_global_step", collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
                        self.policy_train_op = policy_adam.minimize(self.policy_loss, global_step=self.policy_global_step)

            self.score_plh = tf.placeholder(tf.float32, shape=[])
            self.score_sum_t = tf.summary.scalar('score', self.score_plh)
            self.policy_loss_plh = tf.placeholder(tf.float32, shape=[])
            self.policy_loss_sum_t = tf.summary.scalar('policy_loss', self.policy_loss_plh)
            self.v_loss_plh = tf.placeholder(tf.float32, shape=[])
            self.v_loss_sum_t = tf.summary.scalar('v_loss', self.v_loss_plh)
            self.all_summary_t = tf.summary.merge_all()

            self.episode_id, self.inc_ep_id_op = capacities.counter("episode_id")

            # Playing part
            self.pscore_plh = tf.placeholder(tf.float32, shape=[])
            self.pscore_sum_t = tf.summary.scalar('play_score', self.pscore_plh)

        return graph

    def learn_from_episode(self, env, render):
        obs = env.reset()
        act, _ = self.act(obs)

        av_policy_loss = []
        av_v_loss = []
        score = 0
        done = False

        while True:
            if render:
                env.render()

            next_obs, reward, done, info = env.step(act)
            next_act, _ = self.act(next_obs)

            # print([ np.concatenate((obs, [0])) ], act, reward, [ np.concatenate((next_obs, [1 if done else 0])) ], next_act)
            _, policy_loss, _, v_loss = self.sess.run([self.policy_train_op, self.policy_loss , self.v_train_op, self.v_loss], feed_dict={
                self.inputs: [ np.concatenate((obs, [0])) ],
                self.actions: [ [act] ],
                self.rewards: [ reward ],
                self.next_states: [ np.concatenate((next_obs, [1 if done else 0])) ],
                self.next_actions: [ next_act ]
            })
            av_policy_loss.append(policy_loss)
            av_v_loss.append(v_loss)

            score += reward
            obs = next_obs
            act = next_act
            if done:
                break


        summary, _, episode_id = self.sess.run([self.all_summary_t, self.inc_ep_id_op, self.episode_id], feed_dict={
            self.score_plh: score,
            self.policy_loss_plh: np.mean(av_policy_loss),
            self.v_loss_plh: np.mean(av_v_loss),
        })
        self.sw.add_summary(summary, episode_id)

        return
