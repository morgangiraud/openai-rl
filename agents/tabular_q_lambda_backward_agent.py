import numpy as np
import tensorflow as tf

from agents import TabularQAgent, capacities

class TabularQLambdaBackwardAgent(TabularQAgent):
    """
    Agent implementing Backward TD(lambda) tabular Q-learning.
    """
    def set_agent_props(self):
        super(TabularQLambdaBackwardAgent, self).set_agent_props()

        self.lambda_value = self.config['lambda']

    def get_best_config(self, env_name=""):
        return {
            'lr': 0.0001
            , 'lr_decay_steps': 40000
            , 'discount': 0.999
            , 'N0': 220
            , 'min_eps': 0.005
            , 'initial_q_value': 0
            , 'lambda': .99
        }

    @staticmethod
    def get_random_config(fixed_params={}):
        get_lr = lambda: 1e-4 + (.1 - 1e-4) * np.random.random(1)[0]
        get_lr_decay_steps = lambda: np.random.randint(1e3, 5e5)
        get_discount = lambda: 0.5 + (1 - 0.5) * np.random.random(1)[0]
        get_N0 = lambda: np.random.randint(1, 5e3)
        get_min_eps = lambda: 1e-4 + (1e-1 - 1e-4) * np.random.random(1)[0]
        get_initial_q_value = lambda: 0 # int(np.random.random(1)[0] * 200)
        get_lambda = lambda: np.random.random(1)[0]

        random_config = {
            'lr': get_lr()
            , 'lr_decay_steps': get_lr_decay_steps()
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

            self.inputs_plh = tf.placeholder(tf.int32, shape=[None], name="inputs_plh")
            
            q_scope = tf.VariableScope(reuse=False, name='QValues')
            with tf.variable_scope(q_scope):
                self.Qs = tf.get_variable('Qs'
                    , shape=[self.nb_state, self.action_space.n]
                    , initializer=tf.constant_initializer(self.initial_q_value)
                    , dtype=tf.float32
                )
                tf.summary.histogram('Qarray', self.Qs)
                self.q_preds_t = tf.gather(self.Qs, self.inputs_plh)

            policy_scope = tf.VariableScope(reuse=False, name='EpsilonGreedyPolicy')
            with tf.variable_scope(policy_scope):
                self.actions_t = capacities.batch_eps_greedy(
                    self.inputs_plh, self.q_preds_t, self.env.action_space.n, self.N0, self.min_eps, self.nb_state
                )
                self.action_t = self.actions_t[0]
                self.q_value_t = self.q_preds_t[0][self.action_t]

            et_scope = tf.VariableScope(reuse=False, name='EligibilityTraces')
            with tf.variable_scope(et_scope):
                et, update_et_op, self.reset_et_op = capacities.eligibility_traces(self.Qs, self.inputs_plh, self.actions_t, self.discount, self.lambda_value)

            with tf.variable_scope('Training'):
                self.rewards_plh = tf.placeholder(tf.float32, shape=[None], name="rewards_plh")
                self.next_states_plh = tf.placeholder(tf.int32, shape=[None], name="next_states_plh")

                self.targets_t = capacities.get_q_learning_target(self.Qs, self.rewards_plh, self.next_states_plh, self.discount)
                global_step = tf.Variable(0, trainable=False, name="global_step", collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
                inc_global_step = global_step.assign_add(1)
                with tf.control_dependencies([update_et_op, inc_global_step]):
                    self.loss = tf.reduce_sum(self.targets_t[0] * et)
                    self.train_op = tf.assign_add(self.Qs, self.lr * self.targets_t[0] * et)

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

        super(TabularQLambdaBackwardAgent, self).learn_from_episode(env, render)