import numpy as np
import tensorflow as tf

from agents import TabularSigmaAgent, capacities

class TabularSigmaLambdaBackwardAgent(TabularSigmaAgent):
    """
    Agent implementing Backward TD(lambda) tabular Q-learning.
    """
    def set_agent_props(self):
        self.lr = self.config['lr']
        self.lr_decay_steps = self.config['lr_decay_steps']
        self.discount = self.config['discount']
        self.N0 = self.config['N0']
        self.min_eps = self.config['min_eps']
        self.initial_q_value = self.config['initial_q_value']
        self.lambda_value = self.config['lambda']

    def get_best_config(self, env_name=""):
        cartpolev0 = {
            'lr': 0.1
            , 'lr_decay_steps': 30000
            , 'discount': 0.999
            , 'N0': 10
            , 'min_eps': 0.001
            , 'initial_q_value': 0
            , 'lambda': 0.9
        }
        return {
            'CartPole-v0': cartpolev0
        }.get(env_name, cartpolev0)

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
                self.actions_t, self.probs_t = capacities.batch_eps_greedy(
                    self.inputs_plh, self.q_preds_t, self.env.action_space.n, self.N0, self.min_eps, self.nb_state
                )
                self.action_t = self.actions_t[0]
                self.q_value_t = self.q_preds_t[0][self.action_t]

            et_scope = tf.VariableScope(reuse=False, name='EligibilityTraces')
            with tf.variable_scope(et_scope):
                et, update_et_op, self.reset_et_op = capacities.eligibility_traces(self.Qs, self.inputs_plh, self.actions_t, self.discount, self.lambda_value)

            self.episode_id, self.inc_ep_id_op = capacities.counter("episode_id")

            with tf.variable_scope('Training'):
                self.rewards_plh = tf.placeholder(tf.float32, shape=[None], name="rewards_plh")
                self.next_states_plh = tf.placeholder(tf.int32, shape=[None], name="next_states_plh")
                self.next_actions_plh = tf.placeholder(tf.int32, shape=[None], name="next_actions_plh")
                self.next_probs_plh = tf.placeholder(tf.float32, shape=[None, self.action_space.n], name="next_probs_plh")

                sigma = tf.train.inverse_time_decay(tf.constant(1., dtype=tf.float32), self.episode_id, decay_steps=100, decay_rate=0.1)
                tf.summary.scalar('sigma', sigma)

                self.targets_t = capacities.get_sigma_target(self.Qs, sigma, self.rewards_plh, self.next_states_plh, self.next_actions_plh, self.next_probs_plh, self.discount)
                target = self.targets_t[0]
                state_action_pairs = tf.stack([self.inputs_plh, self.actions_t], 1)
                estimate = tf.gather_nd(self.Qs, state_action_pairs)[0]
                err_estimate = target - estimate

                global_step = tf.Variable(0, trainable=False, name="global_step", collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
                lr = tf.train.exponential_decay(tf.constant(self.lr, dtype=tf.float32), global_step, self.lr_decay_steps, 0.5, staircase=True)
                tf.summary.scalar('lr', lr)
                inc_global_step = global_step.assign_add(1)
                with tf.control_dependencies([update_et_op, inc_global_step]):
                    self.loss = tf.reduce_sum(err_estimate * et)
                    self.train_op = tf.assign_add(self.Qs, lr * err_estimate * et)

            self.score_plh = tf.placeholder(tf.float32, shape=[])
            self.score_sum_t = tf.summary.scalar('score', self.score_plh)
            self.loss_plh = tf.placeholder(tf.float32, shape=[])
            self.loss_sum_t = tf.summary.scalar('loss', self.loss_plh)
            self.all_summary_t = tf.summary.merge_all()

            # Playing part
            self.pscore_plh = tf.placeholder(tf.float32, shape=[])
            self.pscore_sum_t = tf.summary.scalar('play_score', self.pscore_plh)

        return graph

    def learn_from_episode(self, env, render=False):
        self.sess.run(self.reset_et_op)

        super(TabularSigmaLambdaBackwardAgent, self).learn_from_episode(env, render)