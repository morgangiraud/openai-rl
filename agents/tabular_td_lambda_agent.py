import numpy as np
import tensorflow as tf

from agents import TabularBasicAgent, capacities
 
class TabularTDLambdaAgent(TabularBasicAgent):
    """
    Agent implementing tabular td(lambda) learning
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
        return {
            'lr': 1e-4
            , 'lr_decay_steps': 40000
            , 'discount': 0.999 # ->1[ improve
            , 'N0': 76 # -> ~ 75 improve
            , 'min_eps': 0.001 # ->0.001[ improve
            , 'initial_q_value': 0
            , 'lambda': 0.9
        }

    @staticmethod
    def get_random_config(fixed_params={}):
        get_lr = lambda: 1e-4 + (1e-1 - 1e-4) * np.random.random(1)[0]
        get_lr_decay_steps = lambda: np.random.randint(1e3, 5e5)
        get_discount = lambda: 0.5 + (1 - 0.5) * np.random.random(1)[0]
        get_N0 = lambda: np.random.randint(1, 5e3)
        get_min_eps = lambda: 1e-4 + (1e-1 - 1e-4) * np.random.random(1)[0]
        get_initial_q_value = lambda: 0 # int(np.random.random(1)[0] * 200)
        get_lambda = lambda: np.random.rand() 

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

    def act(self, obs, done=False):
        state_id = self.phi(obs, done)
        act, estimate = self.sess.run([self.action_t, self.q_value_t], feed_dict={
            self.inputs_plh: [ state_id ]
        })

        return act, state_id, estimate

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

            learning_scope = tf.VariableScope(reuse=False, name='TDLearning')
            with tf.variable_scope(learning_scope):
                self.rewards_plh = tf.placeholder(tf.float32, shape=[None], name="rewards_plh")
                self.targets_plh = tf.placeholder(tf.float32, shape=[None], name="targets_plh")

                self.loss, self.train_op = capacities.tabular_learning_with_lr(
                    self.lr, self.lr_decay_steps, self.Qs, self.inputs_plh, self.actions_t, self.targets_plh
                )

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
        score = 0        
        historyType = np.dtype([('states', 'int32'), ('actions', 'int32'), ('rewards', 'float32'), ('estimates', 'float32')])
        history = np.array([], dtype=historyType)
        done = False
        
        obs = env.reset()
        act, state_id, estimate = self.act(obs)
        while not done:
            if render:
                env.render()
            
            next_obs, reward, done, info = env.step(act)
            next_act, next_state_id, next_estimate = self.act(next_obs, done)

            memory = np.array([(state_id, act, reward, next_estimate)], dtype=historyType)
            history = np.append(history, memory)

            score += reward
            act = next_act
            state_id = next_state_id

        targets = capacities.get_lambda_expected_rewards(history['rewards'], history['estimates'], self.discount, self.lambda_value)
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict={
            self.inputs_plh: history['states'],
            self.actions_t: history['actions'],
            self.targets_plh: targets,
        })

        summary, _, episode_id = self.sess.run([self.all_summary_t, self.inc_ep_id_op, self.episode_id], feed_dict={
            self.score_plh: score,
            self.loss_plh: loss
        })
        self.sw.add_summary(summary, episode_id)