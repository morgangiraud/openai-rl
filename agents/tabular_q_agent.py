import numpy as np
import tensorflow as tf

from agents import TabularBasicAgent, capacities

class TabularQAgent(TabularBasicAgent):
    """
    Agent implementing tabular Q-learning.
    """
    def set_agent_props(self):
        self.lr = self.config['lr']
        self.lr_decay_steps = self.config['lr_decay_steps']
        self.discount = self.config['discount']
        self.N0 = self.config['N0']
        self.min_eps = self.config['min_eps']
        self.initial_q_value = self.config['initial_q_value']

    def get_best_config(self, env_name=""):
        cartpolev0 = {
            'lr': 0.3180045122383971
            , 'lr_decay_steps': 50000
            , 'discount': 0.999
            , 'N0': 11
            , 'min_eps': 0.001
            , 'initial_q_value': 0
        }
        return {
            'CartPole-v0': cartpolev0
        }.get(env_name, cartpolev0)
        

    @staticmethod
    def get_random_config(fixed_params={}):
        get_lr = lambda: 1e-2 + (1 - 1e-2) * np.random.random(1)[0]
        get_lr_decay_steps = lambda: np.random.randint(1e3, 1e5)
        get_discount = lambda: 0.5 + (1 - 0.5) * np.random.random(1)[0]
        get_N0 = lambda: np.random.randint(1, 5e3)
        get_min_eps = lambda: 1e-4 + (1e-1 - 1e-4) * np.random.random(1)[0]
        get_initial_q_value = lambda: 0 # int(np.random.random(1)[0] * 200)

        random_config = {
            'lr': get_lr()
            , 'lr_decay_steps': get_lr_decay_steps()
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
                self.actions_t, self.probs_t = capacities.tabular_eps_greedy(
                    self.inputs_plh, self.q_preds_t, self.env.action_space.n, self.N0, self.min_eps, self.nb_state
                )
                self.action_t = self.actions_t[0]
                self.q_value_t = self.q_preds_t[0][self.action_t]

            learning_scope = tf.VariableScope(reuse=False, name='QLearning')
            with tf.variable_scope(learning_scope):
                self.rewards_plh = tf.placeholder(tf.float32, shape=[None], name="rewards_plh")
                self.next_states_plh = tf.placeholder(tf.int32, shape=[None], name="next_states_plh")

                self.targets_t = capacities.get_q_learning_target(self.Qs, self.rewards_plh, self.next_states_plh, self.discount)
                self.loss, self.train_op = capacities.tabular_learning_with_lr(
                    self.lr, self.lr_decay_steps, self.Qs, self.inputs_plh, self.actions_t, self.targets_t
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

    def act(self, obs):
        state_id = self.phi(obs)
        act = self.sess.run(self.action_t, feed_dict={
            self.inputs_plh: [ state_id ]
        })

        return act, state_id

    def learn_from_episode(self, env, render=False):
        score = 0
        av_loss = []
        done = False

        obs = env.reset()
        while not done:
            if render:
                env.render()

            act, state_id = self.act(obs)
            next_obs, reward, done, info = env.step(act)
            next_state_id = self.phi(next_obs, done)

            loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={
                self.inputs_plh: [ state_id ],
                self.actions_t: [ act ],
                self.rewards_plh: [ reward ],
                self.next_states_plh: [ next_state_id ], 
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