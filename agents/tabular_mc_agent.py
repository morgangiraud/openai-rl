import numpy as np
import tensorflow as tf

from agents import TabularBasicAgent, capacities

class TabularMCAgent(TabularBasicAgent):
    """
    Agent implementing tabular Q-learning.
    """
    def set_agent_props(self):
        self.discount = self.config['discount']
        self.N0 = self.config['N0']
        self.min_eps = self.config['min_eps']
        self.initial_q_value = self.config['initial_q_value']

    def get_best_config(self, env_name=""):
        cartpolev0 = {
            'discount': .99
            , 'N0': 10
            , 'min_eps': 0.001
            , 'initial_q_value': 0
        }
        mountaincarv0 = {
            'discount': 0.99
            , 'N0': 10
            , 'min_eps': 0.001
            , 'initial_q_value': 0 # This is an optimistic initialization
        }
        acrobotv1 = {
            "discount": 0.999
            , "initial_q_value": 0 # This is an optimistic initialization
            , "N0": 100
            , "min_eps": 0.11409578938939571
          }
        return {
            'CartPole-v0': cartpolev0
            , 'MountainCar-v0': mountaincarv0
            , 'Acrobot-v1': acrobotv1
        }.get(env_name, cartpolev0)

    @staticmethod
    def get_random_config(fixed_params={}):
        get_discount = lambda: 0.98 + (1 - 0.98) * np.random.random(1)[0]
        get_N0 = lambda: np.random.randint(1, 1e3)
        get_min_eps = lambda: 1e-4 + (2e-1 - 1e-4) * np.random.random(1)[0]
        get_initial_q_value = lambda: 0

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

            policy_scope = tf.VariableScope(reuse=False, name='Policy')
            with tf.variable_scope(policy_scope):
                if 'UCB' in self.config and self.config['UCB']:
                    self.actions_t, self.probs_t = capacities.tabular_UCB(
                        self.Qs, self.inputs_plh
                    )    
                else:
                    self.actions_t, self.probs_t = capacities.tabular_eps_greedy(
                        self.inputs_plh, self.q_preds_t, self.nb_state, self.env.action_space.n, self.N0, self.min_eps
                    )
                self.action_t = self.actions_t[0]
                self.q_value_t = self.q_preds_t[0][self.action_t]

            learning_scope = tf.VariableScope(reuse=False, name='Learning')
            with tf.variable_scope(learning_scope):
                self.rewards_plh = tf.placeholder(tf.float32, shape=[None], name="rewards_plh")

                self.targets_t = capacities.get_mc_target(self.rewards_plh, self.discount)
                self.loss, self.train_op = capacities.tabular_learning(
                    self.Qs, self.inputs_plh, self.actions_t, self.targets_t
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

    def act(self, obs, done=False):
        state_id = self.phi(obs, done)
        act = self.sess.run(self.action_t, feed_dict={
            self.inputs_plh: [ state_id ]
        })

        return act, state_id

    def learn_from_episode(self, env, render=False):
        score = 0
        episodeType = np.dtype([('states', 'int32'), ('actions', 'int32'), ('rewards', 'float32')])
        episode = np.array([], dtype=episodeType)
        done = False

        obs = env.reset()
        while not done:
            if render:
                env.render()

            act, state_id= self.act(obs)
            obs, reward, done, info = env.step(act)

            memory = np.array([(state_id, act, reward)], dtype=episodeType)
            episode = np.append(episode, memory)

            score += reward

        _, loss = self.sess.run([self.train_op, self.loss], feed_dict={
            self.inputs_plh: episode['states'],
            self.actions_t: episode['actions'],
            self.rewards_plh: episode['rewards'],
        })
        summary, _, episode_id = self.sess.run([self.all_summary_t, self.inc_ep_id_op, self.episode_id], feed_dict={
            self.score_plh: score,
            self.loss_plh: loss
        })
        self.sw.add_summary(summary, episode_id)