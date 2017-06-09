import numpy as np
import tensorflow as tf

from agents import TabularQERAgent, capacities

class TabularQDoubleERAgent(TabularQERAgent):
    """
    Agent implementing tabular Q-learning with experience replay and a second fixed network.
    """
    def __init__(self, config, env):
        super(TabularQDoubleERAgent, self).__init__(config, env)

        self.update_every = config['update_every']

    def get_best_config(self, env_name=""):
        return {
            'lr': 0.03
            , 'lr_decay_steps': 40000
            , 'discount': 0.999
            , 'N0': 75
            , 'min_eps': 0.005
            , 'initial_q_value': 0
            , 'er_batch_size': 783
            , 'er_rm_size': 36916
            , "update_every": 200
        }

    @staticmethod
    def get_random_config(fixed_params={}):
        get_lr = lambda: 1e-3 + (1 - 1e-3) * np.random.random(1)[0]
        get_lr_decay_steps = lambda: np.random.randint(1e3, 1e5)
        get_discount = lambda: 0.5 + (1 - 0.5) * np.random.random(1)[0]
        get_N0 = lambda: np.random.randint(1, 5e3)
        get_min_eps = lambda: 1e-4 + (1e-1 - 1e-4) * np.random.random(1)[0]
        get_initial_q_value = lambda: 0 # int(np.random.random(1)[0] * 200)
        get_er_batch_size = lambda: np.random.randint(16, 1024)
        get_er_rm_size = lambda: np.random.randint(1000, 50000)
        get_update_every = lambda: np.random.randint(1, 1000)

        random_config = {
            'lr': get_lr()
            , 'lr_decay_steps': get_lr_decay_steps()
            , 'discount': get_discount()
            , 'N0': get_N0()
            , 'min_eps': get_min_eps()
            , 'initial_q_value': get_initial_q_value()
            , 'er_batch_size': get_er_batch_size()
            , 'er_rm_size': get_er_rm_size()
            , 'update_every': get_update_every()
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

            fixed_q_scope = tf.VariableScope(reuse=False, name='FixedQValues')
            with tf.variable_scope(fixed_q_scope):
                self.update_fixed_vars_op = capacities.fix_scope(q_scope)

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

            # Experienced replay part
            with tf.variable_scope('Learning'):
                with tf.variable_scope(fixed_q_scope, reuse=True):
                    fixed_Qs = tf.get_variable('Qs')

                self.rewards_plh = tf.placeholder(tf.float32, shape=[None], name="rewards_plh")
                self.next_states_plh = tf.placeholder(tf.int32, shape=[None], name="next_states_plh")

                # Note that we use the fixed Qs to create the targets
                self.targets_t = capacities.get_q_learning_target(fixed_Qs, self.rewards_plh, self.next_states_plh, self.discount)
                self.loss, self.train_op = capacities.tabular_learning_with_lr(
                    self.lr, self.lr_decay_steps, self.Qs, self.inputs_plh, self.actions_t, self.targets_t
                )

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

            memory = np.array([(state_id, act, reward, next_state_id)], dtype=self.replayMemoryDt)
            if self.replayMemory.shape[0] >= self.er_rm_size:
                self.replayMemory = np.delete(self.replayMemory, 0)
            self.replayMemory = np.append(self.replayMemory, memory)

            memories = np.random.choice(self.replayMemory, self.er_batch_size)
            loss, _, event_count, _ = self.sess.run([self.loss, self.inc_event_count_op, self.event_count, self.train_op], feed_dict={
                self.inputs_plh: memories['states'],
                self.actions_t: memories['actions'],
                self.rewards_plh: memories['rewards'],
                self.next_states_plh: memories['next_states'],
            })
            if event_count % self.update_every == 0:
                self.sess.run(self.update_fixed_vars_op)

            av_loss.append(loss)
            score += reward
            obs = next_obs

        summary, _, episode_id = self.sess.run([self.all_summary_t, self.inc_ep_id_op, self.episode_id], feed_dict={
            self.score_plh: score,
            self.loss_plh: np.mean(av_loss)
        })
        self.sw.add_summary(summary, episode_id)

        return