import numpy as np
import tensorflow as tf

from agents import BasicAgent, capacities
from agents.rnn_cell import LSTMCell
from agents.capacities import get_expected_rewards, build_batches

class CMAgent(BasicAgent):
    """
    Agent implementing CM algorithm
    """
    def set_agent_props(self):
        self.m_params = {
            'env_state_size': self.observation_space.shape[0] + 1
            , 'nb_actions': self.action_space.n
            , 'nb_units': self.config['nb_m_units']
            , 'initial_mean': self.config['initial_mean']
            , 'initial_stddev': self.config['initial_m_stddev']
            , 'lr': self.config['m_lr']
        }

        self.c_params = {
            'nb_inputs': self.observation_space.shape[0] + 1
            , 'nb_units': self.config['nb_units']
            , 'nb_actions': self.action_space.n
            , 'initial_mean': self.config['initial_mean']
            , 'initial_stddev': self.config['initial_stddev']
            , 'lr': self.config['lr']
        }

        self.discount = self.config['discount']

        self.nb_wake_iter = self.config['nb_wake_iter']
        self.nb_sleep_iter = self.config['nb_sleep_iter']

        self.dtKeys = ['states', 'actions', 'rewards', 'next_states']
        self.memoryDt = np.dtype([
            ('states', 'float32', (self.m_params['env_state_size'],))
            , ('actions', 'float32', (1, ))
            , ('rewards', 'float32', (1, ))
            , ('next_states', 'float32', (self.m_params['env_state_size'],))
        ])
        self.episode_histories = []

        self.C_SUMMARIES = "c_summaries"
        self.M_SUMMARIES = "m_summaries"

    def get_best_config(self, env_name=""):
        return {
            'lr': 3e-3
            , 'm_lr': 3e-2
            , 'discount': 0.999
            , 'nb_units': 15
            , 'nb_m_units': 23
            , 'initial_mean': 0.
            , 'initial_stddev': 0.16604040438411888
            , 'initial_m_stddev': 0.28261298970489424
            , 'nb_sleep_iter': 32
            , 'nb_wake_iter': 8
            # , 'random_seed': 1
        }
        
    @staticmethod
    def get_random_config(fixed_params={}):
        get_lr = lambda: 1e-4 + (1e-1 - 1e-4) * np.random.random(1)[0]
        get_discount = lambda: 0.5 + (1 - 0.5) * np.random.random(1)[0]
        get_nb_units = lambda: np.random.randint(10, 100)
        get_initial_mean = lambda: 0
        get_initial_stddev = lambda: 5e-1 * np.random.random(1)[0]
        get_initial_m_stddev = lambda: 1. * np.random.random(1)[0]
        get_nb_iter = lambda: np.random.randint(10, 200)
        
        random_config = {
            'lr': get_lr()
            , 'm_lr': get_lr()
            , 'discount': get_discount()
            , 'nb_units': get_nb_units()
            , 'nb_m_units': get_nb_units()
            , 'initial_mean': get_initial_mean()
            , 'initial_stddev': get_initial_stddev()
            , 'initial_m_stddev': get_initial_m_stddev()
            , 'nb_sleep_iter': get_nb_iter()
            , 'nb_wake_iter': get_nb_iter()
        }
        random_config.update(fixed_params)

        return random_config

    def build_graph(self, graph):
        self.env.seed(self.random_seed)
        np.random.seed(self.random_seed)
        with graph.as_default():
            tf.set_random_seed(self.random_seed)

            # Graph of the  LSTM model of the world
            input_scope = tf.VariableScope(reuse=False, name="inputs")
            with tf.variable_scope(input_scope):
                self.state_input_plh = tf.placeholder(tf.float32, shape=[None, None, self.m_params['env_state_size']], name='state_input_plh')
                self.action_input_plh = tf.placeholder(tf.int32, shape=[None, None, 1], name='action_input_plh')
                self.mask_plh = tf.placeholder(tf.float32, shape=[None, None, 1], name="mask_plh")
                
                input_shape = tf.shape(self.state_input_plh)
                dynamic_batch_size, dynamic_num_steps = input_shape[0], input_shape[1]

                action_input = tf.one_hot(
                    indices=tf.squeeze(self.action_input_plh, 2)
                    , depth=self.m_params['nb_actions']
                )
                m_inputs = tf.concat([self.state_input_plh, action_input], 2, name="m_inputs")

            m_scope = tf.VariableScope(reuse=False, name="m")
            with tf.variable_scope(m_scope):
                m_cell = LSTMCell(
                    num_units=self.m_params['nb_units']
                    , initializer=tf.truncated_normal_initializer(
                        mean=self.m_params['initial_mean']
                        , stddev=self.m_params['initial_stddev']
                    )
                )
                self.m_state = m_cell.zero_state(dynamic_batch_size, dtype=tf.float32)
                m_c_h_states, self.m_final_state = tf.nn.dynamic_rnn(m_cell, m_inputs, initial_state=self.m_state)
                m_c_states, m_h_states = tf.split(value=m_c_h_states, num_or_size_splits=[self.m_params['nb_units'], self.m_params['nb_units']], axis=2)

                # Compute the Controller projection
                WM_proj = tf.get_variable(
                    "WM_proj"
                    , shape=[self.m_params['nb_units'], self.m_params['env_state_size'] + 1]
                    , dtype=tf.float32
                    , initializer=tf.truncated_normal_initializer(
                        mean=self.m_params['initial_mean']
                        , stddev=self.m_params['initial_stddev']
                    )
                )
                m_h_states_mat = tf.reshape(m_h_states, [-1, self.m_params['nb_units']])
                state_reward_preds_mat = tf.matmul(m_h_states_mat, WM_proj) 
                self.state_reward_preds = tf.reshape(state_reward_preds_mat, [dynamic_batch_size, -1, self.m_params['env_state_size'] + 1], name="actions_t")

            m_training_scope = tf.VariableScope(reuse=False, name='m_training')
            with tf.variable_scope(m_training_scope):
                self.m_next_states = tf.placeholder(tf.float32, shape=[None, None, self.m_params['env_state_size']], name="m_next_states")
                self.m_rewards = tf.placeholder(tf.float32, shape=[None, None, 1], name="m_rewards")
                y_true = tf.concat([self.m_rewards, self.m_next_states], 2)

                with tf.control_dependencies([self.state_reward_preds]):
                    self.m_loss = 1 / 2 * tf.reduce_mean(tf.square(self.state_reward_preds - y_true) * self.mask_plh)
                    tf.summary.scalar('m_loss', self.m_loss, collections=[self.M_SUMMARIES])

                m_adam = tf.train.AdamOptimizer(self.m_params['lr'])
                self.m_global_step = tf.Variable(0, trainable=False, name="m_global_step")
                self.m_train_op = m_adam.minimize(self.m_loss, global_step=self.m_global_step)
            
            self.all_m_summary_t = tf.summary.merge_all(key=self.M_SUMMARIES)

            # Graph of the controller
            c_scope = tf.VariableScope(reuse=False, name="c")
            with tf.variable_scope(c_scope):
                c_cell = LSTMCell(
                    num_units=self.c_params['nb_units']
                    , initializer=tf.truncated_normal_initializer(
                        mean=self.c_params['initial_mean']
                        , stddev=self.c_params['initial_stddev']
                    )
                )
                self.c_initial_state = c_cell.zero_state(dynamic_batch_size, dtype=tf.float32)
                c_c_h_states, self.c_final_state = tf.nn.dynamic_rnn(c_cell, self.state_input_plh, initial_state=self.c_initial_state)
                c_c_states, c_h_states = tf.split(value=c_c_h_states, num_or_size_splits=[self.c_params['nb_units'], self.c_params['nb_units']], axis=2)
                
                # Compute the Controller projection
                WC_proj = tf.get_variable(
                    "WC_proj"
                    , shape=[self.c_params['nb_units'], self.c_params['nb_actions']]
                    , dtype=tf.float32
                    , initializer=tf.truncated_normal_initializer(
                        mean=self.c_params['initial_mean']
                        , stddev=self.c_params['initial_stddev']
                    )
                )
                bC_proj = tf.get_variable("bC_proj", shape=[self.c_params['nb_actions']], dtype=tf.float32)
                c_h_states_mat = tf.reshape(c_h_states, [-1, self.c_params['nb_units']])
                logits = tf.matmul(c_h_states_mat, WC_proj)

                probs_t = tf.nn.softmax(logits, 1, name="probs_t")
                self.probs_t = tf.reshape(probs_t, [dynamic_batch_size, -1, self.c_params['nb_actions']], name="actions_t")

                actions_t = tf.cast(tf.multinomial(logits, 1), tf.int32)
                self.actions_t = tf.reshape(actions_t, [dynamic_batch_size, -1, 1], name="actions_t")

                # helper tensor used for inference
                self.action_t = self.actions_t[0, 0, 0]

            # Interface Controller -> Model
            # cm_scope = tf.VariableScope(reuse=False, name="cm")
            # with tf.variable_scope(cm_scope):
                # Compute the activation of the model form the current_state and a potential action from the controller
                # m_inputs_from_c = tf.concat([self.state_input_plh, tf.cast(self.actions_t, tf.float32)], 2, name="m_inputs_2")
                # _, self.m_final_state_tmp = m_cell(m_inputs_from_c, self.m_state)

                # print(self.m_final_state_tmp)
                # # We compute the fast weights for model cell state
                # c_final_c, c_final_h = self.c_final_state
                # m_final_c, m_final_h = self.m_final_state_tmp
                # W_cm = tf.get_variable('W_cm', shape=[self.m_params['nb_units'], self.m_params['nb_units']])
                # b_cm =tf.get_variable('b_cm', shape=[self.m_params['nb_units']])
                # fast_m_cell_weight_update = tf.nn.relu(tf.matmul(c_final_h, W_cm) + b_cm)
                # fast_m_cell_weight = tf.nn.relu(m_final_c + fast_m_cell_weight_update)


            c_training_scope = tf.VariableScope(reuse=False, name='c_training')
            with tf.variable_scope(c_training_scope):
                self.c_rewards_plh = tf.placeholder(tf.float32, shape=[None, None, 1], name="c_rewards_plh")

                baseline = tf.reduce_mean(self.c_rewards_plh)

                batch_size, num_steps = tf.shape(self.actions_t)[0], tf.shape(self.actions_t)[1]
                line_indices = tf.matmul( # Line indice
                    tf.reshape(tf.range(0, batch_size), [-1, 1])
                    , tf.ones([1, num_steps], dtype=tf.int32)
                )
                column_indices = tf.matmul( # Column indice
                    tf.ones([batch_size, 1], dtype=tf.int32)
                    , tf.reshape(tf.range(0, num_steps), [1, -1])
                )
                depth_indices = tf.squeeze(self.actions_t, 2)
                stacked_actions = tf.stack(
                    [line_indices, column_indices, depth_indices], 2
                )

                with tf.control_dependencies([self.probs_t]):
                    log_probs = tf.expand_dims(tf.log(tf.gather_nd(self.probs_t, stacked_actions)), 2)
                    masked_log_probs = log_probs * self.mask_plh
                    self.c_loss = tf.reduce_mean( - tf.reduce_sum(masked_log_probs * (self.c_rewards_plh - baseline), 1))
                    tf.summary.scalar('c_loss', self.c_loss, collections=[self.C_SUMMARIES])

                c_adam = tf.train.AdamOptimizer(self.c_params['lr'])
                self.c_global_step = tf.Variable(0, trainable=False, name="global_step", collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES], dtype=tf.int32)
                self.c_train_op = c_adam.minimize(self.c_loss, global_step=self.c_global_step)

            self.all_c_summary_t = tf.summary.merge_all(key=self.C_SUMMARIES)

            self.score_plh = tf.placeholder(tf.float32, shape=[])
            self.score_sum_t = tf.summary.scalar('av_score', self.score_plh)

            self.episode_id, self.inc_ep_id_op = capacities.counter("episode_id")

            # Playing part
            self.pscore_plh = tf.placeholder(tf.float32, shape=[])
            self.pscore_sum_t = tf.summary.scalar('play_score', self.pscore_plh)

        return graph

    def act(self, obs, prev_c_hidden_state=None):
        state = np.concatenate((obs, [float(False)]))
        if prev_c_hidden_state is None:
            act, c_hidden_state = self.sess.run([self.action_t, self.c_final_state], feed_dict={
                self.state_input_plh: [ [ state ] ]
            })
        else:
            act, c_hidden_state = self.sess.run([self.action_t, self.c_final_state], feed_dict={
                self.state_input_plh: [ [ state ] ]
                , self.c_initial_state: prev_c_hidden_state
            })

        return act, state, c_hidden_state

    def train(self, render=False, save_every=49):
        for i in range(0, self.max_iter, self.nb_wake_iter):
            # Wake phase
            # Collect a batched trajectories
            sequence_history, episode_id = self.collect_samples(self.env, render, self.nb_wake_iter)

            # On-policy learning only
            batches = build_batches(self.dtKeys, sequence_history, len(sequence_history))
            self.train_controller(batches[0])

            # sleep phase
            self.train_model()

            if save_every > 0 and i % save_every > (i + self.nb_wake_iter) % save_every:
                self.save()

    def collect_samples(self, env, render, nb_sequence=1):
        sequence_history = []
        av_score = []
        for i in range(nb_sequence):
            obs = env.reset()
            score = 0
            c_hidden_state = None
            episode_history = np.array([], dtype=self.memoryDt)
            done = False

            while True:
                if render:
                    env.render()

                act, state, c_hidden_state = self.act(obs, c_hidden_state)
                next_obs, reward, done, info = env.step(act)
                next_state = np.concatenate( (next_obs, [float(done)]) )

                memory = np.array([(state, act, reward, next_state)], dtype=self.memoryDt)
                episode_history = np.append(episode_history, memory)

                score += reward
                obs = next_obs
                if done:
                    break
            sequence_history.append(episode_history)
            av_score.append(score)

            self.sess.run(self.inc_ep_id_op)

        summary, episode_id = self.sess.run([self.score_sum_t, self.episode_id], feed_dict={
            self.score_plh: np.mean(score),
        })
        self.sw.add_summary(summary, episode_id)

        self.episode_histories += sequence_history

        return sequence_history, episode_id

    def train_controller(self, batch):
        for i, episode_rewards in enumerate(batch['rewards']):
            batch['rewards'][i] = get_expected_rewards(episode_rewards, self.discount)

        _, c_sum, c_step = self.sess.run([self.c_train_op, self.all_c_summary_t, self.c_global_step], feed_dict={
            self.state_input_plh: batch['states']
            , self.actions_t: batch['actions']
            , self.c_rewards_plh: batch['rewards']
            , self.mask_plh: batch['mask'] 
        })
            
        self.sw.add_summary(c_sum, c_step)

        return

    def train_model(self):
        batches = build_batches(self.dtKeys, self.episode_histories, self.nb_sleep_iter)

        # We train on maximum a hundered episode each time
        for batch in batches:
            _, m_sum, m_step = self.sess.run([self.m_train_op, self.all_m_summary_t, self.m_global_step], feed_dict={
                self.state_input_plh: batch['states']
                , self.action_input_plh: batch['actions']
                , self.m_rewards: batch['rewards']
                , self.m_next_states: batch['next_states']
                , self.mask_plh: batch['mask'] 
            })

            self.sw.add_summary(m_sum, m_step)
