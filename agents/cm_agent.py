import numpy as np
import tensorflow as tf

from agents import BasicAgent, capacities
from agents.rnn_cell import LSTMCell, CMCell
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
            , 'nb_units': 10
            , 'nb_m_units': 50
            , 'initial_mean': 0.
            , 'initial_stddev': 0.16604040438411888
            , 'initial_m_stddev': 0.28261298970489424
            , 'nb_sleep_iter': 32
            , 'nb_wake_iter': 8
            , 'random_seed': 1
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
                self.state_reward_preds, self.m_final_state, self.m_initial_state = capacities.predictive_model(
                    self.m_params, m_inputs, dynamic_batch_size, None, summary_collections=[self.M_SUMMARIES]
                )

            fixed_m_scope = tf.VariableScope(reuse=False, name='FixedM')
            with tf.variable_scope(fixed_m_scope):
                self.update_m_fixed_vars_op = capacities.fix_scope(m_scope)

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
                tf.summary.scalar('m_global_step', self.m_global_step, collections=[self.M_SUMMARIES])
                self.m_train_op = m_adam.minimize(self.m_loss, global_step=self.m_global_step)
            
            self.all_m_summary_t = tf.summary.merge_all(key=self.M_SUMMARIES)

            # Graph of the controller
            c_scope = tf.VariableScope(reuse=False, name="c")
            c_summary_collection = [self.C_SUMMARIES]
            with tf.variable_scope(c_scope):
                # c_cell = LSTMCell(
                #     num_units=self.c_params['nb_units']
                #     , initializer=tf.truncated_normal_initializer(
                #         mean=self.c_params['initial_mean']
                #         , stddev=self.c_params['initial_stddev']
                #     )
                # )
                # self.c_initial_state = c_cell.zero_state(dynamic_batch_size, dtype=tf.float32)
                # c_c_h_states, self.c_final_state = tf.nn.dynamic_rnn(c_cell, self.state_input_plh, initial_state=self.c_initial_state)
                # c_c_states, c_h_states = tf.split(value=c_c_h_states, num_or_size_splits=[self.c_params['nb_units'], self.c_params['nb_units']], axis=2)
                # # Compute the Controller projection
                # self.probs_t, self.actions_t = projection_func(c_h_states)
                m_params = self.m_params
                model_func = lambda m_inputs, m_state: capacities.predictive_model(m_params, m_inputs, dynamic_batch_size, m_state)
                c_params = self.c_params
                projection_func = lambda inputs:capacities.projection(c_params, inputs)
                cm_cell = CMCell(
                    num_units=self.c_params['nb_units'], m_units=self.m_params['nb_units']
                    , fixed_model_scope=fixed_m_scope, model_func=model_func
                    , projection_func=projection_func, num_proj=self.c_params['nb_actions']
                    , initializer=tf.truncated_normal_initializer(
                        mean=self.c_params['initial_mean']
                        , stddev=self.c_params['initial_stddev']
                    )
                )

                self.cm_initial_state = cm_cell.zero_state(dynamic_batch_size, dtype=tf.float32)
                probs_and_actions_t, self.cm_final_state = tf.nn.dynamic_rnn(cm_cell, self.state_input_plh, initial_state=self.cm_initial_state)
                self.probs_t, actions_t = tf.split(value=probs_and_actions_t, num_or_size_splits=[self.c_params['nb_actions'], 1], axis=2)
                self.actions_t = tf.cast(actions_t, tf.int32)
                # helper tensor used for inference
                self.action_t = self.actions_t[0, 0, 0]

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
                    tf.summary.scalar('c_loss', self.c_loss, collections=c_summary_collection)

                c_adam = tf.train.AdamOptimizer(self.c_params['lr'])
                self.c_global_step = tf.Variable(0, trainable=False, name="global_step", collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES], dtype=tf.int32)
                tf.summary.scalar('c_global_step', self.c_global_step, collections=c_summary_collection)
                self.c_train_op = c_adam.minimize(self.c_loss, global_step=self.c_global_step)

            self.all_c_summary_t = tf.summary.merge_all(key=self.C_SUMMARIES)

            self.score_plh = tf.placeholder(tf.float32, shape=[])
            self.score_sum_t = tf.summary.scalar('score', self.score_plh)

            self.episode_id, self.inc_ep_id_op = capacities.counter("episode_id")
            self.episode_id_sum = tf.summary.scalar('episode_id', self.episode_id)
            self.time, self.inc_time_op = capacities.counter("time")

            # Playing part
            self.pscore_plh = tf.placeholder(tf.float32, shape=[])
            self.pscore_sum_t = tf.summary.scalar('play_score', self.pscore_plh)

        return graph

    def act(self, obs, cm_prev_state=None):
        state = np.concatenate((obs, [float(False)]))
        if cm_prev_state is None:
            act, cm_prev_state = self.sess.run([self.action_t, self.cm_final_state], feed_dict={
                self.state_input_plh: [ [ state ] ]
            })
        else:
            act, cm_prev_state = self.sess.run([self.action_t, self.cm_final_state], feed_dict={
                self.state_input_plh: [ [ state ] ]
                , self.cm_initial_state: cm_prev_state
            })

        return act, state, cm_prev_state

    def train(self, render=False, save_every=49):
        self.sess.run(self.update_m_fixed_vars_op)
        for i in range(0, self.max_iter, self.nb_wake_iter):
            # Wake phase
            # Collect a batched trajectories
            sequence_history = self.collect_samples(self.env, render, self.nb_wake_iter)

            # On-policy learning only
            batches = build_batches(self.dtKeys, sequence_history, len(sequence_history))
            self.train_controller(batches[0])

            # sleep phase
            # If you train the model before the controller
            # The controller is not learning on-policy anymore
            self.train_model(sequence_history)

            if save_every > 0 and i % save_every > (i + self.nb_wake_iter) % save_every:
                self.save()

    def collect_samples(self, env, render, nb_sequence=1):
        # print('Collecting samples')
        sequence_history = []
        av_score = []
        for i in range(nb_sequence):
            obs = env.reset()
            score = 0
            cm_prev_state = None
            episode_history = np.array([], dtype=self.memoryDt)
            done = False

            while True:
                if render:
                    env.render()

                act, state, cm_prev_state = self.act(obs, cm_prev_state)
                next_obs, reward, done, info = env.step(act)

                memory = np.array([(
                    state, 
                    act, 
                    reward, 
                     np.concatenate( (next_obs, [float(done)]) )
                )], dtype=self.memoryDt)
                episode_history = np.append(episode_history, memory)

                score += reward
                obs = next_obs
                if done:
                    break
            sequence_history.append(episode_history)
            av_score.append(score)

            episode_id_sum, _, time, _ = self.sess.run([self.episode_id_sum, self.inc_ep_id_op, self.time, self.inc_time_op])
            self.sw.add_summary(episode_id_sum, time)

        summary = self.sess.run(self.score_sum_t, feed_dict={
            self.score_plh: np.mean(score),
        })
        self.sw.add_summary(summary, time)

        self.episode_histories += sequence_history

        return sequence_history

    def train_controller(self, batch):
        # print('Training controller')
        for i, episode_rewards in enumerate(batch['rewards']):
            batch['rewards'][i] = get_expected_rewards(episode_rewards, self.discount)

        _, c_sum, time, _ = self.sess.run([self.c_train_op, self.all_c_summary_t, self.time, self.inc_time_op], feed_dict={
            self.state_input_plh: batch['states']
            , self.actions_t: batch['actions']
            , self.c_rewards_plh: batch['rewards']
            , self.mask_plh: batch['mask'] 
        })
            
        self.sw.add_summary(c_sum, time)

        return

    def train_model(self, sequence_history):
        # print('Training model')
        batches = build_batches(self.dtKeys, self.episode_histories, self.nb_sleep_iter)
        seq_batches = build_batches(self.dtKeys, sequence_history, self.nb_sleep_iter)

        for batch in seq_batches + batches[:2]:
            _, m_sum, time, _ = self.sess.run([self.m_train_op, self.all_m_summary_t, self.time, self.inc_time_op], feed_dict={
                self.state_input_plh: batch['states']
                , self.action_input_plh: batch['actions']
                , self.m_rewards: batch['rewards']
                , self.m_next_states: batch['next_states']
                , self.mask_plh: batch['mask'] 
            })

            self.sw.add_summary(m_sum, time)

        self.sess.run(self.update_m_fixed_vars_op)
