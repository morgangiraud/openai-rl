import numpy as np
import tensorflow as tf

from agents import BasicAgent, capacities
from agents.capacities import get_expected_rewards, build_batches

class PolicyAgent(BasicAgent):
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
        self.lr = self.config['lr']
        self.discount = self.config['discount']
        self.batch_size = 8

        self.dtKeys = ['states', 'actions', 'rewards']
        self.memoryDt = np.dtype([
            ('states', 'float32', (self.policy_params['nb_inputs'],))
            , ('actions', 'int32', (1,))
            , ('rewards', 'float32', (1,))
        ])

    def get_best_config(self, env_name=""):
        return {
            'lr': 3e-3
            , 'discount': 0.99
            , 'nb_units': 41
            , 'initial_mean': 0.
            , 'initial_stddev': 0.3
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
        np.random.seed(self.random_seed)
        with graph.as_default():
            tf.set_random_seed(self.random_seed)

            # Dims: bs x num_steps x state_size
            self.inputs = tf.placeholder(tf.float32, shape=[None, None, self.policy_params['nb_inputs']], name='inputs')
            input_shape = tf.shape(self.inputs)
            dynamic_batch_size, dynamic_num_steps = input_shape[0], input_shape[1]

            policy_scope = tf.VariableScope(reuse=False, name='Policy')
            with tf.variable_scope(policy_scope):
                policy_inputs = tf.reshape(self.inputs, [-1, self.policy_params['nb_inputs']])
                probs, actions = capacities.policy(self.policy_params, policy_inputs)
                self.probs = tf.reshape(probs, [dynamic_batch_size, dynamic_num_steps, self.policy_params['nb_outputs']])
                self.actions = tf.reshape(actions, [dynamic_batch_size, dynamic_num_steps, 1])
            self.action_t = self.actions[0, 0, 0]

            with tf.variable_scope('Training'):
                self.rewards = tf.placeholder(tf.float32, shape=[None, None, 1], name="reward")
                self.mask_plh = tf.placeholder(tf.float32, shape=[None, None, 1], name="mask_plh")

                baseline = tf.reduce_mean(self.rewards)
                
                batch_size, num_steps = tf.shape(self.actions)[0], tf.shape(self.actions)[1]
                line_indices = tf.matmul( # Line indice
                    tf.reshape(tf.range(0, batch_size), [-1, 1])
                    , tf.ones([1, num_steps], dtype=tf.int32)
                )
                column_indices = tf.matmul( # Column indice
                    tf.ones([batch_size, 1], dtype=tf.int32)
                    , tf.reshape(tf.range(0, num_steps), [1, -1])
                )
                depth_indices = tf.cast(tf.squeeze(self.actions, 2), tf.int32)
                stacked_actions = tf.stack(
                    [line_indices, column_indices, depth_indices], 2
                )

                log_probs = tf.expand_dims(tf.log(tf.gather_nd(self.probs, stacked_actions)), 2)
                # We want to average on sequence
                self.loss = tf.reduce_mean( - tf.reduce_sum((log_probs * (self.rewards - baseline)) * self.mask_plh, 1))

                adam = tf.train.AdamOptimizer(self.lr)
                self.global_step = tf.Variable(0, trainable=False, name="global_step", collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
                self.train_op = adam.minimize(self.loss, global_step=self.global_step)

            self.score_plh = tf.placeholder(tf.float32, shape=[])
            self.score_sum_t = tf.summary.scalar('av_score', self.score_plh)
            self.loss_plh = tf.placeholder(tf.float32, shape=[])
            self.loss_sum_t = tf.summary.scalar('loss', self.loss_plh)
            self.all_summary_t = tf.summary.merge_all()

            self.episode_id, self.inc_ep_id_op = capacities.counter("episode_id")

            # Playing part
            self.pscore_plh = tf.placeholder(tf.float32, shape=[])
            self.pscore_sum_t = tf.summary.scalar('play_score', self.pscore_plh)

        return graph

    def act(self, obs):
        state = np.concatenate((obs, [float(False)]))
        act = self.sess.run(self.action_t, feed_dict={
            self.inputs: [ [ state ] ]
        })

        return (act, state)

    def train(self, render=False, save_every=49):
        for i in range(0, self.max_iter, self.batch_size):
            # Collect a batched trajectories
            sequence_history, episode_id = self.collect_samples(self.env, render, self.batch_size)

            # On-policy learning only
            batches = build_batches(self.dtKeys, sequence_history, len(sequence_history))
            self.train_controller(batches[0])

            if save_every > 0 and i % save_every > (i + self.batch_size) % save_every:
                self.save()


    def collect_samples(self, env, render, nb_sequence=1):
        sequence_history = []
        av_score = []
        for i in range(nb_sequence):
            obs = env.reset()
            score = 0
            history = np.array([], dtype=self.memoryDt)
            done = False

            while True:
                if render:
                    env.render()

                act, state = self.act(obs)
                next_obs, reward, done, info = env.step(act)

                memory = np.array([(state, act, reward)], dtype=self.memoryDt)
                history = np.append(history, memory)

                score += reward
                obs = next_obs
                if done:
                    break
            sequence_history.append(history)
            av_score.append(score)

            self.sess.run(self.inc_ep_id_op)

        summary, episode_id = self.sess.run([self.score_sum_t, self.episode_id], feed_dict={
            self.score_plh: np.mean(score),
        })
        self.sw.add_summary(summary, episode_id)

        return sequence_history, episode_id

    def train_controller(self, batch):
        for i, episode_rewards in enumerate(batch['rewards']):
            batch['rewards'][i] = get_expected_rewards(episode_rewards, self.discount)

        _, loss = self.sess.run([self.train_op, self.loss], feed_dict={
            self.inputs: batch['states']
            , self.actions: batch['actions']
            , self.rewards: batch['rewards']
            , self.mask_plh: batch['mask'] 
        })
            
        summary, episode_id = self.sess.run([self.loss_sum_t, self.episode_id], feed_dict={
            self.loss_plh: np.mean(loss),
        })
        self.sw.add_summary(summary, episode_id)

        return


class ActorQCriticAgent(PolicyAgent):
    """
    Agent implementing Policy gradient using Monte-Carlo control
    """
    def set_agent_props(self):
        super(ActorQCriticAgent, self).set_agent_props()

        self.critic_params = {
            'nb_inputs': self.observation_space.shape[0] + 1
            , 'nb_units': self.config['nb_units']
            , 'nb_outputs': self.action_space.n
            , 'initial_mean': self.config['initial_mean']
            , 'initial_stddev': self.config['initial_stddev']
        }
        self.critic_lr = self.config['critic_lr']
        self.nb_critic_iter = self.config['nb_critic_iter']

        self.dtKeys = ['states', 'actions', 'rewards', 'next_states', 'next_actions']
        self.memoryDt = np.dtype([
            ('states', 'float32', (self.policy_params['nb_inputs'],))
            , ('actions', 'int32', (1,))
            , ('rewards', 'float32', (1,))
            , ('next_states', 'float32', (self.policy_params['nb_inputs'],))
            , ('next_actions', 'int32', (1,)),
        ])

    def get_best_config(self, env_name=""):
        return {
            'lr': 3e-3
            , 'discount': 0.99
            , 'nb_units': 41
            , 'initial_mean': 0.
            , 'initial_stddev': 0.3
            , 'critic_lr': 1e-3
        }

    @staticmethod
    def get_random_config(fixed_params={}):
        get_lr = lambda: 1e-4 + (1e-1 - 1e-4) * np.random.random(1)[0]
        get_discount = lambda: 0.5 + (1 - 0.5) * np.random.random(1)[0]
        get_nb_units = lambda: np.random.randint(10, 100)
        get_initial_mean = lambda: 0
        get_initial_stddev = lambda: 5e-1 * np.random.random(1)[0]
        get_nb_critic_iter = lambda: np.random.randint(4, 50)

        random_config = {
            'lr': get_lr()
            , 'discount': get_discount()
            , 'nb_units': get_nb_units()
            , 'initial_mean': get_initial_mean()
            , 'initial_stddev': get_initial_stddev()
            , 'critic_lr': get_lr()
            ,'nb_critic_iter': get_nb_critic_iter()
        }
        random_config.update(fixed_params)

        return random_config

    def build_graph(self, graph):
        np.random.seed(self.random_seed)
        with graph.as_default():
            tf.set_random_seed(self.random_seed)

            self.inputs = tf.placeholder(tf.float32, shape=[None, None, self.policy_params['nb_inputs']], name='inputs')
            input_shape = tf.shape(self.inputs)
            dynamic_batch_size, dynamic_num_steps = input_shape[0], input_shape[1]
            inputs_mat = tf.reshape(self.inputs, [-1, self.policy_params['nb_inputs']])

            policy_scope = tf.VariableScope(reuse=False, name='Policy')
            with tf.variable_scope(policy_scope):
                probs, actions = capacities.policy(self.policy_params, inputs_mat)
                self.probs = tf.reshape(probs, [dynamic_batch_size, dynamic_num_steps, self.policy_params['nb_outputs']])
                self.actions = tf.reshape(actions, [dynamic_batch_size, dynamic_num_steps, 1])
            self.action_t = self.actions[0, 0, 0]

            critic_scope = tf.VariableScope(reuse=False, name='QValues')
            with tf.variable_scope(critic_scope):
                critic_values_mat = capacities.value_f(self.critic_params, inputs_mat)
                self.critic_values = tf.reshape(critic_values_mat, [dynamic_batch_size, dynamic_num_steps, self.critic_params['nb_outputs']])

            fixed_critic_scope = tf.VariableScope(reuse=False, name='FixedQValues')
            with tf.variable_scope(fixed_critic_scope):
                self.update_fixed_vars_op = capacities.fix_scope(critic_scope)

            with tf.variable_scope('Training'):
                self.expected_rewards = tf.placeholder(tf.float32, shape=[None, None, 1], name="reward")
                self.mask_plh = tf.placeholder(tf.float32, shape=[None, None, 1], name="mask_plh")

                baseline = tf.reduce_mean(self.expected_rewards)

                batch_size, num_steps = tf.shape(self.actions)[0], tf.shape(self.actions)[1]
                line_indices = tf.matmul( # Line indice
                    tf.reshape(tf.range(0, batch_size), [-1, 1])
                    , tf.ones([1, num_steps], dtype=tf.int32)
                )
                column_indices = tf.matmul( # Column indice
                    tf.ones([batch_size, 1], dtype=tf.int32)
                    , tf.reshape(tf.range(0, num_steps), [1, -1])
                )
                depth_indices = tf.cast(tf.squeeze(self.actions, 2), tf.int32)
                stacked_actions = tf.stack(
                    [line_indices, column_indices, depth_indices], 2
                )

                qs = tf.expand_dims(tf.gather_nd(self.critic_values, stacked_actions), 2)
                log_probs = tf.expand_dims(tf.log(tf.gather_nd(self.probs, stacked_actions)), 2)
                self.policy_loss = tf.reduce_mean( - tf.reduce_sum((log_probs * (tf.stop_gradient(qs) - baseline)) * self.mask_plh, 1))

                adam = tf.train.AdamOptimizer(self.lr)
                self.train_policy_op = adam.minimize(self.policy_loss)

                self.rewards = tf.placeholder(tf.float32, shape=[None, None, 1], name="reward")
                self.next_states = tf.placeholder(tf.float32, shape=[None, None, self.critic_params['nb_inputs']], name="next_states")
                self.next_actions = tf.placeholder(tf.int32, shape=[None, None, 1], name="next_actions")
                with tf.variable_scope(fixed_critic_scope, reuse=True):
                    next_states_mat = tf.reshape(self.next_states, [-1, self.critic_params['nb_inputs']])
                    next_critic_values_mat = capacities.value_f(self.critic_params, next_states_mat)
                    next_critic_values = tf.reshape(next_critic_values_mat, [dynamic_batch_size, dynamic_num_steps, self.critic_params['nb_outputs']])

                batch_size, num_steps = tf.shape(self.next_actions)[0], tf.shape(self.next_actions)[1]
                line_indices = tf.matmul( # Line indice
                    tf.reshape(tf.range(0, batch_size), [-1, 1])
                    , tf.ones([1, num_steps], dtype=tf.int32)
                )
                column_indices = tf.matmul( # Column indice
                    tf.ones([batch_size, 1], dtype=tf.int32)
                    , tf.reshape(tf.range(0, num_steps), [1, -1])
                )
                depth_indices = tf.cast(tf.squeeze(self.next_actions, 2), tf.int32)
                next_stacked_actions = tf.stack(
                    [line_indices, column_indices, depth_indices], 2
                )
                next_qs = tf.expand_dims(tf.gather_nd(next_critic_values, next_stacked_actions), 2)
                target_qs1 = tf.stop_gradient(self.rewards + self.discount * next_qs)
                target_qs2 = self.rewards
                stacked_targets = tf.stack([tf.squeeze(target_qs1, 2), tf.squeeze(target_qs2, 2)], 2)

                batch_size, num_steps = tf.shape(self.next_states)[0], tf.shape(self.next_states)[1]
                line_indices = tf.matmul( # Line indice
                    tf.reshape(tf.range(0, batch_size), [-1, 1])
                    , tf.ones([1, num_steps], dtype=tf.int32)
                )
                column_indices = tf.matmul( # Column indice
                    tf.ones([batch_size, 1], dtype=tf.int32)
                    , tf.reshape(tf.range(0, num_steps), [1, -1])
                )
                depth_indices = tf.cast(self.next_states[:, :, -1], tf.int32)
                select_targets = tf.stack(
                    [line_indices, column_indices, depth_indices], 2
                )

                target_qs = tf.expand_dims(tf.gather_nd(stacked_targets, select_targets), 2)
                self.critic_loss = 1/2 * tf.reduce_sum(tf.square(target_qs - qs) * self.mask_plh)

                adam = tf.train.AdamOptimizer(self.critic_lr)
                self.global_step = tf.Variable(0, trainable=False, name="global_step", collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
                self.train_critic_op = adam.minimize(self.critic_loss, global_step=self.global_step)

            self.policy_loss_plh = tf.placeholder(tf.float32, shape=[])
            self.policy_loss_sum_t = tf.summary.scalar('policy_loss', self.policy_loss_plh)
            self.critic_loss_plh = tf.placeholder(tf.float32, shape=[])
            self.critic_loss_sum_t = tf.summary.scalar('critic_loss', self.critic_loss_plh)
            # self.loss_plh = tf.placeholder(tf.float32, shape=[])
            # self.loss_sum_t = tf.summary.scalar('loss', self.loss_plh)
            self.all_summary_t = tf.summary.merge_all()

            self.score_plh = tf.placeholder(tf.float32, shape=[])
            self.score_sum_t = tf.summary.scalar('av_score', self.score_plh)

            self.episode_id, self.inc_ep_id_op = capacities.counter("episode_id")

            # Playing part
            self.pscore_plh = tf.placeholder(tf.float32, shape=[])
            self.pscore_sum_t = tf.summary.scalar('play_score', self.pscore_plh)

        return graph

    def collect_samples(self, env, render, nb_sequence=1):
        sequence_history = []
        av_score = []
        for i in range(nb_sequence):
            obs = env.reset()
            act, state = self.act(obs)
            score = 0
            history = np.array([], dtype=self.memoryDt)
            done = False

            while True:
                if render:
                    env.render()

                next_obs, reward, done, info = env.step(act)
                next_act, _ = self.act(next_obs)

                memory = np.array([(
                    np.concatenate((obs, [float(False)]))
                    , act
                    , reward
                    , np.concatenate((next_obs, [float(done)]))
                    , next_act
                )], dtype=self.memoryDt)
                history = np.append(history, memory)

                score += reward
                obs = next_obs
                act = next_act
                if done:
                    break
            sequence_history.append(history)
            av_score.append(score)

            self.sess.run(self.inc_ep_id_op)

        summary, episode_id = self.sess.run([self.score_sum_t, self.episode_id], feed_dict={
            self.score_plh: np.mean(score),
        })
        self.sw.add_summary(summary, episode_id)

        return sequence_history, episode_id

    def train_controller(self, batch):
        expected_rewards = []
        for i, episode_rewards in enumerate(batch['rewards']):
            expected_rewards.append(get_expected_rewards(episode_rewards, self.discount))

        # Fit Critic
        av_critic_loss = []
        for i in range(self.nb_critic_iter):
            _, critic_loss = self.sess.run([self.train_critic_op, self.critic_loss], feed_dict={
                self.inputs: batch['states']
                , self.actions: batch['actions']
                , self.expected_rewards: expected_rewards
                , self.rewards: batch['rewards']
                , self.mask_plh: batch['mask']
                , self.next_states: batch['next_states']
                , self.next_actions: batch['next_actions']
            })
            av_critic_loss.append(critic_loss)
        self.sess.run(self.update_fixed_vars_op)

        _, policy_loss = self.sess.run([self.train_policy_op, self.policy_loss], feed_dict={
            self.inputs: batch['states']
            , self.actions: batch['actions']
            , self.expected_rewards: expected_rewards
            , self.rewards: batch['rewards']
            , self.mask_plh: batch['mask']
            , self.next_states: batch['next_states']
            , self.next_actions: batch['next_actions']
        })
            
        summary, _, episode_id = self.sess.run([self.all_summary_t, self.inc_ep_id_op, self.episode_id], feed_dict={
            self.policy_loss_plh: policy_loss,
            self.critic_loss_plh: np.mean(av_critic_loss),
        })
        self.sw.add_summary(summary, episode_id)

        return

class ActorCriticAgent(ActorQCriticAgent):
    """
    Agent implementing Policy gradient using Monte-Carlo control
    """
    def set_agent_props(self):
        super(ActorCriticAgent, self).set_agent_props()

        self.critic_params = {
            'nb_inputs': self.observation_space.shape[0] + 1
            , 'nb_units': self.config['nb_units']
            , 'nb_outputs': 1
            , 'initial_mean': self.config['initial_mean']
            , 'initial_stddev': self.config['initial_stddev']
        }
        self.critic_lr = self.config['critic_lr']
        self.nb_critic_iter = self.config['nb_critic_iter']

        self.dtKeys = ['states', 'actions', 'rewards', 'next_states']
        self.memoryDt = np.dtype([
            ('states', 'float32', (self.policy_params['nb_inputs'],))
            , ('actions', 'int32', (1,))
            , ('rewards', 'float32', (1,))
            , ('next_states', 'float32', (self.policy_params['nb_inputs'],))
        ])

    def get_best_config(self, env_name=""):
        return {
            'lr': 3e-3
            , 'discount': 0.99
            , 'nb_units': 41
            , 'initial_mean': 0.
            , 'initial_stddev': 0.3
            , 'critic_lr': 1e-2
        }

    @staticmethod
    def get_random_config(fixed_params={}):
        get_lr = lambda: 1e-4 + (1e-1 - 1e-4) * np.random.random(1)[0]
        get_discount = lambda: 0.5 + (1 - 0.5) * np.random.random(1)[0]
        get_nb_units = lambda: np.random.randint(10, 100)
        get_initial_mean = lambda: 0
        get_initial_stddev = lambda: 5e-1 * np.random.random(1)[0]
        get_nb_critic_iter = lambda: np.random.randint(4, 50)

        random_config = {
            'lr': get_lr()
            , 'discount': get_discount()
            , 'nb_units': get_nb_units()
            , 'initial_mean': get_initial_mean()
            , 'initial_stddev': get_initial_stddev()
            , 'critic_lr': get_lr()
            ,'nb_critic_iter': get_nb_critic_iter()
        }
        random_config.update(fixed_params)

        return random_config

    def build_graph(self, graph):
        np.random.seed(self.random_seed)
        with graph.as_default():
            tf.set_random_seed(self.random_seed)

            self.inputs = tf.placeholder(tf.float32, shape=[None, None, self.policy_params['nb_inputs']], name='inputs')
            input_shape = tf.shape(self.inputs)
            dynamic_batch_size, dynamic_num_steps = input_shape[0], input_shape[1]
            inputs_mat = tf.reshape(self.inputs, [-1, self.policy_params['nb_inputs']])

            policy_scope = tf.VariableScope(reuse=False, name='Policy')
            with tf.variable_scope(policy_scope):
                probs, actions = capacities.policy(self.policy_params, inputs_mat)
                self.probs = tf.reshape(probs, [dynamic_batch_size, dynamic_num_steps, self.policy_params['nb_outputs']])
                self.actions = tf.reshape(actions, [dynamic_batch_size, dynamic_num_steps, 1])
            self.action_t = self.actions[0, 0, 0]

            critic_scope = tf.VariableScope(reuse=False, name='QValues')
            with tf.variable_scope(critic_scope):
                critic_values_mat = capacities.value_f(self.critic_params, inputs_mat)
                self.critic_values = tf.reshape(critic_values_mat, [dynamic_batch_size, dynamic_num_steps, self.critic_params['nb_outputs']])

            fixed_critic_scope = tf.VariableScope(reuse=False, name='FixedQValues')
            with tf.variable_scope(fixed_critic_scope):
                self.update_fixed_vars_op = capacities.fix_scope(critic_scope)

            with tf.variable_scope('Training'):
                self.expected_rewards = tf.placeholder(tf.float32, shape=[None, None, 1], name="reward")
                self.mask_plh = tf.placeholder(tf.float32, shape=[None, None, 1], name="mask_plh")

                batch_size, num_steps = tf.shape(self.actions)[0], tf.shape(self.actions)[1]
                line_indices = tf.matmul( # Line indice
                    tf.reshape(tf.range(0, batch_size), [-1, 1])
                    , tf.ones([1, num_steps], dtype=tf.int32)
                )
                column_indices = tf.matmul( # Column indice
                    tf.ones([batch_size, 1], dtype=tf.int32)
                    , tf.reshape(tf.range(0, num_steps), [1, -1])
                )
                depth_indices = tf.cast(tf.squeeze(self.actions, 2), tf.int32)
                stacked_actions = tf.stack(
                    [line_indices, column_indices, depth_indices], 2
                )
                
                log_probs = tf.expand_dims(tf.log(tf.gather_nd(self.probs, stacked_actions)), 2)
                self.policy_loss = tf.reduce_mean( - tf.reduce_sum((log_probs * (self.expected_rewards - tf.stop_gradient(self.critic_values))) * self.mask_plh, 1))

                adam = tf.train.AdamOptimizer(self.lr)
                self.train_policy_op = adam.minimize(self.policy_loss)

                self.rewards = tf.placeholder(tf.float32, shape=[None, None, 1], name="reward")
                self.next_states = tf.placeholder(tf.float32, shape=[None, None, self.critic_params['nb_inputs']], name="next_states")
                with tf.variable_scope(fixed_critic_scope, reuse=True):
                    next_states_mat = tf.reshape(self.next_states, [-1, self.critic_params['nb_inputs']])
                    next_critic_values_mat = capacities.value_f(self.critic_params, next_states_mat)
                    next_critic_values = tf.reshape(next_critic_values_mat, [dynamic_batch_size, dynamic_num_steps, self.critic_params['nb_outputs']])

                target_critics1 = tf.stop_gradient(self.rewards + self.discount * next_critic_values)
                target_critics2 = self.rewards
                stacked_targets = tf.stack([tf.squeeze(target_critics1, 2), tf.squeeze(target_critics2, 2)], 2)

                batch_size, num_steps = tf.shape(self.next_states)[0], tf.shape(self.next_states)[1]
                line_indices = tf.matmul( # Line indice
                    tf.reshape(tf.range(0, batch_size), [-1, 1])
                    , tf.ones([1, num_steps], dtype=tf.int32)
                )
                column_indices = tf.matmul( # Column indice
                    tf.ones([batch_size, 1], dtype=tf.int32)
                    , tf.reshape(tf.range(0, num_steps), [1, -1])
                )
                depth_indices = tf.cast(self.next_states[:, :, -1], tf.int32)
                select_targets = tf.stack(
                    [line_indices, column_indices, depth_indices], 2
                )

                target_critics = tf.expand_dims(tf.gather_nd(stacked_targets, select_targets), 2)
                self.critic_loss = 1/2 * tf.reduce_sum(tf.square(target_critics - self.critic_values) * self.mask_plh)

                adam = tf.train.AdamOptimizer(self.critic_lr)
                self.global_step = tf.Variable(0, trainable=False, name="global_step", collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
                self.train_critic_op = adam.minimize(self.critic_loss, global_step=self.global_step)

            self.policy_loss_plh = tf.placeholder(tf.float32, shape=[])
            self.policy_loss_sum_t = tf.summary.scalar('policy_loss', self.policy_loss_plh)
            self.critic_loss_plh = tf.placeholder(tf.float32, shape=[])
            self.critic_loss_sum_t = tf.summary.scalar('critic_loss', self.critic_loss_plh)
            # self.loss_plh = tf.placeholder(tf.float32, shape=[])
            # self.loss_sum_t = tf.summary.scalar('loss', self.loss_plh)
            self.all_summary_t = tf.summary.merge_all()

            self.score_plh = tf.placeholder(tf.float32, shape=[])
            self.score_sum_t = tf.summary.scalar('av_score', self.score_plh)

            self.episode_id, self.inc_ep_id_op = capacities.counter("episode_id")

            # Playing part
            self.pscore_plh = tf.placeholder(tf.float32, shape=[])
            self.pscore_sum_t = tf.summary.scalar('play_score', self.pscore_plh)

        return graph

    def collect_samples(self, env, render, nb_sequence=1):
        sequence_history = []
        av_score = []
        for i in range(nb_sequence):
            obs = env.reset()
            score = 0
            history = np.array([], dtype=self.memoryDt)
            done = False

            while True:
                if render:
                    env.render()

                act, state = self.act(obs)
                next_obs, reward, done, info = env.step(act)

                memory = np.array([(
                    state
                    , act
                    , reward
                    , np.concatenate((next_obs, [float(done)]))
                )], dtype=self.memoryDt)
                history = np.append(history, memory)

                score += reward
                obs = next_obs
                if done:
                    break
            sequence_history.append(history)
            av_score.append(score)

            self.sess.run(self.inc_ep_id_op)

        summary, episode_id = self.sess.run([self.score_sum_t, self.episode_id], feed_dict={
            self.score_plh: np.mean(score),
        })
        self.sw.add_summary(summary, episode_id)

        return sequence_history, episode_id

    def train_controller(self, batch):
        expected_rewards = []
        for i, episode_rewards in enumerate(batch['rewards']):
            expected_rewards.append(get_expected_rewards(episode_rewards, self.discount))

        # Fit Critic
        av_critic_loss = []
        for i in range(self.nb_critic_iter):
            _, critic_loss = self.sess.run([self.train_critic_op, self.critic_loss], feed_dict={
                self.inputs: batch['states']
                , self.actions: batch['actions']
                , self.expected_rewards: expected_rewards
                , self.rewards: batch['rewards']
                , self.mask_plh: batch['mask']
                , self.next_states: batch['next_states']
            })
            av_critic_loss.append(critic_loss)
        self.sess.run(self.update_fixed_vars_op)

        _, policy_loss = self.sess.run([self.train_policy_op, self.policy_loss], feed_dict={
            self.inputs: batch['states']
            , self.actions: batch['actions']
            , self.expected_rewards: expected_rewards
            , self.rewards: batch['rewards']
            , self.mask_plh: batch['mask']
            , self.next_states: batch['next_states']
        })
            
        summary, _, episode_id = self.sess.run([self.all_summary_t, self.inc_ep_id_op, self.episode_id], feed_dict={
            self.policy_loss_plh: policy_loss,
            self.critic_loss_plh: np.mean(av_critic_loss),
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

class TDACAgent(PolicyAgent):
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
