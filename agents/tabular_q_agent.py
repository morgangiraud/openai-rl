import numpy as np
import tensorflow as tf

from agents import BasicAgent, phis

class TabularQAgent(BasicAgent):
    """
    Agent implementing tabular Q-learning.
    """
    def __init__(self, config, env):
        config.update(phis.getPhiConfig(config['env_name'], config['debug']))
        super(TabularQAgent, self).__init__(config, env)

        self.N0 = config['N0']
        self.min_eps = config['min_eps']

        self.graph = self.buildGraph(tf.Graph())

        self.sess = tf.Session(graph=self.graph)
        self.sw = tf.summary.FileWriter(self.result_dir, self.sess.graph)
        self.init()

    def buildGraph(self, graph):
        with graph.as_default():
            self.N0_t = tf.constant(self.N0, tf.float32, name='N_0')
            self.N_s = tf.Variable(tf.ones(shape=[self.nb_state]), name='N_s', trainable=False)
            self.min_eps_t = tf.constant(self.min_eps, tf.float32, name='min_eps')
            self.discount = 1

            with tf.variable_scope('Qs'):
                self.Qs = tf.Variable(
                    initial_value=np.zeros([self.nb_state, self.action_space.n])
                    , name = "Qs"
                    , dtype=tf.float32
                )
                self.inputs = tf.placeholder(tf.int32, shape=[], name="inputs")
                self.q_preds = self.Qs[self.inputs]

            with tf.variable_scope('Policy'):
                eps = tf.maximum(self.N0_t / (self.N0_t + self.N_s[self.inputs]), self.min_eps_t)
                update_N_s = tf.scatter_add(self.N_s, self.inputs, 1)
                cond = tf.greater(tf.random_uniform([], 0, 1), eps)
                pred_action = tf.cast(tf.argmax(self.q_preds, 0), tf.int32)
                random_action = tf.random_uniform([], 0, self.env.action_space.n, dtype=tf.int32)
                with tf.control_dependencies([update_N_s]):
                    self.action_t = tf.cond(cond, lambda: pred_action, lambda: random_action)
                self.q_t = self.q_preds[self.action_t]

            with tf.variable_scope('Training'):
                self.reward = tf.placeholder(tf.float32, shape=[], name="reward")
                self.next_state = tf.placeholder(tf.int32, shape=[], name="nextState")
                next_max_action_t = tf.cast(tf.argmax(self.Qs[self.next_state], 0), tf.int32)
                target_q = tf.stop_gradient(self.reward + self.discount * self.Qs[self.next_state, next_max_action_t])
                self.loss = 1/2 * tf.square(target_q - self.q_t)

                adam = tf.train.AdamOptimizer(self.lr)
                self.global_step = tf.Variable(0, trainable=False)
                self.train_op = adam.minimize(self.loss, global_step=self.global_step)

            self.score_plh = tf.placeholder(tf.float32, shape=[])
            self.score_sum_t = tf.summary.scalar('score', self.score_plh)
            self.loss_plh = tf.placeholder(tf.float32, shape=[])
            self.loss_sum_t = tf.summary.scalar('loss', self.loss_plh)
            self.qs_plh = tf.placeholder(tf.float32, shape=[self.nb_state, self.action_space.n])
            self.qs_sum_t = tf.summary.histogram('Qarray', self.qs_plh)
            self.all_summary_t = tf.summary.merge_all()

            self.episode_id = tf.Variable(0, trainable=False)
            self.inc_ep_id_op = tf.assign(self.episode_id, self.episode_id + 1)

            self.saver = tf.train.Saver()

            self.init_op = tf.global_variables_initializer()

        return graph

    def act(self, obs):
        state_id = self.phi(obs)
        act = self.sess.run(self.action_t, feed_dict={
            self.inputs: state_id
        })

        return act, state_id

    def learnFromEpisode(self, env, render=False):
        obs = env.reset()
        score = 0
        av_loss = []
        done = False

        while True:
            if render:
                env.render()

            act, state_id = self.act(obs)
            next_obs, reward, done, info = env.step(act)

            next_state_id = self.phi(next_obs, done)
            loss, qs, _ = self.sess.run([self.loss, self.Qs, self.train_op], feed_dict={
                self.inputs: state_id,
                self.action_t: act,
                self.reward: reward,
                self.next_state: next_state_id,
            })

            av_loss.append(loss)
            score += reward
            obs = next_obs
            if done:
                break

        summary, _, episode_id = self.sess.run([self.all_summary_t, self.inc_ep_id_op, self.episode_id], feed_dict={
            self.score_plh: score,
            self.loss_plh: np.mean(av_loss),
            self.qs_plh: qs
        })
        self.sw.add_summary(summary, episode_id)

        return

    def play(self, env, render=True):
        obs = env.reset()
        score = 0
        done = False
        while True:
            if render:
                env.render()

            act, state_id = self.act(obs)
            next_obs, reward, done, info = env.step(act)
            score += reward

            obs = next_obs
            if done:
                break

        return

class BackwardTabularQAgent(TabularQAgent):
    """
    Agent implementing Backward TD(lambda) tabular Q-learning.
    """

    def buildGraph(self, graph):
        with graph.as_default():
            self.N0_t = tf.constant(self.N0, tf.float32, name='N_0')
            self.N_s = tf.Variable(tf.ones(shape=[self.nb_state]), name='N_s', trainable=False)
            self.min_eps_t = tf.constant(self.min_eps, tf.float32, name='min_eps')
            self.discount = 1

            with tf.variable_scope('Qs'):
                self.Qs = tf.Variable(
                    initial_value=np.zeros([self.nb_state, self.action_space.n])
                    , name = "Qs"
                    , dtype=tf.float32
                )
                self.inputs = tf.placeholder(tf.int32, shape=[], name="inputs")
                self.q_preds = self.Qs[self.inputs]

            with tf.variable_scope('Policy'):
                eps = tf.maximum(self.N0_t / (self.N0_t + self.N_s[self.inputs]), self.min_eps_t)
                update_N_s = tf.scatter_add(self.N_s, self.inputs, 1)
                cond = tf.greater(tf.random_uniform([], 0, 1), eps)
                pred_action = tf.cast(tf.argmax(self.q_preds, 0), tf.int32)
                random_action = tf.random_uniform([], 0, self.env.action_space.n, dtype=tf.int32)
                with tf.control_dependencies([update_N_s]):
                    self.action_t = tf.cond(cond, lambda: pred_action, lambda: random_action)
                self.q_t = self.q_preds[self.action_t]

            with tf.variable_scope('Training'):
                self.reward = tf.placeholder(tf.float32, shape=[], name="reward")
                self.next_state = tf.placeholder(tf.int32, shape=[], name="nextState")
                next_max_action_t = tf.cast(tf.argmax(self.Qs[self.next_state], 0), tf.int32)
                target_q = tf.stop_gradient(self.reward + self.discount * self.Qs[self.next_state, next_max_action_t])
                self.Et = tf.placeholder(tf.float32, shape=[self.nb_state, self.action_space.n], name="Et")
                self.loss = - tf.stop_gradient(target_q - self.q_t) * self.Et * self.Qs

                adam = tf.train.AdamOptimizer(self.lr)
                self.global_step = tf.Variable(0, trainable=False)
                self.train_op = adam.minimize(self.loss, global_step=self.global_step)

            self.score_plh = tf.placeholder(tf.float32, shape=[])
            self.score_sum_t = tf.summary.scalar('score', self.score_plh)
            self.loss_plh = tf.placeholder(tf.float32, shape=[])
            self.loss_sum_t = tf.summary.scalar('loss', self.loss_plh)
            self.qs_plh = tf.placeholder(tf.float32, shape=[self.nb_state, self.action_space.n])
            self.qs_sum_t = tf.summary.histogram('Qarray', self.qs_plh)
            self.all_summary_t = tf.summary.merge_all()

            self.episode_id = tf.Variable(0, trainable=False)
            self.inc_ep_id_op = tf.assign(self.episode_id, self.episode_id + 1)

            self.saver = tf.train.Saver()

            self.init_op = tf.global_variables_initializer()

        return graph

    def learnFromEpisode(self, env, render=False):
        obs = env.reset()
        score = 0
        av_loss = []
        done = False
        Et = np.zeros([self.nb_state, self.action_space.n])
        visited_sa = []
        lambda_val = 0.9
        while True:
            if render:
                env.render()

            act, state_id = self.act(obs)
            for s, a in visited_sa:
                Et[s, a] *= self.discount * lambda_val
            Et[state_id, act] = 1
            visited_sa.append((state_id, act))

            next_obs, reward, done, info = env.step(act)

            next_state_id = self.phi(next_obs, done)
            loss, qs, _ = self.sess.run([self.loss, self.Qs, self.train_op], feed_dict={
                self.inputs: state_id,
                self.action_t: act,
                self.reward: reward,
                self.next_state: next_state_id,
                self.Et: Et,
            })

            av_loss.append(loss)
            score += reward
            obs = next_obs
            if done:
                break

        summary, _, episode_id = self.sess.run([self.all_summary_t, self.inc_ep_id_op, self.episode_id], feed_dict={
            self.score_plh: score,
            self.loss_plh: np.mean(av_loss),
            self.qs_plh: qs
        })
        self.sw.add_summary(summary, episode_id)

        return

class TabularQplusAgent(TabularQAgent):
    """
    Agent implementing tabular Q-learning with experience replay.
    """
    def __init__(self, config, env):
        self.er_lr = config['er_lr']
        self.er_every = config['er_every']
        self.er_batch_size = config['er_batch_size']
        self.er_epoch_size = config['er_epoch_size']
        self.er_rm_size = config['er_rm_size']

        super(TabularQplusAgent, self).__init__(config, env)

    def buildGraph(self, graph):
        with graph.as_default():
            self.N0_t = tf.constant(self.N0, tf.float32, name='N_0')
            self.N_s = tf.Variable(tf.ones(shape=[self.nb_state]), name='N_s', trainable=False)
            self.min_eps_t = tf.constant(self.min_eps, tf.float32, name='min_eps')
            self.discount = 1
            self.replayMemoryDt = np.dtype([('states', 'int32'), ('actions', 'int32'), ('rewards', 'float32'), ('next_states', 'int32')])
            self.replayMemory = np.array([], dtype=self.replayMemoryDt)

            with tf.variable_scope('Qs'):
                self.Qs = tf.Variable(
                    initial_value=np.zeros([self.nb_state, self.action_space.n])
                    , name = "Qs"
                    , dtype=tf.float32
                )
                self.inputs = tf.placeholder(tf.int32, shape=[], name="inputs")
                self.q_preds = self.Qs[self.inputs]

            with tf.variable_scope('Policy'):
                eps = tf.maximum(self.N0_t / (self.N0_t + self.N_s[self.inputs]), self.min_eps_t)
                update_N_s = tf.scatter_add(self.N_s, self.inputs, 1)
                cond = tf.greater(tf.random_uniform([], 0, 1), eps)
                pred_action = tf.cast(tf.argmax(self.q_preds, 0), tf.int32)
                random_action = tf.random_uniform([], 0, self.env.action_space.n, dtype=tf.int32)
                with tf.control_dependencies([update_N_s]):
                    self.action_t = tf.cond(cond, lambda: pred_action, lambda: random_action)
                self.q_t = self.q_preds[self.action_t]

            with tf.variable_scope('Training'):
                self.reward = tf.placeholder(tf.float32, shape=[], name="reward")
                self.next_state = tf.placeholder(tf.int32, shape=[], name="nextState")
                next_max_action_t = tf.cast(tf.argmax(self.Qs[self.next_state], 0), tf.int32)
                target_q = tf.stop_gradient(self.reward + self.discount * self.Qs[self.next_state, next_max_action_t])
                self.loss = 1/2 * tf.square(target_q - self.q_t)

                adam = tf.train.AdamOptimizer(self.lr)
                self.global_step = tf.Variable(0, trainable=False)
                self.train_op = adam.minimize(self.loss, global_step=self.global_step)
            # Experienced replay part
            with tf.variable_scope('ExperienceReplay'):
                self.er_inputs = tf.placeholder(tf.int32, shape=[None], name="ERInputs")
                self.er_actions = tf.placeholder(tf.int32, shape=[None], name="ERInputs")
                self.er_rewards = tf.placeholder(tf.float32, shape=[None], name="ERReward")
                self.er_next_states = tf.placeholder(tf.int32, shape=[None], name="ERNextState")
                er_next_max_action_t = tf.cast(tf.argmax(tf.gather(self.Qs, self.er_next_states), 1), tf.int32)
                er_next_sa_pairs = tf.stack([self.er_next_states, er_next_max_action_t], 1)
                er_target_qs = tf.stop_gradient(self.er_rewards + self.discount * tf.gather_nd(self.Qs, er_next_sa_pairs))
                er_sa_pairs = tf.stack([self.er_inputs, self.er_actions], 1)
                er_qs = tf.gather_nd(self.Qs, er_sa_pairs)
                er_loss = 1/2 * tf.reduce_sum(tf.square(er_target_qs - er_qs))
                er_adam = tf.train.AdamOptimizer(self.er_lr)
                self.er_global_step = tf.Variable(0, trainable=False)
                self.er_train_op = er_adam.minimize(er_loss, global_step=self.er_global_step)

            self.score_plh = tf.placeholder(tf.float32, shape=[])
            self.score_sum_t = tf.summary.scalar('score', self.score_plh)
            self.loss_plh = tf.placeholder(tf.float32, shape=[])
            self.loss_sum_t = tf.summary.scalar('loss', self.loss_plh)
            self.qs_plh = tf.placeholder(tf.float32, shape=[self.nb_state, self.action_space.n])
            self.qs_sum_t = tf.summary.histogram('Qarray', self.qs_plh)
            self.all_summary_t = tf.summary.merge_all()

            self.episode_id = tf.Variable(0, trainable=False)
            self.inc_ep_id_op = tf.assign(self.episode_id, self.episode_id + 1)

            self.saver = tf.train.Saver()

            self.init_op = tf.global_variables_initializer()

        return graph

    def learnFromEpisode(self, env, render=False):
        obs = env.reset()
        score = 0
        av_loss = []
        done = False

        while True:
            if render:
                env.render()

            act, state_id = self.act(obs)
            next_obs, reward, done, info = env.step(act)
            
            next_state_id = self.phi(next_obs, done)
            loss, qs, _ = self.sess.run([self.loss, self.Qs, self.train_op], feed_dict={
                self.inputs: state_id,
                self.action_t: act,
                self.reward: reward,
                self.next_state: next_state_id,
            })
            memory = np.array([(state_id, act, reward, next_state_id)], dtype=self.replayMemoryDt)
            if self.replayMemory.shape[0] >= self.er_rm_size:
                self.replayMemory = np.delete(self.replayMemory, 0)
            self.replayMemory = np.append(self.replayMemory, memory)

            av_loss.append(loss)
            score += reward
            obs = next_obs
            if done:
                break

        summary, _, episode_id = self.sess.run([self.all_summary_t, self.inc_ep_id_op, self.episode_id], feed_dict={
            self.score_plh: score,
            self.loss_plh: np.mean(av_loss),
            self.qs_plh: qs
        })
        self.sw.add_summary(summary, episode_id)

        # Experiencing replays
        if episode_id % self.er_every == 0:
            for i in range(self.er_epoch_size):
                memories = np.random.choice(self.replayMemory, self.er_batch_size)
                _ = self.sess.run(self.er_train_op, feed_dict={
                    self.er_inputs: memories['states'],
                    self.er_actions: memories['actions'],
                    self.er_rewards: memories['rewards'],
                    self.er_next_states: memories['next_states'],
                })

        return

class TabularFixedQplusAgent(TabularQAgent):
    """
    Agent implementing tabular Q-learning with fixed Qs and experience replay.
    """
    def __init__(self, config, env):
        self.er_lr = config['er_lr']
        self.er_every = config['er_every']
        self.er_batch_size = config['er_batch_size']
        self.er_epoch_size = config['er_epoch_size']
        self.er_rm_size = config['er_rm_size']

        super(TabularFixedQplusAgent, self).__init__(config, env)

    def buildGraph(self, graph):
        with graph.as_default():
            self.N0_t = tf.constant(self.N0, tf.float32, name='N_0')
            self.N_s = tf.Variable(tf.ones(shape=[self.nb_state]), name='N_s', trainable=False)
            self.min_eps_t = tf.constant(self.min_eps, tf.float32, name='min_eps')
            self.discount = 1
            self.replayMemoryDt = np.dtype([('states', 'int32'), ('actions', 'int32'), ('rewards', 'float32'), ('next_states', 'int32')])
            self.replayMemory = np.array([], dtype=self.replayMemoryDt)

            with tf.variable_scope('Qs'):
                self.Qs = tf.Variable(
                    initial_value=np.zeros([self.nb_state, self.action_space.n])
                    , name = "Qs"
                    , dtype=tf.float32
                )
                self.inputs = tf.placeholder(tf.int32, shape=[], name="inputs")
                self.q_preds = self.Qs[self.inputs]

            with tf.variable_scope('FixedQs'):
                self.fixed_Qs = tf.Variable(
                    initial_value=np.zeros([self.nb_state, self.action_space.n])
                    , name = "fixedQs"
                    , trainable=False
                    , dtype=tf.float32
                )
                self.update_fixed_Q_op = tf.assign(self.fixed_Qs, self.Qs)

            with tf.variable_scope('Policy'):
                eps = tf.maximum(self.N0_t / (self.N0_t + self.N_s[self.inputs]), self.min_eps_t)
                update_N_s = tf.scatter_add(self.N_s, self.inputs, 1)
                cond = tf.greater(tf.random_uniform([], 0, 1), eps)
                pred_action = tf.cast(tf.argmax(self.q_preds, 0), tf.int32)
                random_action = tf.random_uniform([], 0, self.env.action_space.n, dtype=tf.int32)
                with tf.control_dependencies([update_N_s]):
                    self.action_t = tf.cond(cond, lambda: pred_action, lambda: random_action)
                self.q_t = self.q_preds[self.action_t]

            with tf.variable_scope('Training'):
                self.reward = tf.placeholder(tf.float32, shape=[], name="reward")
                self.next_state = tf.placeholder(tf.int32, shape=[], name="nextState")
                next_max_action_t = tf.cast(tf.argmax(self.Qs[self.next_state], 0), tf.int32)
                target_q = tf.stop_gradient(self.reward + self.discount * self.Qs[self.next_state, next_max_action_t])
                self.loss = 1/2 * tf.square(target_q - self.q_t)

                adam = tf.train.AdamOptimizer(self.lr)
                self.global_step = tf.Variable(0, trainable=False)
                self.train_op = adam.minimize(self.loss, global_step=self.global_step)

            with tf.variable_scope('ExperienceReplay'):
                self.er_inputs = tf.placeholder(tf.int32, shape=[None], name="ERInputs")
                self.er_actions = tf.placeholder(tf.int32, shape=[None], name="ERInputs")
                self.er_rewards = tf.placeholder(tf.float32, shape=[None], name="ERReward")
                self.er_next_states = tf.placeholder(tf.int32, shape=[None], name="ERNextState")
                er_next_max_action_t = tf.cast(tf.argmax(tf.gather(self.fixed_Qs, self.er_next_states), 1), tf.int32)
                er_next_sa_pairs = tf.stack([self.er_next_states, er_next_max_action_t], 1)
                er_target_qs = tf.stop_gradient(self.er_rewards + self.discount * tf.gather_nd(self.fixed_Qs, er_next_sa_pairs))
                er_sa_pairs = tf.stack([self.er_inputs, self.er_actions], 1)
                er_qs = tf.gather_nd(self.Qs, er_sa_pairs)
                er_loss = 1/2 * tf.reduce_sum(tf.square(er_target_qs - er_qs))
                er_adam = tf.train.AdamOptimizer(self.er_lr)
                self.er_global_step = tf.Variable(0, trainable=False)
                self.er_train_op = er_adam.minimize(er_loss, global_step=self.er_global_step)

            self.score_plh = tf.placeholder(tf.float32, shape=[])
            self.score_sum_t = tf.summary.scalar('score', self.score_plh)
            self.loss_plh = tf.placeholder(tf.float32, shape=[])
            self.loss_sum_t = tf.summary.scalar('loss', self.loss_plh)
            self.qs_plh = tf.placeholder(tf.float32, shape=[self.nb_state, self.action_space.n])
            self.qs_sum_t = tf.summary.histogram('Qarray', self.qs_plh)
            self.all_summary_t = tf.summary.merge_all()

            self.episode_id = tf.Variable(0, trainable=False)
            self.inc_ep_id_op = tf.assign(self.episode_id, self.episode_id + 1)

            self.saver = tf.train.Saver()

            self.init_op = tf.global_variables_initializer()

        return graph

    def learnFromEpisode(self, env, render=False):
        obs = env.reset()
        score = 0
        av_loss = []
        done = False

        while True:
            if render:
                env.render()

            act, state_id = self.act(obs)
            next_obs, reward, done, info = env.step(act)
            
            next_state_id = self.phi(next_obs, done)
            loss, qs, _ = self.sess.run([self.loss, self.Qs, self.train_op], feed_dict={
                self.inputs: state_id,
                self.action_t: act,
                self.reward: reward,
                self.next_state: next_state_id,
            })
            memory = np.array([(state_id, act, reward, next_state_id)], dtype=self.replayMemoryDt)
            if self.replayMemory.shape[0] >= self.er_rm_size:
                self.replayMemory = np.delete(self.replayMemory, 0)
            self.replayMemory = np.append(self.replayMemory, memory)

            av_loss.append(loss)
            score += reward
            obs = next_obs
            if done:
                break

        summary, _, episode_id = self.sess.run([self.all_summary_t, self.inc_ep_id_op, self.episode_id], feed_dict={
            self.score_plh: score,
            self.loss_plh: np.mean(av_loss),
            self.qs_plh: qs
        })
        self.sw.add_summary(summary, episode_id)

        # Experiencing replays
        if episode_id % self.er_every == 0:
            self.sess.run(self.update_fixed_Q_op)
            for i in range(self.er_epoch_size):
                memories = np.random.choice(self.replayMemory, self.er_batch_size)
                _ = self.sess.run(self.er_train_op, feed_dict={
                    self.er_inputs: memories['states'],
                    self.er_actions: memories['actions'],
                    self.er_rewards: memories['rewards'],
                    self.er_next_states: memories['next_states'],
                })

        return

class BackwardTabularFixedQplusAgent(TabularQAgent):
    """
    Agent implementing backward TD(lambda) tabular Q-learning with fixed Qs and experience replay.
    """
    def __init__(self, config, env):
        self.er_lr = config['er_lr']
        self.er_every = config['er_every']
        self.er_batch_size = config['er_batch_size']
        self.er_epoch_size = config['er_epoch_size']
        self.er_rm_size = config['er_rm_size']

        super(BackwardTabularFixedQplusAgent, self).__init__(config, env)

    def buildGraph(self, graph):
        with graph.as_default():
            self.N0_t = tf.constant(self.N0, tf.float32, name='N_0')
            self.N_s = tf.Variable(tf.ones(shape=[self.nb_state]), name='N_s', trainable=False)
            self.min_eps_t = tf.constant(self.min_eps, tf.float32, name='min_eps')
            self.discount = 1
            self.replayMemoryDt = np.dtype([('states', 'int32'), ('actions', 'int32'), ('rewards', 'float32'), ('next_states', 'int32')])
            self.replayMemory = np.array([], dtype=self.replayMemoryDt)

            with tf.variable_scope('Qs'):
                self.Qs = tf.Variable(
                    initial_value=np.zeros([self.nb_state, self.action_space.n])
                    , name = "Qs"
                    , dtype=tf.float32
                )
                self.inputs = tf.placeholder(tf.int32, shape=[], name="inputs")
                self.q_preds = self.Qs[self.inputs]

            with tf.variable_scope('FixedQs'):
                self.fixed_Qs = tf.Variable(
                    initial_value=np.zeros([self.nb_state, self.action_space.n])
                    , name = "fixedQs"
                    , trainable=False
                    , dtype=tf.float32
                )
                self.update_fixed_Q_op = tf.assign(self.fixed_Qs, self.Qs)

            with tf.variable_scope('Policy'):
                eps = tf.maximum(self.N0_t / (self.N0_t + self.N_s[self.inputs]), self.min_eps_t)
                update_N_s = tf.scatter_add(self.N_s, self.inputs, 1)
                cond = tf.greater(tf.random_uniform([], 0, 1), eps)
                pred_action = tf.cast(tf.argmax(self.q_preds, 0), tf.int32)
                random_action = tf.random_uniform([], 0, self.env.action_space.n, dtype=tf.int32)
                with tf.control_dependencies([update_N_s]):
                    self.action_t = tf.cond(cond, lambda: pred_action, lambda: random_action)
                self.q_t = self.q_preds[self.action_t]

            with tf.variable_scope('Training'):
                self.reward = tf.placeholder(tf.float32, shape=[], name="reward")
                self.next_state = tf.placeholder(tf.int32, shape=[], name="nextState")
                next_max_action_t = tf.cast(tf.argmax(self.Qs[self.next_state], 0), tf.int32)
                target_q = tf.stop_gradient(self.reward + self.discount * self.Qs[self.next_state, next_max_action_t])
                self.Et = tf.placeholder(tf.float32, shape=[self.nb_state, self.action_space.n], name="Et")
                self.loss = - tf.stop_gradient(target_q - self.q_t) * self.Et * self.Qs
                
                adam = tf.train.AdamOptimizer(self.lr)
                self.global_step = tf.Variable(0, trainable=False)
                self.train_op = adam.minimize(self.loss, global_step=self.global_step)

            with tf.variable_scope('ExperienceReplay'):
                self.er_inputs = tf.placeholder(tf.int32, shape=[None], name="ERInputs")
                self.er_actions = tf.placeholder(tf.int32, shape=[None], name="ERInputs")
                self.er_rewards = tf.placeholder(tf.float32, shape=[None], name="ERReward")
                self.er_next_states = tf.placeholder(tf.int32, shape=[None], name="ERNextState")
                er_next_max_action_t = tf.cast(tf.argmax(tf.gather(self.fixed_Qs, self.er_next_states), 1), tf.int32)
                er_next_sa_pairs = tf.stack([self.er_next_states, er_next_max_action_t], 1)
                er_target_qs = tf.stop_gradient(self.er_rewards + self.discount * tf.gather_nd(self.fixed_Qs, er_next_sa_pairs))
                er_sa_pairs = tf.stack([self.er_inputs, self.er_actions], 1)
                er_qs = tf.gather_nd(self.Qs, er_sa_pairs)
                er_loss = 1/2 * tf.reduce_sum(tf.square(er_target_qs - er_qs))
                er_adam = tf.train.AdamOptimizer(self.er_lr)
                self.er_global_step = tf.Variable(0, trainable=False)
                self.er_train_op = er_adam.minimize(er_loss, global_step=self.er_global_step)


            self.score_plh = tf.placeholder(tf.float32, shape=[])
            self.score_sum_t = tf.summary.scalar('score', self.score_plh)
            self.loss_plh = tf.placeholder(tf.float32, shape=[])
            self.loss_sum_t = tf.summary.scalar('loss', self.loss_plh)
            self.qs_plh = tf.placeholder(tf.float32, shape=[self.nb_state, self.action_space.n])
            self.qs_sum_t = tf.summary.histogram('Qarray', self.qs_plh)
            self.all_summary_t = tf.summary.merge_all()

            self.episode_id = tf.Variable(0, trainable=False)
            self.inc_ep_id_op = tf.assign(self.episode_id, self.episode_id + 1)

            self.saver = tf.train.Saver()

            self.init_op = tf.global_variables_initializer()

        return graph

    def learnFromEpisode(self, env, render=False):
        obs = env.reset()
        score = 0
        av_loss = []
        done = False

        Et = np.zeros([self.nb_state, self.action_space.n])
        visited_sa = []
        lambda_val = 0.9

        while True:
            if render:
                env.render()

            act, state_id = self.act(obs)
            next_obs, reward, done, info = env.step(act)

            for s, a in visited_sa:
                Et[s, a] *= self.discount * lambda_val
            Et[state_id, act] = 1
            visited_sa.append((state_id, act))
            

            next_state_id = self.phi(next_obs, done)
            loss, qs, _ = self.sess.run([self.loss, self.Qs, self.train_op], feed_dict={
                self.inputs: state_id,
                self.action_t: act,
                self.reward: reward,
                self.next_state: next_state_id,
                self.Et: Et,
            })
            memory = np.array([(state_id, act, reward, next_state_id)], dtype=self.replayMemoryDt)
            if self.replayMemory.shape[0] >= self.er_rm_size:
                self.replayMemory = np.delete(self.replayMemory, 0)
            self.replayMemory = np.append(self.replayMemory, memory)

            av_loss.append(loss)
            score += reward
            obs = next_obs
            if done:
                break

        summary, _, episode_id = self.sess.run([self.all_summary_t, self.inc_ep_id_op, self.episode_id], feed_dict={
            self.score_plh: score,
            self.loss_plh: np.mean(av_loss),
            self.qs_plh: qs
        })
        self.sw.add_summary(summary, episode_id)

        # Experiencing replays
        if episode_id % self.er_every == 0:
            self.sess.run(self.update_fixed_Q_op)
            for i in range(self.er_epoch_size):
                memories = np.random.choice(self.replayMemory, self.er_batch_size)
                _ = self.sess.run(self.er_train_op, feed_dict={
                    self.er_inputs: memories['states'],
                    self.er_actions: memories['actions'],
                    self.er_rewards: memories['rewards'],
                    self.er_next_states: memories['next_states'],
                })

        return



