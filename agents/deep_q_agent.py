import numpy as np
import tensorflow as tf

from agents import BasicAgent, capacities

class DeepFixedQPlusAgent(BasicAgent):
    """
    Agent implementing 2-layer NN Q-learning, using experience replay, fixed Q Network and TD(0).
    """

    def __init__(self, config, env):
        super(DeepFixedQPlusAgent, self).__init__(config, env)

        self.nb_units = config['nb_units']

        self.N0 = config['N0']
        self.min_eps = config['min_eps']

        self.er_every = config['er_every']
        self.er_batch_size = config['er_batch_size']
        self.er_epoch_size = config['er_epoch_size']
        self.er_rm_size = config['er_rm_size']

        self.graph = self.buildGraph(tf.Graph())

        gpu_options = tf.GPUOptions(allow_growth=True)
        sessConfig = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=sessConfig, graph=self.graph)
        self.sw = tf.summary.FileWriter(self.result_dir, self.sess.graph)
        self.init()

    def net(self, inputs, net_scope, scope_reuse=False):
        with tf.variable_scope(net_scope) as scope:
            if scope_reuse:
                scope.reuse_variables()

            W1 = tf.get_variable('W1'
                , shape=[self.observation_space.shape[0] + 1, self.nb_units]
                , initializer=tf.random_normal_initializer(stddev=1e-2)
                , collections=[tf.GraphKeys.GLOBAL_VARIABLES, "net"]
            )
            b1 = tf.get_variable('b1'
                , shape=[self.nb_units]
                , initializer=tf.zeros_initializer()
                , collections=[tf.GraphKeys.GLOBAL_VARIABLES, "net"]
            )
            a1 = tf.nn.relu(tf.matmul(inputs, W1) + b1)

            W2 = tf.get_variable('W2'
                , shape=[self.nb_units, self.nb_units]
                , initializer=tf.random_normal_initializer(stddev=1e-2)
                , collections=[tf.GraphKeys.GLOBAL_VARIABLES, "net"]
            )
            b2 = tf.get_variable('b2'
                , shape=[self.nb_units]
                , initializer=tf.zeros_initializer()
                , collections=[tf.GraphKeys.GLOBAL_VARIABLES, "net"]
            )
            a2 = tf.nn.relu(tf.matmul(a1, W2) + b2)

            W3 = tf.get_variable('W3'
                , shape=[self.nb_units, self.action_space.n]
                , initializer=tf.random_normal_initializer(stddev=1e-2)
                , collections=[tf.GraphKeys.GLOBAL_VARIABLES, "net"]
            )
            b3 = tf.get_variable('b3'
                , shape=[self.action_space.n]
                , initializer=tf.zeros_initializer()
                , collections=[tf.GraphKeys.GLOBAL_VARIABLES, "net"]
            )
            out = tf.matmul(a2, W3) + b3

        return out

    def buildGraph(self, graph):
        with graph.as_default():
            self.N0_t = tf.constant(self.N0, tf.float32, name='N_0')
            self.N = tf.Variable(0., dtype=tf.float32, name='N', trainable=False)
            self.min_eps_t = tf.constant(self.min_eps, tf.float32, name='min_eps')
            self.discount = 1
            self.replayMemoryDt = np.dtype([('states', 'float32', (5,)), ('actions', 'int32'), ('rewards', 'float32'), ('next_states', 'float32', (5,))])
            self.replayMemory = np.array([], dtype=self.replayMemoryDt)
            
            # Model
            net_scope = tf.VariableScope(reuse=False, name='Net')
            self.inputs = tf.placeholder(tf.float32, shape=[None, self.observation_space.shape[0] + 1], name='inputs')
            self.q_preds = tf.squeeze(self.net(self.inputs, net_scope, False))

            self.action_t = capacities.epsGreedy(
                self.inputs, self.q_preds, self.env.action_space.n, self.N0, self.min_eps
            )
            self.q_t = self.q_preds[self.action_t]

            fixed_net_scope = tf.VariableScope(reuse=False, name='FixedNet')
            with tf.variable_scope(fixed_net_scope):
                self.update_fixed_vars_op = []
                for var in tf.get_collection("net"):
                    fixed = tf.get_variable(var.name.split("/")[-1].split(":")[0], shape=var.get_shape())    
                    assign_op = tf.assign(fixed, var)
                    self.update_fixed_vars_op.append(assign_op)

            with tf.variable_scope('Training'):
                self.reward = tf.placeholder(tf.float32, shape=[], name="reward")
                self.next_state = tf.placeholder(tf.float32, shape=[1, self.observation_space.shape[0] + 1], name="nextState")
                next_q_preds_t = tf.squeeze(self.net(self.next_state, net_scope, True))
                next_max_action_t = tf.cast(tf.argmax(next_q_preds_t, 0), tf.int32)
                target_q1 = tf.stop_gradient(self.reward + self.discount * next_q_preds_t[next_max_action_t])
                target_q2 = self.reward
                is_done = tf.equal(self.next_state[0, 4], 1)
                target_q = tf.cond(is_done, lambda: target_q2, lambda: target_q1)
                with tf.control_dependencies([target_q]):
                    self.loss = 1/2 * tf.square(target_q - self.q_t)

                adam = tf.train.AdamOptimizer(self.lr)
                self.global_step = tf.Variable(0, trainable=False, name="global_step", collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
                self.train_op = adam.minimize(self.loss, global_step=self.global_step)

            with tf.variable_scope('ExperienceReplay'):
                self.er_inputs = tf.placeholder(tf.float32, shape=[None, self.observation_space.shape[0] + 1], name="ERInputs")
                self.er_actions = tf.placeholder(tf.int32, shape=[None], name="ERInputs")
                self.er_rewards = tf.placeholder(tf.float32, shape=[None], name="ERReward")
                self.er_next_states = tf.placeholder(tf.float32, shape=[None, self.observation_space.shape[0] + 1], name="ERNextState")

                er_qs_preds = self.net(self.er_inputs, net_scope, True)
                er_sa_pairs = tf.stack([tf.range(0, tf.shape(self.er_actions)[0]), self.er_actions], 1)
                er_qs = tf.gather_nd(er_qs_preds, er_sa_pairs)

                er_next_qs_preds = self.net(self.er_next_states, fixed_net_scope, True)
                er_next_max_action_t = tf.cast(tf.argmax(er_next_qs_preds, 1), tf.int32)
                er_next_sa_pairs = tf.stack([tf.range(0, tf.shape(self.er_next_states)[0]), er_next_max_action_t], 1)
                er_next_qs = tf.gather_nd(er_next_qs_preds, er_next_sa_pairs)

                er_target_qs1 = tf.stop_gradient(self.er_rewards + self.discount * er_next_qs)
                er_target_qs2 = self.er_rewards
                scan_res = tf.scan(
                    lambda indice, next_state: (indice[0] + 1, tf.cond(tf.equal(next_state[4], 1), lambda: er_target_qs2[indice[0]], lambda: er_target_qs1[indice[0]]))
                    , self.er_next_states
                    , initializer=(tf.constant(0, tf.int32), tf.constant(0., tf.float32))
                )
                er_target_qs = scan_res[1]
                er_loss = 1/2 * tf.reduce_sum(tf.square(er_target_qs - er_qs))
                er_adam = tf.train.AdamOptimizer(self.lr)
                self.er_train_op = er_adam.minimize(er_loss)

            self.score_plh = tf.placeholder(tf.float32, shape=[])
            self.score_sum_t = tf.summary.scalar('score', self.score_plh)
            self.loss_plh = tf.placeholder(tf.float32, shape=[])
            self.loss_sum_t = tf.summary.scalar('loss', self.loss_plh)
            self.all_summary_t = tf.summary.merge_all()

            self.episode_id = tf.Variable(0, trainable=False)
            self.inc_ep_id_op = tf.assign(self.episode_id, self.episode_id + 1)

            self.saver = tf.train.Saver()

            self.init_op = tf.global_variables_initializer()

            # Playing part
            self.pscore_plh = tf.placeholder(tf.float32, shape=[])
            self.pscore_sum_t = tf.summary.scalar('play_score', self.pscore_plh)

        return graph

    def act(self, obs):
        state = [ np.concatenate((obs, [0])) ]
        act = self.sess.run(self.action_t, feed_dict={
            self.inputs: state
        })

        return (act, state)

    def learnFromEpisode(self, env, render):
        obs = env.reset()
        score = 0
        av_loss = []
        done = False
        
        while True:
            if render:
                env.render()

            act, state = self.act(obs)
            next_obs, reward, done, info = env.step(act)

            loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={
                self.inputs: [ np.concatenate((obs, [0])) ],
                self.action_t: act,
                self.reward: reward,
                self.next_state: [ np.concatenate((next_obs, [1 if done else 0])) ]
            })
            memory = np.array([(np.concatenate((obs, [0])), act, reward, np.concatenate((next_obs, [1 if done else 0])))], dtype=self.replayMemoryDt)
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
            self.loss_plh: np.mean(av_loss)
        })
        self.sw.add_summary(summary, episode_id)

        # Experiencing replays
        if episode_id % self.er_every == 0:
            self.sess.run(self.update_fixed_vars_op)
            for i in range(self.er_epoch_size):
                memories = np.random.choice(self.replayMemory, self.er_batch_size)
                _ = self.sess.run(self.er_train_op, feed_dict={
                    self.er_inputs: memories['states'],
                    self.er_actions: memories['actions'],
                    self.er_rewards: memories['rewards'],
                    self.er_next_states: memories['next_states'],
                })

        return 


class DQNAgent(BasicAgent):
    """
    Agent implementing The DQN (Experience replay abd fixed Q-Net).
    """

    def __init__(self, config, env):
        super(DQNAgent, self).__init__(config, env)

        self.nb_units = config['nb_units']

        self.N0 = config['N0']
        self.min_eps = config['min_eps']

        self.er_every = config['er_every']
        self.er_batch_size = config['er_batch_size']
        self.er_epoch_size = config['er_epoch_size']
        self.er_rm_size = config['er_rm_size']

        self.graph = self.buildGraph(tf.Graph())

        gpu_options = tf.GPUOptions(allow_growth=True)
        sessConfig = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=sessConfig, graph=self.graph)
        self.sw = tf.summary.FileWriter(self.result_dir, self.sess.graph)
        self.init()

    def net(self, inputs, scope, scope_reuse=False):
        with tf.variable_scope(scope, reuse=scope_reuse):
            W1 = tf.get_variable('W1'
                , shape=[self.observation_space.shape[0] + 1, self.nb_units]
                , initializer=tf.random_normal_initializer(stddev=1e-2)
                , collections=[tf.GraphKeys.GLOBAL_VARIABLES, "net"]
            )
            b1 = tf.get_variable('b1'
                , shape=[self.nb_units]
                , initializer=tf.zeros_initializer()
                , collections=[tf.GraphKeys.GLOBAL_VARIABLES, "net"]
            )
            a1 = tf.nn.relu(tf.matmul(inputs, W1) + b1)

            W2 = tf.get_variable('W2'
                , shape=[self.nb_units, self.nb_units]
                , initializer=tf.random_normal_initializer(stddev=1e-2)
                , collections=[tf.GraphKeys.GLOBAL_VARIABLES, "net"]
            )
            b2 = tf.get_variable('b2'
                , shape=[self.nb_units]
                , initializer=tf.zeros_initializer()
                , collections=[tf.GraphKeys.GLOBAL_VARIABLES, "net"]
            )
            a2 = tf.nn.relu(tf.matmul(a1, W2) + b2)

            W3 = tf.get_variable('W3'
                , shape=[self.nb_units, self.action_space.n]
                , initializer=tf.random_normal_initializer(stddev=1e-2)
                , collections=[tf.GraphKeys.GLOBAL_VARIABLES, "net"]
            )
            b3 = tf.get_variable('b3'
                , shape=[self.action_space.n]
                , initializer=tf.zeros_initializer()
                , collections=[tf.GraphKeys.GLOBAL_VARIABLES, "net"]
            )
            out = tf.matmul(a2, W3) + b3

        return out

    def buildGraph(self, graph):
        with graph.as_default():
            self.N0_t = tf.constant(self.N0, tf.float32, name='N_0')
            self.N = tf.Variable(0., dtype=tf.float32, name='N', trainable=False)
            self.min_eps_t = tf.constant(self.min_eps, tf.float32, name='min_eps')
            self.discount = 1
            self.replayMemoryDt = np.dtype([('states', 'float32', (5,)), ('actions', 'int32'), ('rewards', 'float32'), ('next_states', 'float32', (5,))])
            self.replayMemory = np.array([], dtype=self.replayMemoryDt)
            
            # Model
            net_scope = tf.VariableScope(reuse=False, name='Net')
            self.inputs = tf.placeholder(tf.float32, shape=[None, self.observation_space.shape[0] + 1], name='inputs')
            self.q_preds = tf.squeeze(self.net(self.inputs, net_scope, False))

            self.action_t = capacities.epsGreedy(
                self.inputs, self.q_preds, self.env.action_space.n, self.N0, self.min_eps
            )
            self.q_t = self.q_preds[self.action_t]

            fixed_net_scope = tf.VariableScope(reuse=False, name='FixedNet')
            with tf.variable_scope(fixed_net_scope):
                self.update_fixed_vars_op = []
                for var in tf.get_collection("net"):
                    fixed = tf.get_variable(var.name.split("/")[-1].split(":")[0], shape=var.get_shape())    
                    assign_op = tf.assign(fixed, var)
                    self.update_fixed_vars_op.append(assign_op)

            with tf.variable_scope('ExperienceReplay'):
                self.er_inputs = tf.placeholder(tf.float32, shape=[None, self.observation_space.shape[0] + 1], name="ERInputs")
                self.er_actions = tf.placeholder(tf.int32, shape=[None], name="ERInputs")
                self.er_rewards = tf.placeholder(tf.float32, shape=[None], name="ERReward")
                self.er_next_states = tf.placeholder(tf.float32, shape=[None, self.observation_space.shape[0] + 1], name="ERNextState")

                er_qs_preds = self.net(self.er_inputs, net_scope, True)
                er_sa_pairs = tf.stack([tf.range(0, tf.shape(self.er_actions)[0]), self.er_actions], 1)
                er_qs = tf.gather_nd(er_qs_preds, er_sa_pairs)

                er_next_qs_preds = self.net(self.er_next_states, fixed_net_scope, True)
                er_next_max_action_t = tf.cast(tf.argmax(er_next_qs_preds, 1), tf.int32)
                er_next_sa_pairs = tf.stack([tf.range(0, tf.shape(self.er_next_states)[0]), er_next_max_action_t], 1)
                er_next_qs = tf.gather_nd(er_next_qs_preds, er_next_sa_pairs)

                er_target_qs1 = tf.stop_gradient(self.er_rewards + self.discount * er_next_qs)
                er_target_qs2 = self.er_rewards
                scan_res = tf.scan(
                    lambda indice, next_state: (indice[0] + 1, tf.cond(tf.equal(next_state[4], 1), lambda: er_target_qs2[indice[0]], lambda: er_target_qs1[indice[0]]))
                    , self.er_next_states
                    , initializer=(tf.constant(0, tf.int32), tf.constant(0., tf.float32))
                )
                er_target_qs = scan_res[1]
                self.er_loss = 1/2 * tf.reduce_sum(tf.square(er_target_qs - er_qs))
                er_adam = tf.train.AdamOptimizer(self.lr)
                self.global_step = tf.Variable(0, trainable=False, name="global_step", collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
                self.er_train_op = er_adam.minimize(self.er_loss, global_step=self.global_step)

            self.score_plh = tf.placeholder(tf.float32, shape=[])
            self.score_sum_t = tf.summary.scalar('score', self.score_plh)
            self.loss_plh = tf.placeholder(tf.float32, shape=[])
            self.loss_sum_t = tf.summary.scalar('loss', self.loss_plh)
            self.all_summary_t = tf.summary.merge_all()

            self.episode_id, self.inc_ep_id_op = capacities.counter()
            self.event_count, self.inc_event_count_op = capacities.counter()

            self.saver = tf.train.Saver()

            self.init_op = tf.global_variables_initializer()

            # Playing part
            self.pscore_plh = tf.placeholder(tf.float32, shape=[])
            self.pscore_sum_t = tf.summary.scalar('play_score', self.pscore_plh)

        return graph

    def act(self, obs):
        state = np.concatenate( (obs, [0]) )
        act = self.sess.run(self.action_t, feed_dict={
            self.inputs: [ state ]
        })

        return (act, state)

    def learnFromEpisode(self, env, render):
        obs = env.reset()
        score = 0
        av_loss = []
        done = False
        
        while True:
            if render:
                env.render()

            act, state = self.act(obs)
            next_obs, reward, done, info = env.step(act)
            next_state = np.concatenate( (next_obs, [1. if done else 0.]) )

            if self.replayMemory.shape[0] >= self.er_rm_size:
                self.replayMemory = np.delete(self.replayMemory, 0)
            memory = np.array([(state, act, reward, next_state)], dtype=self.replayMemoryDt)
            self.replayMemory = np.append(self.replayMemory, memory)  

            memories = np.random.choice(self.replayMemory, self.er_batch_size)
            loss, _, event_count, _ = self.sess.run([self.er_loss, self.inc_event_count_op, self.event_count, self.er_train_op], feed_dict={
                self.er_inputs: memories['states'],
                self.er_actions: memories['actions'],
                self.er_rewards: memories['rewards'],
                self.er_next_states: memories['next_states'],
            })
            if event_count % self.er_every == 0:
                self.sess.run(self.update_fixed_vars_op)

            av_loss.append(loss)
            score += reward
            obs = next_obs
            if done:
                break

        summary, _, episode_id = self.sess.run([self.all_summary_t, self.inc_ep_id_op, self.episode_id], feed_dict={ 
            self.score_plh: score,
            self.loss_plh: np.mean(av_loss)
        })
        self.sw.add_summary(summary, episode_id)

        return 


