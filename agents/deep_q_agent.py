import numpy as np
import tensorflow as tf

from agents import BasicAgent, capacities

class DeepQAgent(BasicAgent):
    """
    Agent implementing tabular Q-learning.
    """

    def __init__(self, config, env):
        super(DeepQAgent, self).__init__(config, env)

        self.nb_units = config['nb_units']

        self.N0 = config['N0']
        self.min_eps = config['min_eps']

        self.er_lr = config['er_lr']
        self.er_every = config['er_every']
        self.er_batch_size = config['er_batch_size']
        self.er_epoch_size = config['er_epoch_size']
        self.er_rm_size = config['er_rm_size']

        self.graph = self.buildGraph(tf.Graph())

        self.sess = tf.Session(graph=self.graph)
        self.sw = tf.summary.FileWriter(self.result_dir, self.sess.graph)
        self.init()

    def buildGraph(self, graph):
        with graph.as_default():
            self.N0_t = tf.constant(self.N0, tf.float32, name='N_0')
            self.N = tf.Variable(0., dtype=tf.float32, name='N', trainable=False)
            self.min_eps_t = tf.constant(self.min_eps, tf.float32, name='min_eps')
            self.discount = 1
            self.replayMemoryDt = np.dtype([('states', 'float32', (5,)), ('actions', 'int32'), ('rewards', 'float32'), ('next_states', 'float32', (5,))])
            self.replayMemory = np.array([], dtype=self.replayMemoryDt)
            
            # Model
            with tf.variable_scope('Qs'):
                self.inputs = tf.placeholder(tf.float32, shape=[None, self.observation_space.shape[0] + 1], name='inputs')

                W1 = tf.get_variable('W1', shape=[self.observation_space.shape[0] + 1, self.nb_units], initializer=tf.random_normal_initializer(stddev=1e-2))
                b1 = tf.get_variable('b1', shape=[self.nb_units], initializer=tf.zeros_initializer())
                a1 = tf.nn.relu(tf.matmul(self.inputs, W1) + b1)

                W2 = tf.get_variable('W2', shape=[self.nb_units, self.action_space.n], initializer=tf.random_normal_initializer(stddev=1e-2))
                b2 = tf.get_variable('b2', shape=[self.action_space.n], initializer=tf.zeros_initializer())
                out = tf.matmul(a1, W2) + b2

                self.q_preds = tf.squeeze(out)
                
            self.action_t = capacities.epsGreedy(
                self.inputs, self.q_preds, self.env.action_space.n, self.N0, self.min_eps
            )
            self.q_t = self.q_preds[self.action_t]

            with tf.variable_scope('FixedQs'):
                f_W1 = tf.get_variable('f_W1', shape=[self.observation_space.shape[0] + 1, self.nb_units])
                f_b1 = tf.get_variable('f_b1', shape=[self.nb_units])
                f_W2 = tf.get_variable('f_W2', shape=[self.nb_units, self.action_space.n])
                f_b2 = tf.get_variable('f_b2', shape=[self.action_space.n])
                self.update_f_W1_op = tf.assign(f_W1, W1)
                self.update_f_b1_op = tf.assign(f_b1, b1)
                self.update_f_W2_op = tf.assign(f_W2, W2)
                self.update_f_b2_op = tf.assign(f_b2, b2)

            with tf.variable_scope('Training'):
                self.reward = tf.placeholder(tf.float32, shape=[], name="reward")
                self.next_state = tf.placeholder(tf.float32, shape=[1, self.observation_space.shape[0] + 1], name="nextState")
                next_a1 = tf.nn.relu(tf.matmul(self.next_state, W1) + b1)
                next_q_preds = tf.squeeze(tf.matmul(next_a1, W2) + b2)
                next_max_action_t = tf.cast(tf.argmax(next_q_preds, 0), tf.int32)
                tartget_q1 = tf.stop_gradient(self.reward + self.discount * next_q_preds[next_max_action_t])
                tartget_q2 = self.reward
                target_q = tf.cond(tf.equal(self.next_state[0, 4], 1), lambda: tartget_q2, lambda: tartget_q1)
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

                er_a1 = tf.nn.relu(tf.matmul(self.er_inputs, W1) + b1)
                er_qs_preds = tf.matmul(er_a1, W2) + b2
                er_sa_pairs = tf.stack([tf.range(0, tf.shape(self.er_actions)[0]), self.er_actions], 1)
                er_qs = tf.gather_nd(er_qs_preds, er_sa_pairs)

                er_next_a1 = tf.nn.relu(tf.matmul(self.er_next_states, f_W1) + f_b1)
                er_next_qs_preds = tf.matmul(er_next_a1, f_W2) + f_b2
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
                er_adam = tf.train.AdamOptimizer(self.er_lr)
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
            self.sess.run([self.update_f_W1_op, self.update_f_b1_op, self.update_f_W2_op, self.update_f_b2_op])
            for i in range(self.er_epoch_size):
                memories = np.random.choice(self.replayMemory, self.er_batch_size)
                _ = self.sess.run(self.er_train_op, feed_dict={
                    self.er_inputs: memories['states'],
                    self.er_actions: memories['actions'],
                    self.er_rewards: memories['rewards'],
                    self.er_next_states: memories['next_states'],
                })

        return 


