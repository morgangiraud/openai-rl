import numpy as np
import tensorflow as tf

from agents import BasicAgent, capacities

def getExpectedRewards(episodeRewards):
    expected_reward = [0] * len(episodeRewards)
    for i in range(len(episodeRewards)):
        for j in range(i + 1):
            expected_reward[j] += episodeRewards[i]

    return np.reshape(expected_reward, (len(expected_reward), 1))

class DeepMCPolicyAgent(BasicAgent):
    """
    Agent implementing Policy gradient using Monte-Carlo control
    """

    def __init__(self, config, env):
        # Best conf: 
        # nb_units: 50
        # lr: 1e-3
        super(DeepMCPolicyAgent, self).__init__(config, env)

        self.policy_params = {
            'nb_inputs': self.observation_space.shape[0] + 1
            , 'nb_units': config['nb_units']
            , 'nb_actions': self.action_space.n
        }

        self.graph = self.buildGraph(tf.Graph())

        gpu_options = tf.GPUOptions(allow_growth=True)
        sessConfig = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=sessConfig, graph=self.graph)
        self.sw = tf.summary.FileWriter(self.result_dir, self.sess.graph)
        self.init()

    
    def buildGraph(self, graph):
        with graph.as_default():
            self.discount = 1
            
            # Model
            self.inputs = tf.placeholder(tf.float32, shape=[None, self.observation_space.shape[0] + 1], name='inputs')

            policy_scope = tf.VariableScope(reuse=False, name='Policy')
            with tf.variable_scope(policy_scope):
                self.probs, self.actions = capacities.policy(self.policy_params, self.inputs)
            self.action_t = tf.squeeze(self.actions, 1)[0]

            with tf.variable_scope('Training'):
                self.rewards = tf.placeholder(tf.float32, shape=[None, 1], name="reward")
                stacked_actions = tf.stack([tf.range(0, tf.shape(self.actions)[0]), tf.squeeze(self.actions, 1)], 1)
                log_probs = tf.log(tf.gather_nd(self.probs, stacked_actions))
                # log_probs = tf.Print(log_probs, data=[tf.shape(self.probs), tf.shape(self.actions), tf.shape(log_probs)], message="tf.shape(log_probs):")
                self.loss = - tf.reduce_sum(log_probs * self.rewards)

                adam = tf.train.AdamOptimizer(self.lr)
                self.global_step = tf.Variable(0, trainable=False, name="global_step", collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
                self.train_op = adam.minimize(self.loss, global_step=self.global_step)

            self.score_plh = tf.placeholder(tf.float32, shape=[])
            self.score_sum_t = tf.summary.scalar('score', self.score_plh)
            self.loss_plh = tf.placeholder(tf.float32, shape=[])
            self.loss_sum_t = tf.summary.scalar('loss', self.loss_plh)
            self.all_summary_t = tf.summary.merge_all()

            self.episode_id, self.inc_ep_id_op = capacities.counter()

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
        historyType = np.dtype([('states', 'float32', (env.observation_space.shape[0] + 1,)), ('actions', 'int32', (1,)), ('rewards', 'float32')])
        history = np.array([], dtype=historyType)
        done = False
        
        while True:
            if render:
                env.render()

            act, state = self.act(obs)
            next_obs, reward, done, info = env.step(act)

            memory = np.array([(np.concatenate((obs, [0])), act, reward)], dtype=historyType)
            history = np.append(history, memory)

            score += reward
            obs = next_obs
            if done:
                break

        expected_reward = getExpectedRewards(history['rewards'])
        
        # Learning
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict={
            self.inputs: history['states'],
            self.actions: history['actions'],
            self.rewards: expected_reward,
        })
        summary, _, episode_id = self.sess.run([self.all_summary_t, self.inc_ep_id_op, self.episode_id], feed_dict={ 
            self.score_plh: score,
            self.loss_plh: loss
        })
        self.sw.add_summary(summary, episode_id)

        return 


class ActorCriticAgent(BasicAgent):
    """
    Agent implementing Actor critic using REINFORCE 
    """

    def __init__(self, config, env):
        # Best conf: 
        super(ActorCriticAgent, self).__init__(config, env)

        self.policy_params = {
            'nb_inputs': self.observation_space.shape[0] + 1
            , 'nb_units': config['nb_units']
            , 'nb_actions': self.action_space.n
        }
        self.q_params = {
            'nb_inputs': self.observation_space.shape[0] + 1
            , 'nb_units': config['nb_units']
            , 'nb_actions': self.action_space.n
        }
        self.discount = 1
        self.policy_lr = self.lr
        self.q_lr = self.lr

        self.graph = self.buildGraph(tf.Graph())

        gpu_options = tf.GPUOptions(allow_growth=True)
        sessConfig = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=sessConfig, graph=self.graph)
        self.sw = tf.summary.FileWriter(self.result_dir, self.sess.graph)
        self.init()
    
    def buildGraph(self, graph):
        with graph.as_default():            
            # Model
            self.inputs = tf.placeholder(tf.float32, shape=[None, self.observation_space.shape[0] + 1], name='inputs')

            policy_scope = tf.VariableScope(reuse=False, name='Policy')
            with tf.variable_scope(policy_scope):
                self.probs, self.actions = capacities.policy(self.policy_params, self.inputs)
            self.action_t = tf.squeeze(self.actions, 1)[0]
            # self.action_t = tf.Print(self.action_t, data=[self.probs, self.action_t], message="self.probs, self.action_t:")

            q_scope = tf.VariableScope(reuse=False, name='QValues')
            with tf.variable_scope(q_scope):
                self.q_values = capacities.q_value(self.q_params, self.inputs)
            self.q = self.q_values[0, tf.stop_gradient(self.action_t)]

            with tf.control_dependencies([self.probs, self.q]):
                with tf.variable_scope('Training'):
                    stacked_actions = tf.stack([tf.range(0, tf.shape(self.actions)[0]), tf.squeeze(self.actions, 1)], 1)
                    log_probs = tf.log(tf.gather_nd(self.probs, stacked_actions))
                    qs = tf.gather_nd(self.q_values, stacked_actions)

                    self.policy_loss = - tf.reduce_sum(log_probs * tf.stop_gradient(qs))
                    policy_adam = tf.train.AdamOptimizer(self.policy_lr)
                    self.policy_global_step = tf.Variable(0, trainable=False, name="policy_global_step", collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
                    self.policy_train_op = policy_adam.minimize(self.policy_loss, global_step=self.policy_global_step)

                with tf.variable_scope('QTraining'):
                    self.reward = tf.placeholder(tf.float32, shape=[], name="reward")
                    self.next_state = tf.placeholder(tf.float32, shape=[None, self.observation_space.shape[0] + 1], name="next_state")
                    self.next_act = tf.placeholder(tf.int32, shape=[], name="next_act")
                    
                    with tf.variable_scope(q_scope, reuse=True):
                        next_q_values = capacities.q_value(self.q_params, self.next_state)
                    next_q = tf.stop_gradient(next_q_values[0, self.next_act])

                    target_q1 = self.reward + self.discount * next_q
                    target_q2 = self.reward
                    is_next_done = tf.equal(self.next_state[0, 4], 1)
                    target_q = tf.cond(is_next_done, lambda: target_q2, lambda: target_q1)

                    with tf.control_dependencies([target_q]):
                        self.q_loss = 1/2 * tf.square(target_q - self.q)

                    q_adam = tf.train.AdamOptimizer(self.q_lr)
                    self.q_global_step = tf.Variable(0, trainable=False, name="q_global_step")
                    self.q_train_op = q_adam.minimize(self.q_loss, global_step=self.q_global_step)

            self.score_plh = tf.placeholder(tf.float32, shape=[])
            self.score_sum_t = tf.summary.scalar('score', self.score_plh)
            self.policy_loss_plh = tf.placeholder(tf.float32, shape=[])
            self.policy_loss_sum_t = tf.summary.scalar('policy_loss', self.policy_loss_plh)
            self.q_loss_plh = tf.placeholder(tf.float32, shape=[])
            self.q_loss_sum_t = tf.summary.scalar('q_loss', self.q_loss_plh)
            self.all_summary_t = tf.summary.merge_all()

            self.episode_id, self.inc_ep_id_op = capacities.counter()

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
        act, _ = self.act(obs)

        av_policy_loss = []
        av_q_loss = []
        score = 0
        done = False
        
        while True:
            if render:
                env.render()

            next_obs, reward, done, info = env.step(act)
            next_act, _ = self.act(next_obs)

            # print([ np.concatenate((obs, [0])) ], act, reward, [ np.concatenate((next_obs, [1 if done else 0])) ], next_act)
            _, policy_loss, _, q_loss = self.sess.run([self.policy_train_op, self.policy_loss , self.q_train_op, self.q_loss], feed_dict={
                self.inputs: [ np.concatenate((obs, [0])) ],
                self.actions: [ [act] ],
                self.reward: reward,
                self.next_state: [ np.concatenate((next_obs, [1 if done else 0])) ],
                self.next_act: next_act
            })
            av_policy_loss.append(policy_loss)
            av_q_loss.append(q_loss)

            score += reward
            obs = next_obs
            act = next_act
            if done:
                break


        summary, _, episode_id = self.sess.run([self.all_summary_t, self.inc_ep_id_op, self.episode_id], feed_dict={ 
            self.score_plh: score,
            self.policy_loss_plh: np.mean(av_policy_loss),
            self.q_loss_plh: np.mean(av_q_loss),
        })
        self.sw.add_summary(summary, episode_id)

        return 


