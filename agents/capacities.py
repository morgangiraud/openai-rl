import numpy as np
import tensorflow as tf

# All those capacities will be applied in the current default graph
# Use it this way:
# with my_graph.as_default():
#     my_capacity()

def epsGreedy(inputs_t, q_preds_t, nb_actions, N0, min_eps, nb_state=None):
    with tf.variable_scope('EpsilonGreedyPolicy'):
        N0_t = tf.constant(N0, tf.float32, name='N0')
        min_eps_t = tf.constant(min_eps, tf.float32, name='min_eps')

        if nb_state == None:
            N = tf.Variable(0., trainable=False, dtype=tf.float32, name='N')
            update_N = tf.assign(N, N + 1)
            eps = tf.maximum(N0_t / (N0_t + N), min_eps_t, name="eps")
        else:
            N = tf.Variable(tf.ones(shape=[nb_state]), name='N', trainable=False)
            update_N = tf.scatter_add(N, inputs_t, 1)
            eps = tf.maximum(N0_t / (N0_t + N[inputs_t]), min_eps_t, name="eps")
        cond = tf.greater(tf.random_uniform([], 0, 1), eps)
        pred_action = tf.cast(tf.argmax(q_preds_t, 0), tf.int32)
        random_action = tf.random_uniform([], 0, nb_actions, dtype=tf.int32)
        with tf.control_dependencies([update_N]):
            action_t = tf.cond(cond, lambda: pred_action, lambda: random_action)

    return action_t

def MSETabularQLearning(Qs_t, discount, q_preds, action_t, optimizer):
    with tf.variable_scope('MSEQLearning'):
        q_t = q_preds[action_t]
        reward = tf.placeholder(tf.float32, shape=[], name="reward")
        next_state = tf.placeholder(tf.int32, shape=[], name="nextState")
        next_max_action_t = tf.cast(tf.argmax(Qs_t[next_state], 0), tf.int32)
        target_q = tf.stop_gradient(reward + discount * Qs_t[next_state, next_max_action_t], name='target_q')
        loss = 1/2 * tf.square(target_q - q_t)

        global_step = tf.Variable(0, trainable=False, name="global_step", collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
        train_op = optimizer.minimize(loss, global_step=global_step)

    return (reward, next_state, loss, train_op)

def episodeCount():
    episode_id = tf.Variable(0, trainable=False)
    inc_ep_id_op = tf.assign(episode_id, episode_id + 1)

    return (episode_id, inc_ep_id_op)

