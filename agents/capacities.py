import numpy as np
import tensorflow as tf

# All those capacities will be applied in the current default graph
# Use it this way:
# with my_graph.as_default():
#     my_capacity()

def eps_greedy(inputs_t, q_preds_t, nb_actions, N0, min_eps, nb_state=None):
    reusing_scope = tf.get_variable_scope().reuse

    N0_t = tf.constant(N0, tf.float32, name='N0')
    min_eps_t = tf.constant(min_eps, tf.float32, name='min_eps')

    if nb_state == None:
        N = tf.Variable(1., trainable=False, dtype=tf.float32, name='N')
        eps = tf.maximum(N0_t / (N0_t + N), min_eps_t, name="eps")
        update_N = tf.assign(N, N + 1)
        if reusing_scope is False:
            tf.summary.scalar('N', N)
    else:
        N = tf.Variable(tf.ones(shape=[nb_state]), name='N', trainable=False)
        eps = tf.maximum(N0_t / (N0_t + N[inputs_t]), min_eps_t, name="eps")
        update_N = tf.scatter_add(N, inputs_t, 1)
        if reusing_scope is False:
            tf.summary.histogram('N', N)
    cond = tf.greater(tf.random_uniform([], 0, 1), eps)
    pred_action = tf.cast(tf.argmax(q_preds_t, 0), tf.int32)
    random_action = tf.random_uniform([], 0, nb_actions, dtype=tf.int32)

    with tf.control_dependencies([update_N]): # Force the update call
        action_t = tf.where(cond, pred_action, random_action)

    return action_t

def batch_eps_greedy(inputs_t, q_preds_t, nb_actions, N0, min_eps, nb_state=None):
    reusing_scope = tf.get_variable_scope().reuse

    N0_t = tf.constant(N0, tf.float32, name='N0')
    min_eps_t = tf.constant(min_eps, tf.float32, name='min_eps')

    if nb_state == None:
        N = tf.Variable(1., trainable=False, dtype=tf.float32, name='N')
        eps = tf.maximum(N0_t / (N0_t + N), min_eps_t, name="eps")
        update_N = tf.assign(N, N + 1)
        if reusing_scope is False:
            tf.summary.scalar('N', N)
    else:
        N = tf.Variable(tf.ones(shape=[nb_state]), name='N', trainable=False)
        eps = tf.maximum(
            N0_t / (N0_t + tf.squeeze(tf.nn.embedding_lookup(N, inputs_t), 1))
            , min_eps_t
            , name="eps"
        )
        update_N = tf.scatter_nd_add(N, inputs_t, tf.ones(shape=[tf.shape(inputs_t)[0]]))
        if reusing_scope is False:
            tf.summary.histogram('N', N)

    nb_samples = tf.shape(q_preds_t)[0]

    conditions = tf.greater(tf.random_uniform([nb_samples], 0, 1), eps)
    max_actions = tf.cast(tf.argmax(q_preds_t, 1), tf.int32)
    random_actions = tf.random_uniform(shape=[nb_samples], minval=0, maxval=nb_actions, dtype=tf.int32)

    with tf.control_dependencies([update_N]): # Force the update call
        actions_t = tf.where(conditions, max_actions, random_actions)

    return actions_t

def get_expected_rewards(episodeRewards, discount=.99):
    expected_reward = [0] * len(episodeRewards)
    for t in range(len(episodeRewards) - 1, -1, -1):
        if t == len(episodeRewards) - 1:
            expected_reward[t] = episodeRewards[t]
        else:
            expected_reward[t] = discount * expected_reward[t + 1] + episodeRewards[t]

    return expected_reward

def get_n_step_expected_rewards(episodeRewards, estimates, discount=.99, n_step=0):
    expected_reward = [0] * len(episodeRewards)
    for t in range(len(episodeRewards)):
        if t + n_step < len(episodeRewards):
            expected_reward[t] = estimates[t + n_step]
            for t_2 in range(t+n_step, t - 1, -1):
                expected_reward[t] = episodeRewards[t_2] + discount * expected_reward[t]
        else:
            for t_2 in range(len(episodeRewards) - 1, t - 1, -1):
                expected_reward[t] = episodeRewards[t_2] + discount * expected_reward[t]

    return expected_reward

def get_lambda_expected_rewards(episodeRewards, estimates, discount=.99, lambda_value=.9):
    if lambda_value == 1.: # In this case this leads to MC 
        return get_expected_rewards(episodeRewards, discount)

    expected_reward = np.array([0.] * len(episodeRewards))
    for i in range(len(episodeRewards)):
        rewards = np.concatenate( (episodeRewards[:len(episodeRewards)-i], np.zeros(i)) )
        n_step_returns = np.array(get_n_step_expected_rewards(rewards, estimates, discount, i))
        if i == len(episodeRewards) - 1:
            expected_reward += lambda_value**i * n_step_returns
        else:
            expected_reward += (1-lambda_value) * lambda_value**i * n_step_returns

    return expected_reward

def eligibility_traces(inputs, action_t, et_shape, discount, lambda_value):
    with tf.variable_scope("EligibilityTraces"):
        et = tf.Variable(
            initial_value=np.zeros(et_shape)
            , name="EligibilityTraces"
            , dtype=tf.float32
            , trainable=False
        )
        tf.summary.histogram('ETarray', et)
        dec_et_op = tf.assign(et, discount * lambda_value * et)
        with tf.control_dependencies([dec_et_op]):
            update_et_op = tf.scatter_nd_update(et, indices=[[inputs, action_t]], updates=[1.])

        reset_et_op = et.assign(np.zeros(et_shape))

    return (et, update_et_op, reset_et_op)


def mse_tabular_q_learning(Qs_t, reward, next_state, discount, q_preds, action_t):
    # reusing_scope = tf.get_variable_scope().reuse

    next_max_action_t = tf.cast(tf.argmax(Qs_t[next_state], 0), tf.int32)
    target_q = tf.stop_gradient(reward + discount * Qs_t[next_state, next_max_action_t], name='target_q')

    q_t = q_preds[action_t]
    loss = tf.reduce_mean(tf.square(target_q - q_t))

    return loss

def counter(name):
    count_t = tf.Variable(0, trainable=False, dtype=tf.int32, name=name)
    inc_count_op = count_t.assign_add(1)

    return (count_t, inc_count_op)

def fix_scope(from_scope):
    update_fixed_vars_op = []
    for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=from_scope.name):
        fixed = tf.get_variable(
            var.name.split("/")[-1].split(":")[0]
            , shape=var.get_shape()
            , trainable=False
        )
        assign_op = tf.assign(fixed, var)
        update_fixed_vars_op.append(assign_op)

    return update_fixed_vars_op


def policy(network_params, inputs):
    reusing_scope = tf.get_variable_scope().reuse
    
    W1 = tf.get_variable('W1'
        , shape=[ network_params['nb_inputs'], network_params['nb_units'] ]
        , initializer=tf.random_normal_initializer(mean=network_params['initial_mean'], stddev=network_params['initial_stddev'])
    )
    if reusing_scope is False:
        tf.summary.histogram('W1', W1)
    b1 = tf.get_variable('b1'
        , shape=[ network_params['nb_units'] ]
        , initializer=tf.zeros_initializer()
    )
    if reusing_scope is False:
        tf.summary.histogram('b1', b1)
    a1 = tf.nn.relu(tf.matmul(inputs, W1) + b1)

    W2 = tf.get_variable('W2'
        , shape=[ network_params['nb_units'], network_params['nb_units'] ]
        , initializer=tf.random_normal_initializer(mean=network_params['initial_mean'], stddev=network_params['initial_stddev'])
    )
    if reusing_scope is False:
        tf.summary.histogram('W2', W2)
    b2 = tf.get_variable('b2'
        , shape=[ network_params['nb_units'] ]
        , initializer=tf.zeros_initializer()
    )
    if reusing_scope is False:
        tf.summary.histogram('b2', b2)
    a2 = tf.nn.relu(tf.matmul(a1, W2) + b2)

    W3 = tf.get_variable('W3'
        , shape=[ network_params['nb_units'], network_params['nb_outputs'] ]
        , initializer=tf.random_normal_initializer(mean=network_params['initial_mean'], stddev=network_params['initial_stddev'])
    )
    if reusing_scope is False:
        tf.summary.histogram('W3', W3)
    b3 = tf.get_variable('b3'
        , shape=[ network_params['nb_outputs'] ]
        , initializer=tf.zeros_initializer()
    )
    if reusing_scope is False:
        tf.summary.histogram('b3', b3)
    logits = tf.matmul(a2, W3) + b3
    probs_t = tf.nn.softmax(logits)

    actions_t = tf.cast(tf.multinomial(logits, 1), tf.int32)

    return (probs_t, actions_t)


def value_f(network_params, inputs):
    reusing_scope = tf.get_variable_scope().reuse

    W1 = tf.get_variable('W1'
        , shape=[ network_params['nb_inputs'], network_params['nb_units'] ]
        , initializer=tf.random_normal_initializer(mean=network_params['initial_mean'], stddev=network_params['initial_stddev'])
    )
    if reusing_scope is False:
        tf.summary.histogram('W1', W1)
    b1 = tf.get_variable('b1'
        , shape=[ network_params['nb_units'] ]
        , initializer=tf.zeros_initializer()
    )
    if reusing_scope is False:
        tf.summary.histogram('b1', b1)
    a1 = tf.nn.relu(tf.matmul(inputs, W1) + b1)
    # a1 = tf.matmul(inputs, W1) + b1

    W2 = tf.get_variable('W2'
        , shape=[ network_params['nb_units'], network_params['nb_units'] ]
        , initializer=tf.random_normal_initializer(mean=network_params['initial_mean'], stddev=network_params['initial_stddev'])
    )
    if reusing_scope is False:
        tf.summary.histogram('W2', W2)
    b2 = tf.get_variable('b2'
        , shape=[ network_params['nb_units'] ]
        , initializer=tf.zeros_initializer()
    )
    if reusing_scope is False:
        tf.summary.histogram('b2', b2)
    a2 = tf.nn.relu(tf.matmul(a1, W2) + b2)

    W3 = tf.get_variable('W3'
        , shape=[ network_params['nb_units'], network_params['nb_outputs'] ]
        , initializer=tf.random_normal_initializer(mean=network_params['initial_mean'], stddev=network_params['initial_stddev'])
    )
    if reusing_scope is False:
        tf.summary.histogram('W3', W3)
    b3 = tf.get_variable('b3'
        , shape=[ network_params['nb_outputs'] ]
        , initializer=tf.zeros_initializer()
    )
    if reusing_scope is False:
        tf.summary.histogram('b3', b3)
    values = tf.matmul(a2, W3) + b3

    return values
