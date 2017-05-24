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
        action_t = tf.cond(cond, lambda: pred_action, lambda: random_action)

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

    max_actions = tf.cast(tf.expand_dims(tf.argmax(q_preds_t, 1), -1), tf.int32)
    random_actions = tf.random_uniform(shape=[nb_samples, 1], minval=0, maxval=nb_actions, dtype=tf.int32)
    stacked_actions = tf.stack([random_actions, max_actions], 1)

    conditions = tf.cast(tf.greater(tf.random_uniform([nb_samples], 0, 1), eps), tf.int32)
    selects = tf.stack([tf.range(0, nb_samples), conditions], 1)

    with tf.control_dependencies([update_N]): # Force the update call
        actions_t = tf.gather_nd(stacked_actions, selects)

    return actions_t

def get_expected_rewards(episodeRewards, discount=1):
    expected_reward = [0] * len(episodeRewards)
    for i in range(len(episodeRewards)):
        for j in range(i + 1):
            expected_reward[j] += discount**(i-j) * episodeRewards[i]

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
