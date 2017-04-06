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
            N = tf.Variable(1., trainable=False, dtype=tf.float32, name='N')
            eps = tf.maximum(N0_t / (N0_t + N), min_eps_t, name="eps")
            update_N = tf.assign(N, N + 1)
            tf.summary.scalar('N', N)
        else:
            N = tf.Variable(tf.ones(shape=[nb_state]), name='N', trainable=False)
            eps = tf.maximum(N0_t / (N0_t + N[inputs_t]), min_eps_t, name="eps")
            update_N = tf.scatter_add(N, inputs_t, 1)
            tf.summary.histogram('N', N)
        cond = tf.greater(tf.random_uniform([], 0, 1), eps)
        pred_action = tf.cast(tf.argmax(q_preds_t, 0), tf.int32)
        random_action = tf.random_uniform([], 0, nb_actions, dtype=tf.int32)

        with tf.control_dependencies([update_N]): # Force the update call
            action_t = tf.cond(cond, lambda: pred_action, lambda: random_action)

    return action_t

def eligibilityTraces(inputs, action_t, et_shape, discount, lambda_value):
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

# def tabularQValue(nb_state, nb_action):
#     scope = tf.VariableScope('TabularQValue')
#     with tf.variable_scope(scope, reuse=False):
#         Q = tf.get_variable(
#             'Q'
#             , shape=[nb_state, nb_action]
#             , initializer=tf.zeros_initializer()
#             , dtype=tf.float32
#         )

#     def apply(inputs_t):
#         with tf.variable_scope(scope, reuse=True):
#             Q = tf.get_variable('Q')
#             out = tf.nn.embedding_lookup(Q, inputs_t)

#         return out

#     return apply


def MSETabularQLearning(Qs_t, discount, q_preds, action_t, optimizer=None):
    with tf.variable_scope('MSEQLearning'):
        reward = tf.placeholder(tf.float32, shape=[], name="reward")
        next_state = tf.placeholder(tf.int32, shape=[], name="nextState")

        q_t = q_preds[action_t]

        next_max_action_t = tf.cast(tf.argmax(Qs_t[next_state], 0), tf.int32)
        target_q = tf.stop_gradient(reward + discount * Qs_t[next_state, next_max_action_t], name='target_q')

        loss = tf.reduce_mean(tf.square(target_q - q_t))

    with tf.variable_scope('Training'):
        global_step = tf.Variable(0, trainable=False, name="global_step", collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
        if optimizer == None:
            learning_rate = tf.train.inverse_time_decay(1., global_step, 1, 0.001, staircase=False, name="decay_lr")
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)

    return (reward, next_state, loss, train_op)

def counter(name):
    count_t = tf.Variable(0, trainable=False, dtype=tf.int32, name=name)
    inc_count_op = count_t.assign_add(1)

    return (count_t, inc_count_op)

def fixScope(from_scope):
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
    W1 = tf.get_variable('W1'
        , shape=[ network_params['nb_inputs'], network_params['nb_units'] ]
        , initializer=tf.random_normal_initializer(stddev=1e-2)
    )
    b1 = tf.get_variable('b1'
        , shape=[ network_params['nb_units'] ]
        , initializer=tf.zeros_initializer()
    )
    a1 = tf.nn.relu(tf.matmul(inputs, W1) + b1)

    W2 = tf.get_variable('W2'
        , shape=[ network_params['nb_units'], network_params['nb_units'] ]
        , initializer=tf.random_normal_initializer(stddev=1e-2)
    )
    b2 = tf.get_variable('b2'
        , shape=[ network_params['nb_units'] ]
        , initializer=tf.zeros_initializer()
    )
    a2 = tf.nn.relu(tf.matmul(a1, W2) + b2)

    W3 = tf.get_variable('W3'
        , shape=[ network_params['nb_units'], network_params['nb_outputs'] ]
        , initializer=tf.random_normal_initializer(stddev=1e-2)
    )
    b3 = tf.get_variable('b3'
        , shape=[ network_params['nb_outputs'] ]
        , initializer=tf.zeros_initializer()
    )
    logits = tf.matmul(a2, W3) + b3
    probs_t = tf.nn.softmax(logits)

    actions_t = tf.cast(tf.multinomial(logits, 1), tf.int32)

    return (probs_t, actions_t)


def value_f(network_params, inputs):
    W1 = tf.get_variable('W1'
        , shape=[ network_params['nb_inputs'], network_params['nb_units'] ]
        , initializer=tf.random_normal_initializer(stddev=1e-2)
    )
    b1 = tf.get_variable('b1'
        , shape=[ network_params['nb_units'] ]
        , initializer=tf.zeros_initializer()
    )
    a1 = tf.nn.relu(tf.matmul(inputs, W1) + b1)
    # a1 = tf.matmul(inputs, W1) + b1

    W2 = tf.get_variable('W2'
        , shape=[ network_params['nb_units'], network_params['nb_units'] ]
        , initializer=tf.random_normal_initializer(stddev=1e-2)
    )
    b2 = tf.get_variable('b2'
        , shape=[ network_params['nb_units'] ]
        , initializer=tf.zeros_initializer()
    )
    a2 = tf.nn.relu(tf.matmul(a1, W2) + b2)
    # a2 = tf.matmul(a1, W2) + b2

    W3 = tf.get_variable('W3'
        , shape=[ network_params['nb_units'], network_params['nb_outputs'] ]
        , initializer=tf.random_normal_initializer(stddev=1e-2)
    )
    b3 = tf.get_variable('b3'
        , shape=[ network_params['nb_outputs'] ]
        , initializer=tf.zeros_initializer()
    )
    values = tf.matmul(a2, W3) + b3

    return values
