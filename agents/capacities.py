import numpy as np
import tensorflow as tf

from agents.rnn_cell import LSTMCell

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

def tabular_eps_greedy(inputs_t, q_preds_t, nb_states, nb_actions, N0, min_eps):
    reusing_scope = tf.get_variable_scope().reuse

    Ns = tf.get_variable('Ns', shape=[nb_states], dtype=tf.float32, trainable=False, initializer=tf.zeros_initializer())
    if reusing_scope is False:
        tf.summary.histogram('Ns', Ns)
    update_Ns = tf.scatter_add(Ns, inputs_t, tf.ones_like(inputs_t, dtype=tf.float32))

    eps = tf.maximum(
        N0 / (N0 + tf.gather(Ns, inputs_t))
        , min_eps
        , name="eps"
    )
    
    nb_samples = tf.shape(q_preds_t)[0]
    max_actions = tf.cast(tf.argmax(q_preds_t, 1), tf.int32) 
    probs_t = tf.sparse_to_dense(
        sparse_indices=tf.stack([tf.range(nb_samples), max_actions], 1)
        , output_shape=[nb_samples, nb_actions]
        , sparse_values=1 - eps
        , default_value=0.
    ) + tf.expand_dims(eps / nb_actions, 1)

    conditions = tf.greater(tf.random_uniform([nb_samples], 0, 1), eps)
    random_actions = tf.random_uniform(shape=[nb_samples], minval=0, maxval=nb_actions, dtype=tf.int32)
    with tf.control_dependencies([update_Ns]): # Force the update call
        actions_t = tf.where(conditions, max_actions, random_actions)

    return actions_t, probs_t

def tabular_UCB(Qs_t, inputs_t):
    reusing_scope = tf.get_variable_scope().reuse

    timestep = tf.get_variable("timestep", shape=[], dtype=tf.int32, trainable=False, initializer=tf.zeros_initializer())
    inc_t = tf.assign_add(timestep, 1)

    # State Action count
    Nsa_t = tf.get_variable("Nsa", shape=Qs_t.get_shape(), dtype=tf.float32, trainable=False, initializer=tf.ones_initializer())
    if reusing_scope is False:
        tf.summary.histogram('Nsa', Nsa_t)

    qs_t = tf.gather(Qs_t, inputs_t)
    nsa_t = tf.gather(Nsa_t, inputs_t)

    # We make sure we increment the timestep before computing the values of each actions
    with tf.control_dependencies([inc_t]):
        values_t = qs_t + ( (2 * tf.log(tf.cast(timestep, tf.float32))) / nsa_t )**(1/2)
    actions_t = tf.cast(tf.argmax(values_t, 1), dtype=tf.int32)
    probs_t = tf.one_hot(actions_t, depth=tf.shape(Qs_t)[1])

    state_action_pairs = tf.stack([inputs_t, actions_t], 1)
    update_Nsa = tf.scatter_nd_add(Nsa_t, state_action_pairs, tf.ones_like(inputs_t, dtype=tf.float32))

    with tf.control_dependencies([update_Nsa]): # Force the update call
        actions_t = tf.identity(actions_t)

    return actions_t, probs_t

def eligibility_traces(Qs_t, states_t, actions_t, discount, lambda_value):
    et = tf.get_variable(
        "eligibilitytraces"
        , shape=Qs_t.get_shape()
        , dtype=tf.float32
        , trainable=False
        , initializer=tf.zeros_initializer()
    )
    tf.summary.histogram('eligibilitytraces', et)
    dec_et_op = tf.assign(et, discount * lambda_value * et)
    with tf.control_dependencies([dec_et_op]):
        state_action_pairs = tf.stack([states_t, actions_t], 1)
        update_et_op = tf.scatter_nd_update(et, indices=state_action_pairs, updates=tf.ones_like(states_t, dtype=tf.float32))

    reset_et_op = et.assign(tf.zeros_like(et, dtype=tf.float32))

    return (et, update_et_op, reset_et_op)

def eligibility_dutch_traces(Qs_t, states_t, actions_t, lr, discount, lambda_value):
    # Beware this trace has to be used with a different learning rule
    et = tf.get_variable(
        "eligibilitytraces"
        , shape=Qs_t.get_shape()
        , dtype=tf.float32
        , trainable=False
        , initializer=tf.zeros_initializer()
    )
    tf.summary.histogram('eligibilitytraces', et)
    state_action_pairs = tf.stack([states_t, actions_t], 1)
    current_trace = tf.gather_nd(et, state_action_pairs)
    updates = 1 - lr * discount * lambda_value * current_trace
    with tf.control_dependencies([updates]):
        dec_et_op = tf.assign(et, discount * lambda_value * et)
        with tf.control_dependencies([dec_et_op]):
            update_et_op = tf.scatter_nd_add(et, indices=state_action_pairs, updates=updates)

    reset_et_op = et.assign(tf.zeros_like(et, dtype=tf.float32))

    return (et, update_et_op, reset_et_op)

def get_expected_rewards(episode_rewards, discount=.99):
    expected_reward = [0] * len(episode_rewards)
    for t in range(len(episode_rewards) - 1, -1, -1):
        if t == len(episode_rewards) - 1:
            expected_reward[t] = episode_rewards[t]
        else:
            expected_reward[t] = discount * expected_reward[t + 1] + episode_rewards[t]

    return expected_reward

def tf_get_n_step_expected_rewards(episode_rewards_t, estimates_t, discount, n_step):
    ep_r_shape_0 = tf.shape(episode_rewards_t)[0]
    i = tf.matmul(
        tf.expand_dims(tf.cast(tf.range(ep_r_shape_0), dtype=tf.float32), 1), 
        tf.ones((1, ep_r_shape_0))
    )
    j = tf.transpose(i)
    reward_coefs = (discount**(i-j)) * tf.cast(i >= j, tf.float32) * tf.cast(i - j <= n_step, tf.float32)
    permut_matrix = discount**(n_step + 1) * tf.cast(i > j + n_step, tf.float32) * tf.cast(i <= j + n_step + 1, tf.float32)
    all_n_step_expected_rewards = tf.squeeze(tf.matmul(tf.expand_dims(episode_rewards_t, 0), reward_coefs)) + tf.matmul(tf.expand_dims(estimates_t, 0), permut_matrix)

    return all_n_step_expected_rewards

def get_n_step_expected_rewards(episode_rewards, estimates, discount=.99, n_step=1):
    expected_reward = [0] * len(episode_rewards)
    for t in range(len(episode_rewards)):
        if t + n_step <= len(episode_rewards):
            expected_reward[t] = estimates[t + n_step - 1]
            for t_2 in range(t + n_step - 1, t - 1, -1):
                expected_reward[t] = episode_rewards[t_2] + discount * expected_reward[t]
        else:
            for t_2 in range(len(episode_rewards) - 1, t - 1, -1):
                expected_reward[t] = episode_rewards[t_2] + discount * expected_reward[t]

    return expected_reward

def get_n_step_expected_rewards_mat(episode_rewards, estimates, discount=.99, n_step=1):
    expected_reward = [0] * len(episode_rewards)
    rewards_coef = np.fromfunction(lambda i,j: discount**(i-j) * (i >= j) * (i - j < n_step), (len(episode_rewards), len(episode_rewards)))
    permut = np.fromfunction(lambda i,j: (i > j + n_step) * (i <= j + n_step), (len(episode_rewards), len(episode_rewards)))

    return np.dot(episode_rewards, rewards_coef) + discount**(n_step) * np.dot(estimates, permut)

def get_lambda_expected_rewards(episode_rewards, estimates, discount=.99, lambda_value=.9):
    if lambda_value == 1.: # In this case this leads to MC 
        return get_expected_rewards(episode_rewards, discount)

    expected_reward = np.array([0.] * len(episode_rewards))
    for i in range(1, len(episode_rewards) + 1):
        n_step_returns = np.array(get_n_step_expected_rewards(episode_rewards, estimates, discount, i))
        if i == len(episode_rewards):
            expected_reward += lambda_value**(i-1) * np.array(get_expected_rewards(episode_rewards, discount))
        else:
            expected_reward += (1-lambda_value) * lambda_value**(i-1) * n_step_returns

    return expected_reward

def get_mc_target(rewards_t, discount):
    discounts = discount ** tf.cast(tf.range(tf.shape(rewards_t)[0]), dtype=tf.float32)
    epsilon = 1e-7
    return tf.cumsum(rewards_t * discounts, reverse=True) / (discounts + epsilon)

def get_td_target(Qs_t, rewards_t, next_states_t, next_actions_t, discount):
    state_action_pairs = tf.stack([next_states_t, next_actions_t], 1)
    next_estimates = tf.gather_nd(Qs_t, state_action_pairs)
    return tf.stop_gradient(rewards_t + discount * next_estimates, name='td_target')

def get_td_n_target(Qs_t, rewards_t, next_state_t, next_action_t, discount, n_step):
    cond = tf.greater(tf.shape(rewards_t)[0] > n_step)
    estimate = tf.where(cond, discount**(n_step + 1) * Qs_t[next_state_t, next_action_t], 0)
    discounts = discount ** tf.cast(tf.range(n_step + 1), dtype=tf.float32)
    return tf.stop_gradient(tf.reduce_sum(rewards_t[- (n_step + 1):] * discounts) + estimate)

def get_q_learning_target(Qs_t, rewards_t, next_states_t, discount):
    next_qs = tf.gather(Qs_t, next_states_t)
    next_max_actions_t = tf.cast(tf.argmax(next_qs, 1), tf.int32)
    state_action_pairs = tf.stack([next_states_t, next_max_actions_t], 1)
    next_estimates = tf.gather_nd(Qs_t, state_action_pairs)
    return tf.stop_gradient(rewards_t + discount * next_estimates, name='q_learning_target')

def get_expected_sarsa_target(Qs_t, rewards_t, next_states_t, next_probs_t, discount):
    next_qs = tf.gather(Qs_t, next_states_t)
    next_estimates = tf.reduce_sum(next_qs * next_probs_t, 1)
    return tf.stop_gradient(rewards_t + discount * next_estimates, name='expected_sarsa_target')

def get_sigma_target(Qs_t, sigma, rewards_t, next_states_t, next_actions_t, next_probs_t, discount):
    next_qs = tf.gather(Qs_t, next_states_t)
    next_expected_sarsa_estimates = tf.reduce_sum(next_qs * next_probs_t, 1)

    state_action_pairs = tf.stack([next_states_t, next_actions_t], 1)
    next_td_estimates = tf.gather_nd(Qs_t, state_action_pairs)

    next_estimates = sigma * next_td_estimates + (1 - sigma) * next_expected_sarsa_estimates
    return tf.stop_gradient(rewards_t + discount * next_estimates, name='sigma_target')

def tabular_learning(Qs_t, states_t, actions_t, targets):
    reusing_scope = tf.get_variable_scope().reuse

    state_action_pairs = tf.stack([states_t, actions_t], 1)
    estimates = tf.gather_nd(Qs_t, state_action_pairs)
    err_estimates = targets - estimates
    loss = tf.reduce_mean(err_estimates)

    Nsa = tf.get_variable(
        "Nsa"
        , shape=Qs_t.get_shape()
        , dtype=tf.float32
        , trainable=False
        , initializer=tf.zeros_initializer()
    )
    if reusing_scope is False:
        tf.summary.histogram('Nsa', Nsa)

    update_Nsa = tf.scatter_nd_add(Nsa, state_action_pairs, tf.ones_like(states_t, dtype=tf.float32))
    global_step = tf.Variable(0, trainable=False, name="global_step", collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
    inc_global_step = global_step.assign_add(1)
    with tf.control_dependencies([update_Nsa, inc_global_step]):
        epsilon = 1e-7
        lr = (1 / ( epsilon + tf.gather_nd(Nsa, state_action_pairs)) )
        updates = lr * err_estimates
        train_op = tf.scatter_nd_add(Qs_t, state_action_pairs, updates)

    return loss, train_op

def tabular_learning_with_lr(init_lr, decay_steps, Qs_t, states_t, actions_t, targets):
    reusing_scope = tf.get_variable_scope().reuse

    state_action_pairs = tf.stack([states_t, actions_t], 1)
    estimates = tf.gather_nd(Qs_t, state_action_pairs)
    err_estimates = targets - estimates
    loss = tf.reduce_mean(err_estimates)

    global_step = tf.Variable(0, trainable=False, name="global_step", collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
    lr = tf.train.exponential_decay(tf.constant(init_lr, dtype=tf.float32), global_step, decay_steps, 0.5, staircase=True)
    if reusing_scope is False:
        tf.summary.scalar('lr', lr)
    inc_global_step = global_step.assign_add(1)
    with tf.control_dependencies([inc_global_step]):
        updates = lr * err_estimates
        train_op = tf.scatter_nd_add(Qs_t, state_action_pairs, updates)

    return loss, train_op

def counter(name):
    count_t = tf.get_variable(name, shape=[], trainable=False, dtype=tf.int32, initializer=tf.zeros_initializer())
    inc_count_op = count_t.assign_add(1)

    return (count_t, inc_count_op)

def fix_scope(from_scope):
    update_fixed_vars_op = []
    for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=from_scope.name):
        fixed = tf.get_variable(
            var.name.split(":")[0].strip(from_scope.name + "/")
            , shape=var.get_shape()
            , initializer=tf.constant_initializer(0.)
            , trainable=False
            , dtype=var.dtype
        )
        assign_op = fixed.assign(var)
        update_fixed_vars_op.append(assign_op)

    return update_fixed_vars_op

def build_batches(dtKeys, sequence_history, batch_size):
    sequence_history = np.random.permutation(sequence_history)

    batches = []
    for i in range(0, len(sequence_history), batch_size):
        history_chunk = sequence_history[i:i + batch_size]
        max_seq_len = len(max(history_chunk, key=lambda x: len(x)))
        
        batch = { key: [] for key in dtKeys }
        batch['mask'] = []
        for history in history_chunk:
            seq_len = len(history)
            for key in dtKeys:
                padded_data = np.lib.pad(history[key], [[0, max_seq_len - seq_len], [0, 0]], 'constant', constant_values=(0))
                batch[key].append(padded_data)
            batch['mask'].append([[1]] * seq_len + [[0]] * (max_seq_len - seq_len))

        batches.append(batch)

    return batches

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

    w_init = tf.random_normal_initializer(mean=network_params['initial_mean'], stddev=network_params['initial_stddev'])
    b_init = tf.zeros_initializer()  

    W1 = tf.get_variable('W1'
        , shape=[ network_params['nb_inputs'], network_params['nb_units'] ]
        , initializer=w_init
    )
    if reusing_scope is False:
        tf.summary.histogram('W1', W1)
    b1 = tf.get_variable('b1'
        , shape=[ network_params['nb_units'] ]
        , initializer=b_init
    )
    if reusing_scope is False:
        tf.summary.histogram('b1', b1)
    a1 = tf.nn.relu(tf.matmul(inputs, W1) + b1)

    W2 = tf.get_variable('W2'
        , shape=[ network_params['nb_units'], network_params['nb_units'] ]
        , initializer=w_init
    )
    if reusing_scope is False:
        tf.summary.histogram('W2', W2)
    b2 = tf.get_variable('b2'
        , shape=[ network_params['nb_units'] ]
        , initializer=b_init
    )
    if reusing_scope is False:
        tf.summary.histogram('b2', b2)
    a2 = tf.nn.relu(tf.matmul(a1, W2) + b2)

    W3 = tf.get_variable('W3'
        , shape=[ network_params['nb_units'], network_params['nb_outputs'] ]
        , initializer=w_init
    )
    if reusing_scope is False:
        tf.summary.histogram('W3', W3)
    b3 = tf.get_variable('b3'
        , shape=[ network_params['nb_outputs'] ]
        , initializer=b_init
    )
    if reusing_scope is False:
        tf.summary.histogram('b3', b3)
    values = tf.matmul(a2, W3) + b3

    return values

def predictive_model(network_params, m_inputs, dynamic_batch_size, m_initial_state=None):
    reusing_scope = tf.get_variable_scope().reuse

    m_cell = LSTMCell(
        num_units=network_params['nb_units']
        , initializer=tf.truncated_normal_initializer(
            mean=network_params['initial_mean']
            , stddev=network_params['initial_stddev']
        )
        , reuse=reusing_scope
    )
    # The first time you call this function, you obtain a placeholder
    # for the initial state
    # The second time (if you reuse the variables), you need to pass in the initial state
    if not reusing_scope:
        m_initial_state = m_cell.zero_state(dynamic_batch_size, dtype=tf.float32)
    m_c_h_states, m_final_state = tf.nn.dynamic_rnn(m_cell, m_inputs, initial_state=m_initial_state)
    m_c_states, m_h_states = tf.split(value=m_c_h_states, num_or_size_splits=[network_params['nb_units'], network_params['nb_units']], axis=2)

    # Compute the Controller projection
    WM_proj = tf.get_variable(
        "WM_proj"
        , shape=[network_params['nb_units'], network_params['env_state_size'] + 1]
        , dtype=tf.float32
        , initializer=tf.truncated_normal_initializer(
            mean=network_params['initial_mean']
            , stddev=network_params['initial_stddev']
        )
    )
    m_h_states_mat = tf.reshape(m_h_states, [-1, network_params['nb_units']])
    state_reward_preds_mat = tf.matmul(m_h_states_mat, WM_proj) 
    state_reward_preds = tf.reshape(state_reward_preds_mat, [dynamic_batch_size, -1, network_params['env_state_size'] + 1], name="actions_t")

    return state_reward_preds, m_final_state, m_initial_state

# This is weird, it seems to trouble the learning
def projection(proj_params, inputs):
    reusing_scope = tf.get_variable_scope().reuse

    input_shape = tf.shape(inputs)
    dynamic_batch_size, dynamic_num_steps = input_shape[0], input_shape[1]

    WC_proj = tf.get_variable(
        "WC_proj"
        , shape=[proj_params['nb_units'], proj_params['nb_actions']]
        , dtype=tf.float32
        , initializer=tf.truncated_normal_initializer(
            mean=proj_params['initial_mean']
            , stddev=proj_params['initial_stddev']
        )
    )
    bC_proj = tf.get_variable("bC_proj", shape=[proj_params['nb_actions']], dtype=tf.float32)
    inputs_mat = tf.reshape(inputs, [-1, proj_params['nb_units']])
    logits = tf.matmul(inputs_mat, WC_proj)

    preds_mat = tf.nn.softmax(logits, 1)
    preds_t = tf.reshape(preds_mat, [dynamic_batch_size, -1, proj_params['nb_actions']], name="preds_t")

    actions_mat = tf.cast(tf.multinomial(logits, 1), tf.int32)
    actions_t = tf.reshape(actions_mat, [dynamic_batch_size, -1, 1], name="actions_t")

    return preds_t, actions_t
