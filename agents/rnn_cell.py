import tensorflow as tf

from tensorflow.python.ops.rnn_cell_impl import _linear

class LSTMCell(tf.nn.rnn_cell.RNNCell):
  """Long short-term memory unit (LSTM) recurrent network cell.

  The implementation is based on:

    http://www.bioinf.jku.at/publications/older/2604.pdf

  S. Hochreiter and J. Schmidhuber.
  "Long Short-Term Memory". Neural Computation, 9(8):1735-1780, 1997.

  """

  def __init__(self, num_units, cell_clip=None,
               initializer=None, num_proj=None, proj_clip=None,
               forget_bias=1.0, state_is_tuple=True,
               activation=None, reuse=None, trainable=True):
    """Initialize the parameters for an LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell
      cell_clip: (optional) A float value, if provided the cell state is clipped
        by this value prior to the cell output activation.
      initializer: (optional) The initializer to use for the weight and
        projection matrices.
      num_proj: (optional) int, The output dimensionality for the projection
        matrices.  If None, no projection is performed.
      proj_clip: (optional) A float value.  If `num_proj > 0` and `proj_clip` is
        provided, then the projected values are clipped elementwise to within
        `[-proj_clip, proj_clip]`.
      forget_bias: Biases of the forget gate are initialized by default to 1
        in order to reduce the scale of forgetting at the beginning of
        the training. Must set it manually to `0.0` when restoring from
        CudnnLSTM trained checkpoints.
      state_is_tuple: If True, accepted and returned states are 2-tuples of
        the `c_state` and `m_state`.  If False, they are concatenated
        along the column axis.  This latter behavior will soon be deprecated.
      activation: Activation function of the inner states.  Default: `tanh`.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.

      When restoring from CudnnLSTM-trained checkpoints, must use
      CudnnCompatibleLSTMCell instead.
    """
    super(LSTMCell, self).__init__(_reuse=reuse)
    if not state_is_tuple:
      logging.warn("%s: Using a concatenated state is slower and will soon be "
                   "deprecated.  Use state_is_tuple=True.", self)

    self._num_units = num_units
    self._cell_clip = cell_clip
    self._initializer = initializer
    self._num_proj = num_proj
    self._proj_clip = proj_clip
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._activation = activation or tf.tanh

    if num_proj:
      self._state_size = (
          tf.nn.rnn_cell.LSTMStateTuple(num_units, num_proj)
          if state_is_tuple else num_units + num_proj)
      self._output_size = num_units + num_proj # cell units + projected h units
    else:
      self._state_size = (
          tf.nn.rnn_cell.LSTMStateTuple(num_units, num_units)
          if state_is_tuple else 2 * num_units)
      self._output_size = num_units + num_units # cell units + h units

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

  def call(self, inputs, state):
    """Run one step of LSTM.

    Args:
      inputs: input Tensor, 2D, batch x num_units.
      state: if `state_is_tuple` is False, this must be a state Tensor,
        `2-D, batch x state_size`.  If `state_is_tuple` is True, this must be a
        tuple of state Tensors, both `2-D`, with column sizes `c_state` and
        `m_state`.

    Returns:
      A tuple containing:

      - A `2-D, [batch x output_dim]`, Tensor representing the output of the
        LSTM after reading `inputs` when previous state was `state`.
        Here output_dim is:
           num_proj if num_proj was set,
           num_units otherwise.
      - Tensor(s) representing the new state of LSTM after reading `inputs` when
        the previous state was `state`.  Same type and shape(s) as `state`.

    Raises:
      ValueError: If input size cannot be inferred from inputs via
        static shape inference.
    """
    num_proj = self._num_units if self._num_proj is None else self._num_proj
    sigmoid = tf.sigmoid

    if self._state_is_tuple:
      (c_prev, m_prev) = state
    else:
      c_prev = tf.slice(state, [0, 0], [-1, self._num_units])
      m_prev = tf.slice(state, [0, self._num_units], [-1, num_proj])

    dtype = inputs.dtype
    input_size = inputs.get_shape().with_rank(2)[1]
    if input_size.value is None:
      raise ValueError("Could not infer input size from inputs.get_shape()[-1]")
    scope = tf.get_variable_scope()
    with tf.variable_scope(scope, initializer=self._initializer) as unit_scope:
      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      lstm_matrix = _linear([inputs, m_prev], 4 * self._num_units, bias=True)
      i, j, f, o = tf.split(
          value=lstm_matrix, num_or_size_splits=4, axis=1)

      c = (sigmoid(f + self._forget_bias) * c_prev + sigmoid(i) *
           self._activation(j))

      if self._cell_clip is not None:
        # pylint: disable=invalid-unary-operand-type
        c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)
        # pylint: enable=invalid-unary-operand-type
      
      m = sigmoid(o) * self._activation(c)

      if self._num_proj is not None:
        with tf.variable_scope("projection") as proj_scope:
          m = _linear(m, self._num_proj, bias=False)

        if self._proj_clip is not None:
          # pylint: disable=invalid-unary-operand-type
          m = clip_ops.clip_by_value(m, -self._proj_clip, self._proj_clip)
          # pylint: enable=invalid-unary-operand-type

    outputs = tf.concat([c, m], 1)
    new_state = (tf.nn.rnn_cell.LSTMStateTuple(c, m) if self._state_is_tuple else
                 tf.concat([c, m], 1))
    return outputs, new_state
 
class CMCell(tf.nn.rnn_cell.RNNCell):

  def __init__(self, num_units, m_units,
               fixed_model_scope, model_func,
               projection_func, num_proj, cell_clip=None,
               initializer=None, forget_bias=1.0, state_is_tuple=True,
               activation=None, reuse=None,
               use_query_and_answer_topology=True):

    super(CMCell, self).__init__(_reuse=reuse)
    if not state_is_tuple:
      logging.warn("%s: Using a concatenated state is slower and will soon be "
                   "deprecated.  Use state_is_tuple=True.", self)

    self._num_units = num_units
    self._m_units = m_units
    self._num_proj = num_proj
    self._cell_clip = cell_clip
    self._initializer = initializer
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._activation = activation or tf.tanh

    self._state_size = (
        tf.nn.rnn_cell.LSTMStateTuple(2*num_units, 2*m_units)
        if state_is_tuple else 2 * num_units)
    self._output_size = num_proj + 1 # nb_actions + selected action

    self._model_func = model_func
    self._fixed_model_scope = fixed_model_scope
    self._projection_func = projection_func

    self._use_query_and_answer_topology = use_query_and_answer_topology
    

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

  def zero_state(self, batch_size, dtype):
      c_state, m_state = super(CMCell, self).zero_state(batch_size, dtype)

      c_state = tf.nn.rnn_cell.LSTMStateTuple(
        tf.slice(c_state, [0, 0], [-1, self._num_units])
        , tf.slice(c_state, [0, self._num_units], [-1, self._num_units])
      )
      m_state = tf.nn.rnn_cell.LSTMStateTuple(
        tf.slice(m_state, [0, 0], [-1, self._m_units])
        , tf.slice(m_state, [0, self._m_units], [-1, self._m_units])
      )

      return tf.nn.rnn_cell.LSTMStateTuple(c_state, m_state)

  def _call_controller_cell(self, inputs, state):
    """Run one step of LSTM.

    Args:
      inputs: input Tensor, 2D, batch x num_units.
      state: if `state_is_tuple` is False, this must be a state Tensor,
        `2-D, batch x state_size`.  If `state_is_tuple` is True, this must be a
        tuple of state Tensors, both `2-D`, with column sizes `c_state` and
        `m_state`.

    Returns:
      A tuple containing:

      - A `2-D, [batch x output_dim]`, Tensor representing the output of the
        LSTM after reading `inputs` when previous state was `state`.
        Here output_dim is:
           num_proj if num_proj was set,
           num_units otherwise.
      - Tensor(s) representing the new state of LSTM after reading `inputs` when
        the previous state was `state`.  Same type and shape(s) as `state`.

    Raises:
      ValueError: If input size cannot be inferred from inputs via
        static shape inference.
    """
    sigmoid = tf.sigmoid

    if self._state_is_tuple:
      (c_prev, m_prev) = state
    else:
      c_rev, m_prev = tf.split(value=state, num_or_size_splits=2, axis=1)

    dtype = inputs.dtype
    input_size = inputs.get_shape().with_rank(2)[1]
    if input_size.value is None:
      raise ValueError("Could not infer input size from inputs.get_shape()[-1]")
    scope = tf.get_variable_scope()
    with tf.variable_scope(scope, initializer=self._initializer) as unit_scope:
      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      lstm_matrix = _linear([inputs, m_prev], 4 * self._num_units, bias=True)
      i, j, f, o = tf.split(
          value=lstm_matrix, num_or_size_splits=4, axis=1)

      c = (sigmoid(f + self._forget_bias) * c_prev + sigmoid(i) *
           self._activation(j))

      if self._cell_clip is not None:
        # pylint: disable=invalid-unary-operand-type
        c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)
        # pylint: enable=invalid-unary-operand-type
      
      h = sigmoid(o) * self._activation(c)
      

    outputs = tf.concat([c, h], 1)
    new_state = (tf.nn.rnn_cell.LSTMStateTuple(c, h) if self._state_is_tuple else
                 tf.concat([c, h], 1))
    return outputs, new_state
 
  def _call_controller_cell_mul(self, inputs, state):
    """Long short-term memory cell (LSTM).

    Args:
      inputs: `2-D` tensor with shape `[batch_size x input_size]`.
      state: An `LSTMStateTuple` of state tensors, each shaped
        `[batch_size x self.state_size]`, if `state_is_tuple` has been set to
        `True`.  Otherwise, a `Tensor` shaped
        `[batch_size x 2 * self.state_size]`.

    Returns:
      A pair containing the new hidden state, and the new state (either a
        `LSTMStateTuple` or a concatenated state, depending on
        `state_is_tuple`).
    """
    sigmoid = tf.sigmoid
    # Parameters of gates are concatenated into one multiply for efficiency.
    if self._state_is_tuple:
      c_prev, h_prev = state
    else:
      c_prev, h_prev = tf.split(value=state, num_or_size_splits=2, axis=1)

    dtype = inputs.dtype
    input_size = inputs.get_shape().with_rank(2)[1]
    if input_size.value is None:
      raise ValueError("Could not infer input size from inputs.get_shape()[-1]")

    scope = tf.get_variable_scope()
    with tf.variable_scope(scope, initializer=self._initializer) as outer_scope:
      W_hi = tf.get_variable('W_hi', [self._num_units, self._num_units])
      W_xi = tf.get_variable('W_xi', [inputs.shape[1].value, self._num_units])
      W_hj = tf.get_variable('W_hj', [self._num_units, self._num_units])
      W_xj = tf.get_variable('W_xj', [inputs.shape[1].value, self._num_units])
      W_hf = tf.get_variable('W_hf', [self._num_units, self._num_units])
      W_xf = tf.get_variable('W_xf', [inputs.shape[1].value, self._num_units])
      W_ho = tf.get_variable('W_ho', [self._num_units, self._num_units])
      W_xo = tf.get_variable('W_xo', [inputs.shape[1].value, self._num_units])
        
      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      i = tf.matmul(h_prev, W_hi) * tf.matmul(inputs, W_xi)
      j = tf.matmul(h_prev, W_hj) * tf.matmul(inputs, W_xj)
      f = tf.matmul(h_prev, W_hf) * tf.matmul(inputs, W_xf)
      o = tf.matmul(h_prev, W_ho) * tf.matmul(inputs, W_xo)

      c = (c_prev * sigmoid(f + self._forget_bias) + sigmoid(i) *
           self._activation(j))
      h = self._activation(c) * sigmoid(o)


    outputs = tf.concat([c, h], 1)
    new_state = (tf.nn.rnn_cell.LSTMStateTuple(c, h) if self._state_is_tuple else
                 tf.concat([c, h], 1))
    return outputs, new_state

  def call(self, inputs, cm_state):
    if self._state_is_tuple:
      (c_state, m_state) = cm_state
    else:
      c_state = tf.slice(cm_state, [0, 0], [-1, self._num_units * 2])
      m_state = tf.slice(cm_state, [0, self._num_units * 2], [-1, self._m_units])

    # We use the controller cell to predict the action distribution
    c_scope = tf.get_variable_scope()
    with tf.variable_scope(c_scope):
      # outputs: [controller memory cell, controller state]
      # new_c_state: LSTMTuple([controller memory cell, controller state])
      # c_inputs = tf.concat([inputs, tf.ones(tf.shape(inputs)) * -1], 1)
      c_inputs = inputs
      c_tmp_outputs, c_tmp_state = self._call_controller_cell(c_inputs, c_state)
    c_tmp_c, c_tmp_h = tf.split(value=c_tmp_outputs, num_or_size_splits=[self._num_units, self._num_units], axis=1)


    c_projection_scope = tf.VariableScope(reuse=False, name="c_projection")
    with tf.variable_scope(c_projection_scope):
      action_distribution, _ = self._projection_func(c_tmp_h)

    if self._use_query_and_answer_topology:
      query_scope = tf.VariableScope(reuse=False, name="query")
      with tf.variable_scope(query_scope):
        m_c_state, m_h_state = m_state

        a1 = tf.layers.dense(
          tf.concat([m_c_state, c_tmp_h], 1), self._num_units, tf.nn.relu)
        a2 = tf.layers.dense(a1, self._num_units, tf.nn.relu)
        edited_m_c_state = tf.layers.dense(a2, m_c_state.get_shape()[1], tf.tanh)

      edited_m_state = tf.nn.rnn_cell.LSTMStateTuple(edited_m_c_state, m_h_state)

    with tf.variable_scope(self._fixed_model_scope, reuse=True):
      model_inputs = tf.concat([
        tf.expand_dims(inputs, 1), 
        tf.expand_dims(action_distribution, 1)]
      , 2)
      if self._use_query_and_answer_topology:
        state_reward_preds_seq, m_tmp_state, _ = self._model_func(model_inputs, edited_m_state)
      else:
        state_reward_preds_seq, m_tmp_state, _ = self._model_func(model_inputs, m_state)
      state_reward_preds = tf.squeeze(state_reward_preds_seq, 1)
    state_preds = state_reward_preds[:, :-1]

    if self._use_query_and_answer_topology:
      answer_scope = tf.VariableScope(reuse=False, name="answer")
      with tf.variable_scope(answer_scope):
        m_tmp_c_state, m_tmp_h_state = m_tmp_state

        a1 = tf.layers.dense(tf.concat([m_tmp_h_state, state_reward_preds], 1), self._num_units, tf.nn.relu)
        a2 = tf.layers.dense(a1, self._num_units, tf.nn.relu)
        answer = tf.layers.dense(a2, inputs.get_shape()[1])

    # We use the next state, to finally predict the new distribution over action
    # And the selected action
    with tf.variable_scope(c_scope, reuse=True):
      if not self._use_query_and_answer_topology:
        # answer = tf.concat([state_preds, tf.ones(tf.shape(state_preds))], 1)
        answer = state_preds
      outputs, c_final_state = self._call_controller_cell(answer, c_tmp_state)
    c, h = tf.split(value=outputs, num_or_size_splits=[self._num_units, self._num_units], axis=1)

    with tf.variable_scope(c_projection_scope, reuse=True):
      preds_t, actions_t = self._projection_func(h)

    # Finally we use the chosen action, the input state, and the current model state
    # To compute the next model state
    with tf.variable_scope(self._fixed_model_scope, reuse=True):
      action_input = tf.one_hot(indices=actions_t, depth=self._num_proj)
      model_inputs = tf.concat([tf.expand_dims(inputs, 1), action_input], 2)

      _, m_final_state, _ = self._model_func(model_inputs, m_state)
      
    final_outputs = tf.concat([preds_t, tf.cast(actions_t, tf.float32)], 1)

    new_cm_state = (tf.nn.rnn_cell.LSTMStateTuple(c_final_state, m_final_state) if self._state_is_tuple else
          tf.concat([c_final_state, m_final_state], 1))

    return final_outputs, new_cm_state
 