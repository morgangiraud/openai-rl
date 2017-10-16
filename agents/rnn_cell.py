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
               activation=None, reuse=None):

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
      c_prev = tf.slice(state, [0, 0], [-1, self._num_units])
      m_prev = tf.slice(state, [0, self._num_units], [-1, self._num_units])

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
 
  def call(self, input_state, cm_state):
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
      c_tmp_outputs, c_tmp_state = self._call_controller_cell(input_state, c_state)
    c_tmp_c, c_tmp_h = tf.split(value=c_tmp_outputs, num_or_size_splits=[self._num_units, self._num_units], axis=1)

    c_projection_scope = tf.VariableScope(reuse=False, name="c")
    with tf.variable_scope(c_projection_scope):
      tmp_preds_t, _ = self._projection_func(tf.expand_dims(c_tmp_h, 1))

    # We use the action distribution, previous model state and input state
    # To compute the prediction for the next state
    with tf.variable_scope(self._fixed_model_scope, reuse=True):
      model_inputs = tf.concat([tf.expand_dims(input_state, 1), tmp_preds_t], 2)
      # It's invisible but we have a placeholder for the model initial state
      state_reward_preds_seq, _, _ = self._model_func(model_inputs, m_state)
      state_reward_preds = tf.squeeze(state_reward_preds_seq, 1)
    state_preds = state_reward_preds[:, :-1]

    # We use the next state, to finally predict the new distribution over action
    # And the selected action
    with tf.variable_scope(c_scope, reuse=True):
      outputs, c_final_state = self._call_controller_cell(state_preds, c_tmp_state)
    c, h = tf.split(value=outputs, num_or_size_splits=[self._num_units, self._num_units], axis=1)

    with tf.variable_scope(c_projection_scope, reuse=True):
      preds_t, actions_t = self._projection_func(tf.expand_dims(h, 1))

    # Finally we use the chosen action, the input state, and the current model state
    # To compute the next model state
    with tf.variable_scope(self._fixed_model_scope, reuse=True):
      action_input = tf.one_hot(
        indices=tf.squeeze(actions_t, 2), depth=self._num_proj
      )
      model_inputs = tf.concat([tf.expand_dims(input_state, 1), action_input], 2)
      _, m_final_state, _ = self._model_func(model_inputs, m_state)
      

    final_outputs = tf.concat(
      [tf.squeeze(preds_t, 1), tf.squeeze(tf.cast(actions_t, tf.float32), 1)]
      , 1
      )

    new_cm_state = (tf.nn.rnn_cell.LSTMStateTuple(c_final_state, m_final_state) if self._state_is_tuple else
          tf.concat([c_final_state, m_final_state], 1))

    return final_outputs, new_cm_state
 

