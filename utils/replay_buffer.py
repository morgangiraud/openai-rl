import tensorflow as tf

# Inspired by http://web.stanford.edu/class/cs20si/lectures/slides_14.pdf

class ReplayBuffer:

    def __init__(self, templates, capacity):
        self.capacity = capacity
        self.buffers = self._create_buffers(templates)
        self.index = tf.Variable(0, dtype=tf.int32, trainable=False)

    def _create_buffers(self, templates):
        buffers = {}
        for name, dtype, shape in templates:
            final_shape = tf.TensorShape([self.capacity]).concatenate(shape)
            buf = tf.Variable(
                tf.zeros(final_shape, dtype=dtype)
                , trainable=False
                , dtype=dtype
                , name=name
            )
            buffers[name] = buf
        return buffers

    def size(self):
        return tf.minimum(self.index, self.capacity)

    def append(self, tensors):
        position = tf.mod(self.index, self.capacity)
        append_ops = [self.buffers[key][position].assign(tensor) for key, tensor in zip(self.buffers, tensors)]
        with tf.control_dependencies(append_ops):
            inc_index_op = self.index.assign_add(1)

        return inc_index_op

    def sample(self, nb_samples):
        positions = tf.random_uniform( (nb_samples,), 0, self.size() - 1, tf.int32 )
        samples = [tf.gather(b, positions) for key,b in self.buffers.items()]

        return samples

    def get_all_buffers(self):
        size = self.size()
        return [b[:size] for key,b in self.buffers.items()]


class PrioritizedReplayBuffer:

    def __init__(self, templates, capactiy):
        templates = [('priority', tf.float32, ())] + list(templates)
        self.buffer = ReplayBuffer(templates, capactiy)

    def size(self):
        return self.buffer.size()

    def append(self, priority, tensors):
        return self.buffer.append([priority] + list(tensors))

    def sample(self, amount, temperature=1):
        priorities = self.buffer.buffers['priority'].value()[:self.size()]
        logprobs = tf.log(priorities / tf.reduce_sum(priorities)) / temperature
        positions = tf.multinomial(logprobs[None, ...], amount)[0]
        return [ tf.gather(b, positions) for key,b in self.buffer.buffers.items() if key != 'priority' ]
