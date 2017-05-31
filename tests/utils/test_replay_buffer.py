import os, sys, unittest
import numpy as np
import tensorflow as tf

dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir + '/../..')

from utils import replay_buffer

class TestReplayBuffer(unittest.TestCase):

    def test_replay_buffer_init(self):
        templates = [('test1', tf.int32, ()), ('test2', tf.int32, (1,)), ('test3', tf.int32, (2, 2))]
        capacity = 2
        rep_buf = replay_buffer.ReplayBuffer(templates, capacity)

        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        buffers = sess.run(rep_buf.buffers)

        self.assertEqual(np.array_equal(buffers['test1'], [0, 0]), True)
        self.assertEqual(np.array_equal(buffers['test2'], [ [0], [0] ]), True)
        self.assertEqual(np.array_equal(buffers['test3'], [ [[0, 0],[0, 0]], [[0, 0],[0, 0]] ]), True)

    def test_replay_buffer_append(self):
        templates = [('test1', tf.int32, ()), ('test2', tf.int32, (1,)), ('test3', tf.int32, (2, 2))]
        capacity = 2
        rep_buf = replay_buffer.ReplayBuffer(templates, capacity)
        memory = [tf.constant(1), tf.constant([2]), tf.constant([[1,2], [3,4]])]
        inc_index_op = rep_buf.append(memory)

        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        sess.run(inc_index_op)
        index, buffers = sess.run([rep_buf.index, rep_buf.buffers])

        self.assertEqual(index, 1)
        self.assertEqual(np.array_equal(buffers['test1'], [1, 0]), True)
        self.assertEqual(np.array_equal(buffers['test2'], [ [2], [0] ]), True)
        self.assertEqual(np.array_equal(buffers['test3'], [ [[1, 2],[3, 4]], [[0, 0],[0, 0]] ]), True)

    def test_replay_buffer_sample(self):
        templates = [('test1', tf.int32, ()), ('test2', tf.int32, (1,)), ('test3', tf.int32, (2, 2))]
        capacity = 5
        rep_buf = replay_buffer.ReplayBuffer(templates, capacity)
        memory = [
            tf.Variable(1, dtype=tf.int32, trainable=False), 
            tf.Variable([2], dtype=tf.int32, trainable=False), 
            tf.Variable([[1,2], [3,4]], dtype=tf.int32, trainable=False)
        ]
        with tf.control_dependencies([v.assign(v + 1) for v in memory]):
            inc_index_op = rep_buf.append(memory)
        samples_t = rep_buf.sample(5)

        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        sess.run(inc_index_op)
        sess.run(inc_index_op)
        sess.run(inc_index_op)
        sess.run(inc_index_op)
        sess.run(inc_index_op)
        samples = sess.run(samples_t)

        self.assertEqual(len(samples), 3)
        self.assertEqual(len(samples[0]), 5)

    def test_prioritized_replay_buffer(self):
        templates = [('test1', tf.int32, ()), ('test2', tf.int32, (1,)), ('test3', tf.int32, (2, 2))]
        capacity = 5
        p_rep_buf = replay_buffer.PrioritizedReplayBuffer(templates, capacity)
        memory = [
            tf.Variable(1, dtype=tf.int32, trainable=False), 
            tf.Variable([2], dtype=tf.int32, trainable=False), 
            tf.Variable([[1,2], [3,4]], dtype=tf.int32, trainable=False)
        ]
        priority = tf.Variable(1, dtype=tf.float32, trainable=False, name="memory")
        with tf.control_dependencies([v.assign(v + 1) for v in memory] + [priority.assign(priority + 1)]):
            inc_index_op = p_rep_buf.append(priority, memory)
        samples_t = p_rep_buf.sample(5)

        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        sess.run(inc_index_op)
        sess.run(inc_index_op)
        sess.run(inc_index_op)
        sess.run(inc_index_op)
        sess.run(inc_index_op)
        samples = sess.run(samples_t)

        self.assertEqual(len(samples), 3)
        self.assertEqual(len(samples[0]), 5)
        

if __name__ == "__main__":
    unittest.main()