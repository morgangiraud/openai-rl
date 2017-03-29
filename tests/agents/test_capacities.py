import os, sys, unittest
import numpy as np
import tensorflow as tf

dir = os.path.dirname(os.path.realpath(__file__))

from agents import capacities

class TestCapacities(unittest.TestCase):

    def test_tabular_q_value(self):
        nb_state = 3
        nb_action = 2

        with tf.Graph().as_default():
            inputs = tf.placeholder(tf.int32, shape=[None])
            apply = capacities.tabularQValue(nb_state, nb_action)
            out_t = apply(inputs)

            with tf.variable_scope('test'):
                inputs2 = tf.placeholder(tf.int32, shape=[None, 1])
                out_t2 = apply(inputs2)


            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.assign(tf.trainable_variables()[0], [ [0., 1.], [2., 3.], [4., 5.]]))
                out = sess.run(out_t, feed_dict={
                    inputs: [ 0, 1 ]
                })
                self.assertEqual(np.array_equal(out, [ [0., 1.], [2., 3.] ]), True)

                out2 = sess.run(out_t2, feed_dict={
                    inputs2: [ [1], [2] ]
                })
                self.assertEqual(np.array_equal(out2, [ [ [2., 3.] ], [ [4., 5.] ] ]), True)

    def test_policy(self):
        policy_params = {
            'nb_inputs': 3
            , 'nb_units': 3
            , 'nb_actions': 2
        }
        with tf.Graph().as_default():
            tf.set_random_seed(1)

            inputs = tf.placeholder(tf.float32, shape=[None, 3])
            probs_t, actions_t = capacities.policy(policy_params, inputs)
            

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                
                probs, actions = sess.run([probs_t, actions_t], feed_dict={
                    inputs: [ [ -24, 10, 10 ]]
                })
                self.assertEqual(np.array_equal(np.round(probs, 1), [[ 0.5 , 0.5]]), True)
                self.assertEqual(np.array_equal(actions, [[0]]), True)


if __name__ == "__main__":
    unittest.main()