import os, sys, unittest, timeit
import numpy as np
import tensorflow as tf

dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir + '/../..')

from agents import capacities

class TestCapacities(unittest.TestCase):

    def test_tabular_eps_greedy(self):
        nb_states = 3
        nb_actions = 2

        with tf.Graph().as_default():
            tf.set_random_seed(1)

            inputs = tf.random_uniform(shape=[2], minval=0, maxval=3, dtype=tf.int32)
            # inputs = tf.Print(inputs, data=[inputs], message='inputs')

            Qs = tf.random_uniform(shape=[nb_states, nb_actions])
            # Qs = tf.Print(Qs, data=[Qs], message='Qs', summarize=12)

            q_preds = tf.gather(Qs, inputs)
            # q_preds = tf.Print(q_preds, data=[q_preds], message='q_preds', summarize=6)

            N0 = 100
            actions, probs = capacities.tabular_eps_greedy(inputs, q_preds, nb_actions, N0, 0., nb_states)
            # actions = tf.Print(actions, data=[actions], message='actions')
# 
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                inputs, actions = sess.run([inputs, actions])
                self.assertEqual(np.array_equal(inputs, [ 2, 1 ]), True)
                self.assertEqual(np.array_equal(actions, [ 0,  1 ]), True)

    def test_tabular_UCB(self):
        nb_states = 3
        nb_actions = 2

        with tf.Graph().as_default():
            tf.set_random_seed(1)

            inputs = tf.random_uniform(shape=[1], minval=0, maxval=3, dtype=tf.int32)
            inputs = tf.Print(inputs, data=[inputs], message='inputs')

            Qs = tf.ones([nb_states, nb_actions], dtype=tf.float32)
            Qs = tf.Print(Qs, data=[Qs], message='Qs', summarize=12)

            timestep = tf.Variable(0, dtype=tf.int32, trainable=False, name="timestep")
            inc_t = tf.assign_add(timestep, 1)

            with tf.control_dependencies([inc_t]):
                actions, probs = capacities.tabular_UCB(Qs, inputs, timestep, nb_actions)
            actions = tf.Print(actions, data=[timestep, actions], message='actions')
 
            with tf.Session() as sess:
                print(tf.global_variables())
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
                print('biboup')

                inputs, actions = sess.run([inputs, actions])
                inputs, actions = sess.run([inputs, actions])
                inputs, actions = sess.run([inputs, actions])
                self.assertEqual(np.array_equal(inputs, [ 2, 1 ]), True)
                self.assertEqual(np.array_equal(actions, [ 0,  1 ]), True)

    def test_policy(self):
        policy_params = {
            'nb_inputs': 3
            , 'nb_units': 3
            , 'nb_outputs': 2
            , 'initial_mean': 0.
            , 'initial_stddev': .1
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

    def test_eligibility_traces(self):
        with tf.Graph().as_default():
            Qs_t = tf.constant([[1, 1], [1, 1], [1, 1]], dtype=tf.float32)
            states_t = tf.placeholder(tf.int32, shape=[None])
            actions_t = tf.placeholder(tf.int32, shape=[None])
            discount = .9
            lambda_value = .9

            et, update_et_op, reset_et_op = capacities.eligibility_traces(Qs_t, states_t, actions_t, discount, lambda_value)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                _ = sess.run([update_et_op], feed_dict={
                    states_t: [0],
                    actions_t: [1]
                })
                self.assertEqual(np.array_equal(sess.run(et), [[ 0. , 1.], [ 0. , 0.], [ 0. , 0.]]), True)
                _ = sess.run([reset_et_op])
                self.assertEqual(np.array_equal(sess.run(et), [[ 0. , 0.], [ 0. , 0.], [ 0. , 0.]]), True)

    def test_get_mc_target(self):
        discount = .5
        rewards = [1, 1, 2]

        rewards_plh = tf.placeholder(tf.float32, shape=[None])
        target_t = capacities.get_mc_target(rewards_plh, discount)
        with tf.Session() as sess:
            target = sess.run(target_t, feed_dict={rewards_plh: rewards})
            # print('10000 call:', timeit.timeit(lambda: sess.run(target_t, feed_dict={rewards_plh: rewards}), number=10000))

        self.assertEqual(np.sum(np.isclose(target, [2, 2, 2])) == 3, True)

    def test_get_expected_rewards_with_discount(self):
        rewards = [1, 1, 2]
        discount = .5
        
        expected_rewards = capacities.get_expected_rewards(rewards, discount)

        self.assertEqual(np.array_equal(expected_rewards, [2, 2, 2]), True)

    def test_get_n_step_expected_rewards(self):
        rewards = [1, 1, 2, 5]
        estimates = [0.1, 0.2, 0.3, 0]
        discount = 1.
        expected_rewards_td0 = capacities.get_n_step_expected_rewards(rewards, estimates, discount, 1)
        expected_rewards_td1 = capacities.get_n_step_expected_rewards(rewards, estimates, discount, 2)
        expected_rewards_mc = capacities.get_n_step_expected_rewards(rewards, estimates, discount, 5)

        self.assertEqual(np.array_equal(expected_rewards_td0, [1.1, 1.2, 2.3, 5]), True)
        self.assertEqual(np.array_equal(expected_rewards_td1, [2.2, 3.3, 7, 5]), True)
        self.assertEqual(np.array_equal(expected_rewards_mc, [9, 8, 7, 5]), True)

    def test_get_n_step_expected_rewards_with_discount(self):
        rewards = [1, 1, 2, 5]
        estimates = [0.1, 0.2, 0.3, 0]
        discount = .5
        expected_rewards_td0 = capacities.get_n_step_expected_rewards(rewards, estimates, discount, 1)
        expected_rewards_td1 = capacities.get_n_step_expected_rewards(rewards, estimates, discount, 2)
        expected_rewards_mc = capacities.get_n_step_expected_rewards(rewards, estimates, discount, 5)

        self.assertEqual(np.array_equal(expected_rewards_td0, [1.05, 1.1, 2.15, 5]), True)
        self.assertEqual(np.array_equal(expected_rewards_td1, [1.55, 2.075, 4.5, 5]), True)
        self.assertEqual(np.array_equal(expected_rewards_mc, [2.625, 3.25, 4.5, 5]), True)

    def test_get_lambda_expected_rewards(self):
        rewards = [1, 1, 2, 5]
        estimates = [0.1, 0.2, 0.3, 0]
        discount = 1.
        expected_rewards_td0 = capacities.get_lambda_expected_rewards(rewards, estimates, discount, 0.)
        expected_rewards_lambda = capacities.get_lambda_expected_rewards(rewards, estimates, discount, 0.5)
        expected_rewards_mc = capacities.get_lambda_expected_rewards(rewards, estimates, discount, 1.)

        self.assertEqual(np.array_equal(expected_rewards_td0, [1.1, 1.2, 2.3, 5]), True)
        self.assertEqual(np.sum(np.isclose(expected_rewards_lambda, [2.7625, 3.425, 4.65, 5])) == 4, True)
        self.assertEqual(np.array_equal(expected_rewards_mc, [9, 8, 7, 5]), True)

if __name__ == "__main__":
    unittest.main()