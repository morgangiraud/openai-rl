import os, sys, unittest
import numpy as np
import tensorflow as tf

dir = os.path.dirname(os.path.realpath(__file__))

from agents.deep_policy_agent import getExpectedRewards

class TestCapacities(unittest.TestCase):

    def test_get_expected_rewards(self):
        rewards = [1, 1, 2]
        expected_rewards = getExpectedRewards(rewards)

        self.assertEqual(np.array_equal(expected_rewards, [[4], [3], [2]]), True)


if __name__ == "__main__":
    unittest.main()