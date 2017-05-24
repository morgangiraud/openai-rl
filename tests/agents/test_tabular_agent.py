import gym, os, sys, unittest, shutil
import numpy as np
import tensorflow as tf

dir = os.path.dirname(os.path.realpath(__file__))

from agents import make_agent, get_agent_class

# Silent gym logger
import logging
logging.getLogger("gym").setLevel(logging.WARNING)

class TestTabularAgent(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        shutil.rmtree(dir + '/results/')

    def test_mcagent_act(self):
        config = {
            'agent_name': 'TabularMCAgent'
            , 'env_name': 'CartPole-v0'
            , 'random_seed': 0
            , 'result_dir': dir + '/results'
            , 'discount': 1.
            # 'debug': True
        }
        config.update(get_agent_class(config).get_random_config())

        env = gym.make(config['env_name'])
        env.seed(0)

        agent = make_agent(config, env)
        act, state_id = agent.act(env.reset())

        self.assertEqual(act, 1)
        self.assertEqual(state_id, 65)

    def test_mcagent_learn_from_episode(self):
        config = {
            'agent_name': 'TabularMCAgent'
            , 'env_name': 'CartPole-v0'
            , 'random_seed': 0
            , 'result_dir': dir + '/results'
            , 'discount': 1.
            # 'debug': True
        }
        np.random.seed(0)
        config.update(get_agent_class(config).get_random_config())
        config['discount'] = 1.

        env = gym.make(config['env_name'])
        env.seed(0)

        agent = make_agent(config, env)
        agent.learn_from_episode(env)

        qs = agent.sess.run(agent.Qs)
        self.assertEqual(qs[141][0], 17.)
        self.assertEqual(qs[141][1], 15.5)


if __name__ == "__main__":
    unittest.main()