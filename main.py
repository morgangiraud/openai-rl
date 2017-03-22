import gym, os, time, json
import tensorflow as tf
import numpy as np

from agents import make_agent

dir = os.path.dirname(os.path.realpath(__file__))

flags = tf.app.flags

# Agent
flags.DEFINE_string('agent_name', 'DeepQAgent', 'Name of the agent')
flags.DEFINE_integer('max_iter', 2000, 'Number of training step')
flags.DEFINE_float('lr', 1e-3, 'Learning rate')

# Policy
flags.DEFINE_integer('N0', 100, 'Offset used in the decay algorithm of espilon')
flags.DEFINE_float('min_eps', 1e-2, 'Limit after which the decay stops')

# Experience replay
flags.DEFINE_float('er_lr', 1e-3, 'Experience replay learning rate')
flags.DEFINE_integer('er_every', 20, 'If the model can handle experience replay, use it every n episode_id')
flags.DEFINE_integer('er_batch_size', 16, 'Batch size of the experience replay learning')
flags.DEFINE_integer('er_epoch_size', 200, 'Number of sampled contained in an epoch of experience replay')
flags.DEFINE_integer('er_rm_size', 20000, 'Size of the replay memory buffer')

# Environment
flags.DEFINE_string('env_name', 'CartPole-v0', 'The name of gym environment to use')
flags.DEFINE_boolean('debug', False, 'Debug mode')

flags.DEFINE_string('result_dir', dir + '/results/' + flags.FLAGS.env_name + '/' + flags.FLAGS.agent_name + '/' + str(int(time.time())), 'Name of the directory to store/log the agent (if it exists, the agent will be loaded from it)')

flags.DEFINE_boolean('play', False, 'Load an agents for playing')
# flags.DEFINE_integer('random_seed', 123, 'Value of random seed')

def main(_):
    config = flags.FLAGS.__flags.copy()
    # if os.path.isfile(config['result_dir'] + '/config.json'):
    #     print('Warning: loading config from file: %s' % (config['result_dir'] + '/config.json'))
    #     with open(config['result_dir'] + '/config.json', 'r') as f:
    #         config = json.load(f)
    print(config)

    env = gym.make(config['env_name'])

    agent = make_agent(config, env)

    if config['play']:
        agent.play(env)
    else:
        agent.train()
        agent.save()


if __name__ == '__main__':
  tf.app.run()