import gym, os, time, random, sys
import tensorflow as tf
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


from agents import make_agent, get_agent_class

dir = os.path.dirname(os.path.realpath(__file__))

flags = tf.app.flags

# HP search
flags.DEFINE_boolean('randomsearch', False, 'Perform a random search fixing one HP at a time')
flags.DEFINE_boolean('fullsearch', False, 'Perform a full search of hyperparameter space (hyperband -> lr search -> hyperband with best lr)')
flags.DEFINE_string('fixed_params', "{}", 'JSON inputs to fix some params in a random search, ex: \'{"lr": 0.001}\'')
# Hyperband
flags.DEFINE_boolean('hyperband', False, 'Perform a hyperband search of hyperparameters')
flags.DEFINE_boolean('dry_run', False, 'Perform a hyperband dry_run')
flags.DEFINE_integer('nb_process', 4, 'Number of parallel process to perform a hyperband search')
flags.DEFINE_integer('games_per_epoch', 100, 'Number of parallel process to perform a hyperband search')

# Agent
flags.DEFINE_string('agent_name', 'DQNAgent', 'Name of the agent')
flags.DEFINE_boolean('best', False, 'Use the best known configuration')
flags.DEFINE_float('initial_q_value', 0., 'Initial Q values in the Tabular case')
flags.DEFINE_float('initial_mean', 0., 'Initial mean for NN')
flags.DEFINE_float('initial_stddev', 1e-2, 'Initial standard deviation for NN')
flags.DEFINE_float('lambda', .9, 'Lambda parameters used with eligibility traces')
flags.DEFINE_float('discount', .999, 'Discount factor')
flags.DEFINE_float('lr', 1e-3, 'Learning rate')
flags.DEFINE_integer('lr_decay_steps', 50000, 'Learning rate decay steps for tabular methods')
flags.DEFINE_integer('nb_units', 20, 'Number of hidden units in Deep learning agents')
flags.DEFINE_float('q_scale_lr', 1., 'For actor critic agents, scale variables between q loss and policy loss')
flags.DEFINE_integer('n_step', 4, 'Number of step used in TD(n) algorithm')

# Policy
flags.DEFINE_integer('N0', 100, 'Offset used in the decay algorithm of epsilon')
flags.DEFINE_float('min_eps', 1e-2, 'Limit after which the decay stops')

# Experience replay
flags.DEFINE_integer('er_batch_size', 512, 'Batch size of the experience replay learning')
flags.DEFINE_integer('er_epoch_size', 50, 'Number of sampled contained in an epoch of experience replay')
flags.DEFINE_integer('er_rm_size', 20000, 'Size of the replay memory buffer')
flags.DEFINE_integer('update_every', 20, 'Update the fixed Q network every chosen step')

# Environment
flags.DEFINE_string('env_name', 'CartPole-v0', 'The name of gym environment to use')
flags.DEFINE_boolean('debug', False, 'Debug mode')
flags.DEFINE_integer('max_iter', 2000, 'Number of training step')

flags.DEFINE_string('result_dir', dir + '/results/' + flags.FLAGS.env_name + '/' + flags.FLAGS.agent_name + '/' + str(int(time.time())), 'Name of the directory to store/log the agent (if it exists, the agent will be loaded from it)')

flags.DEFINE_boolean('play', False, 'Load an agent for playing')
flags.DEFINE_integer('play_nb', 10, 'Number of games to play')
flags.DEFINE_integer('random_seed', random.randint(0, sys.maxsize), 'Value of random seed')

def main(_):
    config = flags.FLAGS.__flags.copy()

    env = gym.make(config['env_name'])
    agent = make_agent(config, env)

    qs = agent.sess.run(agent.Qs)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = np.linspace(0, len(qs) - 1, len(qs))
    y = np.linspace(0, len(qs[0]) - 1, len(qs[0]))
    xpos, ypos = np.meshgrid(x, y, indexing='ij')
    xpos = xpos.flatten('F')
    ypos = ypos.flatten('F')
    num_elements = len(qs) * len(qs[0])
    zpos = np.zeros(num_elements)
    dx = np.ones(num_elements)
    dy = np.ones(num_elements)
    dz = np.abs(np.array(qs).flatten('F'))

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='#00ceaa')
    plt.savefig('test.png')

if __name__ == '__main__':
    tf.app.run()
