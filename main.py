import gym, os, time, json, random, sys
import tensorflow as tf
import numpy as np

from agents import make_agent, get_agent_class
from hpsearch.hyperband import Hyperband, run_params
from hpsearch import fullsearch
from hpsearch import randomsearch

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
flags.DEFINE_float('initial_stddev', 1e-1, 'Initial standard deviation for NN')
flags.DEFINE_float('lambda', .9, 'Lambda parameters used with eligibility traces')
flags.DEFINE_float('discount', .999, 'Discount factor')
flags.DEFINE_float('lr', 1e-3, 'Learning rate for actor/controller network')
flags.DEFINE_float('m_lr', 1e-3, 'Learning rate for the predictive model/critic')
flags.DEFINE_integer('lr_decay_steps', 50000, 'Learning rate decay steps for tabular methods')
flags.DEFINE_integer('nb_units', 20, 'Number of hidden units in Deep learning agents')
flags.DEFINE_float('critic_lr', 1e-3, 'For Learning rate for the critic part of AC agents')
flags.DEFINE_float('nb_critic_iter', 16, 'Number of iteration to fit the critic')
flags.DEFINE_integer('n_step', 4, 'Number of step used in TD(n) algorithm')
flags.DEFINE_float('reg_coef', 1e-2, 'regularization coefficient')
# CM agents
flags.DEFINE_integer('nb_sleep_iter', 100, 'Used in CM models: number of step used to train the actor in the environment')
flags.DEFINE_integer('nb_wake_iter', 50, 'Used in CM models: number of step used to train the predictive model of the environment')
flags.DEFINE_float('initial_m_stddev', 2e-1, 'Initial standard deviation for the predictive model')
flags.DEFINE_integer('nb_m_units', 50, 'Number of hidden units for the predictive model')

# Policy
flags.DEFINE_boolean('UCB', False, 'Use the UCB policy for tabular agents')
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
flags.DEFINE_integer('random_seed', random.randint(0, 2**32 - 1), 'Value of random seed')

def main(_):
    config = flags.FLAGS.__flags.copy()
    config["fixed_params"] = json.loads(config["fixed_params"])

    # if os.path.isfile(config['result_dir'] + '/config.json'):
    #     print("Overriding shell configuration with the one found in " + config['result_dir'])
    #     with open(config['result_dir'] + '/config.json', 'r') as f:
    #         config = json.loads(f.read())

    if config['hyperband']:
        print('Starting hyperband search')

        config['result_dir_prefix'] = dir + '/results/hyperband/' + str(int(time.time()))

        get_params = get_agent_class(config).get_random_config
        hb = Hyperband( get_params, run_params )
        results = hb.run(config, skip_last=True, dry_run=config['dry_run'])

        if not os.path.exists(config['result_dir_prefix']):
            os.makedirs(config['result_dir_prefix'])
        with open(config['result_dir_prefix'] + '/hb_results.json', 'w') as f:
            json.dump(results, f)

    elif config['fullsearch']:
        print('*** Starting full search')
        config['result_dir_prefix'] = dir + '/results/fullsearch/' + str(int(time.time())) + '-' + config['agent_name']
        os.makedirs(config['result_dir_prefix'])
        
        print('*** Starting first pass: full random search')
        summary = fullsearch.first_pass(config)
        with open(config['result_dir_prefix'] + '/fullsearch_results1.json', 'w') as f:
            json.dump(summary, f)

        print('*** Starting second pass: Learning rate search')
        best_agent_config = summary['results'][0]['params']
        summary = fullsearch.second_pass(config, best_agent_config)
        with open(config['result_dir_prefix'] + '/fullsearch_results2.json', 'w') as f:
            json.dump(summary, f)

        print('*** Starting third pass: Hyperband search with best lr')
        best_lr = summary['results'][0]['lr']
        summary = fullsearch.third_pass(config, best_lr)
        with open(config['result_dir_prefix'] + '/fullsearch_results3.json', 'w') as f:
            json.dump(summary, f)
    elif config['randomsearch']:
        print('*** Starting random search')
        config['result_dir_prefix'] = dir + '/results/randomsearch/' + str(int(time.time())) + '-' + config['agent_name']
        os.makedirs(config['result_dir_prefix'])

        summary = randomsearch.search(config)
        with open(config['result_dir_prefix'] + '/fullsearch_results1.json', 'w') as f:
            json.dump(summary, f)
    else:
        env = gym.make(config['env_name'])
        agent = make_agent(config, env)

        if config['play']:
            for i in range(config['play_nb']):
                agent.play(env)
        else:
            agent.train()
            agent.save()


if __name__ == '__main__':
  tf.app.run()