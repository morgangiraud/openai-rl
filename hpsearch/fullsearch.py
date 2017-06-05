import copy, os, sys, multiprocessing, time, shutil
import concurrent.futures
import tensorflow as tf
import numpy as np
import gym

dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir + '/..')

from agents import make_agent, get_agent_class
from hpsearch.hyperband import Hyperband, run_params
from hpsearch.utils import get_score_stat


def exec_first_pass(counter, config, params):
    start_time = time.time()

    config['result_dir'] = config['result_dir_prefix'] + '/run-' + str(counter).zfill(3)

    try:
        # We create the agent
        env = gym.make(config['env_name'])
        agent = make_agent(config, env)

        # We train the agent
        agent.train(save_every=-1)
        mean_score, stddev_score = get_score_stat(config['result_dir'])
        result = {
            'params': params
            , 'mean_score': mean_score
            , 'stddev_score': stddev_score
        }

        seconds = int( round( time.time() - start_time ))
        print("Run: {} | {}, mean_score {}".format(counter, time.ctime(), mean_score))
        print("%d seconds." % seconds )
    except:
        result = {
            'params': params
            , 'mean_score': 0
            , 'stddev_score': 0
            , 'error': str(sys.exc_info()[0])
            , 'error_message': str(sys.exc_info()[1])
        }
    
    if os.path.exists(config['result_dir']):
        shutil.rmtree(config['result_dir'])

    return result

def first_pass(config):
    config = copy.deepcopy(config)

    config['result_dir_prefix'] = config['result_dir_prefix'] + '/first-pass'
    if config['debug']:
        print('Removing fixed params')
    config["fixed_params"] = {}
    config['max_iter'] = 5 if config['debug'] else 150
    if config['debug']:
        print('Overriding max_iter params to %d' % config['max_iter'])
    dry_run = True if config['debug'] else False

    get_params = get_agent_class(config).get_random_config

    results = []
    futures = []
    with concurrent.futures.ProcessPoolExecutor(min(multiprocessing.cpu_count(), config['nb_process'])) as executor:
        nb_config = 5 if config['debug'] else 1000
        for i in range(nb_config): 
            params = get_params()
            config.update(params)

            futures.append(executor.submit(exec_first_pass, i, copy.deepcopy(config), params))
        concurrent.futures.wait(futures)
    
    results = []
    for future in futures:
        results.append(future.result())
        
    return {
        'results': sorted(results, key=lambda result: result['mean_score'], reverse=True)
    }

def exec_second_pass(config):
    start_time = time.time()

    config['result_dir'] = config['result_dir_prefix'] + '/lr-' + str(config['lr'])

    try:
        # We create the agent
        env = gym.make(config['env_name'])
        agent = make_agent(config, env)

        # We train the agent
        agent.train(save_every=-1)
        mean_score, stddev_score = get_score_stat(config['result_dir'])
        result = {
            'lr': config['lr']
            , 'mean_score': mean_score
            , 'stddev_score': stddev_score
        }

        seconds = int( round( time.time() - start_time ))
        print("Run with lr: {} | {}".format(config['lr'], time.ctime()))
        print("%d seconds." % seconds )
    except:
        result = {
            'lr': config['lr']
            , 'mean_score': 0
            , 'stddev_score': 0
            , 'error': str(sys.exc_info()[0])
            , 'error_message': str(sys.exc_info()[1])
        }

    return result

def second_pass(config, best_agent_config):
    config = copy.deepcopy(config)

    config.update(best_agent_config)
    config['result_dir_prefix'] = config['result_dir_prefix'] + '/second-pass'
    config['max_iter'] = 5 if config['debug'] else 500
    futures = []
    with concurrent.futures.ProcessPoolExecutor(min(multiprocessing.cpu_count(), config['nb_process'])) as executor:
        if config['debug']:
            lrs = [1e-4, 1e-2, 1]
        else:
            lrs = [1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3, 1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2, 1e-1, 2e-1, 3e-1, 4e-1, 5e-1, 6e-1, 7e-1, 8e-1, 9e-1, 1]
        for lr in lrs:
            config['lr'] = lr
            futures.append(executor.submit(exec_second_pass, copy.deepcopy(config)))
        concurrent.futures.wait(futures)

    results = []
    for future in futures:
        results.append(future.result())


    return {
        'best_agent_config': best_agent_config
        , 'results': sorted(results, key=lambda result: result['mean_score'], reverse=True)
    }

def third_pass(config, best_lr):
    config = copy.deepcopy(config)

    config["fixed_params"] = {'lr': best_lr}
    config['result_dir_prefix'] = config['result_dir_prefix'] + '/third-pass'
    config['games_per_epoch'] =  5 if config['debug'] else 100
    dry_run = True if config['debug'] else False

    get_params = get_agent_class(config).get_random_config
    hb = Hyperband( get_params, run_params )

    summary = hb.run(config, skip_last=True, dry_run=dry_run)


    return summary