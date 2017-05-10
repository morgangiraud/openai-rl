import copy, os, sys, multiprocessing, time
import concurrent.futures
import tensorflow as tf
import numpy as np
import gym

dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir + '/..')

from agents import make_agent, get_agent_class
from hpsearch.utils import get_score_stat


def search(config):
    get_params = get_agent_class(config).get_random_config
    params_keys = list(get_params().keys())
    nb_hp_params = len(params_keys)

    if config['debug']:
        print('*** Number of hyper-parameters: %d' % nb_hp_params)

    fixed_params = {}
    all_fixed_params = []
    all_results = []
    for i in range(nb_hp_params):
        all_fixed_params.append(copy.deepcopy(fixed_params))

        config['max_iter'] = 5 if config['debug'] else 150 * (1 + i/2)
        results = []
        futures = []
        with concurrent.futures.ProcessPoolExecutor(min(multiprocessing.cpu_count(), config['nb_process'])) as executor:
            nb_config = 5 if config['debug'] else 1000 / (1 + i/2)
            for j in range(nb_config): 
                params = get_params(fixed_params)
                config.update(params)

                futures.append(executor.submit(test_params, i * nb_config + j, copy.deepcopy(config), params))
            concurrent.futures.wait(futures)
        for future in futures:
            results.append(future.result())
            
        all_results += results
        results = sorted(results, key=lambda result: result['mean_score'], reverse=True)
        best_params = results[0]['params']
        key = np.random.choice(params_keys, 1)[0]
        params_keys.remove(key)
        fixed_params[key] = best_params[key]
    all_fixed_params.append(copy.deepcopy(fixed_params))

    return { 
        'best_params': best_params
        , 'all_fixed_params': all_fixed_params
        , 'results': sorted(all_results, key=lambda result: result['mean_score'], reverse=True)
    }

    
def test_params(counter, config, params):
    start_time = time.time()

    config['result_dir'] = config['result_dir_prefix'] + '/run-' + str(counter).zfill(3)

    # We create the agent
    env = gym.make(config['env_name'])
    agent = make_agent(config, env)

    # We train the agent
    agent.train(save_every=-1)
    mean_score, stddev_score = get_score_stat(config['result_dir'])

    seconds = int( round( time.time() - start_time ))
    print("Run: {} | {}, mean_score {}".format(counter, time.ctime(), mean_score))
    print("%d seconds." % seconds )

    return {
        'params': params
        , 'mean_score': mean_score
        , 'stddev_score': stddev_score
    }    