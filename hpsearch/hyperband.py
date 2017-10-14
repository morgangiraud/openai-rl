#############################################################################
# Taken from https://github.com/zygmuntz/hyperband/blob/master/hyperband.py #
#############################################################################
# n_i: number of configurations
# r_i: number of iterations/epochs
# max_iter = 81        s=4             s=3             s=2             s=1             s=0
# eta = 3              n_i   r_i       n_i   r_i       n_i   r_i       n_i   r_i       n_i   r_i
# B = 5*max_iter       ---------       ---------       ---------       ---------       ---------
#                       81    1         27    3         9     9         6     27        5     81
#                       27    3         9     9         3     27        2     81
#                       9     9         3     27        1     81
#                       3     27        1     81
#                       1     81
import gym, os, shutil, re, multiprocessing, json, copy
import concurrent.futures
import numpy as np

from random import random
from math import log, ceil
from time import time, ctime

from agents import make_agent
from hpsearch.utils import get_stats

dir = os.path.dirname(os.path.realpath(__file__))

def execute_run(counter, func, n_iterations, params, main_config, dry_run):    
    start_time = time()

    if dry_run:
        result = { 'loss': random(), 'log_loss': random(), 'auc': random()}
    else:
        result = func( n_iterations, params, main_config )   # <---

    seconds = int( round( time() - start_time ))
    if not dry_run:
        print("Run {} with params {} | {}".format(counter, params['id'], ctime()))
        print("%d seconds." % seconds )

    return counter, result, n_iterations, params, seconds

class Hyperband:

    def __init__( self, get_params_function, try_params_function ):
        self.get_params = get_params_function
        self.try_params = try_params_function

        self.max_iter = 81 # 244     # maximum iterations per configuration
        self.eta = 3            # defines configuration downsampling rate (default = 3)

        self.logeta = lambda x: log( x ) / log( self.eta )
        self.s_max = int( self.logeta( self.max_iter ))
        self.B = ( self.s_max + 1 ) * self.max_iter

        self.results = [] # list of dicts
        self.counter = 0
        self.best_loss = np.inf
        self.best_counter = -1

        print("*** max_iter: %d, eta: %d, s_max: %d, B %d" % (self.max_iter, self.eta, self.s_max, self.B))

    # can be called multiple times
    def run( self, main_config, skip_last = 0, dry_run = False ):
        for s in reversed( range( self.s_max + 1 )):

            # initial number of configurations
            n = int( ceil( self.B / self.max_iter / ( s + 1 ) * self.eta ** s ))

            # initial number of iterations per config
            r = self.max_iter * self.eta ** ( -s )

            # n random configurations
            T = [ self.get_params(main_config['fixed_params']) for i in range( n )]
            for i, params in enumerate(T):
                params['id'] = i

            for i in range(( s + 1 ) - int( skip_last )): # changed from s + 1

                # Run each of the n configs for <iterations>
                # and keep best (n_configs / eta) configurations

                n_configs = n * self.eta ** ( -i )
                n_iterations = r * self.eta ** ( i )

                print("\n*** {} configurations x {:.1f} iterations each".format(
                    int(n_configs), n_iterations
                ))

                val_losses = []
                early_stops = []

                futures = []
                with concurrent.futures.ProcessPoolExecutor(min(multiprocessing.cpu_count(), main_config['nb_process'])) as executor:
                    for t in T:
                        self.counter += 1
                        futures.append(executor.submit(execute_run, self.counter, self.try_params, n_iterations, t, main_config, dry_run))
                    concurrent.futures.wait(futures)

                for future in futures:
                    counter, result, n_iterations, t, seconds = future.result()

                    assert( type( result ) == dict )
                    assert( 'loss' in result )

                    loss = result['loss']
                    val_losses.append( loss )

                    early_stop = result.get( 'early_stop', False )
                    early_stops.append( early_stop )

                    # keeping track of the best result so far (for display only)
                    # could do it be checking results each time, but hey
                    if loss < self.best_loss:
                        self.best_loss = loss
                        self.best_counter = counter

                    result['counter'] = counter
                    result['seconds'] = seconds
                    result['params'] = t
                    result['iterations'] = n_iterations

                    self.results.append( result )

                if not dry_run:
                    print("*** Set %d-%d finished | best so far: %4f (run %d)" % (
                        s, i, self.best_loss, self.best_counter
                    ))

                # select a number of best configurations for the next loop
                # filter out early stops, if any
                indices = np.argsort( val_losses )
                T = [ T[i] for i in indices if not early_stops[i]]
                T = T[ 0:int( n_configs / self.eta )]

            results = sorted(self.results, key=lambda result: result['loss'])
            config = T[0]

        return {'results': results, 'best_counter':self.best_counter}

def run_params(nb_epochs, params, main_config):
    config = copy.deepcopy(main_config)
    config.update(params)
    config['result_dir'] = config['result_dir_prefix'] + '/' + config['env_name'] + '/' + config['agent_name'] + '/run-' + str(config['id']).zfill(3)
    config['max_iter'] = int(nb_epochs) * config['games_per_epoch']

    # If we are reusing a configuration, we remove its folder before next training
    if os.path.exists(config['result_dir']):
        shutil.rmtree(config['result_dir'])

    try:
        # We create the agent
        env = gym.make(config['env_name'])
        agent = make_agent(config, env)
        
        # We train the agent
        agent.train(save_every=-1)
        stats = get_stats(config['result_dir'], ["score"])
        mean_score = np.mean(stats['score'])
        stddev_score = np.sqrt(np.var(stats['score']))
        result = {
            'run_id': run_id
            , 'params': params
            , 'loss': -mean_score
            , 'mean_score': mean_score
            , 'stddev_score': stddev_score
        }
        agent.save()
    except Exception as inst:
        result = {
            'loss': 0
            , 'mean_score': 0
            , 'stddev_score': 0
            , 'error': str(sys.exc_info()[0])
            , 'error_message': str(sys.exc_info()[1])
        }

    # If we are training for less than 9 epochs, we remove the folder
    if nb_epochs < 9 and os.path.exists(config['result_dir']):
        shutil.rmtree(config['result_dir'])

    return result

