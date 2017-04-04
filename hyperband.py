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
import gym, os, shutil, re, multiprocessing, json
import concurrent.futures
import numpy as np

from random import random
from math import log, ceil
from time import time, ctime

from agents import make_agent

dir = os.path.dirname(os.path.realpath(__file__))

def execute_run(counter, func, n_iterations, t, dry_run):
    start_time = time()
    t['run'] = counter

    if dry_run:
        result = { 'loss': random(), 'log_loss': random(), 'auc': random()}
    else:
        result = func( n_iterations, t )   # <---

    seconds = int( round( time() - start_time ))
    print("Run {} | {}".format(counter, ctime()))
    print("%d seconds." % seconds )

    return counter, result, n_iterations, t, seconds

class Hyperband:
  
    def __init__( self, get_params_function, try_params_function ):
        self.get_params = get_params_function
        self.try_params = try_params_function
        
        self.max_iter = 81    # maximum iterations per configuration
        self.eta = 3      # defines configuration downsampling rate (default = 3)

        self.logeta = lambda x: log( x ) / log( self.eta )
        self.s_max = int( self.logeta( self.max_iter ))
        self.B = ( self.s_max + 1 ) * self.max_iter

        self.results = [] # list of dicts
        self.counter = 0
        self.best_loss = np.inf
        self.best_counter = -1
    

    # can be called multiple times
    def run( self, main_config, skip_last = 0, dry_run = False ):
        for s in reversed( range( self.s_max + 1 )):
          
            # initial number of configurations
            n = int( ceil( self.B / self.max_iter / ( s + 1 ) * self.eta ** s ))  

            # initial number of iterations per config
            r = self.max_iter * self.eta ** ( -s )    

            # n random configurations
            T = [ self.get_params() for i in range( n )] 
          
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

                executor = concurrent.futures.ProcessPoolExecutor(min(multiprocessing.cpu_count(), main_config['hb_nb_child']))
                futures = []

                for t in T:
                    self.counter += 1
                    futures.append(executor.submit(execute_run, self.counter, self.try_params, n_iterations, t, dry_run))
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
            with open(config['result_dir_prefix'] + '/hb_results.json', 'w') as f:
                json.dump({'results': results, 'best_counter':self.best_counter}, f)

        return {'results': results, 'best_counter':self.best_counter}
  
def make_get_params(config):
    get_N0 = lambda: np.random.randint(1, 1000)
    get_min_eps = lambda: 1e-4 + (1e-1 - 1e-4) * np.random.random(1)[0]

    get_er_every = lambda: np.random.randint(1, 1000)
    get_er_batch_size = lambda: np.random.randint(16, 1024)
    get_er_epoch_size = lambda: np.random.randint(1, 200)
    get_er_rm_size = lambda: np.random.randint(10000, 40000)

    def get_tabular_params():
        get_lr = lambda: 1e-3 + (1 - 1e-3) * np.random.random(1)[0]

        return {
            'lr': get_lr(),
            'N0': get_N0(),
            'min_eps': get_min_eps(),
            'er_every': get_er_every(),
            'er_batch_size': get_er_batch_size(),
            'er_epoch_size': get_er_epoch_size(),
            'er_rm_size': get_er_rm_size(),
            'discount': config['discount'],
            'debug': config['debug'],
            'agent_name': config['agent_name'],
            'env_name': config['env_name'],
            'result_dir_prefix': config['result_dir_prefix'], 
        }

    def get_deep_params():
        get_lr = lambda: 1e-4 + (1e-2 - 1e-4) * np.random.random(1)[0]
        get_nb_units = lambda: np.random.randint(10, 100)
        
        return {
            'lr': get_lr(),
            'nb_units': get_nb_units(),
            'N0': get_N0(),
            'min_eps': get_min_eps(),
            'er_every': get_er_every(),
            'er_batch_size': get_er_batch_size(),
            'er_epoch_size': get_er_epoch_size(),
            'er_rm_size': get_er_rm_size(),
            'discount': config['discount'],
            'debug': config['debug'],
            'agent_name': config['agent_name'],
            'env_name': config['env_name'],
            'result_dir_prefix': config['result_dir_prefix'], 
        }

    if re.compile("tabular", re.I).search(config['agent_name']):
        return get_tabular_params
    else:
        return get_deep_params


def run_params(nb_epoch, config):
    config['max_iter'] = int(nb_epoch) * 100
    config['result_dir'] = config['result_dir_prefix'] + '/' + config['env_name'] + '/' + config['agent_name'] + '/run-' + str(config['run'])

    # If we are reusing a configuration, we remove its folder before next training
    if os.path.exists(config['result_dir']):
        shutil.rmtree(config['result_dir'])

    # We create the agent
    env = gym.make(config['env_name'])
    agent = make_agent(config, env)

    # We tran the agent
    agent.train()
    agent.save()

    # If we are training for less than 9 epochs, we remove the folder
    if nb_epoch < 9:
        shutil.rmtree(config['result_dir'])

    # We test the agent and get the mean score for metrics
    score = []
    for i in range(200):
        score.append(agent.play(env, render=False))
    loss = - np.mean(score) # Hyperbands want a loss

    return {
        'loss': loss
    }

