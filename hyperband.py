#############################################################################
# Taken from https://github.com/zygmuntz/hyperband/blob/master/hyperband.py #
#############################################################################

import gym, os
import numpy as np

from random import random
from math import log, ceil
from time import time, ctime

from agents import make_agent

dir = os.path.dirname(os.path.realpath(__file__))

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
    def run( self, skip_last = 0, dry_run = False ):
    
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
                    n_configs, n_iterations
                ))

                val_losses = []
                early_stops = []

                for t in T:
              
                    self.counter += 1
                    print("\n{} | {} | best so far: {:.4f} (run {})\n".format(
                        self.counter, ctime(), self.best_loss, self.best_counter 
                    ))

                    start_time = time()

                    if dry_run:
                        result = { 'loss': random(), 'log_loss': random(), 'auc': random()}
                    else:
                        result = self.try_params( n_iterations, t )   # <---

                    assert( type( result ) == dict )
                    assert( 'loss' in result )

                    seconds = int( round( time() - start_time ))
                    print("\n{} seconds.".format( seconds ))

                    loss = result['loss'] 
                    val_losses.append( loss )

                    early_stop = result.get( 'early_stop', False )
                    early_stops.append( early_stop )

                    # keeping track of the best result so far (for display only)
                    # could do it be checking results each time, but hey
                    if loss < self.best_loss:
                        self.best_loss = loss
                        self.best_counter = self.counter

                    result['counter'] = self.counter
                    result['seconds'] = seconds
                    result['params'] = t
                    result['iterations'] = n_iterations

                    self.results.append( result )
            
                # select a number of best configurations for the next loop
                # filter out early stops, if any
                indices = np.argsort( val_losses )
                T = [ T[i] for i in indices if not early_stops[i]]
                T = T[ 0:int( n_configs / self.eta )]
    
        return self.results
  
def make_get_params(config):
    get_lr = lambda: 1e-5 + (1e-1 - 1e-5) * np.random.random(1)[0]
    get_nb_units = lambda: np.random.randint(1, 200)

    get_N0 = lambda: np.random.randint(1, 1000)
    get_min_eps = lambda: 1e-4 + (1e-1 - 1e-4) * np.random.random(1)[0]

    get_er_lr = lambda: 1e-5 + (1e-1 - 1e-5) * np.random.random(1)[0]
    get_er_every = lambda: np.random.randint(1, 1000)
    get_er_batch_size = lambda: np.random.randint(4, 1024)
    get_er_epoch_size = lambda: np.random.randint(20, 200)
    get_er_rm_size = lambda: np.random.randint(5000, 40000)

    def get_params():
        return {
            'lr': get_lr(),
            'nb_units': get_nb_units(),
            'N0': get_N0(),
            'min_eps': get_min_eps(),
            'er_lr': get_er_lr(),
            'er_every': get_er_every(),
            'er_batch_size': get_er_batch_size(),
            'er_epoch_size': get_er_epoch_size(),
            'er_rm_size': get_er_rm_size(),
            'debug': config['debug'],
            'agent_name': config['agent_name'],
            'env_name': config['env_name'],
            'result_dir_prefix': config['result_dir_prefix'], 
        }

    return get_params

def make_run_params(env_name, agent_name):
    def run_params(nb_epoch, config):
        # Max number of epochs is 81
        config['max_iter'] = int(nb_epoch) * 100
        config['result_dir'] = config['result_dir_prefix'] + '/' + config['env_name'] + '/' + config['agent_name'] + '/' + str(int(time()))

        # We create the agent
        env = gym.make(env_name)
        agent = make_agent(config, env)

        # We tran the agent
        agent.train()
        agent.save()

        # We test the agent and get the mean score for metrics
        score = []
        for i in range(50):
            score.append(agent.play(env, render=False))
        loss = - np.mean(score) # Hyperbands want a loss

        return {
            'loss': loss
        }

    return run_params
