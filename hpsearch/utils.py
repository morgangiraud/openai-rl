import os
import tensorflow as tf 
import numpy as np

def get_score_stat(result_dir):
    eventFile = [f for f in os.listdir(result_dir) if os.path.isfile(os.path.join(result_dir, f)) and 'events' in f][0]
    scores = []
    try:
        for events in tf.train.summary_iterator(os.path.join(result_dir, eventFile)):
            for v in events.summary.value:
                if v.tag == "score":
                    scores.append(v.simple_value)
    except:
        pass

    return ( np.mean(scores), np.sqrt(np.var(scores)) )