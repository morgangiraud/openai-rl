import os, json
import tensorflow as tf 
import numpy as np

def get_stats(result_dir, tag_names):
    eventFile = [f for f in os.listdir(result_dir) if os.path.isfile(os.path.join(result_dir, f)) and 'events' in f][0]
    stats = { key: [] for key in tag_names }
    try:
        for events in tf.train.summary_iterator(os.path.join(result_dir, eventFile)):
            for v in events.summary.value:
                if v.tag in tag_names:
                    stats[v.tag].append(v.simple_value)
    except:
        pass

    return stats

def add_metrics_to_search(root_results_dir, metrics=[]):
    root, subdirs, filenames = next(os.walk(root_results_dir))
    with open(os.path.join(root, filenames[0])) as f:
        json_data = f.read()
        f.close()
    results = json.loads(json_data)['results']
    for i, result in enumerate(results):
        for key in metrics:
            results[i][key] = 2**32 - 1
    for subdir in subdirs:
        print('handling %s' % subdir)
        run_id = subdir.split('-')[1]
        stats = get_stats(os.path.join(root,subdir), metrics)
        for i, result in enumerate(results):
            if result["run_id"] == run_id:
                for key in metrics:
                    results[i][key] = np.mean(stats[key])
                break

    sorted_by_key_results = {key: None for key in metrics }
    for key in metrics:
        sorted_by_key_results[key] = sorted(results, key=lambda result: result[key])
    
    summary = { "best_" + key: sorted_by_key_results[key][0]['params'] for key in metrics }
    summary['results'] = sorted_by_key_results
    with open(os.path.join(root, 'repaired-' + filenames[0]), 'w') as f:
        f.write(json.dumps(summary))
        