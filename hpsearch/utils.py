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

def repair_randomsearch_ids(root_results_dir):
    root, subdirs, filenames = next(os.walk(root_results_dir))
    with open(os.path.join(root, filenames[0])) as f:
        json_data = f.read()
        f.close()
    results = json.loads(json_data)['results']
    for i, result in enumerate(results):
        results[i]["mean_m_loss"] = 2**32 - 1
    for subdir in subdirs:
        print('handling file %s' % subdir)
        stats = get_stats(os.path.join(root,subdir), ['score', 'm_training/m_loss'])
        mean_score = np.mean(stats['score'])
        for i, result in enumerate(results):
            if result["mean_score"] == mean_score:
                results[i]["run_id"] = int(subdir.split('-')[1])
                results[i]["mean_m_loss"] = np.mean(stats['m_training/m_loss'][-100:])
                break

    c_results = sorted(results, key=lambda result: result['mean_score'], reverse=True)
    m_results = sorted(results, key=lambda result: result['mean_m_loss'])

    with open(os.path.join(root, 'repaired-' + filenames[0]), 'w') as f:
        f.write(json.dumps({
            'best_c_params': c_results[0]['params']
            , 'best_m_params': m_results[0]['params']
            , 'results': results
        }))
        f.close()
        