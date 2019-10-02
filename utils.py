import json
import logging
import os


def recursive_update(d, u):
    '''
    Recursively update values in a dict (and dicts inside)
    '''
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = recursive_update(d.get(k, {}), v)
        elif v is not None:
            d[k] = v
    return d


def parse_json(json_file):
    if not os.path.isabs(json_file):
        json_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'config_files', json_file)
    with open(json_file, 'r') as f:
        data = json.load(f)
    default_values = {
        'nonlocal': True,
        'weight_decay': 1e-4,
        'learning_scheduler': {
            'type': 'reduce_lr',
            'params': {
                "step_per_iter": True
            }
        },
        'num_epochs': 30,
        'batch_size': 8
    }
    default_values = recursive_update(default_values, data)

    return default_values


def setup_logger(logger_name, log_file):
    log = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(message)s')
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    streamHandler.setLevel(logging.INFO)

    log.setLevel(logging.DEBUG)
    log.addHandler(fileHandler)
    log.addHandler(streamHandler)
