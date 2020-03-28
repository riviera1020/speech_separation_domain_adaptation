import os
import yaml
import torch
import _pickle as cPickle

DEV = torch.device('cpu')
DEBUG = False

NCOL = 100

def set_device(use_cuda):
    global DEV
    use_cuda = use_cuda and torch.cuda.is_available()
    DEV = torch.device("cuda" if use_cuda else "cpu")

def set_debug(is_debug):
    global DEBUG
    DEBUG = is_debug

def read_config(path, local_path):

    config = yaml.load(open(path), Loader=yaml.FullLoader)
    path_conf = yaml.load(open(local_path), Loader=yaml.FullLoader)

    for key in path_conf:
        path = path_conf[key]
        if 'data' not in config:
            config['data'] = {}
        config['data'][key] = path
    return config

def read_scale(path):
    with open(os.path.join(path, 'scale')) as f:
        scale = f.readlines()[0].rstrip()
        scale = float(scale)
    return scale

def inf_data_gen(loader):
    while True:
        for s in loader:
            yield s
