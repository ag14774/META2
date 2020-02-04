import argparse
import hashlib
import json
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid


def read_json(fname):
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


class ConfigIterator(object):
    SEP = '|'
    CV_TRIGGER = '!'
    '''
    Keys that start with '!' need to be set to a list that contains the
    hyperparameters of that key
    '''

    def __init__(self, config):
        self.config = deepcopy(config)
        self.config['name'] = self.config['name'].rsplit('.')[0]
        root = {}
        root['MASTER_ROOT'] = self.config
        df = pd.io.json.json_normalize(root, sep=self.SEP)
        self.config_norm = df.to_dict(orient='records')[0]
        key_vals = list(self.config_norm.items())
        for key, val in key_vals:
            oldkey = key
            if f'{self.SEP}{self.CV_TRIGGER}' in key:
                keys = key.split(self.SEP)
                for i, k in enumerate(keys):
                    if k.startswith(self.CV_TRIGGER):
                        keys[i] = k[1:]
                key = self.SEP.join(keys)
                self.config_norm[key] = self.config_norm.pop(oldkey)
            else:
                self.config_norm[key] = [val]
        self.grid = ParameterGrid(self.config_norm)

    def denormalise(self, config):
        res = {}
        for k, v in config.items():
            root = res
            nested_keys = k.split(self.SEP)
            for i, n in enumerate(nested_keys):
                if n not in root:
                    root[n] = {}
                if i != (len(nested_keys) - 1):
                    root = root[n]
            root[nested_keys[-1]] = v
        return res

    def __getitem__(self, idx):
        conf = self.grid[idx]
        conf = self.denormalise(conf)['MASTER_ROOT']
        name = conf['name']
        hash = hashlib.md5(json.dumps(
            conf, sort_keys=True).encode('utf-8')).hexdigest()
        conf['name'] = f'{name}.{hash}'
        return conf

    def __len__(self):
        return len(self.grid)


if __name__ == '__main__':
    args = argparse.ArgumentParser(
        description='Generate configs for hyperparameter search')
    args.add_argument('-c',
                      '--config',
                      default=None,
                      type=str,
                      help='config file path (default: None)')
    args.add_argument('-n',
                      '--num',
                      default=None,
                      type=int,
                      help='Number of configs to generate')
    args.add_argument('-s', '--seed', default=42, type=int, help='Random seed')

    args = args.parse_args()
    cfg_path = Path(args.config)
    config = read_json(cfg_path)

    citer = ConfigIterator(config)

    cfg_folder = cfg_path.parent / cfg_path.stem
    cfg_folder.mkdir(parents=True, exist_ok=True)
    for f in cfg_folder.glob('*.json'):
        f.unlink()

    if args.num:
        if args.seed:
            np.random.seed(args.seed)
        choices = np.random.choice(len(citer), size=args.num, replace=False)
        for i in choices:
            cfg_name = cfg_folder / f'{cfg_path.stem}_{i}.json'
            write_json(citer[i], cfg_name)
    else:
        for i, c in enumerate(citer):
            cfg_name = cfg_folder / f'{cfg_path.stem}_{i}.json'
            write_json(c, cfg_name)
