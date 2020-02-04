import copy
import logging
import os
from datetime import datetime
from functools import reduce
from operator import getitem
from pathlib import Path
import inspect
import builtins
from utils.util import get_global_rank
from logger.logger import setup_logging
from utils.util import read_json, write_json


class Key2Func(object):
    def __init__(self, f):
        self.f = f

    def __getitem__(self, key):
        return self.f(key)


def createResolutionCallback(frames_up=0):
    """
    Source: Taken from pytorch internal code (cannot be imported)
    Creates a function which, given a string variable name,
    returns the value of the variable in the scope of the caller of
    the function which called createResolutionCallback (by default).
    This is used to enable access in-scope Python variables inside
    TorchScript fragments.
    frames_up is number of additional frames to go up on the stack.
    The default value is 0, which correspond to the frame of the caller
    of createResolutionCallback. Also for example, if frames_up is set
    to 1, then the frame of the caller's caller of createResolutionCallback
    will be taken.
    For example, the following program prints 2::
        def bar():
            cb = createResolutionCallback(1)
            print(cb("foo"))
        def baz():
            foo = 2
            bar()
        baz()
    """
    frame = inspect.currentframe()
    i = 0
    while i < frames_up + 1:
        frame = frame.f_back
        i += 1

    f_locals = frame.f_locals
    f_globals = frame.f_globals

    def env(key):
        if key in f_locals:
            return f_locals[key]
        elif key in f_globals:
            return f_globals[key]
        elif hasattr(builtins, key):
            return getattr(builtins, key)
        else:
            return None

    return env


def parse_dict_or_list(dict_or_list, frames_up=1):
    '''
    It parses a dictionary resolving to variables any strings that start with
    '!' and to path environment variables any strings that begin with '$(ENV)'
    Args:
        arg_dict: Dictionary to parse.
        frames_up: How many frames up in the stack to look for a variable
    '''
    if isinstance(dict_or_list, dict):
        for k, v in dict_or_list.items():
            dict_or_list[k] = parse_value(v, frames_up=frames_up + 1)
    elif isinstance(dict_or_list, list):
        for i, v in enumerate(dict_or_list):
            dict_or_list[i] = parse_value(v, frames_up=frames_up + 1)
    return dict_or_list


def parse_value(v, frames_up=1):
    res = v
    if isinstance(v, str):
        if v.startswith('!'):
            # ! infornt of line means treat this as a command
            cb = Key2Func(createResolutionCallback(frames_up))
            command = v[1:]
            try:
                res = eval(command, None, cb)
            except Exception:
                res = v
        elif v.startswith('$('):
            # $() means this is a path and the env variable in () should
            # be joined with the rest of the path
            env_var, rest = v[2:].split(')', 1)
            if rest.startswith('/'):
                rest = rest[1:]
            value = os.path.join(os.getenv(env_var, default=""), rest)
            res = value
    elif isinstance(v, dict) or isinstance(v, list):
        res = parse_dict_or_list(v, frames_up=frames_up + 1)
    return res


def attach_constructor(module_cfg, module):
    """
    finds a function handle with the name given as 'type' in config,
    and returns the instance initialized with corresponding
    keyword args given as 'args'.
    """
    if not isinstance(module_cfg, list) and not isinstance(module_cfg, dict):
        return

    for m in module_cfg:
        if isinstance(module_cfg, dict):
            attach_constructor(module_cfg[m], module)
        else:
            attach_constructor(m, module)

    if isinstance(module_cfg, list):
        return

    if 'type' not in module_cfg:
        return

    if "args" not in module_cfg:
        module_cfg['args'] = {}

    if module_cfg['type'] is None:
        module_cfg['constructor'] = None
        return

    if 'constructor' in module_cfg:
        return

    if not isinstance(module, list):
        module = [module]

    for m in module:
        try:
            constructor = getattr(m, module_cfg['type'])
            break
        except AttributeError as e:
            if m == module[-1]:
                raise e

    module_cfg['constructor'] = constructor


def initialize_from_module(module_cfg, *args):
    """
    finds a function handle with the name given as 'type' in config,
    and returns the instance initialized with corresponding
    keyword args given as 'args'.
    """
    if isinstance(module_cfg, list):
        res = []
        for m in module_cfg:
            res.append(initialize_from_module(m, *args))
        return res

    if module_cfg['constructor'] is None:
        return None
    else:
        constructor = module_cfg['constructor']
        return constructor(*args, **module_cfg['args'])


class ConfigParser:
    def __init__(self, args, options='', timestamp=True, disable_write=False):
        # parse default and custom cli options
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        args = args.parse_args()

        self.resume = None
        if args.device:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        if args.resume:
            self.resume = Path(args.resume)
            self.cfg_fname = self.resume.parent / 'config.json'
        if args.config:
            self.cfg_fname = Path(args.config)
        msg_no_cfg = ("Configuration file need to be specified. "
                      "Add '-c config.json', for example.")
        assert self.cfg_fname is not None, msg_no_cfg

        # load config file and apply custom cli options
        config = read_json(self.cfg_fname)
        self.__config = _update_config(config, options, args)
        self.__raw = copy.deepcopy(self.__config)

        # set save_dir where trained model and log will be saved.
        save_dir = Path(
            parse_value(self.config['trainer']['extra_args']['save_dir']))
        timestamp = datetime.now().strftime(
            r'%m%d_%H%M%S') if timestamp else ''

        exper_name = self.config['name']
        self.__save_dir = save_dir / 'models' / exper_name / timestamp
        self.__log_dir = save_dir / 'log' / exper_name / timestamp

        if not disable_write:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir.mkdir(parents=True, exist_ok=True)

            # save updated config file to the checkpoint dir
            if get_global_rank() == 0:
                write_json(self.config, self.save_dir / 'config.json')

        # configure logging module
        setup_logging(self.log_dir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }
        logger = self.get_logger('config')
        logger.info(f"Experiment name: {exper_name}")

    def initialize(self, name, module, *args):
        """
        finds a function handle with the name given as 'type' in config,
        and returns the instance initialized with corresponding
        keyword args given as 'args'.
        """
        frames_up = 1
        module_cfg = self.config[name]
        module_cfg = parse_dict_or_list(module_cfg, frames_up=frames_up + 1)
        attach_constructor(module_cfg, module)
        return initialize_from_module(module_cfg, *args)

    def initialize_from_dict(self, module_cfg, module, *args):
        frames_up = 1
        module_cfg = parse_dict_or_list(module_cfg, frames_up=frames_up + 1)
        attach_constructor(module_cfg, module)
        return initialize_from_module(module_cfg, *args)

    def __getitem__(self, name):  # TODO: parse only env variables here
        return parse_value(self.config[name], frames_up=2)

    def get_logger(self, name, verbosity=2):
        msg_verbosity = (
            'verbosity option {} is invalid. Valid options are {}.').format(
                verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    # setting read-only attributes
    @property
    def config(self):
        return self.__config

    @property
    def save_dir(self):
        return self.__save_dir

    @property
    def log_dir(self):
        return self.__log_dir

    @property
    def raw(self):
        return self.__raw


# helper functions used to update config dict with custom cli options


def _update_config(config, options, args):
    for opt in options:
        value = getattr(args, _get_opt_name(opt.flags))
        if value is not None:
            _set_by_path(config, opt.target, value)
    return config


def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')


def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)
