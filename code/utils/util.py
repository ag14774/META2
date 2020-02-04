import hashlib
import json
import os
import shutil
import urllib
import urllib.request
from collections import OrderedDict
from datetime import datetime
from functools import partial
from pathlib import Path

import torch.cuda
import torch.distributed as dist


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def fetch_url(url):
    try:
        response = urllib.request.urlopen(url)
        data = response.read()  # a `bytes` object
        text = data.decode('utf-8')
        return text.strip()
    except urllib.error.URLError:
        return None


def download_file(url, filename=None):
    if filename is None:
        filename = url.split('/')[-1]
    try:
        with urllib.request.urlopen(url) as response:
            with open(filename, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
    except urllib.error.URLError:
        return None
    return 1


def strip_header(lines, startswith='#'):
    for i, line in enumerate(lines):
        if not line.startswith('#'):
            break
    return lines[i:]


def md5sum(filename):
    filename = Path(filename)
    with filename.open('rb') as f:
        d = hashlib.md5()
        for buf in iter(partial(f.read, 128), b''):
            d.update(buf)
    return d.hexdigest()


def check_file_integrity(filename, md5_path):
    filename = Path(filename)
    md5_path = Path(md5_path)
    if filename.is_file() and md5_path.is_file():
        md5_file = md5sum(filename)
        with md5_path.open('r') as f:
            correct_md5 = f.readline().strip()
        if md5_file == correct_md5:
            return True
        else:
            return False
    else:
        return False


def create_md5_file(filename, md5_path):
    filename = Path(filename)
    md5_path = Path(md5_path)

    md5_file = md5sum(filename)
    with md5_path.open('w') as f:
        f.write(md5_file)


def dict_from_md5_file(md5list):
    """
    Takes as input a file as a list of lines
    Returns a dict that maps filenames to MD5 hashes
    """
    file2hash = {}
    for line in md5list:
        line = line.split(' ')
        hash = line[0]
        file = line[-1].lstrip('./')
        file2hash[file] = hash

    return file2hash


def get_global_rank(default_value=0):
    """
    Returns rank of process in distributed setting.
    Returns 0 if not in a distributed environment.
    """
    if dist.is_initialized():
        return dist.get_rank()
    else:
        return int(os.getenv("RANK", default_value))


def get_world_size():
    if dist.is_initialized():
        return dist.get_world_size()
    else:
        return int(os.getenv("WORLD_SIZE", 1))


def get_local_size():
    return int(os.getenv("LOCAL_SIZE", torch.cuda.device_count()))


def is_distributed():
    return get_world_size() > 1


class Timer:
    def __init__(self):
        self.cache = datetime.now()

    def check(self):
        now = datetime.now()
        duration = now - self.cache
        self.cache = now
        return duration.total_seconds()

    def reset(self):
        self.cache = datetime.now()
