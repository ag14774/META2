import logging
import logging.config
from pathlib import Path

from utils.util import read_json


def setup_logging(save_dir,
                  log_config='logger_config.json',
                  default_level=logging.INFO):
    """
    Setup logging configuration
    """
    log_config = Path(log_config)
    if not log_config.is_file():
        log_config = Path(__file__).parent / str(log_config)
    if log_config.is_file() and save_dir.exists():
        config = read_json(log_config)
        # modify logging paths based on run config
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = str(save_dir / handler['filename'])

        logging.config.dictConfig(config)
    else:
        print("Warning: logging configuration file is not found in {}.".format(
            log_config))
        logging.basicConfig(level=default_level)
