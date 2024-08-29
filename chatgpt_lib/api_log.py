import logging
import logging.handlers
import os
import traceback

from flask import g
from chatgpt_lib import WORK_DIR
from chatgpt_lib.api_helpers import convert_dt
from chatgpt_lib.api_configs import configuration

log_levels = {
        'critical': logging.CRITICAL,
        'error': logging.ERROR,
        'warning': logging.WARNING,
        'info': logging.INFO,
        'debug': logging.DEBUG,
        'notset': logging.NOTSET
    }


class MakeLogging:
    """Class for logging to console and file"""
    def __init__(self):
        print(self)
        self.log_file_name = ""
        self.log_file_level = ""
        try:
            try:
                self.log_file_name, self.log_file_level = self.get_log_options()
            except Exception:
                pass
            self.logger = logging.getLogger(os.path.basename(__file__))
            for handler in self.logger.handlers:
                self.logger.removeHandler(handler)
            # init console logger with Info level
            self.logger.setLevel(logging.DEBUG)
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter(fmt='\n%(asctime)s %(levelname)s %(message)s'))
            self.logger.addHandler(ch)
            if self.log_file_name:
                # create folders if necessary
                os.makedirs(os.path.dirname(self.log_file_name), exist_ok=True)
                # init rotating file logger with 10MB max , 3 max files
                fh = logging.handlers.RotatingFileHandler(self.log_file_name,
                                                          encoding='utf8',
                                                          maxBytes=10000000,
                                                          backupCount=3)
                fh.setFormatter(logging.Formatter(fmt='\n%(asctime)s %(levelname)s %(message)s'))
                fh.setLevel(self.log_file_level)
                self.logger.addHandler(fh)
        except Exception:
            print('Logging initializing error')
            exit(-1)

    def close(self):
        for handler in self.logger.handlers:
            self.logger.removeHandler(handler)

    @staticmethod
    def get_log_options():
        if configuration.config_dict.get('log_file'):
            log_file_name = os.path.join(WORK_DIR, convert_dt(configuration.config_dict.get('log_file')))
        else:
            log_file_name = ''
        log_file_level = log_levels[configuration.config_dict.get('log_file_level', 'debug').lower()]
        return log_file_name, log_file_level


log_obj = MakeLogging()


def log(lvl, msg, *args, no_sql=False, no_str=False, new_request=False, **kwargs):
    """
    Log message.

    :param lvl: str
    :param msg: str
    :param no_sql: bool. True if no log to sql.
    :param no_str: bool. True if no log to console and file
    :param new_request: bool. True if this is logging of new HTTP request
    """
    # log to MySQL
    if not no_sql:
        try:
            g.log_sql.log(lvl, msg, new_request)
        except Exception:
            pass
    # log to file
    if not no_str:
        lvl = log_levels[lvl.lower()]
        try:
            log_obj.logger.log(lvl, msg, *args, **kwargs)
        except Exception as exc:
            print('Undefined error: ' + repr(exc) + '\n' + traceback.format_exc())


def keyb_exc(func, *args, **kwargs):
    try:
        func(*args, **kwargs)
    except KeyboardInterrupt:
        log('error', 'Keyboard interrupt')
        exit(-1)
    except Exception as exc:
        log('error', 'Undefined error: ' + repr(exc) + '\n' + traceback.format_exc())
