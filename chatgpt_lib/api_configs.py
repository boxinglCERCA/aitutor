import codecs
import os
from datetime import datetime
from traceback import format_exc
from chatgpt_lib.api_helpers import get_ip_from_docker_name
from chatgpt_lib import WORK_DIR


class SGRELConfigParser:
    def __init__(self, config_folder: str = 'conf', str_sections: tuple = (), parse_mysql: bool = True,
                 init: bool = True):
        """Initialize config object from config_file

        :param config_folder: str
        :param str_sections: tuple of sections where to search for list of raw lines
        :param parse_mysql: bool. True if mysql config is necessary
        :param init: bool. False if just object to be created
        """
        if not init:
            return
        config_folder = os.path.join(WORK_DIR, config_folder)
        self.settings_table = 'api_settings'
        self.settings_table_last_update = None
        self.settings_dict = dict()
        # dict {url name: url, ...}
        self.urls_dict = dict()
        # dict {url name: page_title, ...}
        self.urls_titles = dict()
        # dict {url: url name, ...}
        self.urls_reverse = dict()
        # dict {url name: roles set, ...}
        self.urls_roles = dict()
        # dict {url name: options set, ...}
        self.urls_options = dict()
        # dict {url_name: meta}
        self.urls_meta = dict()
        self.str_sections = str_sections
        # `api_settings`.`data_timestamp` to check
        self.data_timestamp = datetime(1999, 2, 3, 4, 5, 6, 7).timestamp()
        error_msg = ''
        try:
            error_msg = 'parsing config.ini'
            self.all_cfg = dict()
            self.parse_config(codecs.open(os.path.join(config_folder, 'config.ini'), 'r', encoding='utf8'))
            self.config_dict = self.all_cfg['MainSection']
            self.release = self.get_yes_no_value("release", True)

            if parse_mysql:
                error_msg = 'parsing mysql settings'
                # check mysql ini settings
                self.check_important_keys((
                        'mysql_login',
                        'mysql_password',
                        'mysql_address',
                        'mysql_port',
                        'mysql_db_name'))
                # populate mysql credentials dictionary
                cd = self.config_dict
                self.mysql_config = {'host': cd['mysql_address'],
                                     'user': cd['mysql_login'],
                                     'port': int(cd['mysql_port']),
                                     'password': cd['mysql_password'],
                                     'charset': 'utf8',
                                     'db': cd['mysql_db_name']}
                # get ip address of MySQL server if it is in another docker container
                if 'docker_container' in self.mysql_config['host']:
                    self.mysql_config['host'] = get_ip_from_docker_name(configuration.mysql_config['host'][17:])

        except Exception:
            print('Config initialization error at %s.' % error_msg)
            print(format_exc())
            exit(-1)

    def parse_config(self, config_file):
        """Parse all configuration file to dict of dicts self.all_cfg"""
        cur_section = ""
        for line in config_file:
            line = line.strip()
            if not line:
                continue
            if line[0] == '#':
                continue
            if line[0] == '[':
                line = line.strip('[]')
                cur_section = line
                self.all_cfg[cur_section] = dict()
                continue
            if cur_section not in self.str_sections:
                key, value = self.get_key_value(line)
                self.all_cfg[cur_section][key] = value
            else:
                if self.all_cfg[cur_section]:
                    self.all_cfg[cur_section].append(line)
                else:
                    self.all_cfg[cur_section] = [line]

    @staticmethod
    def get_key_value(line: str):
        """Parse line in configuration file"""
        opened = False
        key = ''
        value = ''
        cur_item = 'key'
        for symbol in line.strip():
            if opened:
                if symbol in ['"', "'"]:
                    opened = False
                    continue
                if cur_item == 'key':
                    key += symbol
                    continue
                if cur_item == 'value':
                    value += symbol
                    continue
            else:
                if not symbol.strip():
                    continue
                if symbol in ['"', "'"]:
                    opened = True
                    continue
                if symbol == '=':
                    cur_item = 'value'
                    continue
                if cur_item == 'key':
                    key += symbol
                    continue
                if cur_item == 'value':
                    value += symbol
                    continue
        return key, value

    def sections(self):
        return self.all_cfg.keys()

    def check_important_keys(self, important_keys: tuple):
        """Check important keys in the config dictionary"""
        for key in important_keys:
            if key not in self.config_dict:
                print('Config key is missing: ' + key)
                exit(-1)

    def get_yes_no_value(self, c_name: str, default_value: bool = False, c_dict: dict = None):
        """Get True/False from yes/no config value

        :param c_name: str. Name of option
        :param default_value: bool. Default value if no option found.
        :param c_dict: dict. Dictionary that's used. Self.config_dict by default.
        :return: bool
        """
        if c_dict is None:
            c_dict = self.config_dict
        if c_name in c_dict:
            if str(c_dict[c_name])[0].lower() == 'y':
                return True
            else:
                return False
        return default_value


configuration = SGRELConfigParser('', init=False)
