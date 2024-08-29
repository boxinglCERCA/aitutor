from datetime import datetime
import io
import hashlib
import re


def convert_dt(string_: str) -> str:
    """Converting string like 'abcDateTime' to 'abcYYMMDDHHMMSS'

    :param string_: str
    :return: str
    """
    now = datetime.now()
    return string_.replace('Date', now.strftime('%y%m%d')).replace('Time', now.strftime('%H%M%S'))


def find_sub_list(sub_list_, list_) -> int:
    """
    Function to find sub list index in the list

    :param sub_list_: list
    :param list_: list
    :return: int. Index of the sub list in list. Or -1 if not found anything
    """
    sll = len(sub_list_)
    for ind in (i for i, e in enumerate(list_) if e == sub_list_[0]):
        if list_[ind:ind + sll] == sub_list_:
            return ind
    return -1


def d_t() -> str:
    """Return current Date and Time in certain format"""
    return datetime.now().strftime("%y/%m/%d %H:%M:%S")


def sum_lists(*args) -> tuple:
    """
    Sum lists or tuples into 1 tuple

    :return: tuple
    """
    return tuple(sum(x) for x in zip(*args))


def char_tuple(c1, c2):
    """Tuple of characters from `c1` to `c2`, inclusive."""
    return tuple(chr(c_code) for c_code in range(ord(c1), ord(c2)+1))


def get_file_md5(file_obj: io.BytesIO, base_value: bytes = None):
    """
    Calculate MD5 hash from binary file object

    :param file_obj: io.BytesIO
    :param base_value: bytes
    :return: bytes
    """
    md5 = hashlib.md5()
    if base_value is not None:
        md5.update(base_value)
    md5.update(file_obj.read())
    file_obj.seek(0)
    return md5


def get_ip_from_docker_name(host_name_: str):
    """Get ip address of the server by docker's container name"""
    result = '127.0.0.1'
    f_in = open('/etc/hosts')
    f_str = f_in.read()
    f_in.close()
    for ip, host_name in re.findall(r'^([0-9.]+)\t([\w]+)', f_str, re.M):
        if host_name_ == host_name:
            return ip
    return result


def rgb2hex(r_value: int, g_value: int, b_value: int):
    """
    Get RGB HEX #FFFFFF from 255, 255, 255

    :param r_value: int
    :param g_value: int
    :param b_value: int
    :return: str
    """
    return "#{0:02x}{1:02x}{2:02x}".format(r_value, g_value, b_value).upper()


def none_to_na(value_str):
    """
    Convert None values to str 'N/A'

    :param value_str: various
    :return: various
    """
    if value_str is None:
        return 'N/A'
    return value_str


def replace_xml_chars(value_str: str) -> str:
    """
    Safe replace XML tags

    :param value_str: str
    :return: str
    """
    return value_str.encode('ascii', 'xmlcharrefreplace').decode('utf8')
