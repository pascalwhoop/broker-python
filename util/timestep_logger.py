import os
import util.config as cfg
from util.utils import get_now_date_file_ready



def log_message(dispatched_message):
    name_ = type(dispatched_message).__name__
    file_handler = get_file_handler(name_)
    log_message(file_handler, dispatched_message)

def get_file_handler(type_):
    return open(os.path.join(_path, "{}.csv".format(type_)))
