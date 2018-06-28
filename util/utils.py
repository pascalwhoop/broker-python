import datetime as dt
import warnings
import functools
import numpy as np

def list_dim(a):
    """Recursively gives back the dimensions of a list"""
    if not type(a) == list:
        return []
    return [len(a)] + list_dim(a[0])

def get_now_date_file_ready():
    pattern = "%Y-%m-%d--%H-%M-%S"
    return dt.datetime.now().strftime(pattern)



# source: https://stackoverflow.com/questions/2536307/how-do-i-deprecate-python-functions
def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return new_func
