import datetime as dt

def list_dim(a):
    """Recursively gives back the dimensions of a list"""
    if not type(a) == list:
        return []
    return [len(a)] + list_dim(a[0])

def get_now_date_file_ready():
    pattern = "%Y-%m-$d--%H-%M-%S"
    return dt.datetime.now().strftime(pattern)
