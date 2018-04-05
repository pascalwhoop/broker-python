
def list_dim(a):
    """Recursively gives back the dimensions of a list"""
    if not type(a) == list:
        return []
    return [len(a)] + list_dim(a[0])
