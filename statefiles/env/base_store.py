from abc import abstractmethod


class BaseStore(object):
    '''A base class that defines a common structure for all object type stores
    '''


    def __init__(self, key_prop):
        self.key_prop = key_prop
        self._store = {}

    def insert(self, obj):
        # using the key_prop as a key for the store
        self._store[obj[self.key_prop]] = obj

    def find(self, key):
        return self._store[key]
