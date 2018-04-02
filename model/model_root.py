from functools import reduce
from typing import List

class ModelRoot (object):
    def get_values_list(self) -> List:
        return list(self.__dict__.values())

    def __str__(self):
        d = self.__dict__
        lines = map(lambda i: "{} : {}, ".format(i, d[i]), d)
        return reduce(lambda sum, st: sum + st, lines)

