from typing import List


class ModelRoot (object):
    def get_values_list(self) -> List:
        return list(self.__dict__.values())