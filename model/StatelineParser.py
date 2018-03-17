"""
This module helps with the unified way of extracting the parts in a state line
"""

from typing import List


class StatelineParser(object):

    @staticmethod
    def from_state_line(line: str) -> "StatelineParser":
        # TODO what does this do?
        return StatelineParser()

    @staticmethod
    def split_line(line: str) -> List[str]:
        return line.split("::")
