"""
A parser of state files to continuously parse one or many state files and feed
them into a consumer
"""
import os
import re
from typing import List

import util.config as cfg
import env.environment as _env
import statefiles.line_parser_factory as lpf
from util import config


class StateExtractor():
    def __init__(self,component=None):
        self.environment = _env.get_instance()
        self.line_parsers = lpf.get_line_parser(self.environment)
        if component == config.AGENT_COMPONENTS[2]: #wholesale
            self.line_parsers = lpf.get_wholesale_line_parser(self.environment)
        
        self.ignored_states = set()


    def run_through_all_files(self, tickcallback=None, roundcallback=None):
        """high level function to hook into parsing and get messages from environment"""
        files = self.get_state_files()
        for f in files:
            print("parsing file {}".format(f))
            states = self.get_states_from_file(f)
    
            self.parse_state_lines(states, tickcallback)
    
            if roundcallback is not None and callable(roundcallback):
                roundcallback()
    
    
    def get_state_files(self, root_dir=cfg.STATE_FILES_ROOT):
        """
        returns a list of state file paths
        """
        records = os.listdir(root_dir)
        state_files = [os.path.join(root_dir, r, "log", r + ".state") for r in records]
        return state_files
    
    
    def get_states_from_file(self, file, states=None) -> List[str]:
        """Gets a list of specific class lines"""
        state_messages = []
        with open(file) as f:
            for line in f:
                if states is not None and any(s + "::" in line for s in states):
                    state_messages.append(line)
                elif states is None:
                    state_messages.append(line)
    
        return state_messages
    
    
    def get_method(self, msg: str):
        return msg.split("::")[2]
    
    
    def parse_state_lines(self, messages, tick_callback=None):
        """
        Expects a list of messages (lines from the state logs) and updates the environment accordingly.
        It's important to note that the messages need to include all types needed to construct the
        tariffs. That also includes the ticks. It's best to just pass this the whole state file
        :param messages: state lines
        :return: list of
        """
        for msg in messages:
            self.parse_state_line_message(msg, tick_callback)
    
    def parse_state_line_message(self, msg, tick_callback=None):
        # allow for callback after every timeslot is complete
        if "TimeslotUpdate" in msg and tick_callback is not None and callable(tick_callback):
            tick_callback()
    
        msg = msg.strip()
        class_ = self.get_class(msg)
        method = self.get_method(msg)
    
        if class_ not in self.line_parsers:
            self.ignored_states.add((class_, method))
            return
    
        class_handler = self.line_parsers[class_]
        if callable(class_handler):
            class_handler(msg)
        elif class_handler is not None and method in class_handler and callable(class_handler[method]):
            class_handler[method](msg)
        else:
            self.ignored_states.add((class_, method))
    
    def get_class(self, msg: str):
        pattern = re.compile("(org[^:]*)")
        origin = pattern.search(msg)
        return origin.group(0)
    
    
    """Mapping between state origins and corresponding handlers. Most will be None but some are important to the agents
    learning algorithms and will therefore be processed.
    
    Received through `cat *.state | egrep "org[^:]*" -o | sort -u`
    """
