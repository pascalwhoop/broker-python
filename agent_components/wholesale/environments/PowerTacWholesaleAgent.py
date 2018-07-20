from util.learning_utils import TbWriterHelper


class PowerTacWholesaleAgent:
    """Abstract wholesale agent that can act in a `PowerTacEnv`"""
    def __init__(self, tb_log_name):
        self.tb_log_helper = TbWriterHelper(tb_log_name, True)

    def forward(self, env: "PowerTacEnv"):
        """Gets an action based on the environment. The agent is responsible for interpreting the environment"""
        raise NotImplementedError

    def backward(self, env: "PowerTacEnv", action, reward):
        """Receive the reward for the action and learn from historical data"""
        raise NotImplementedError

    def save_model(self):
        raise NotImplementedError