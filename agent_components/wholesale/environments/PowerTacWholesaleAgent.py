from util.learning_utils import TbWriterHelper


class PowerTacWholesaleAgent:
    """Abstract wholesale agent that can act in a `PowerTacEnv`"""
    def __init__(self, full_name):
        self.tb_log_helper = TbWriterHelper(full_name, True)
        self.full_name = full_name

    def forward(self, env: "PowerTacEnv"):
        """Gets an action based on the environment. The agent is responsible for interpreting the environment"""
        raise NotImplementedError

    def backward(self, env: "PowerTacEnv", action, reward):
        """Receive the reward for the action and learn from historical data"""
        raise NotImplementedError

    def save_model(self):
        raise NotImplementedError

    def load_model(self, model_name):
        raise NotImplementedError
