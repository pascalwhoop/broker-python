from gym import Env


class PowerTacEnv(Env):
    def __init__(self):
        super().__init__()

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def render(self, mode='logging'):
        # might implement later
        pass

    def close(self):
        raise NotImplementedError