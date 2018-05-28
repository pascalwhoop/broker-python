import os
"""
Holds all config variables for python broker
"""
ROUNDING_PRECISION = 3
DATETIME_PATTERN   = "%Y-%m-%dT%H:%M:%S.000Z"
TENSORBOARD_PATH   = "tensorboard/"
DATA_PATH          = "data/"
MODEL_PATH         = os.path.join(DATA_PATH, "models")
LOG_PATH           = "log/"
LOG_LEVEL          = "INFO"
#ADAPTER_HOST       = "localhost"
#ADAPTER_PORT       = "1234"
AGENT_COMPONENTS   = ['demand', 'tariff', 'wholesale', 'balancing']
STATE_FILES_ROOT   = "./data/state_files"

GRPC_PORT = 50051

###############################
# Component configuration
###############################
DEMAND_LEARNER = "gru"
WHOLESALE_LEARNER = "rl"
TARIFF_LEARNER = "dense"

# learning config
VALIDATION_SPLIT         = 0.10  #don't go below 10%, otherwise the hotencoding breaks (ugly bug but annoying to fix)
# demand config
DEMAND_LEARNING_USAGE_PATH = '../powertac-tools/logtool-examples/data/'
DEMAND_VALIDATION_PART   = 0.05
DEMAND_FORECAST_DISTANCE = 24
DEMAND_SEQUENCE_LENGTH   = 48  # one week sequences because that's a probable range for patterns
DEMAND_SEQUENCE_STRIDE   = 4  #every 4 timesteps will be used for a new forecasting request
DEMAND_BATCH_SIZE        = 32  # TODO... higher? number of sequences to feed to the model at once and whose errors are added up before propagated
DEMAND_SAMPLING_RATE     = 1  # assuming correlation between hours somewhere in this range (6h ago, 12h ago, 18h ago, 24h ago,..)
#GRU_DEMAND_DATAPOINTS_PER_TS = 17  # number of datapoints in each timestep. That's customer data, weather, usage etc
#GRU_DEMAND_DATAPOINTS_PER_TS = 1  # solely based on previous usage version
DEMAND_DATAPOINTS_PER_TS = 47  # sparse version
DEMAND_GRU_EPOCHS_P_GAME = 20
DEMAND_ONE_WEEK          = 24*7
DEMAND_LOGREG_FEATURES   = True

#wholesale
#WHOLESALE_AGENT_TEACHER_BROKER = 'tactex'
WHOLESALE_MIN_KWH_PRICE = -2.0 #
WHOLESALE_OPEN_FOR_TRADING_PARALLEL = 24
WHOLESALE_LEARNING_USAGE_PATH = '../powertac-tools/logtool-examples/data/'
WHOLESALE_FORECASTS_TYPE = 'perfect' # 'error1', 'error2', ..., 'forecast'
WHOLESALE_HISTORICAL_DATA_LENGTH = 168
WHOLESALE_STEPS_PER_TRIAL = 24
WHOLESALE_FORECAST_ERROR_PER_TS = 0.02



###############################
#logging setup
###############################
def get_log_handlers():
    return {
        'file': {
            'level': LOG_LEVEL,
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'mode': 'a',
            'filename': os.path.join(os.curdir, LOG_PATH, "agent.log")
        }
    }


def get_log_config():
    return {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
        },
        'handlers': {
            'default': {
                'level': LOG_LEVEL,
                'formatter': 'standard',
                'class': 'logging.StreamHandler',
            },
        },
        'loggers': {
            '': {
                'level': LOG_LEVEL,
                'handlers': ['default'],
                'propagate': True,
                'formatters': ['standard']
            },
        }
    }
