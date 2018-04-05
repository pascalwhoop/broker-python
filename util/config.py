import os
"""
Holds all config variables for python broker
"""
ROUNDING_PRECISION = 3
DATETIME_PATTERN   = "%Y-%m-%dT%H:%M:%S.000Z"
LOG_PATH           = "log/"
LOG_LEVEL          = "INFO"
ADAPTER_HOST       = "localhost"
ADAPTER_PORT       = "1234"
AGENT_COMPONENTS   = ['demand','tariff','wholesale','balancing']
STATE_FILES_ROOT   = "./data/state_files"


###############################
#logging setup
###############################
def get_log_handlers():
    return {
        'file' : {
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
            'default' : { 
                'level': LOG_LEVEL,
                'formatter': 'standard',
                'class': 'logging.StreamHandler',
        },
     },
        'loggers': { 
            '': { 
                'level': LOG_LEVEL,
                'handlers':['default'],
                'propagate': True,
                'formatters': ['standard']
            },
        } 
    }
