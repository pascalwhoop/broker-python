# PowerTAC Python Adapter to broker-adapter

##Installation

1. TODO
2. Install all packages with pip, ideally in a virtual environment to avoid conflicts with locally installed packages

##Architecture

- `scripts` holds utility script that can be run independently files
- `notebooks` holds Jupyter notebooks 
- `main.py` is the main entry of the agent and it expects parameters to control what to be started
- `communication` holds the abstraction to the powertac server as well as the pub/sub tooling to let components send
  messages to each other
- `data` and `Graph` are to be ignored in git and they hold training data and tensorboard logs respectively

## Tests

execute `run_tests.sh` which watches the files in the local folder and executes the unit tests when changes have been
detected

## Starting Jupyter Notebooks
start them with `jupyter notebook`. The first cell adds the project to the sys.path so we can include local path
modules. 

## Code Generation for GRPC connector
run the shell script in the root of the project. but first run `pip3 install grpcio-tools` and make sure you have python3 and pip3 installed and in your PATH.

## Todo

- Model definition based on XML messages
- Turning Events into consistent "Environment"
- Storing all messages in local memory
- Pickling ?
- ...
