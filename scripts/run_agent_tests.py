"""Generates a number of calls to the agent CLI so that one doesn't have to manually call them all but rather one may
call this script and then let the computer run for a day (or two)"""

from subprocess import check_output

# MAKE SURE TO CALL THIS FROM THE PROJECT ROOT ( $ python scripts/run_agent_tests.py )


# ENTER ALL COMBINATIONS HERE
# Any combination of the parameters is added to the queue and executed one after another
config_map = {
    "--agent-type": ["dqn", "vpg", "random"],
    "--network" : ["vnn32x2", "bn_vnn32x2"],
    "--action-type": ["continuous", "discrete", "twoarmedbandit"],
    "--preprocessing": ["simple", "simplenorm"],
    "--reward": ["step_close_relative_mprice", "market_relative_prices"],
    "--games": ["10"]
}
AGENT_EXECUTABLE = "agent --log-level ERROR wholesale --tag automatic_testing"


#running the agent once each for each possible combination of the above configurations
def generate_calls():
    parameters = list(config_map.keys())
    calls = []
    loop_through_and_add(calls, AGENT_EXECUTABLE, parameters, 0)
    return calls


def loop_through_and_add(calls, current_call, params, current_param_index):
    """loops through the config_map and generates a combination of each"""
    current_param = params[current_param_index]
    for p in config_map[current_param]:
        call = " ".join([current_call, current_param, p])
        if params[current_param_index] == params[-1]:
            #last in line, iterate through all and add each
            calls.append(call)
        else:
            loop_through_and_add(calls, call, params, current_param_index+1)


def call_all(calls):
    for c in calls:
        print("CALLING: ")
        print(c)
        output = check_output(c.split(' '))
        lines = output.decode('utf-8').split('\n')
        reward = lines[-2]
        #remember this for later
        with open("log/test_all.log", "w+") as f:
            f.write(" ---> ".join([c, reward]) + "\n")

def print_calls(calls):
    print("-" * 80)
    for c in calls:
        print(c)
    print("-" * 80)

def main():
    calls = generate_calls()
    print_calls(calls)
    call_all(calls)


if __name__ == "__main__":
    main()
