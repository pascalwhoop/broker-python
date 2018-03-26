import time as t

def time_function(function, params):
    start = t.time()
    function(*params)
    end = t.time()
    print(end-start)
