import tensorflow
import numpy
import keras


# create learning model (simple NN with a few layers
# I want to classify inputs to an output value (regression) describing the value of a tariff based on a state
# the tariff value should be based on:
    # total sum of charges against customers
    # total sum of payments to customers (good payments such as production balancing)
    # how many customers are subscribed
    # revenue / timeslot


# to learn the model I need to feed it:
    # INPUT environment overview (aka current competition)
    # INPUT tariff values
    # OUTPUT normalized value of "quality" of tariff
        # --> calculate based on revenue made from tariff

# 1. get list of state files
    # 2. iterate over state files
    #
