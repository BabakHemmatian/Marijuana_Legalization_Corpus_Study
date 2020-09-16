# import the required modules, functions and parameters

from config import *

# TODO: inherit all the variables and everything from the neural network
# analysis and check them right before network training
def NN_param_typecheck():
    assert 0 < learning_rate and 1 > learning_rate
    assert type(batchSz) is int
    assert type(embedSz) is int
    assert type(hiddenSz) is int
    assert type(ff1Sz) is int
    assert type(ff2Sz) is int
    assert 0 < keepP and 1 >= keepP
    assert type(l2regularization) is bool
    if l2regularization == True:
        assert 0 < alpha and 1 > alpha
    assert type(early_stopping) is bool
