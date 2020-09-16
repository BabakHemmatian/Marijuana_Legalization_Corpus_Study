from pip._vendor.distlib.compat import raw_input

from defaults import *
from lda_config import *
#TODO: Should this be NN_Utils.py / what do we want to import from here
from NN_Utils import *

# ENTIRE_CORPUS should be True if NN is True
if NN:
    try:
        assert ENTIRE_CORPUS
    except AssertionError:
        raw_input("NN is True. Overriding config settings and setting ENTIRE_CORPUS to True. Press Enter to continue.")
        ENTIRE_CORPUS=True
