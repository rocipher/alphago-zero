import unittest
import logging

from model import *
from hyper_params import *

class TestGoBoard(unittest.TestCase):

    def test_create_model(self):
        logging.debug("start")
        model = SimpleNNModel()
        state = GoBoard.create_zero_state(0)
        val, act_prob = model.predict(state)
        logging.debug(val.shape)
    
    def test_copy(self):
        model = SimpleNNModel()
        new_model = model.copy()        


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    
    unittest.main()