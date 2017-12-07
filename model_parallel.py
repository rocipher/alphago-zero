import numpy as np
import logging
from logging.handlers import QueueHandler
import ctypes
import os
from enum import Enum
from multiprocessing import Manager, Event, Value, Array
from multiprocessing import Queue, Pipe
import multiprocessing
from queue import Empty, Full
import uuid
import time
from collections import deque

from hyper_params import *
from go_board import *
from model import *

class ControlActions(Enum):
    STOP = -1,
    SAVE = 1,
    SAVE_COMPLETED = 2,
    TRAIN = 3,
    TRAIN_COMPLETED = 4

class MultiProcessModelProxy(Model):
    def __init__(self, pipe_id=None,
                    model_file:str=None,                     
                    predict_queue:Queue=None,
                    history_queue=None,
                    results_pipe=None):
        super(MultiProcessModelProxy, self).__init__()

        self.pipe_id = pipe_id
        self.model_file = model_file
        self.predict_queue = predict_queue
        self.history_queue = history_queue        
        self.results_pipe = results_pipe

        logging.info("MultiProcessModelProxy init: %s", model_file)

    def train_on_hist_batch(self, hist_batch, batch_index):
        raise NotImplementedError()

    def predict(self, state):
        self.predict_queue.put((self.pipe_id, state), block=True)
        ret_val, ret_probs = self.results_pipe.recv()
        return ret_val, ret_probs 
           
    def copy(self):
        raise NotImplementedError()

    def save(self, filePath):
        raise NotImplementedError()
        


def init_log_queue(log_queue:Queue):
    if "_LOG_QUEUE_HANDLER" in globals():
        return
    global _LOG_QUEUE_HANDLER    
    _LOG_QUEUE_HANDLER = QueueHandler(log_queue)
    logger = logging.getLogger()
    logger.setLevel(LOG_LEVEL)
    logger.handlers.clear()
    logger.addHandler(_LOG_QUEUE_HANDLER)
    mp_logger = multiprocessing.log_to_stderr()    
    mp_logger.setLevel(LOG_LEVEL)

def _run_model_prediction(model, buffer:list):    
    states = [state for _, state in buffer]
    values, act_probs = model.predict(states)
    results = {}
    for value, act_prob, (client_id, _) in zip(values, act_probs, buffer):
        results[client_id] = (value, act_prob)
    return results

def _train_model(model:MultiProcessModelProxy, history_queue, iter_index:int, num_train_iter:int):
    model.set_seed(iter_index)
    for train_iter_idx in np.arange(iter_index*num_train_iter, (iter_index+1)*num_train_iter):
        logging.debug("Network train: train_batch_idx: %04d", train_iter_idx)
        batch_indices = np.random.choice(len(history_queue), size=BATCH_SIZE)
        batch = [history_queue[indx] for indx in batch_indices]
        model.train_on_hist_batch(batch, train_iter_idx)


def run_loop(model_proxy:MultiProcessModelProxy, control_pipe:Pipe, clients:dict):    
    logging.info("run_loop: model_file: %s", model_proxy.model_file)

    # this method runs on its own thread
    model = SimpleNNModel()
    if model_proxy.model_file is not None:
        model.load(model_proxy.model_file)

    local_predict_buffer = []
    should_stop = False        

    while not should_stop:  
        try:    
            msg = model_proxy.predict_queue.get(block=True, timeout=1.)
            local_predict_buffer.append(msg)            
            while len(local_predict_buffer) < PREDICTION_QUEUE_BATCH_SIZE:
                msg = model_proxy.predict_queue.get(block=False)
                local_predict_buffer.append(msg)
        except Empty:
            pass

        if  len(local_predict_buffer) > 0:         
            logging.debug("run prediction: %d", len(local_predict_buffer))
            results = _run_model_prediction(model, local_predict_buffer)
            for client_id, result in results.items():
                clients[client_id].send(result)
            local_predict_buffer = []

        if control_pipe.poll(0):
            act, args = control_pipe.recv()
            logging.info("Control pipe received: act=%s, args=%s", act, args)
            if act == ControlActions.SAVE:
                model.save(args)
                control_pipe.send((ControlActions.SAVE_COMPLETED, None))
            elif act == ControlActions.TRAIN:
                iter_index = args
                _train_model(model, model_proxy.history_queue, iter_index, NUM_TRAIN_LOOP_ITER)
                control_pipe.send((ControlActions.TRAIN_COMPLETED, None))
            elif act == ControlActions.STOP:
                should_stop = True
            else:
                raise ValueError("Unknown action %d" % (act))
    #end while
