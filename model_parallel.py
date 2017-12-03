import numpy as np
import logging
from logging.handlers import QueueHandler
import ctypes
import os
from enum import Enum
from multiprocessing import Manager, Event, Value, Array
from multiprocessing import Queue
from queue import Empty, Full
import uuid
import time

from hyper_params import *
from go_board import *
from model import *

class ControlActions(Enum):
    SAVE = 1,
    SAVE_COMPLETED = 2

class MultiProcessModelProxy(Model):
    def __init__(self, model_file:str=None, 
                    predict_queue:Queue=None,
                    history_queue=None,
                    stop_event:Event=None,
                    results:dict=None,
                    train_model_event:Event=None,
                    train_completed_event:Event=None,
                    prediction_ready_event:Event=None,
                    control_pipe=None,
                    log_queue=None):
        super(MultiProcessModelProxy, self).__init__()

        self.model_file = model_file
        self.predict_queue = predict_queue
        self.history_queue = history_queue        
        self.stop_event = stop_event
        self.results = results
        self.train_model_event = train_model_event
        self.train_completed_event = train_completed_event
        self.control_pipe = control_pipe
        self.prediction_ready_event = prediction_ready_event
        self.log_queue = log_queue
        self.train_batch_index = 0

        logging.info("MultiProcessModelProxy init: %s", model_file)

    def train_on_hist_batch(self, hist_batch, batch_index):
        raise NotImplementedError()

    def predict(self, state):
        result_id = uuid.uuid4()
        self.predict_queue.put((result_id, state), block=True)        
        while True:
            try:
                self.prediction_ready_event.wait()
                ret_val, ret_probs = self.results[result_id]
                del self.results[result_id]
                break
            except KeyError:
                pass
        return ret_val, ret_probs 
        
    def train(self):
        self.train_completed_event.clear()
        self.train_model_event.set()
        self.train_completed_event.wait()            
    
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
    logger.addHandler(_LOG_QUEUE_HANDLER)

def _run_model_prediction(model, results:dict, buffer:list):
    states = [state for _, state in buffer]
    values, act_probs = model.predict(states)
    for value, act_prob, (result_id, _) in zip(values, act_probs, buffer):
        results[result_id] = (value, act_prob)

def _train_model(model:MultiProcessModelProxy, history_queue, start_batch_index:int, num_train_iter:int):
    for train_iter_idx in np.arange(start_batch_index, start_batch_index+num_train_iter):
        logging.debug("Network train: %d", train_iter_idx)
        batch_indices = np.random.choice(np.arange(len(history_queue)), size=BATCH_SIZE)
        batch = [history_queue[indx] for indx in batch_indices]
        model.train_on_hist_batch(batch, train_iter_idx)            


def run_loop(model_proxy:MultiProcessModelProxy):    
    logging.info("run_loop: model_file: %s", model_proxy.model_file)

    # this method runs on it's own thread
    model = SimpleNNModel()
    if model_proxy.model_file is not None:
        model.load(model_proxy.model_file)

    local_predict_buffer = []
    should_stop = False        
    while not should_stop:  
        try:
            msg = model_proxy.predict_queue.get(block=True, timeout=1)
            local_predict_buffer.append(msg)
            while len(local_predict_buffer) < PREDICTION_QUEUE_BATCH_SIZE:
                msg = model_proxy.predict_queue.get(block=False)
                local_predict_buffer.append(msg)
        except Empty:
            pass            

        if  len(local_predict_buffer) > 0:            
            logging.debug("run prediction: %d", len(local_predict_buffer))
            _run_model_prediction(model, model_proxy.results, local_predict_buffer)
            local_predict_buffer = []
            model_proxy.prediction_ready_event.set()
            model_proxy.prediction_ready_event.clear()
                    
        if model_proxy.control_pipe and model_proxy.control_pipe.poll(0):
            act, args = model_proxy.control_pipe.recv()
            if act == ControlActions.SAVE:
                model.save(args)
                model_proxy.control_pipe.send(ControlActions.SAVE_COMPLETED)
            else:
                raise ValueError("Unknown action %d" % (act))

        should_train = model_proxy.train_model_event.wait(0)
        if should_train:
            _train_model(model, model_proxy.history_queue, model_proxy.train_batch_index, NUM_TRAIN_LOOP_ITER)
            model_proxy.train_batch_index += NUM_TRAIN_LOOP_ITER
            model_proxy.train_model_event.clear()
            model_proxy.train_completed_event.set()
        
        should_stop = model_proxy.stop_event.wait(0)
