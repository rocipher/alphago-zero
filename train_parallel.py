import gc
import sys
import logging
from logging.handlers import QueueHandler, QueueListener
import numpy as np
import datetime
import signal
import ctypes
from itertools import repeat, cycle, product
import traceback
from multiprocessing.pool import Pool
from multiprocessing import Manager, Queue
from multiprocessing import Process, Pipe, Event
import multiprocessing

import model_parallel
from model_parallel import ControlActions, MultiProcessModelProxy
import hyper_params as hp
import game

from collections import namedtuple
from namedlist import namedlist

ModelWorker = namedtuple("ModelWorker", ["process", "model_proxy", "control_pipe", "client_pipes"])


class WorkersState:
    def __init__(self, model_proxies:list=[], manager:Manager=None):
        self.model_proxies = model_proxies
        self.available_queue = manager.Queue()
        for worker_proxy_id in range(len(model_proxies)):
            self.available_queue.put(worker_proxy_id)
    
    def aquire_available_model_proxy(self):
        worker_proxy_id = self.available_queue.get(block=True)
        model_proxy = self.model_proxies[worker_proxy_id]
        return worker_proxy_id, model_proxy

    def release_model_proxy(self, worker_proxy_id):
        self.available_queue.put(worker_proxy_id)        


def pool_worker_init(log_queue):
    signal.signal(signal.SIGINT, signal.SIG_IGN)    
    model_parallel.init_log_queue(log_queue)    


def self_play_worker(iter_index: int, game_index:int, history_queue:Queue, worker_state:WorkersState):
    np.random.seed(int(hp.SEED * 1e6 + iter_index * 1e4 + game_index))
        
    worker_proxy_id, model_proxy = worker_state.aquire_available_model_proxy()
    logging.info("Self-play game %d using model proxy: %d", game_index, worker_proxy_id)
    result = game.self_play(game_index, model_proxy, history_queue, 
                    noise_alpha=hp.DIRICHLET_ALPHA, temperatures=hp.TEMPERATURES)
    worker_state.release_model_proxy(worker_proxy_id)
    return result

    
def multi_player_worker(iter_index: int, game_index:int, model_a_state:WorkersState, model_b_state:WorkersState):
    np.random.seed(int(hp.SEED * 1e6 + iter_index * 1e4 + game_index))

    model_a_id, model_a = model_a_state.aquire_available_model_proxy()
    model_b_id, model_b = model_b_state.aquire_available_model_proxy()
    result = game.two_player_play(game_index, model_a, model_b)
    model_a_state.release_model_proxy(model_a_id)
    model_b_state.release_model_proxy(model_b_id)
    return result

def nn_worker(log_queue:Queue, model_proxy:MultiProcessModelProxy, control_pipe:Pipe, clients:dict):
    model_parallel.init_log_queue(log_queue)
    np.random.seed(hp.SEED)
    model_parallel.run_loop(model_proxy, control_pipe, clients)

def spawn_nn_process(model_file:str, manager:Manager, 
                     history_queue, log_queue: Queue,
                     num_clients:int):
    parent_ctrl_pipe, child_ctrl_pipe = Pipe()    
    send_pipe_ends = {}
    recv_pipe_ends = {}
    for client_id in range(num_clients):
        recv_pipe_ends[client_id], send_pipe_ends[client_id] = Pipe(duplex=False)
    pl_model = MultiProcessModelProxy(pipe_id=None,  
                                    model_file=model_file,
                                    predict_queue=manager.Queue(maxsize=hp.PREDICTION_QUEUE_BATCH_SIZE),
                                    history_queue=history_queue)
    nn_process = Process(target=nn_worker, args=(log_queue, pl_model, parent_ctrl_pipe, send_pipe_ends))
    nn_process.daemon = True
    nn_process.start()
    return ModelWorker(nn_process, pl_model, child_ctrl_pipe, recv_pipe_ends)


def spawn_workers(num_workers, *args):
    workers = []
    for pred_model_ix in range(num_workers):
        model_worker = spawn_nn_process(*args)
        workers.append(model_worker)
    return workers

def stop_and_join_workers(*workers):
    try:
        for worker in workers:
            worker.control_pipe.send((ControlActions.STOP, None))
        for worker in workers:
            worker.process.join(1.)
    finally:
        terminate_workers(*workers)

def terminate_workers(*workers):
    for worker in workers:
        worker.process.terminate()

def save_worker_model(model_worker, model_file):
    model_worker.control_pipe.send((ControlActions.SAVE, model_file))
    if model_worker.control_pipe.poll(30.):
        save_result, _ = model_worker.control_pipe.recv()
        if save_result != ControlActions.SAVE_COMPLETED:
            raise Exception("Save model failed: %s" % model_file)
    else:
        logging.error("Model save action didn't receive an answer! model_file: %s", model_file)


def spawn_predict_proxies(num_workers:int, num_proxies_per_worker:int, model_file:str, 
                        manager:Manager, history_queue:Queue, log_queue:Queue):
    workers = spawn_workers(num_workers, model_file, manager, history_queue, log_queue, num_proxies_per_worker)
    wstate = WorkersState(model_proxies=[MultiProcessModelProxy(pipe_id=pipe_id,
                                                                predict_queue=w.model_proxy.predict_queue,
                                                                results_pipe=pipe)
                                                    for w in workers
                                                    for pipe_id, pipe in w.client_pipes.items()],
                            manager=manager)
    return workers, wstate
    

def train_loop(manager:Manager, log_queue:Queue):
    pool = Pool(processes=hp.NUM_POOL_WORKERS, 
                initializer=pool_worker_init,   
                initargs=(log_queue,),
                maxtasksperchild=hp.MAX_GAMES_PER_POOL_WORKER)

    history_queue = manager.list()    

    self_play_model_file = hp.SELF_PLAY_MODEL_FILE
    trained_model_file = hp.TRAIN_MODEL_FILE

    num_clients_per_predict_worker = hp.NUM_POOL_WORKERS // hp.NUM_PREDICT_WORKERS

    self_play_predict_workers, self_play_model_wstate = spawn_predict_proxies(hp.NUM_PREDICT_WORKERS, 
                        num_clients_per_predict_worker, self_play_model_file, manager, history_queue, log_queue)

    train_workers, train_model_wstate = spawn_predict_proxies(1, hp.NUM_POOL_WORKERS, self_play_model_file, 
                                            manager, history_queue, log_queue)
    assert len(train_workers) == 1
    train_worker = train_workers[0]
    
    try:
        for iter_index in range(hp.START_ITER, hp.START_ITER+hp.NUM_ITER):
            # self-play
            logging.info("Iter %d: Starting self-play", iter_index)            
            self_play_results = pool.starmap(self_play_worker, 
                                    zip(repeat(iter_index), range(hp.NUM_GAMES), 
                                    repeat(history_queue), repeat(self_play_model_wstate)))
            logging.info("Iter %d, self-play results: %s", iter_index, self_play_results)

            # train NN
            logging.info("Iter %d: Starting network train", iter_index)
            train_worker.control_pipe.send((ControlActions.TRAIN, iter_index))
            act, result = train_worker.control_pipe.recv()
            assert act == ControlActions.TRAIN_COMPLETED
            logging.info("Iter %d: Ended network train", iter_index)

            # eval
            logging.info("Iter %d: Starting evaluation", iter_index)
            eval_results = pool.starmap(multi_player_worker, zip(
                                    repeat(iter_index), range(hp.NUM_EVAL_GAMES),
                                    repeat(train_model_wstate), repeat(self_play_model_wstate)))
            logging.info("Iter %d: Evaluation end: results: %s", iter_index, eval_results)
            outcomes = np.array([outcome for _,outcome in eval_results])
            trained_model_win_ratio = np.sum(outcomes == hp.OUTCOME_WIN_PLAYER_1)/len(outcomes)

            logging.info("Iter %d evaluation: trained_model_win percent : %.2f%%", iter_index, trained_model_win_ratio*100)

            if trained_model_win_ratio > hp.MIN_MODEL_REPLACEMENT_WIN_RATIO:
                stop_and_join_workers(*self_play_predict_workers)
                del self_play_model_wstate                
                del self_play_predict_workers
                                
                self_play_model_file = "%s/model-best-%s-%d-%.0f.h5" % (hp.OUTPUT_DIR, log_id, iter_index, trained_model_win_ratio*100)
                save_worker_model(train_worker, self_play_model_file)                
                
                self_play_predict_workers, self_play_model_wstate = spawn_predict_proxies(hp.NUM_PREDICT_WORKERS, 
                        num_clients_per_predict_worker, self_play_model_file, manager, history_queue, log_queue)

            trained_model_file = "%s/model-train-%s-%d-%.0f.h5" % (hp.OUTPUT_DIR, log_id, iter_index, trained_model_win_ratio*100)
            save_worker_model(train_worker, trained_model_file)

            gc.collect()
        # end iter loop
    except Exception as e:
        if e is KeyboardInterrupt or e is SystemExit:
            logging.info("Terminated by user.")
        else:
            logging.error("Error: %s", e)
        pool.terminate()
        pool.join()
        terminate_workers(train_worker, *self_play_predict_workers)
        raise e
    else:
        stop_and_join_workers(train_worker, *self_play_predict_workers)        
        logging.info('Done successfully.')
    finally:
        pool.close()        
        

def logger_init(manager: Manager):
    log_queue = manager.Queue()

    formatter = logging.Formatter("%(levelname)s: %(asctime)s - %(process)s - %(message)s")
    file_handler = logging.FileHandler('%s/%s.log' % (hp.OUTPUT_DIR, log_id))
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    queue_listener = QueueListener(log_queue, file_handler, stream_handler)
    queue_listener.start()

    logger = logging.getLogger()
    logger.setLevel(hp.LOG_LEVEL)
    logger.handlers.clear()
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    mp_logger = multiprocessing.log_to_stderr()
    mp_logger.setLevel(hp.LOG_LEVEL)

    return queue_listener, log_queue


if __name__ == "__main__":
    log_id = "ago-%s-%dx%d-%d" % (datetime.datetime.now().strftime("%m%d%H%M"),
                                 hp.BOARD_SIZE, hp.BOARD_SIZE, hp.MCTS_STEPS)
    manager = Manager()
    logq_listener, log_queue = logger_init(manager)

    train_loop(manager, log_queue)

    logq_listener.stop()    

    logging.info("Done.")
    
