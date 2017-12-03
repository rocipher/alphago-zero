import gc
import sys
import logging
from logging.handlers import QueueHandler, QueueListener
import numpy as np
import datetime
import signal
from itertools import repeat
import traceback
from multiprocessing.pool import Pool
from multiprocessing import Manager, Queue
from multiprocessing import Process, Pipe, Event

import model_parallel
from model_parallel import ControlActions, MultiProcessModelProxy
import hyper_params as hp
import game

def pool_worker_init():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def nn_worker(log_queue:Queue, model_proxy:MultiProcessModelProxy):
    model_parallel.init_log_queue(log_queue)
    model_parallel.run_loop(model_proxy)

def self_play_worker(game_index:int, model_proxy:MultiProcessModelProxy):
    model_parallel.init_log_queue(model_proxy.log_queue)
    result = game.self_play(game_index, model_proxy, model_proxy.history_queue, 
                noise_alpha=hp.DIRICHLET_ALPHA, temperatures=hp.TEMPERATURES)
    return result
    
def multi_player_worker(game_index:int, model_a:MultiProcessModelProxy, model_b:MultiProcessModelProxy):
    model_parallel.init_log_queue(model_a.log_queue)
    result = game.two_player_play(game_index, model_a, model_b)
    return result

def spawn_nn_process(model_file:str, manager:Manager, history_queue, log_queue: Queue):
    parent_ctrl_pipe, child_ctrl_pipe = Pipe()    
    pl_model = MultiProcessModelProxy(model_file=model_file,
                                        predict_queue=manager.Queue(maxsize=hp.PREDICTION_QUEUE_BATCH_SIZE),
                                        history_queue=history_queue,
                                        stop_event=manager.Event(),
                                        results=manager.dict(),
                                        train_model_event=manager.Event(),
                                        train_completed_event=manager.Event(),
                                        control_pipe=parent_ctrl_pipe,                                        
                                        prediction_ready_event=manager.Event(),
                                        log_queue=log_queue)
    nn_process = Process(target=nn_worker, args=(log_queue, pl_model))
    nn_process.daemon = True
    nn_process.start()
    return nn_process, pl_model, child_ctrl_pipe


def train_loop(manager:Manager, log_queue:Queue):    
    pool = Pool(processes=hp.NUM_WORKERS, initializer=pool_worker_init)

    history_queue = manager.list()    

    best_model_file = hp.MODEL_FILE
    nn_train_proc_best, pl_model_best, ctrl_pipe_best = spawn_nn_process(
                best_model_file, manager, history_queue, log_queue)

    trained_model_file = hp.MODEL_FILE
    nn_train_proc_train, pl_model_train, ctrl_pipe_train = spawn_nn_process(
                trained_model_file, manager, history_queue, log_queue)    
    
    try:
        for iter_index in range(hp.NUM_ITER):
            # self-play
            logging.info("Iter %d: Starting self-play", iter_index)
            self_play_results = pool.starmap(self_play_worker, zip(range(hp.NUM_GAMES), repeat(pl_model_best)))
            logging.info("Iter %d, self-play results: %s", iter_index, self_play_results)

            # train NN
            logging.info("Iter %d: Starting network train", iter_index)
            pl_model_train.train()
            logging.info("Iter %d: Ended network train", iter_index)

            # eval
            logging.info("Iter %d: Starting evaluation", iter_index)
            eval_results = pool.starmap(multi_player_worker, zip(range(hp.NUM_EVAL_GAMES), repeat(pl_model_train), repeat(pl_model_best)))
            logging.info("Iter %d: Evaluation end: results: %s", iter_index, eval_results)
            outcomes = np.array([outcome for _,outcome in eval_results])
            trained_model_win_ratio = np.sum(outcomes == hp.OUTCOME_WIN_PLAYER_1)/len(outcomes)

            logging.info("Iter %d evaluation: trained_model_win percent : %.2f%%", iter_index, trained_model_win_ratio*100)

            if trained_model_win_ratio > hp.MIN_MODEL_REPLACEMENT_WIN_RATIO:
                pl_model_best.stop_event.set()
                nn_train_proc_best.join(1.)
                nn_train_proc_best.terminate()
                
                best_model_file = "%s/model-%s-%d-%.0f.h5" % (hp.OUTPUT_DIR, log_id, iter_index, trained_model_win_ratio*100)
                ctrl_pipe_train.send((ControlActions.SAVE, best_model_file))
                if ctrl_pipe_train.poll(30.):
                    save_result = ctrl_pipe_train.recv()
                    if save_result != ControlActions.SAVE_COMPLETED:
                        raise Exception("Save model failed.")
                else:
                    logging.error("Model save action didn't receive an answer! iter_index: %d, new_model_file: %s", iter_index, best_model_file)

                nn_train_proc_best, pl_model_best, ctrl_pipe_best = spawn_nn_process(
                            best_model_file, manager, history_queue, log_queue)

            gc.collect()
        # end iter loop
    except Exception as e:
        if e is KeyboardInterrupt or e is SystemExit:
            logging.info("Terminated by user.")
        else:
            logging.error("Error: %s", e)
        pool.terminate()
        pool.join()
        nn_train_proc_train.terminate()
        nn_train_proc_best.terminate()
        raise e
    else:
        pl_model_train.stop_event.set()
        pl_model_best.stop_event.set()
        
        nn_train_proc_train.join()
        nn_train_proc_best.join()

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
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return queue_listener, log_queue


if __name__ == "__main__":
    log_id = "ago-%s-%dx%d-%d" % (datetime.datetime.now().strftime("%m%d%H%M"),
                                 hp.BOARD_SIZE, hp.BOARD_SIZE, hp.MCTS_STEPS)
    manager = Manager()
    logq_listener, log_queue = logger_init(manager)

    train_loop(manager, log_queue)

    logq_listener.stop()    

    logging.info("Done.")
    
