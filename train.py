import gc
import logging
import numpy as np
import datetime
from queue import Queue
from multiprocessing import Process
from collections import deque

from model import *
from hyper_params import *
from game import *
import go_board

def evaluate(model_a, model_b, num_games=1):
    logging.info("Evaluation start:")
    results = []    
    for eval_game_index in np.arange(num_games):
        start_player, outcome = two_player_play(eval_game_index, model_a, model_b)        
        results.append((start_player, outcome))
    logging.info("Evaluation end: results: %s", results)
    return results

def train_loop():
    best_model = SimpleNNModel()
    trained_model = best_model.copy()

    np.random.seed(SEED)

    history = deque(maxlen=MAX_GAMES_HISTORY_SIZE)
    #game_history is a list of tuples (pi, s, z)
    # pi - the probability distribution over actions at state s
    # s - state
    # z - outcome of the game
    
    for iter_index in np.arange(NUM_ITER):
        logging.debug("Iter: %d", iter_index)
        
        for game_index in range(NUM_GAMES):
            self_play(game_index, best_model, history, noise_alpha=DIRICHLET_ALPHA, temperatures=TEMPERATURES)
                
        for train_iter_idx in np.arange(NUM_TRAIN_LOOP_ITER):
            logging.debug("Network train: %d", train_iter_idx)
            batch = np.random.choice(history, size=BATCH_SIZE)
            trained_model.train_on_hist_batch(batch, train_iter_idx)
            
        eval_results = evaluate(trained_model, best_model, num_games=NUM_EVAL_GAMES)
        outcomes = np.array([outcome for _, outcome in eval_results])
        trained_model_win_ratio = np.sum(outcomes == OUTCOME_WIN_PLAYER_1)/len(outcomes)

        logging.info("Iter %d evaluation: trained_model_win percent : %.2f%%", iter_index, trained_model_win_ratio*100)

        if trained_model_win_ratio > MIN_MODEL_REPLACEMENT_WIN_RATIO:
            old_best = best_model
            best_model = trained_model.copy()
            best_model.save("%s/model-%s-%d-%.0f.h5" % (OUTPUT_DIR, log_id, iter_index, trained_model_win_ratio*100))
            del old_best

        gc.collect()

    
if __name__ == "__main__":
    log_id = "ago-%dx%dx%d-%s" % (BOARD_SIZE, BOARD_SIZE, MCTS_STEPS, datetime.datetime.now().strftime("%m%d%H%M"))
    log_file_handler = logging.FileHandler('%s/%s.log' % (OUTPUT_DIR, log_id))

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger()
    logger.addHandler(log_file_handler)

    train_loop()
