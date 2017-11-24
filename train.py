import gc
import logging
import numpy as np
import datetime
from collections import deque

from model import *
from hyper_params import *
from game import *

def evaluate(model_a, model_b, num_games=1):
    results = []
    for eval_game_index in np.arange(num_games):
        start_player = np.random.choice(2)
        logging.info("Eval start game %d: start player: %d", eval_game_index, start_player)
        starting_state = GoBoard.create_zero_state(start_player)
        player_1 = GoPlayer(starting_state, model=model_a, noise_alpha=0.0, temperatures=[(0, np.inf, 0.0)])
        player_2 = GoPlayer(starting_state, model=model_b, noise_alpha=0.0, temperatures=[(0, np.inf, 0.0)])
        outcome, _ = play_game(player_1=player_1, player_2=player_2)            
        logging.info("Eval end game %d: %d", eval_game_index, outcome)
        results.append((start_player, outcome))
    return results

def actions_uniform_distrib():
    return np.ones(ACTION_SPACE_SIZE)/ACTION_SPACE_SIZE

def train_loop():
    best_model = SimpleNNModel()

    np.random.seed(SEED)

    history = deque(maxlen=MAX_GAMES_HISTORY_SIZE)
    #game_history is a list of tuples (pi, s, z)
    # pi - the probability distribution over actions at state s
    # s - state
    # z - outcome of the game
    
    # initialize the history with one move from and an empty state and uniform action distribution
    history.append(GameMove(GoBoard.create_zero_state(0), 
                            actions_uniform_distrib(), 
                            GoBoard.OUTCOME_DRAW))
    for iter_index in np.arange(NUM_ITER):
        logging.debug("Iter: %d", iter_index)

        # sample starting states
        starting_states = np.random.choice(history, size=NUM_GAMES)
        
        # for every starting state, play the game to end using mcts
        for game_index, start_move in enumerate(starting_states):
            logging.info("Game: %d: start player: %d, start state: \n%s", game_index, start_move.state.player, GoBoard.to_pretty_print(start_move.state.pos[-1]))
            player = GoPlayer(start_move.state, model=best_model, noise_alpha=DIRICHLET_ALPHA, temperatures=TEMPERATURES)
            outcome, current_game_history = play_game(player_1=player, player_2=None)
            for game_move in current_game_history:
                history.append(game_move)
            logging.info("Game: %d, outcome: %d, end position:\n%s\n", 
                            game_index, outcome, 
                            GoBoard.to_pretty_print(current_game_history[-1].state.pos[-1]))
        
        new_model = best_model.copy()        
        for train_iter_idx in np.arange(NUM_TRAIN_LOOP_ITER):
            logging.debug("Network train: %d", train_iter_idx)
            batch = np.random.choice(history, size=BATCH_SIZE)
            new_model.train_on_hist_batch(batch)
            
        eval_results = evaluate(new_model, best_model, num_games=NUM_EVAL_GAMES)
        outcomes = np.array([outcome for _, outcome in eval_results])
        new_model_win_ratio = np.sum(outcomes == GoBoard.OUTCOME_WIN_PLAYER_1)/len(outcomes)

        logging.info("Iter %d evaluation: new_model_win percent : %.2f%%", iter_index, new_model_win_ratio*100)

        if new_model_win_ratio > MIN_MODEL_REPLACEMENT_WIN_RATIO:
            old_best = best_model
            best_model = new_model
            best_model.save("%s/model-%s-%d-%.0f.h5" % (OUTPUT_DIR, log_id, iter_index, new_model_win_ratio*100))
            del old_best

        gc.collect()

    
if __name__ == "__main__":
    log_id = "ago-%dx%dx%d-%s" % (BOARD_SIZE, BOARD_SIZE, MCTS_STEPS, datetime.datetime.now().strftime("%m%d%H%M"))
    log_file_handler = logging.FileHandler('%s/%s.log' % (OUTPUT_DIR, log_id))

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.addHandler(log_file_handler)

    train_loop()
