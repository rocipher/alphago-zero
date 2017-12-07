import logging
import numpy as np
import time

from go_board import *
from go_tree import *
from model import *
from hyper_params import *
import go_board

class GameMove():
    def __init__(self, state=None, action_distribution=None, value=None):
        self.state = state
        self.action_distribution = action_distribution
        self.value = value


def self_play(game_index:int, model:Model, history_queue=None, noise_alpha=0.0, temperatures=[(0, np.inf, 0.0)]):
    start_player = np.random.choice(2)
    starting_state = go_board.create_zero_state(start_player)
    start_time = time.time()
    logging.info("Game: %d: start player: %d", game_index, starting_state.player)
    player = GoPlayer(starting_state, model=model, noise_alpha=noise_alpha, temperatures=temperatures)
    outcome, game_history = play_game(player_1=player, player_2=None)
    if history_queue is not None:
        for game_move in game_history:
            history_queue.append(game_move)
    end_time = time.time()
    logging.info("Game: %d, outcome: %d, num_moves: %d, game_time(s): %.0f, end position:\n%s\n", 
                    game_index, outcome, len(game_history), (end_time-start_time),
                    go_board.to_pretty_print(game_history[-1].state.pos[-1]))
    return start_player, outcome


def two_player_play(game_index: int, model_a:Model, model_b:Model):
    start_player = np.random.choice(2)
    start_time = time.time()
    logging.info("Eval start game %d: start player: %d", game_index, start_player)
    starting_state = go_board.create_zero_state(start_player)
    player_1 = GoPlayer(starting_state, model=model_a, noise_alpha=0.0, temperatures=[(0, np.inf, 0.0)])
    player_2 = GoPlayer(starting_state, model=model_b, noise_alpha=0.0, temperatures=[(0, np.inf, 0.0)])
    outcome, moves = play_game(player_1=player_1, player_2=player_2)
    end_time = time.time()
    logging.info("Eval end game %d, num_moves: %d: outcome: %d, game_time(s): %.0f, end position:\n%s", 
            game_index, len(moves), outcome, (end_time-start_time),
            go_board.to_pretty_print(moves[-1].state.pos[-1]))
    return start_player, outcome


def play_game(player_1=None, player_2=None):
    """
        Plays the game from the starting_state to the end.
        For each new state, it executes MCTS for `mcts_steps` steps.

        Args:

        Returns:
            A vector of tuples (pi, s, z) with the moves played where 
                - pi is a probability distribution of the states
                - s is state
                - z is the outcome from the current player's perspective                
    """

    is_self_play = player_2 is None
    players = { PLAYER_1: player_1, PLAYER_2: player_2 if not is_self_play else player_1 }
    
    # play
    game_history = []
    current_state = player_1.root.state
    outcome = None
    move_index = 0
    while outcome is None:
        move_index += 1        
        current_player = current_state.player
        other_player = 1-current_player
        act_distrib, action_taken, value, outcome = players[current_player].play(move_index)        
        if not is_self_play:
            players[other_player].opponent_played(move_index, action_taken)
        game_history.append(GameMove(current_state, act_distrib, None))
        current_state = players[current_player].root.state        
        
        logging.debug("Move %d, Player %d: action (%d, %d) -> value: %.2f, otc: %s\n%s",
                        move_index, 1-current_state.player, *go_board.get_action_coords(action_taken), 
                        value, outcome, go_board.to_pretty_print(current_state.pos[-1]))

    for game_move in game_history:
        game_move.value = GoPlayer.get_reward(outcome, game_move.state.player)

    return outcome, game_history

