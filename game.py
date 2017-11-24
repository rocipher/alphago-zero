import logging
import numpy as np

from go_board import *
from go_tree import *
from model import *
from hyper_params import *


class GameMove():
    def __init__(self, state=None, action_distribution=None, value=None):
        self.state = state
        self.action_distribution = action_distribution
        self.value = value

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

    self_play = player_2 is None
    players = { PLAYER_1: player_1, PLAYER_2: player_2 if not self_play else player_1 }
    
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
        if not self_play:
            players[other_player].opponent_played(move_index, action_taken)
        game_history.append(GameMove(current_state, act_distrib, value))
        current_state = players[current_player].root.state
        
        current_player_root = players[current_player].root
        logging.debug("Move %d, Player %d: action (%d, %d) -> (n, q, score): next: (%d, %.2f, %.2f)\n%s",
                        move_index, current_state.player, *GoBoard.get_action_coords(action_taken), 
                        current_player_root.n, current_player_root.q, current_player_root.score,
                        GoBoard.to_pretty_print(current_state.pos[-1]))

    return outcome, game_history
