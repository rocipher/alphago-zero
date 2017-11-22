import logging
import numpy as np

from go_board import *
from go_tree import *
from model import *
from hyper_params import *


class GameMove():
    def __init__(self, state=None, action_distribution=None, outcome=None):
        self.state = state
        self.action_distribution = action_distribution
        self.outcome = outcome


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
    players = { PLAYER_1: player_1, PLAYER_2: player_1 if self_play else player_2 }
    
    # play
    game_history = []
    current_state = player_1.root.state
    outcome = None
    move_index = 0
    while outcome is None:
        move_index += 1        
        current_player = current_state.player
        other_player = 1-current_player
        act_distrib, action_taken, outcome = players[current_player].play(move_index)        
        if not self_play:
            players[other_player].opponent_played(move_index, action_taken)
        game_history.append(GameMove(current_state, act_distrib, None))
        current_state = players[current_player].root.state
        
        current_player_root = players[current_player].root
        logging.debug("Move %d, Player %d: action (%d, %d) -> (n, q, score): next: (%d, %.2f, %.2f)\n%s",
                        move_index, current_state.player, *GoBoard.get_action_coords(action_taken), 
                        current_player_root.n, current_player_root.q, current_player_root.score,
                        GoBoard.to_pretty_print(current_state.pos[-1]))

    # distribute the outcome to the played states
    for game_move in game_history:
        if outcome == GoBoard.OUTCOME_DRAW:
            game_move.outcome = REWARD_DRAW
        else:            
            if game_move.state.player==0 and outcome==GoBoard.OUTCOME_WIN_PLAYER_1:
                game_move.outcome = REWARD_WIN
            else:
                game_move.outcome = REWARD_LOOSE
    return outcome, game_history
        



    # # play
    # move_index = 0
    # cp_tree = player_1_tree if starting_state.player == PLAYER_1 else player_2_tree
    # while cp_tree.root.outcome is None:
    #     move_index += 1                
    #     cp_tree.run_mcts(MCTS_STEPS)        
    #     act_distrib = cp_tree.count_probabilites(cp_tree.root)
    #     game_history.append(GameMove(cp_tree.root.state, act_distrib, None))
    
    #     if noise_alpha > 0.0:
    #         # add Dir(0.03) dirichlet noise for additional exploration 
    #         dirichlet_alpha = np.zeros(ACTION_SPACE_SIZE, dtype=np.float)
    #         dirichlet_alpha[act_distrib>0.0] = noise_alpha                
    #         dirichlet_noise = np.random.dirichlet(alpha=dirichlet_alpha)
    #         act_distrib = (1.0-EPS)*act_distrib + EPS*dirichlet_noise
        
    #     # action distribution sanity check        
    #     assert np.abs(np.sum(act_distrib)-1.0)<1e-7, "Invalid act_distrib: sum=%.7f, n=%d, action_n:%s, distrib: %s" % (np.sum(act_distrib), tree.root.n, np.sum([act.n for act in tree.root.actions.values()]), act_distrib)
    #     assert np.product(act_distrib.shape) == (ACTION_SPACE_SIZE,)

    #     if move_index < sample_before_move_index:
    #         # sample next action acording to the mcts distribution
    #         next_action = np.random.choice(ACTION_SPACE_SIZE, p=act_distrib)
    #     else:
    #         next_action = np.argmax(act_distrib)

    #     player_1_tree.play_action(next_action)
    #     if player_2_tree != player_1_tree:
    #         player_2_tree.play_action(next_action)

    #     # player_1_tree.make_root(player_1_tree.actions[next_action])        
    #     # player_2_tree.make_root(player_2_tree.actions[next_action])

    #     next_node = tree.root.actions[next_action]

    #     logging.debug("Move %d, Player %d: action (%d, %d) -> (n, q, score): next: (%d, %.2f, %.2f)\n%s",
    #                     move_index, tree.root.state.player, *GoBoard.get_action_coords(next_action), 
    #                     next_node.n, next_node.q, next_node.score,
    #                     GoBoard.to_pretty_print(next_node.state.pos[-1]))

    #     #prune the tree by making the selected action its root
    #     tree.make_root(next_node)
    
    # game_outcome = tree.root.outcome

    # # distribute the outcome to the played states
    # for game_move in game_history:
    #     if game_outcome == GoBoard.OUTCOME_DRAW:
    #         game_move.outcome = REWARD_DRAW
    #     else:            
    #         if game_move.state.player==0 and game_outcome==GoBoard.OUTCOME_WIN_PLAYER_1:
    #             game_move.outcome = REWARD_WIN
    #         else:
    #             game_move.outcome = REWARD_LOOSE

    # return game_outcome, game_history

         



