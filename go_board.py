
import numpy as np
from hyper_params import *
from collections import namedtuple
import itertools

GameState = namedtuple('GameState', ['player', 'pos'])

def sample_dihedral_transformation(state):
    transformations = [
                        # rotations
                        (lambda m: m, lambda m: m),
                        (lambda m: np.rot90(m, k=1), lambda m: np.rot90(m, k=-1)),
                        (lambda m: np.rot90(m, k=2), lambda m: np.rot90(m, k=-2)),
                        (lambda m: np.rot90(m, k=3), lambda m: np.rot90(m, k=-3)),
                        # reflections
                        (lambda m: np.flipud(m), lambda m: np.flipud(m)),
                        (lambda m: np.flipud(np.rot90(m, k=1)), lambda m: np.rot90(np.flipud(m), k=-1)),
                        (lambda m: np.flipud(np.rot90(m, k=1)), lambda m: np.rot90(np.flipud(m), k=-2)),
                        (lambda m: np.flipud(np.rot90(m, k=1)), lambda m: np.rot90(np.flipud(m), k=-3))
                      ]
    transform_indx = np.random.choice(len(transformations))
    sample_transform_direct, sample_transform_inverse = transformations[transform_indx]
    new_state = create_zero_state(state.player)        
    for ix, px in itertools.product(range(state.pos.shape[0]), range(state.pos.shape[1])):
        new_state.pos[ix, px, :, :] = sample_transform_direct(state.pos[ix, px, :, :])
    return new_state, sample_transform_inverse


def create_zero_state(player):
    return GameState(player, np.zeros((STATE_HIST_SIZE, 2, BOARD_SIZE, BOARD_SIZE), dtype=np.bool))


def next_state(state: GameState, action) -> GameState:
    """
        Returns the next state after an action is taken
        Returns: new_state, outcome
    """
    current_player = state.player
    other_player = 1-current_player

    if not is_pass_action(action):
        row, col = get_action_coords(action)
        if state.pos[-1, current_player, row, col] | state.pos[-1, other_player, row, col]:
            # intersection alreay played
            return None, None

    new_state = create_zero_state(other_player)
    # shift history position to the left
    new_state.pos[0:STATE_HIST_SIZE-1, :, :, :] = state.pos[1:STATE_HIST_SIZE, :, :, :]
    # initialize the new position equal to the previous position
    new_state.pos[-1, :, :, :] = state.pos[-1, :, :, :]
    if not is_pass_action(action):
        new_position = next_position(new_state.pos[-1], current_player, action)
        if new_position is None:
            # invalid position, suicide?
            return None, None
        # check for ko
        if np.all(new_position == state.pos[-2]):
            return None, None
        new_state.pos[-1, :, :, :] = new_position            

    # check for game end
    twice_passed = np.all(new_state.pos[-3] == new_state.pos[-2]) and \
                    np.all(new_state.pos[-2] == new_state.pos[-1])
    board_full = is_board_full(new_state.pos[-1])
    if twice_passed or board_full:
        outcome = calc_game_outcome(new_state.pos[-1])
    else:
        outcome = None
    return new_state, outcome

def to_pretty_print(pos):
    board_str = ""
    for row in np.arange(BOARD_SIZE):
        board_line_str = np.array(['-']*BOARD_SIZE)
        board_line_str[pos[PLAYER_1, row]] = "x"
        board_line_str[pos[PLAYER_2, row]] = "O"
        board_str += "".join(board_line_str) + "\n"
    return board_str

def make_pos_from_matrix(mat, pl_1=1, pl_2=2):
    return np.array([mat == pl_1, mat == pl_2])

def is_board_full(pos):
    return np.all(pos[PLAYER_1] | pos[PLAYER_2])

def next_position(position, player, action):
    new_position = position.copy()
    row, col = get_action_coords(action)
    other_player = 1-player
    # place stone
    new_position[player, row, col] = 1
    
    near_opponent_stone = (row-1>=0 and new_position[other_player, row-1, col] == 1) or \
                            (row+1<BOARD_SIZE and new_position[other_player, row+1, col] == 1) or \
                            (col-1>=0 and new_position[other_player, row, col-1] == 1) or \
                            (col+1<BOARD_SIZE and new_position[other_player, row, col+1] == 1)
    near_own_stone = (row-1>=0 and new_position[player, row-1, col] == 1) or \
                    (row+1<BOARD_SIZE and new_position[player, row+1, col] == 1) or \
                    (col-1>=0 and new_position[player, row, col-1] == 1) or \
                    (col+1<BOARD_SIZE and new_position[player, row, col+1] == 1)
    # only check liberties if the opponent has any adjacent stone
    if near_opponent_stone or near_own_stone:
        # check and remove opponents stones without liberties                
        num_taken_opponent, num_taken_own = make_liberties_in_place(new_position, player)
        # check for suicide
        if num_taken_own>0:
            return None

    return new_position

def make_liberties_in_place(pos, current_player):
    assert pos.shape == (2, BOARD_SIZE, BOARD_SIZE)
    other_player = 1-current_player
    board = 3 - 2*pos[current_player, :, :] - pos[other_player, :, :]
    liberties = list(zip(*np.where(board == 3)))
    while len(liberties)>0:
        new_liberties = []
        for r, c in liberties:
            for vr, vc in [(r+1, c), (r-1, c), (r, c+1), (r, c-1)]:
                if not 0<=vr<BOARD_SIZE or not 0<=vc<BOARD_SIZE: continue
                if board[vr, vc]<3 and (board[r, c]==3 or board[r, c]-3==board[vr, vc]):
                    board[vr, vc] += 3
                    new_liberties.append((vr, vc))                    
        liberties = new_liberties

    oth_pl_taken_rows, oth_pl_taken_cols = np.where(board == 2)        
    curr_pl_taken_rows, curr_pl_taken_cols = np.where(board == 1)        
    pos[other_player, oth_pl_taken_rows, oth_pl_taken_cols] = 0
    
    oth_taken_set = set(zip(oth_pl_taken_rows, oth_pl_taken_cols))
    curr_taken_with_new_liberties = [(r, c) for r, c in zip(curr_pl_taken_rows, curr_pl_taken_cols) \
                                    if len(oth_taken_set.intersection(set([(r+1, c), (r-1, c), (r, c+1), (r, c-1)])))>0]
    return oth_pl_taken_rows.size, curr_pl_taken_rows.size-len(curr_taken_with_new_liberties)


def get_action_coords(action):
    return action // BOARD_SIZE, action % BOARD_SIZE


def is_pass_state(state):
    # if the last two states are exactly the same then it's a pass
    return np.all(state.pos[-2, :, :, :] == state.pos[-1, :, :, :])


def is_pass_action(action):
    return action == ACTION_SPACE_SIZE-1


def encode_action(row=None, col=None):
    if row is None and col is None:
        return ACTION_SPACE_SIZE-1
    else:
        return row*BOARD_SIZE + col


def maybe_valid_actions(state):
    current_player = state.player
    other_player = 1-current_player
    board = state.pos[-1, current_player, :, :] | state.pos[-1, other_player, :, :]
    possible_positions = list(zip(*np.where(board == 0)))
    valid_acts = [encode_action(r, c) for r, c in possible_positions] + \
                 [encode_action(None)]
    return valid_acts


def board_reach(pos, curr_player):
    oth_player = 1-curr_player
    board = 1*pos[curr_player, :, :] + 2*pos[oth_player, :, :]
    while True:
        rows, cols = np.where(board == 1)
        if rows.size == 0:                
            break        
        cells = {}
        for r, c in zip(rows, cols):
            cells[(r, c)]=1
            if r+1<BOARD_SIZE: cells[(r+1, c)]=1
            if r-1>=0: cells[(r-1, c)]=1
            if c+1<BOARD_SIZE: cells[(r, c+1)]=1
            if c-1>=0: cells[(r, c-1)]=1                
        for r, c in cells.keys():
            board[r, c] += 1
    return board > 0


    padded_board = np.pad(board, 1, mode='constant', constant_values=2)
    while True:
        rows, cols = np.where(padded_board == 1)
        if rows.size == 0:
            break
        all_rows = np.concatenate([rows, rows-1, rows+1, rows, rows])
        all_cols = np.concatenate([cols, cols, cols, cols-1, cols+1])
        padded_board[all_rows, all_cols] += 1
    return padded_board[1:-1, 1:-1] > 0


def calc_teritories_count(pos):        
    player_1_reach = board_reach(pos, PLAYER_1)
    player_2_reach = board_reach(pos, PLAYER_2)
    player_2_teritory_count = np.sum(~player_1_reach)
    player_1_teritory_count = np.sum(~player_2_reach)
    return [player_1_teritory_count, player_2_teritory_count]


def calc_game_outcome(position):
    player_1_terr_cnt, player_2_terr_cnt = calc_teritories_count(position)
    player_1_stones = np.sum(position[PLAYER_1])
    player_2_stones = np.sum(position[PLAYER_2])

    player_1_score = player_1_terr_cnt+player_1_stones
    player_2_score = player_2_stones+player_2_terr_cnt

    if player_1_score == player_2_score:
        return OUTCOME_DRAW
    elif player_1_score > player_2_score:
        return OUTCOME_WIN_PLAYER_1
    else:
        return OUTCOME_WIN_PLAYER_2

