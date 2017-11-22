
import numpy as np
from hyper_params import *
from collections import namedtuple
import itertools

GameState = namedtuple('GameState', ['player', 'pos'])

class GoBoard():
    OUTCOME_DRAW = 0
    OUTCOME_WIN_PLAYER_1 = 1
    OUTCOME_WIN_PLAYER_2 = 2

    @staticmethod
    def sample_dihedral_transformation(state):
        transformations = [
                           # rotations
                           lambda m: m,
                           lambda m: np.rot90(m, k=1),
                           lambda m: np.rot90(m, k=2),
                           lambda m: np.rot90(m, k=3),                           
                           # reflections
                           lambda m: np.flipud(m),
                           lambda m: np.flipud(np.rot90(m, k=1)),                           
                           lambda m: np.flipud(np.rot90(m, k=2)),
                           lambda m: np.flipud(np.rot90(m, k=3))]
        sample_transform = np.random.choice(transformations)
        new_state = GoBoard.create_zero_state(state.player)        
        for ix, px in itertools.product(range(state.pos.shape[0]), range(state.pos.shape[1])):
            new_state.pos[ix, px, :, :] = sample_transform(state.pos[ix, px, :, :])
        return new_state


    @staticmethod
    def create_zero_state(player):
        return GameState(player, np.zeros((STATE_HIST_SIZE, 2, BOARD_SIZE, BOARD_SIZE), dtype=np.bool))


    @staticmethod
    def next_state(state: GameState, action) -> GameState:
        """
            Returns the next state after an action is taken
            Returns: new_state, outcome
        """
        new_state = GoBoard.create_zero_state(1-state.player)
        # shift history position to the left
        new_state.pos[0:STATE_HIST_SIZE-1, :, :, :] = state.pos[1:STATE_HIST_SIZE, :, :, :]
        # initialize the new position equal to the previous position
        new_state.pos[-1, :, :, :] = state.pos[-1, :, :, :]
        if not GoBoard.is_pass_action(action):
            new_state.pos[-1, :, :, :] = GoBoard.next_position(new_state.pos[-1],
                                                               state.player,
                                                               action)

        # check for game end
        twice_passed = np.all(new_state.pos[-3] == new_state.pos[-2]) and \
                       np.all(new_state.pos[-2] == new_state.pos[-1])
        board_full = GoBoard.is_board_full(new_state.pos[-1])
        if twice_passed or board_full:
            outcome = GoBoard.calc_game_outcome(new_state.pos[-1])
        else:
            outcome = None
        return new_state, outcome

    @staticmethod
    def to_pretty_print(pos):
        board_str = ""
        for row in np.arange(BOARD_SIZE):
            board_line_str = np.array(['-']*BOARD_SIZE)
            board_line_str[pos[PLAYER_1, row]] = "x"
            board_line_str[pos[PLAYER_2, row]] = "O"
            board_str += "".join(board_line_str) + "\n"
        return board_str

    @staticmethod
    def make_pos_from_matrix(mat):
        return np.array([mat == 1, mat == 2])

    
    @staticmethod
    def is_board_full(pos):
        return np.all(pos[PLAYER_1] | pos[PLAYER_2])

    @staticmethod
    def next_position(position, player, action):
        new_position = position.copy()
        row, col = GoBoard.get_action_coords(action)
        # place stone
        new_position[player, row, col] = 1
        board_full = GoBoard.is_board_full(new_position)
        # check and remove opponents stones without liberties
        if not board_full:
            GoBoard.make_liberties_in_place(new_position, player)
        return new_position


    @staticmethod
    def make_liberties_in_place(pos, current_player):
        assert pos.shape == (2, BOARD_SIZE, BOARD_SIZE)
        other_player = 1-current_player
        board = pos[current_player, :, :] + -1*pos[other_player, :, :]
        padded_board = np.pad(board, 1, mode='constant', constant_values=1)
        while True:
            rows, cols = np.where(padded_board == 0)
            if rows.size == 0:
                break
            all_rows = np.concatenate([rows, rows-1, rows+1, rows, rows])
            all_cols = np.concatenate([cols, cols, cols, cols-1, cols+1])
            padded_board[all_rows, all_cols] += 1

        taken_x, taken_y = np.where(padded_board[1:-1, 1:-1] < 0)
        pos[other_player, taken_x, taken_y] = 0


    @staticmethod
    def get_action_coords(action):
        return action // BOARD_SIZE, action % BOARD_SIZE


    @staticmethod
    def is_pass_state(state):
        # if the last two states are exactly the same then it's a pass
        return np.all(state.pos[-2, :, :, :] == state.pos[-1, :, :, :])


    @staticmethod
    def is_pass_action(action):
        return action == ACTION_SPACE_SIZE-1


    @staticmethod
    def encode_action(row=None, col=None):
        if row is None and col is None:
            return ACTION_SPACE_SIZE-1
        else:
            return row*BOARD_SIZE + col


    @staticmethod
    def valid_actions(state):
        current_player = state.player
        other_player = 1-current_player

        stones_taken_prev_move = state.pos[-1, current_player, :, :] != state.pos[-2, current_player, :, :]
        current_player_prev_move_taken = np.sum(stones_taken_prev_move)
        check_for_ko = current_player_prev_move_taken == 1

        board = state.pos[-1, current_player, :, :] + state.pos[-1, other_player, :, :]
        possible_positions = list(zip(*np.where(board == 0)))

        if check_for_ko:
            check_r, check_c = np.where(stones_taken_prev_move == 1)
            assert len(check_r) == 1
            assert (check_r, check_c) in possible_positions

            new_action = GoBoard.encode_action(check_r, check_c)
            possible_ko_pos = GoBoard.next_position(state.pos[-1, :, :, :], current_player, new_action)

            if np.all(possible_ko_pos == state.pos[-2]):
                possible_positions.remove((check_r, check_c))

        valid_acts = [GoBoard.encode_action(r, c) for r, c in possible_positions] + \
                     [GoBoard.encode_action(None)]
        return valid_acts


    @staticmethod
    def board_reach(pos, curr_player):
        oth_player = 1-curr_player
        board = 1*pos[curr_player, :, :] + 2*pos[oth_player, :, :]
        padded_board = np.pad(board, 1, mode='constant', constant_values=2)
        while True:
            rows, cols = np.where(padded_board == 1)
            if rows.size == 0:
                break
            all_rows = np.concatenate([rows, rows-1, rows+1, rows, rows])
            all_cols = np.concatenate([cols, cols, cols, cols-1, cols+1])
            padded_board[all_rows, all_cols] += 1
        return padded_board[1:-1, 1:-1] > 0


    @staticmethod
    def calc_teritories_count(pos):        
        player_1_reach = GoBoard.board_reach(pos, PLAYER_1)
        player_2_reach = GoBoard.board_reach(pos, PLAYER_2)
        player_2_teritory_count = np.sum(~player_1_reach)
        player_1_teritory_count = np.sum(~player_2_reach)
        return [player_1_teritory_count, player_2_teritory_count]


    @staticmethod
    def calc_game_outcome(position):
        player_1_terr_cnt, player_2_terr_cnt = GoBoard.calc_teritories_count(position)
        player_1_stones = np.sum(position[PLAYER_1])
        player_2_stones = np.sum(position[PLAYER_2])

        player_1_score = player_1_terr_cnt+player_1_stones
        player_2_score = player_2_stones+player_2_terr_cnt

        if player_1_score == player_2_score:
            return GoBoard.OUTCOME_DRAW
        elif player_1_score > player_2_score:
            return GoBoard.OUTCOME_WIN_PLAYER_1
        else:
            return GoBoard.OUTCOME_WIN_PLAYER_2

