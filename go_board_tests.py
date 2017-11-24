import unittest
from hyper_params import *
from go_board import *

class TestGoBoard(unittest.TestCase):

    def test_make_pos_from_matrix(self):
        mat_pos = np.array([
            [0, 2, 0, 0, 1],
            [2, 1, 2, 0, 1],
            [0, 1, 2, 0, 0],
            [0, 2, 0, 1, 0],
            [0, 0, 0, 0, 0]
        ])
        pos = make_pos_from_matrix(mat_pos)
        self.assertEqual(pos.shape, (2, BOARD_SIZE, BOARD_SIZE))


    def test_make_liberties_in_place(self):
        mat_pos = np.array([
            [0, 2, 0, 0, 1],
            [2, 1, 2, 0, 1],
            [2, 1, 2, 0, 0],
            [0, 2, 0, 1, 0],
            [0, 0, 0, 0, 0]
        ])        
        pos = make_pos_from_matrix(mat_pos)        
        prev_pos = pos.copy()
        make_liberties_in_place(pos, 2-1)
        self.assertTrue(np.sum(prev_pos[0] != pos[0]) == 2)

    def test_check_suicide(self):
        mat_pos = np.array([
            [0, 2, 0, 0, 1],
            [2, 1, 2, 0, 0],
            [2, 0, 2, 0, 2],
            [0, 2, 0, 1, 0],
            [0, 0, 0, 0, 0]
        ])        
        pos = make_pos_from_matrix(mat_pos)
        new_pos = next_position(pos, PLAYER_1, encode_action(2, 1))
        self.assertTrue(new_pos is None)

    def test_next_state(self):
        mat_positions = [np.array([
            [0, 2, 1, 0, 0],
            [2, 1, 0, 1, 0],
            [0, 2, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
            ]), np.array([
            [0, 2, 1, 0, 0],
            [2, 0, 2, 1, 0],
            [0, 2, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
            ])]        
        state = create_zero_state(0)
        # set the last 2 states
        state.pos[-2:] = np.array([make_pos_from_matrix(pos) for pos in mat_positions])
        new_state, outcome = next_state(state, encode_action(3, 0))
        self.assertTrue(new_state.player == 1)
        self.assertTrue(outcome is None)

    def test_next_state_end_game(self):
        mat_positions = [np.array([
            [0, 2, 1, 1, 1],
            [2, 1, 0, 0, 0],
            [0, 2, 1, 1, 1],
            [0, 0, 2, 2, 2],
            [0, 0, 0, 2, 0]
            ]), np.array([
            [0, 2, 1, 1, 1],
            [2, 1, 0, 0, 0],
            [0, 2, 1, 1, 1],
            [0, 0, 2, 2, 2],
            [0, 0, 0, 2, 0]
            ])]        
        state = create_zero_state(0)
        # set the last 2 states
        state.pos[-2:] = np.array([make_pos_from_matrix(pos) for pos in mat_positions])
        new_state, outcome = next_state(state, encode_action(None))
        self.assertTrue(new_state.player == 1)
        self.assertTrue(outcome == OUTCOME_WIN_PLAYER_2)


    def test_valid_actions(self):
        mat_positions = [np.array([
            [0, 2, 1, 0, 0],
            [2, 1, 0, 1, 0],
            [0, 2, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
            ]), np.array([
            [0, 2, 1, 0, 0],
            [2, 0, 2, 1, 0],
            [0, 2, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
            ])]        

        state = create_zero_state(PLAYER_1)
        # set the last 2 states
        state.pos[-2:] = np.array([make_pos_from_matrix(pos) for pos in mat_positions])
        self.assertTrue(state.pos.shape == (STATE_HIST_SIZE, 2, BOARD_SIZE, BOARD_SIZE))

        valid_acts = maybe_valid_actions(state)
        #print(valid_acts)
        brd = np.zeros((BOARD_SIZE, BOARD_SIZE))        
        for act in valid_acts:
            new_state, outome = next_state(state, act)
            if new_state is not None and not is_pass_action(act):
                brd[get_action_coords(act)] = 1
        self.assertTrue(brd[1, 1] == 0)        
        #print(brd)        


if __name__ == '__main__':
    unittest.main()