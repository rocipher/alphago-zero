import numpy as np

SEED = 0
OUTPUT_DIR = "./out"

NUM_GAMES = 20
NUM_TRAIN_LOOP_ITER = 50
NUM_ITER = 5
BATCH_SIZE = 5
STATE_HIST_SIZE = 8
NUM_EVAL_GAMES = 5
MIN_MODEL_REPLACEMENT_WIN_RATIO = 0.55 # 55%
MCTS_STEPS = 50
MAX_GAMES_HISTORY_SIZE = 20000

PLAYER_1 = 0
PLAYER_2 = 1

REWARD_DRAW = 0.0
REWARD_WIN = 1.0
REWARD_LOOSE = -1.0
BOARD_SIZE = 5
# first 3% of the moves have temperature 1.0 (exploratory), the rest have 0.0 temperature
#ZERO_TEMP_MOVE_INDEX = int(BOARD_SIZE*BOARD_SIZE/2*0.16) # 16% of the max nr of moves
ZERO_TEMP_MOVE_INDEX = 10
TEMPERATURES = [(0, ZERO_TEMP_MOVE_INDEX, 1.0), (ZERO_TEMP_MOVE_INDEX, np.inf, 1e-8)]
TEMPERATURE_MIN = 1e-3

ACTION_SPACE_SIZE = BOARD_SIZE*BOARD_SIZE+1
DIRICHLET_ALPHA = 0.03
EPS = 0.25
C_PUCT = 1.0

CONV_FILTERS = 16
CONV_KERNEL = (3, 3)
NUM_RESIDUAL_BLOCKS = 2
DENSE_SIZE = 16

# NUM_GAMES = 25000
# NUM_TRAIN_LOOP_ITER = 1000
# NUM_ITER = 10
# BATCH_SIZE = 2048
# NUM_EVAL_GAMES = 400
# MIN_MODEL_REPLACEMENT_WIN_RATIO = 0.55 # 55%
# MCTS_STEPS = 1600
# MAX_GAMES_HISTORY_SIZE = 500000

# REWARD_DRAW = 0.0
# REWARD_WIN = 1.0
# REWARD_LOOSE = -1.0
# TEMPERATURES = [(0, 30, 1.0), (30, np.inf, 1e-8)]
# BOARD_SIZE = 5
# ACTION_SPACE_SIZE = BOARD_SIZE*BOARD_SIZE+1
# DIRICHLET_ALPHA = 0.03
# EPS = 0.25
# C_PUCT = 1.0