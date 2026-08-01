import torch
from pathlib import Path

# dataset 
DATASET = "flow_past_cylinder_domain"

# LID DRIVEN

# RE_MEAN = 1191.50125            # with P
# RE_STD  = 739.8270885769103     # with P

# RE_MAX  = 2499.1                # with P

# U_MEAN = 0.00834795
# U_STD  = 0.23277222

# V_MEAN = 0.0002897293
# V_STD  = 0.19554001

# P_MEAN = -0.03955434
# P_STD  = 0.3204666


RE_MEAN = 23.0               
RE_STD  = 12.84523257866513  

RE_MAX  = 45                 

U_MEAN = 0.2247134
U_STD  = 0.17602053

V_MEAN = -0.00048188222
V_STD  = 0.028765377

P_MEAN = 0.023107985
P_STD  = 0.03080877

P_MEAN = 0.023107985
P_STD  = 0.03080877


# dimention
FOURIER_FEATURES   = 4
FOURIER_DIMENSIONS = 2 * 2 * FOURIER_FEATURES
C                  = 3 # u, v, P


# splits and epochs
TRAIN_SPLIT      = 0.85
TRAIN_BATCH_SIZE = 8
TEST_BATCH_SIZE  = 8
EPOCHS           = 1_000

# dimensions
GRID_SIZE  = 64
PATCH_SIZE = 8
PATCH_DIM  = (PATCH_SIZE * PATCH_SIZE * C) + FOURIER_DIMENSIONS # +64 because positional embedding was done in the dataset itself
                                                                # tgt: (patch_row * patch_col, C * patch_h * patch_w + (2 * 2 * num_freq))
D_MODEL    = 512

# parameters
N         = 6 # 6
NUM_HEADS = 8 # 8
D_FF      = 1024
DROPOUT   = 0.01

# extra
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_EVERY = 20

# save 
MODEL_NAME = "flow_past_cylinder.pt"

# root files
ROOT = Path(__file__).resolve().parent.parent