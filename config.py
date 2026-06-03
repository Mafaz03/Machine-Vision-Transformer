import torch

# dataset 
RE_MEAN = 1175.7935384615384    # with P
RE_STD  = 744.9444946064118     # with P

RE_MAX  = 2499.1                # with P

U_MEAN = 0.008343593
U_STD  = 0.23272848

V_MEAN = 0.00028657727
V_STD  = 0.19424047

P_MEAN = 0.26744577
P_STD  = 0.01844339


# dimention
FOURIER_FEATURES   = 16
FOURIER_DIMENSIONS = 2 * 2 * FOURIER_FEATURES
C                  = 3 # u, v, P


# splits and epochs
TRAIN_SPLIT      = 0.85
TRAIN_BATCH_SIZE = 8
TEST_BATCH_SIZE  = 8
EPOCHS           = 500

# dimensions
GRID_SIZE  = 64
PATCH_SIZE = 8
PATCH_DIM  = (PATCH_SIZE * PATCH_SIZE * C) + FOURIER_DIMENSIONS # +64 because positional embedding was done in the dataset itself
                                                                # tgt: (patch_row * patch_col, C * patch_h * patch_w + (2 * 2 * num_freq))
D_MODEL    = 1024

# parameters
N         = 6
NUM_HEADS = 8
D_FF      = 1024
DROPOUT   = 0.01

# extra
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_EVERY = 20