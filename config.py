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



# PATCH_SIZE       = 


# splits and epochs
TRAIN_SPLIT      = 0.85
TRAIN_BATCH_SIZE = 8
TEST_BATCH_SIZE  = 8

EPOCHS           = 150