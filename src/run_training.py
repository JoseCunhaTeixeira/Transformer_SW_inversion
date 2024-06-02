"""
Author : José CUNHA TEIXEIRA
License : SNCF Réseau, UMR 7619 METIS
Date : April 30, 2024
"""





import numpy as np

from folders import PATH_INPUT
from sequences import InputSequenceFormat, OutputSequenceFormat 
from transformer import Transformer
from misc import load_X, load_y





### UNCOMMENT TO RUN ON CPUs ----------------------------------------------------------------------
# from tensorflow.config import set_visible_devices
# from tensorflow.config.threading import set_intra_op_parallelism_threads, set_inter_op_parallelism_threads

# set_visible_devices([], 'GPU')
# N_jobs = 16
# set_intra_op_parallelism_threads(N_jobs)
# set_inter_op_parallelism_threads(N_jobs)
### -----------------------------------------------------------------------------------------------





### SEQUENCES' FORMATS ----------------------------------------------------------------------------
input_sequence_format = InputSequenceFormat()

output_sequence_format = OutputSequenceFormat()
word_to_index = output_sequence_format.vocab.word_to_index
index_to_word = output_sequence_format.vocab.index_to_word
### -----------------------------------------------------------------------------------------------





### LOAD DATA -------------------------------------------------------------------------------------
data_folder = f'{PATH_INPUT}/training_data7/'
N_samples = 1000000 # Number of samples to load

X, X_noise = load_X(data_folder, N_samples, noise=False)
y = load_y(data_folder, N_samples, word_to_index, N_layers=4)


print('\033[93m\nDATASET EXAMPLE\033[0m')
print(f'{X[0,...] = }')
print(f'{y[0,...] = }')
for el in y[0,...]:
  print(index_to_word[el])

print('\033[93m\nDATASET SHAPE\033[0m')
print(f'{X.shape = }')
print(f'{y.shape = }')


# # Uncomment to test model with only one sample
# shape = X.shape
# first_sample_X = X[0, ...].reshape(1, X.shape[1], 1)
# X = np.full(shape, first_sample_X)

# shape_y = y.shape
# first_sample_y = y[0, ...].reshape(1, y.shape[1])
# y = np.full(shape_y, first_sample_y)
### -----------------------------------------------------------------------------------------------





### SPLIT DATA ------------------------------------------------------------------------------------
train_ratio = 0.80
train_end_idx = int(train_ratio*X.shape[0])

X_train = X[:train_end_idx, ...]
y_train = y[:train_end_idx, ...]

N_samples_train = X_train.shape[0]


val_ratio = (1 - train_ratio) / 2
val_start_idx = train_end_idx
val_end_idx = val_start_idx + int(val_ratio*X.shape[0]) + 1

X_val = X[val_start_idx:val_end_idx, ...]
y_val = y[val_start_idx:val_end_idx, ...]

N_samples_val = X_val.shape[0]


test_start_idx = val_end_idx

X_test = X[test_start_idx:, ...]
y_test = y[test_start_idx:, ...]

N_samples_test = X_test.shape[0]


del X, y, N_samples, data_folder


print('\033[93m\nTRAINING DATASET EXAMPLE\033[0m')
print(f'{X_train[0,...] = }')
print(f'{y_train[0,...] = }')
for el in y_train[0,...]:
  print(index_to_word[el])

print('\033[93m\nVALIDATION DATASET\033[0m')
print(f'{X_val[0,...] = }')
print(f'{y_val[0,...] = }')
for el in y_val[0,...]:
  print(index_to_word[el])

print('\033[93m\nTESTING DATASET EXAMPLE\033[0m')
print(f'{X_test[0,...] = }')
print(f'{y_test[0,...] = }')
for el in y_test[0,...]:
  print(index_to_word[el])


print('\033[93m\nTRAINING DATASET SHAPE\033[0m')
print(f'{X_train.shape = }')
print(f'{y_train.shape = }')

print('\033[93m\nTESTING DATASET SHAPE\033[0m')
print(f'{X_test.shape = }')
print(f'{y_test.shape = }')

print('\033[93m\nVALIDATION DATASET SHAPE\033[0m')
print(f'{X_val.shape = }')
print(f'{y_val.shape = }\n\n')
### -----------------------------------------------------------------------------------------------





### MODEL -----------------------------------------------------------------------------------------
transformer = Transformer(input_sequence_format, output_sequence_format)
### -----------------------------------------------------------------------------------------------





### TRAINING --------------------------------------------------------------------------------------
transformer.train(X_train, y_train,
                  X_val, y_val,
                  epochs=100, batch_size=64)
### -----------------------------------------------------------------------------------------------





### SAVE ------------------------------------------------------------------------------------------
transformer.save_model()
### -----------------------------------------------------------------------------------------------





### INFERENCE TEST --------------------------------------------------------------------------------
N_samples_to_evaluate = 2000
if N_samples_to_evaluate > N_samples_test:
    N_samples_to_evaluate = N_samples_test
    
transformer.evaluate(X_test[:N_samples_to_evaluate,...], y_test[:N_samples_to_evaluate,...])
### -----------------------------------------------------------------------------------------------