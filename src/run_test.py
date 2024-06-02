"""
Author : José CUNHA TEIXEIRA
License : SNCF Réseau, UMR 7619 METIS
Date : April 30, 2024
"""





from keras.metrics import Accuracy

from pickle import load

from folders import PATH_INPUT, PATH_MODELS
from misc import load_X, load_y





### UNCOMMENT TO RUN ON CPUs ----------------------------------------------------------------------
# from tensorflow.config import set_visible_devices
# from tensorflow.config.threading import set_intra_op_parallelism_threads, set_inter_op_parallelism_threads

# set_visible_devices([], 'GPU')
# N_jobs = 16
# set_intra_op_parallelism_threads(N_jobs)
# set_inter_op_parallelism_threads(N_jobs)
### -----------------------------------------------------------------------------------------------





### LOAD MODEL ------------------------------------------------------------------------------------
MODEL_NAME = '[]'

path_model = f'{PATH_MODELS}/{MODEL_NAME}/{MODEL_NAME}_model'
print(f'\nLoading model : {path_model}')

with open(f'{path_model}.pkl', 'rb') as f:
    transformer = load(f)
### -----------------------------------------------------------------------------------------------





### SEQUENCES' FORMATS ----------------------------------------------------------------------------
input_sequence_format = transformer.input_sequence_format
min_freq = input_sequence_format.vocab.min
max_freq = input_sequence_format.vocab.max

output_sequence_format = transformer.output_sequence_format
word_to_index = output_sequence_format.vocab.word_to_index
index_to_word = output_sequence_format.vocab.index_to_word
len_output_seq = output_sequence_format.length
### -----------------------------------------------------------------------------------------------





### LOAD DATA -------------------------------------------------------------------------------------
data_folder = f'{PATH_INPUT}/training_data6/'
N_samples = 1000000 # Number of samples to load

X = load_X(data_folder, N_samples)
y = load_y(data_folder, N_samples, word_to_index, N_layers=4)


print('\033[93m\nDATASET EXAMPLE\033[0m')
print(f'{X[0,...] = }')
print(f'{y[0,...] = }')
for el in y[0,...]:
  print(index_to_word[el])

print('\033[93m\nDATASET SHAPE\033[0m')
print(f'{X.shape = }')
print(f'{y.shape = }')
### -----------------------------------------------------------------------------------------------





### SPLIT DATA ------------------------------------------------------------------------------------
train_ratio = 0.80
test_start_idx = int((train_ratio + (1 - train_ratio) / 2)*X.shape[0])

X_test = X[test_start_idx:, ...]
y_test = y[test_start_idx:, ...]

N_samples_test = X_test.shape[0]


del X, y, N_samples, data_folder


print('\033[93m\nTESTING DATASET EXAMPLE\033[0m')
print(f'{X_test[0,...] = }')
print(f'{y_test[0,...] = }')
for el in y_test[0,...]:
  print(index_to_word[el])


print('\033[93m\nTESTING DATASET SHAPE\033[0m')
print(f'{X_test.shape = }')
print(f'{y_test.shape = }')
### -----------------------------------------------------------------------------------------------





### INFERENCE TEST --------------------------------------------------------------------------------
N_samples_to_evaluate = 2000
if N_samples_to_evaluate > N_samples_test:
    N_samples_to_evaluate = N_samples_test

transformer.evaluate(X_test[:N_samples_to_evaluate,...], y_test[:N_samples_to_evaluate,...])
### -----------------------------------------------------------------------------------------------





# ### INFERENCE -------------------------------------------------------------------------------------
# for i_sample in range(X_test.shape[0]):
#   input_seq = X_test[i_sample, ...].reshape(1, X_test.shape[1], 1)
#   target_seq = y_test[i_sample, 1:-1]

#   decoded_seq = transformer.decode_seq(input_seq)
#   # decoded_seq = transformer.decode_seq_restrictive(input_seq)

#   m = Accuracy()
#   accuracy = m(target_seq, decoded_seq).numpy()

#   print('\n\n----------------------------------------------')
#   print('Decoded Sequence:\n', list(decoded_seq))
#   print('\nTarget Sequence:\n', list(target_seq))
#   print('\nTARGET | DECODED - O/X:')
#   for i in range(len(target_seq)):
#       symbol = 'O' if target_seq[i] == decoded_seq[i] else 'X'
#       print(f'{i+1}: {index_to_word[target_seq[i]]} | {index_to_word[decoded_seq[i]]} - {symbol}')
#   print(f'\nAccuracy: {accuracy*100} %')
#   print('----------------------------------------------\n\n')

#   input('Press Enter to continue...')
# ### -----------------------------------------------------------------------------------------------