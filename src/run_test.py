"""
Author : José CUNHA TEIXEIRA
License : SNCF Réseau, UMR 7619 METIS, Sorbonne Université
Date : April 30, 2024
"""





from keras.metrics import Accuracy

import sys
from pickle import load

from folders import PATH_INPUT, PATH_MODELS
from misc import load_data





### PARAMS ----------------------------------------------------------------------------------------
model_id = '[202407170928]'
data_folder = f'{PATH_INPUT}/training_data/Grand_Est/'
### -----------------------------------------------------------------------------------------------





### LOAD MODEL ------------------------------------------------------------------------------------
path_model = f'{PATH_MODELS}/{model_id}/{model_id}_model'
print(f'\nLoading model : {path_model}')

with open(f'{path_model}.pkl', 'rb') as f:
    transformer = load(f)
### -----------------------------------------------------------------------------------------------





### FORMATS ---------------------------------------------------------------------------------------
params = transformer.params

if params['model_params']['trained'] == False:
  print(f'\033[1;33mERROR: Model {model_id} was not trained yet.\033[0m')
  sys.exit

max_N_layers = params['data_params']['max_N_layers']

min_freq = params['input_seq_format']['min_freq']
max_freq = params['input_seq_format']['max_freq']
min_vel = params['input_seq_format']['min_vel']
max_vel = params['input_seq_format']['max_vel']

word_to_index = params['output_seq_format']['word_to_index']
index_to_word = params['output_seq_format']['index_to_word']
len_output_seq = params['output_seq_format']['length']
### -----------------------------------------------------------------------------------------------





### LOAD DATA -------------------------------------------------------------------------------------
X, min_vel, max_vel, y, N_samples = load_data(data_folder,
                                              word_to_index,
                                              max_N_layers,
                                              min_freq, max_freq,
                                              min_vel=min_vel, max_vel=max_vel,
                                              noise=False)

if min_vel != params['input_seq_format']['min_vel'] or max_vel != params['input_seq_format']['max_vel']:
    raise ValueError("Min and max velocities differ from the ones used to train the model.")


print('\033[93m\nDATASET SHAPE\033[0m')
print(f'{X.shape = }')
print(f'{y.shape = }')
### -----------------------------------------------------------------------------------------------





### SPLIT DATA ------------------------------------------------------------------------------------
N_samples_val = 100_000
N_samples_test = 50_000

train_end_idx = int(X.shape[0]) - N_samples_val - N_samples_test
X_train = X[:train_end_idx, ...]
y_train = y[:train_end_idx, ...]
N_samples_train = X_train.shape[0]


val_start_idx = train_end_idx
val_end_idx = val_start_idx + N_samples_val
X_val = X[val_start_idx:val_end_idx, ...]
y_val = y[val_start_idx:val_end_idx, ...]
N_samples_val = X_val.shape[0]


test_start_idx = val_end_idx
X_test = X[test_start_idx:, ...]
y_test = y[test_start_idx:, ...]
N_samples_test = X_test.shape[0]


del X, y, N_samples, X_val, y_val, N_samples_val, X_train, y_train, N_samples_train


print('\033[92m\nTESTING DATASET SHAPE:\033[0m')
print(f'{X_test.shape = }')
print(f'{y_test.shape = }\n\n')
### -----------------------------------------------------------------------------------------------





### INFERENCE TEST --------------------------------------------------------------------------------
# transformer.evaluate(X_test, y_test, data_name=data_folder.split('/')[-2])
# or
eval = transformer.model.evaluate([X_test, y_test[:, :-1]], y_test[:, 1:], verbose=1)
print(f"\n\n\033[92mTESTING LOSS:\033[0m {eval[0]}")
print(f"\033[92mTESTING ACCURACY:\033[0m {eval[1]}\n")
### -----------------------------------------------------------------------------------------------








# ### INFERENCE TEST (by hand) --------------------------------------------------------------------
# for i_sample in range(X_test.shape[0]):
#   input_seq = X_test[i_sample, ...].reshape(1, X_test.shape[1], 1)
#   target_seq = y_test[i_sample, 1:-1]

#   # decoded_seq = transformer.decode_seq(input_seq)
#   decoded_seq = transformer.decode_seq_restrictive(input_seq)

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