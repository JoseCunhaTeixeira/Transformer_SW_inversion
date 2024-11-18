"""
Author : José CUNHA TEIXEIRA
License : SNCF Réseau, UMR 7619 METIS, Sorbonne Université
Date : April 30, 2024
"""





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
                                            noise=True)

if min_vel != params['input_seq_format']['min_vel'] or max_vel != params['input_seq_format']['max_vel']:
    raise ValueError("Min and max velocities differ from the ones used to train the model.")
  

print('\033[92m\nDATASET SHAPE:\033[0m')
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


del X, y, N_samples, N_samples_test


print('\033[92m\nTRAINING DATASET SHAPE:\033[0m')
print(f'{X_train.shape = }')
print(f'{y_train.shape = }')
### -----------------------------------------------------------------------------------------------





### TRAINING --------------------------------------------------------------------------------------
transformer.train(X_train, y_train,
                  epochs=5, batch_size=64,
                  data_name=data_folder.split('/')[-2])
### -----------------------------------------------------------------------------------------------





### SAVE ------------------------------------------------------------------------------------------
transformer.save_model()
### -----------------------------------------------------------------------------------------------