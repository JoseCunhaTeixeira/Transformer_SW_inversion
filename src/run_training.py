"""
Author : José CUNHA TEIXEIRA
License : SNCF Réseau, UMR 7619 METIS, Sorbonne Université
Date : April 30, 2024
"""





from folders import PATH_INPUT
from transformer import Transformer
from misc import load_data, make_vocab, make_index_representation, make_forbidden_tokens
import numpy as np
import json





### PARAMS ----------------------------------------------------------------------------------------
data_folder = f'{PATH_INPUT}/training_data/Grand_Est/'

min_freq = 15 # minimum frequency to invert [Hz]
max_freq = 50 # maximum frequency to invert [Hz]
### -----------------------------------------------------------------------------------------------





### FORMATS ---------------------------------------------------------------------------------------
with open(f'{data_folder}/params.json', 'r') as f:
  data_params = json.load(f)


freqs = np.array(data_params['freqs'])
i_min = np.where(freqs == min_freq)[0][0]
i_max = np.where(freqs == max_freq)[0][0]
freqs = freqs[i_min:i_max+1]
d_freq = float(freqs[1] - freqs[0])

max_N_layers = data_params['max_N_layers']
output_seq_vocab = make_vocab(data_params)
output_seq_length = max_N_layers*6 + 4
word_to_index, index_to_word = make_index_representation(output_seq_vocab)
forbidden_tokens = make_forbidden_tokens(data_params, word_to_index, output_seq_length)
### -----------------------------------------------------------------------------------------------





### LOAD DATA -------------------------------------------------------------------------------------
X, min_vel, max_vel, y, N_samples = load_data(data_folder,
                                              word_to_index,
                                              max_N_layers,
                                              min_freq, max_freq,
                                              noise=False,
                                              )

print('\033[93m\nDATASET EXAMPLE\033[0m')
print(f'{X[0,...] = }')
print(f'{y[0,...] = }')
for el in y[0,...]:
  print(index_to_word[el])

print('\033[93m\nDATASET SHAPE\033[0m')
print(f'{X.shape = }')
print(f'{y.shape = }')

print('\033[93m\nVELOCITY NORMALIZATION\033[0m')
print(f'{min_vel = }')
print(f'{max_vel = }\n\n')
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

print('\033[92m\nVALIDATION DATASET SHAPE:\033[0m')
print(f'{X_val.shape = }')
print(f'{y_val.shape = }\n\n')
### -----------------------------------------------------------------------------------------------





### -----------------------------------------------------------------------------------------------
params = {
  'data_params' : data_params,

  'input_seq_format' : {
    'length' : len(freqs),
    'freqs' : freqs.tolist(),
    'N_freqs' : len(freqs),
    'min_freq' : min_freq,
    'max_freq' : max_freq,
    'd_freq' : d_freq,
    'min_vel' : min_vel,
    'max_vel' : max_vel,
  },

  'output_seq_format' : {
    'length' : output_seq_length,
    'vocab' : output_seq_vocab,
    'vocab_size' : len(output_seq_vocab),
    'word_to_index' : word_to_index,
    'index_to_word' : index_to_word,
    'forbidden_tokens' : forbidden_tokens,
  },
}

transformer = Transformer(params)
### -----------------------------------------------------------------------------------------------





### TRAINING --------------------------------------------------------------------------------------
transformer.train(X_train, y_train,
                  X_val=X_val, y_val=y_val,
                  epochs=100, batch_size=64,
                  data_name=data_folder.split('/')[-2])
### -----------------------------------------------------------------------------------------------





### SAVE ------------------------------------------------------------------------------------------
transformer.save_model()
### -----------------------------------------------------------------------------------------------