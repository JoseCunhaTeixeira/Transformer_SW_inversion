"""
Author : José CUNHA TEIXEIRA
License : SNCF Réseau, UMR 7619 METIS
Date : April 30, 2024
"""





from keras.metrics import Accuracy

import numpy as np
from tqdm import tqdm
from seaborn import heatmap
import matplotlib.pyplot as plt
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
MODEL_NAME = '[202405222142]'

path_model = f'{PATH_MODELS}/{MODEL_NAME}/{MODEL_NAME}_model'
print(f'\nLoading model : {path_model}')

with open(f'{path_model}.pkl', 'rb') as f:
    model = load(f)
### -----------------------------------------------------------------------------------------------





### SEQUENCES' FORMATS ----------------------------------------------------------------------------
input_sequence_format = model.input_sequence_format
min_freq = input_sequence_format.vocab.min
max_freq = input_sequence_format.vocab.max

output_sequence_format = model.output_sequence_format
word_to_index = output_sequence_format.vocab.word_to_index
index_to_word = output_sequence_format.vocab.index_to_word
len_output_seq = output_sequence_format.length
### -----------------------------------------------------------------------------------------------





### LOAD DATA -------------------------------------------------------------------------------------
data_folder = f'{PATH_INPUT}/training_data6/'
N_samples = 800000 # Number of samples to load

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
train_ratio = 0.75
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
words = model.output_sequence_format.vocab.words.copy()
conf_matrix = np.zeros((len(words), len(words)))

accuracies = []

# for i_sample in tqdm(range(X_test.shape[0])):
for i_sample in tqdm(range(2000)):
  input_seq = X_test[i_sample, ...].reshape(1, X_test.shape[1], 1)
  target_seq = y_test[i_sample, 1:-1]

  # decoded_seq = model.decode_seq(input_seq)
  decoded_seq = model.decode_seq_restrictive(input_seq)

  m = Accuracy()
  accuracy = m(target_seq, decoded_seq).numpy()
  accuracies.append(accuracy)

  for decoded_word_idx, target_word_idx in zip(decoded_seq, target_seq):
    conf_matrix[target_word_idx, decoded_word_idx] += 1

accuracies = np.array(accuracies)
accuracy = np.mean(accuracies)
print(f'\nAccuracy: {accuracy*100} %')

fig, ax = plt.subplots(dpi=300, figsize=(16, 9))
ax.scatter(range(len(accuracies)), accuracies*100)
ax.set_xlabel('Sample (#)')
ax.set_ylabel('Accuracy (%)')
ax.set_ylim([0,100])
fig.savefig(f'{PATH_MODELS}/{model.id}/{model.id}_test_accuracies.png', format='png', dpi='figure', bbox_inches='tight')

fig, ax = plt.subplots(depi=300, figsize=(10, 10))
heatmap(conf_matrix, annot=True, cmap='Blues', ax=ax, xticklabels=words, yticklabels=words)
ax.set_xlabel('Decoded Words')
ax.set_ylabel('Expected Words')
ax.set_title('Confusion Matrix')
fig.savefig(f'{PATH_MODELS}/{model.id}/{model.id}_test_confusion_matrix.png', format='png', dpi='figure', bbox_inches='tight')
### -----------------------------------------------------------------------------------------------