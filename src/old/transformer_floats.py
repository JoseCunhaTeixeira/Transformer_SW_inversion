"""
Author : José CUNHA TEIXEIRA
License : SNCF Réseau, UMR 7619 METIS
Date : April 30, 2024
"""



from keras import Model
from keras import ops
from keras.layers import Input, Dense, Dropout, Conv1D, MaxPooling1D
from keras_nlp.layers import TransformerEncoder, TransformerDecoder, TokenAndPositionEmbedding, SinePositionEncoding
from keras_nlp.samplers import GreedySampler
from keras.optimizers import RMSprop, Adam
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
from keras.metrics import Accuracy

import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from folders import PATH_INPUT, PATH_MODELS
from misc import load_X, load_y, index_representation
from sequences import SoilVocab





### UNCOMMENT TO RUN ON CPUs ----------------------------------------------------------------------
# from tensorflow.config import set_visible_devices
# from tensorflow.config.threading import set_intra_op_parallelism_threads, set_inter_op_parallelism_threads

# set_visible_devices([], 'GPU')
# N_jobs = 16
# set_intra_op_parallelism_threads(N_jobs)
# set_inter_op_parallelism_threads(N_jobs)
### -----------------------------------------------------------------------------------------------





### SOILS FORMAT ----------------------------------------------------------------------------------
soil_vocab = SoilVocab()
SOIL_VOCAB_SIZE = soil_vocab.size
word_to_index = soil_vocab.word_to_index
index_to_word = soil_vocab.index_to_word
### -----------------------------------------------------------------------------------------------





### LOAD DATA ----------------------------------------------------------------------------
print('\033[1;32m\033[1m\n\n\n_______________LOADING DATASET_______________\033[0m')



data_folder_train = f'{PATH_INPUT}training_data6/'
N_samples = 800000 # Number of samples to load



# X = load_X(data_folder_train, N_samples)
# y_index, _ = load_y(data_folder_train, N_samples, word_to_index)

# np.save(f'{data_folder_train}X_floats.npy', X)
# np.save(f'{data_folder_train}y_index.npy', y_index)


X = np.load(f'{data_folder_train}X_floats.npy')
y_index = np.load(f'{data_folder_train}y_index.npy')


print('\033[93m\nDATASET EXAMPLE\033[0m')
print(f'{X[0,...] = }')
print(f'{y_index[0,...] = }')
for el in y_index[0,...]:
  print(index_to_word[el])


print('\033[93m\nDATASET SHAPE\033[0m')
print(f'{X.shape = }')
print(f'{y_index.shape = }')



# # Test model with only one sample
# shape = X.shape
# first_sample = X[0, ...].reshape(1, X.shape[1], 1)
# X = np.full(shape, first_sample)

# shape_y = y_index.shape
# first_sample_y = y_index[0, ...].reshape(1, y_index.shape[1])
# y_index = np.full(shape_y, first_sample_y)



# Split into training, validation, and testing dataset
train_ratio = 0.75
train_end_idx = int(train_ratio*X.shape[0])

X_train = X[:train_end_idx, ...]
y_train_index = y_index[:train_end_idx, ...]

N_samples_train = X_train.shape[0]



val_ratio = (1 - train_ratio) / 2
val_start_idx = train_end_idx
val_end_idx = val_start_idx + int(val_ratio*X.shape[0])

X_val = X[val_start_idx:val_end_idx, ...]
y_val_index = y_index[val_start_idx:val_end_idx, ...]

N_samples_val = X_val.shape[0]



test_start_idx = val_end_idx

X_test = X[test_start_idx:, ...]
y_test_index = y_index[test_start_idx:, ...]

N_samples_test = X_test.shape[0]



del X, y_index, N_samples, data_folder_train



print('\033[93m\nTRAINING DATASET EXAMPLE\033[0m')
print(f'{X_train[0,...] = }')
print(f'{y_train_index[0,...] = }')
for el in y_train_index[0,...]:
  print(index_to_word[el])

print('\033[93m\nVALIDATION DATASET\033[0m')
print(f'{X_val[0,...] = }')
print(f'{y_val_index[0,...] = }')
for el in y_val_index[0,...]:
  print(index_to_word[el])

print('\033[93m\nTESTING DATASET EXAMPLE\033[0m')
print(f'{X_test[0,...] = }')
print(f'{y_test_index[0,...] = }')
for el in y_test_index[0,...]:
  print(index_to_word[el])



print('\033[93m\nTRAINING DATASET SHAPE\033[0m')
print(f'{X_train.shape = }')
print(f'{y_train_index.shape = }')

print('\033[93m\nTESTING DATASET SHAPE\033[0m')
print(f'{X_test.shape = }')
print(f'{y_test_index.shape = }')

print('\033[93m\nVALIDATION DATASET SHAPE\033[0m')
print(f'{X_val.shape = }')
print(f'{y_val_index.shape = }')
### -----------------------------------------------------------------------------------------------





### MODEL -----------------------------------------------------------------------------------------
print( '\033[1;32m\033[1m\n\n\n_______________MODEL_______________\033[0m\n')



# Parameters
LEN_INPUT_SEQUENCE = X_train.shape[1]
LEN_OUTPUT_SEQUENCE = y_train_index.shape[1] - 1

# 2 4 8 16 32 64 128 256 512 1024 2048 4096
ENC_EMBED_DIM = 64
DEC_EMBED_DIM = 64

INTERMEDIATE_DIM = 128
NUM_HEADS = 8

ENC_N_LAYERS = 4
DEC_N_LAYERS = 4



print('\033[93m\nMODEL PARAMETERS\033[0m')
print(f'{LEN_INPUT_SEQUENCE = }', f'\n{LEN_OUTPUT_SEQUENCE = }')
print(f'{SOIL_VOCAB_SIZE = }')
print(f'{ENC_EMBED_DIM = }', f'\n{DEC_EMBED_DIM = }')
print(f'{INTERMEDIATE_DIM = }', f'\n{NUM_HEADS = }')
print(f'{ENC_N_LAYERS = }', f'\n{DEC_N_LAYERS = }\n')



# Encoder
encoder_inputs = Input(shape=(LEN_INPUT_SEQUENCE, 1))

c1 = Conv1D(16, 3, activation='relu', padding='same')(encoder_inputs)
c1 = Conv1D(16, 3, activation='relu', padding='same')(c1)
p1 = MaxPooling1D(pool_size=2)(c1)

c2 = Conv1D(32, 3, activation='relu', padding='same')(p1)
c2 = Conv1D(32, 3, activation='relu', padding='same')(c2)
p2 = MaxPooling1D(pool_size=2)(c2)

c3 = Conv1D(64, 3, activation='relu', padding='same')(p2)
c3 = Conv1D(64, 3, activation='relu', padding='same')(c3)

CNN_output = c3


position_encoding = SinePositionEncoding()(CNN_output)
x1 = CNN_output + position_encoding

for _ in range(ENC_N_LAYERS):
  x1 = TransformerEncoder(intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS)(inputs=x1)

encoder_outputs = x1

encoder = Model(encoder_inputs, encoder_outputs)


# Decoder
decoder_inputs = Input(shape=(None, ))

encoded_seq_inputs = Input(shape=(None, ENC_EMBED_DIM))


x2 = TokenAndPositionEmbedding(
    vocabulary_size=SOIL_VOCAB_SIZE,
    sequence_length=LEN_OUTPUT_SEQUENCE,
    embedding_dim=DEC_EMBED_DIM,
    # mask_zero=True,
)(decoder_inputs)

for _ in range(DEC_N_LAYERS):
  x2 = TransformerDecoder(intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS)(decoder_sequence=x2, encoder_sequence=encoded_seq_inputs)

# x2 = Dropout(0.1)(x2)

decoder_outputs = Dense(SOIL_VOCAB_SIZE, activation="softmax")(x2)

decoder = Model([decoder_inputs, encoded_seq_inputs], decoder_outputs)

decoder_outputs = decoder([decoder_inputs, encoder_outputs])


model = Model(
    [encoder_inputs, decoder_inputs],
    decoder_outputs,
)




# Print summary
model.summary()


# Compile model
model.compile(optimizer=RMSprop(learning_rate=1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Plot model
MODEL_NAME = f'[{datetime.now().strftime("%Y%m%d%H%M")}]'

MODEL_PATH = f'{PATH_MODELS}{MODEL_NAME}/'
if not os.path.exists(MODEL_PATH):
  os.makedirs(MODEL_PATH)

# plot_model(model, to_file=f'{MODEL_PATH}{MODEL_NAME}_plot.png', show_shapes=True, show_layer_names=True)
### -----------------------------------------------------------------------------------------------





# TRAINING ----------------------------------------------------------------------------------------
history = model.fit([X_train, y_train_index[:, :-1]], y_train_index[:, 1:],
                          epochs=50,
                          batch_size=64,
                          shuffle=True,
                          validation_data=([X_val, y_val_index[:, :-1]], y_val_index[:, 1:]),
                          callbacks=[EarlyStopping(monitor='val_loss', start_from_epoch=10, patience=10)],
                          verbose=1)


# Plot loss and accuracy curves
fig, ax = plt.subplots(figsize=(16,10), dpi=300)

# Plot Loss
ax.plot(history.history['loss'])
ax.plot(history.history['val_loss'])
ax.set_xlabel("Epochs")
ax.set_ylabel("Categorical Crossentropy Loss")
ax.legend(["Training dataset", "Validation dataset"])

# Plot Accuracy
ax_twin = ax.twinx()
ax_twin.plot(history.history['accuracy'], linestyle='--')
ax_twin.plot(history.history['val_accuracy'], linestyle='--')
ax_twin.set_xlabel("Epochs")
ax_twin.set_ylabel("Accuracy")
ax_twin.legend(["Training dataset", "Validation dataset"])

# Save figure
fig.savefig(f"{MODEL_PATH}{MODEL_NAME}_history.png", format='png', dpi='figure', bbox_inches='tight')
### -----------------------------------------------------------------------------------------------





### SAVE ------------------------------------------------------------------------------------------
model.save(f'{MODEL_PATH}{MODEL_NAME}_model.keras')
### -----------------------------------------------------------------------------------------------






### INFERENCE -------------------------------------------------------------------------------------
def decode_seq(input_seq):
  input_seq = ops.convert_to_tensor(input_seq)

  def next(prompt, cache, index):
      logits = model([input_seq, prompt])[:, index-1, :]
      hidden_states = None
      return logits, hidden_states, cache

  prompt = [word_to_index['[START]']]
  while len(prompt) < LEN_OUTPUT_SEQUENCE:
      prompt.append(word_to_index['[PAD]'])
  prompt = np.array(prompt).reshape(1, len(prompt))
  prompt = ops.convert_to_tensor(prompt)

  decoded_seq  = GreedySampler()(
      next,
      prompt,
      stop_token_ids=[word_to_index['[END]']],
      index=1
      )

  return ops.convert_to_numpy(decoded_seq)[0][1:]


for i_sample in range(N_samples_test):
  input_seq = X_test[i_sample, ...].reshape(1, X_test.shape[1], 1)
  target_seq = y_test_index[i_sample, 1:-1]

  decoded_seq = decode_seq(input_seq)

  m = Accuracy()
  accuracy = m(target_seq[1:-1], decoded_seq[1:-1]).numpy()


  print('\n\n----------------------------------------------')
  print("Decoded Sequence:\n", list(decoded_seq))

  print("\nTarget Sequence:\n", list(target_seq))

  print("\nTARGET | DECODED - O/X:")
  for i in range(0, len(target_seq)):
    if (target_seq[i] == decoded_seq[i]):
      symbol = "O"
    else:
      symbol = "X"
    print(f'{i+1}: {index_to_word[target_seq[i]]} | {index_to_word[decoded_seq[i]]} - {symbol}')

  print(f"\nAccuracy: {accuracy*100} %")
  print('----------------------------------------------\n\n')


  input("Press Enter to continue...")
### -----------------------------------------------------------------------------------------------