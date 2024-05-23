"""
Author : José CUNHA TEIXEIRA
License : SNCF Réseau, UMR 7619 METIS
Date : April 30, 2024
"""



from keras import Model
from keras.layers import Input, Dense, Dropout
from keras_nlp.layers import TransformerEncoder, TransformerDecoder, TokenAndPositionEmbedding
from keras.optimizers import RMSprop, Adam
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
from keras.metrics import Accuracy

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from folders import path_input, path_models
from misc import load_X, load_y, index_representation





### UNCOMMENT TO RUN ON CPUs ----------------------------------------------------------------------
# from tensorflow.config import set_visible_devices
# from tensorflow.config.threading import set_intra_op_parallelism_threads, set_inter_op_parallelism_threads

# set_visible_devices([], 'GPU')
# N_jobs = 16
# set_intra_op_parallelism_threads(N_jobs)
# set_inter_op_parallelism_threads(N_jobs)
### -----------------------------------------------------------------------------------------------





### SOILS FORMAT ----------------------------------------------------------------------------------
### 0 -> 20 m
# soil_vocab = ['_START',
                            
#               'WT',
#               '1',
#               '2',
#               '3',
#               '4',
#               '5',
#               '6',
#               '7',
#               '8',
#               '9',
#               '10',
#               '11',
#               '12',
#               '13',
#               '14',
#               '15',
#               '16',
#               '17',
#               '18',
#               '19',
              
#               '_',

#               'clay',
#               'silt',
#               'loam',
#               'sand',
              
#               '_END']



### 0 -> 10 m
soil_vocab = ['_START',
              
              '0.0',              
              '1.0',
              '1.5',
              '2.0',
              '2.5',
              '3.0',
              '3.5',
              '4.0',
              '4.5',
              '5.0',
              '5.5',
              '6.0',
              '6.5',
              '7.0',
              '7.5',
              '8.0',
              '8.5',
              '9.0',
              '9.5',
              '10.0',
              '10.5',
              '11.0',
              '11.5',
              '12.0',
              '12.5',
              '13.0',
              '13.5',
              '14.0',
              '15.0',

              'WT',

              'clay',
              'silt',
              'loam',
              'sand',
              
              '_END']



SOIL_VOCAB_SIZE = len(soil_vocab)

soil_to_index, index_to_soil = index_representation(soil_vocab)
### -----------------------------------------------------------------------------------------------





### LOAD DATA ----------------------------------------------------------------------------
data_folder_train = f'{path_input}training_data4/'
N_samples = 400000 # Number of samples to load

# ---
# _, X = load_X(data_folder_train, N_samples)
# y_index, y_oneHot = load_y(data_folder_train, N_samples, soil_to_index)

# np.save(f'{data_folder_train}X_index.npy', X)
# np.save(f'{data_folder_train}y_index.npy', y_index)
# np.save(f'{data_folder_train}y_oneHot.npy', y_oneHot)


X = np.load(f'{data_folder_train}X_index.npy')
y_index = np.load(f'{data_folder_train}y_index.npy')
y_oneHot = np.load(f'{data_folder_train}y_oneHot.npy')



print(X[0,...])
print(y_index[0,...])
# ---


print(f'\n{X.shape = }')
print(f'{y_index.shape = }')
print(f'{y_oneHot.shape = }\n')


# # Test model with only one sample
# shape = X.shape
# first_sample = X[0, ...].reshape(1, X.shape[1])
# X = np.full(shape, first_sample)

# shape_y = y_index.shape
# first_sample_y = y_index[0, ...].reshape(1, y_index.shape[1])
# y_index = np.full(shape_y, first_sample_y)

# shape_y = y_oneHot.shape
# first_sample_y = y_oneHot[0, ...].reshape(1, y_oneHot.shape[1], y_oneHot.shape[2])
# y_oneHot = np.full(shape_y, first_sample_y)



# Split into training, validation, and testing dataset
train_ratio = 0.75
train_end_idx = int(train_ratio*X.shape[0])

X_train = X[:train_end_idx, ...]
y_train_index = y_index[:train_end_idx, ...]
y_train_oneHot = y_oneHot[:train_end_idx, ...]

N_samples_train = X_train.shape[0]



val_ratio = (1 - train_ratio) / 2
val_start_idx = train_end_idx
val_end_idx = val_start_idx + int(val_ratio*X.shape[0])

X_val = X[val_start_idx:val_end_idx, ...]
y_val_index = y_index[val_start_idx:val_end_idx, ...]
y_val_oneHot = y_oneHot[val_start_idx:val_end_idx, ...]

N_samples_val = X_val.shape[0]



test_start_idx = val_end_idx

X_test = X[test_start_idx:, ...]
y_test_index = y_index[test_start_idx:, ...]
y_test_oneHot = y_oneHot[test_start_idx:, ...]

N_samples_test = X_test.shape[0]



del X, y_index, y_oneHot, N_samples, data_folder_train



print(f'\n{X_train.shape = }')
print(f'{y_train_index.shape = }')
print(f'{y_train_oneHot.shape = }\n')

print(f'\n{X_test.shape = }')
print(f'{y_test_index.shape = }')
print(f'{y_test_oneHot.shape = }\n')

print(f'\n{X_val.shape = }')
print(f'{y_val_index.shape = }')
print(f'{y_val_oneHot.shape = }\n')
### -----------------------------------------------------------------------------------------------





### MODEL -----------------------------------------------------------------------------------------
# Parameters
LEN_INPUT_SEQUENCE = X_train.shape[1] # Number of fequency samples : 31 for 20m, 21 for 10m
LEN_OUTPUT_SEQUENCE = 11 # Number of layers + WT + _END token

# 2 4 8 16 32 64 128 256 512 1024 2048 4096
ENC_EMBED_DIM = 128 # 512 for 20m, 512 for 10m
DEC_EMBED_DIM = 20 # 16

INTERMEDIATE_DIM = 256 # 2048 for 20m, 4096 for 10m
NUM_HEADS = 16 # 8 for 20m, 8 for 10m

# SOIL_VOCAB_SIZE = 27 for 20m, 25 for 10m
DISP_VOCAB_SIZE = 40000 # 40000 for 20m, 40000 for 10m

N_LAYERS = 1


# Encoder
encoder_inputs = Input(shape=(None, ))

x = TokenAndPositionEmbedding(
    vocabulary_size=DISP_VOCAB_SIZE,
    sequence_length=LEN_INPUT_SEQUENCE,
    embedding_dim=ENC_EMBED_DIM,
)(encoder_inputs)

for _ in range(N_LAYERS):
  x = TransformerEncoder(intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS)(inputs=x)
  
encoder_outputs = x

encoder = Model(encoder_inputs, encoder_outputs)


# Decoder
decoder_inputs = Input(shape=(None, ))

encoded_seq_inputs = Input(shape=(None, ENC_EMBED_DIM))

x = TokenAndPositionEmbedding(
    vocabulary_size=SOIL_VOCAB_SIZE,
    sequence_length=LEN_OUTPUT_SEQUENCE,
    embedding_dim=DEC_EMBED_DIM,
)(decoder_inputs)

for _ in range(N_LAYERS):
  x = TransformerDecoder(intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS)(decoder_sequence=x, encoder_sequence=encoded_seq_inputs)

x = Dropout(0.1)(x)

decoder_outputs = Dense(SOIL_VOCAB_SIZE, activation="softmax")(x)



decoder = Model([decoder_inputs, encoded_seq_inputs], decoder_outputs)

decoder_outputs = decoder([decoder_inputs, encoder_outputs])


model = Model(
    [encoder_inputs, decoder_inputs],
    decoder_outputs,
)


# Print summary
model.summary()


# Compile model
model.compile(optimizer=RMSprop(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])


# Plot model
model_name = f'{path_models}/[{datetime.now().strftime("%Y%m%d%H%M")}]_transformer_inputEmbedding_{N_LAYERS}layers'
# plot_model(model, to_file=f'{model_name}_plot.png', show_shapes=True, show_layer_names=True)
### -----------------------------------------------------------------------------------------------





# TRAINING ----------------------------------------------------------------------------------------
history = model.fit([X_train, y_train_index[:, :-1]], y_train_oneHot[:, 1:],
                          epochs=50,
                          batch_size=32,
                          shuffle=True,
                          validation_data=([X_val, y_val_index[:, :-1]], y_val_oneHot[:, 1:]),
                          # validation_split=0.2,
                          # callbacks=[EarlyStopping(monitor='val_loss', patience=10)],
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
fig.savefig(f"{model_name}_history.png", format='png', dpi='figure', bbox_inches='tight')
### -----------------------------------------------------------------------------------------------





### SAVE ------------------------------------------------------------------------------------------
model.save(f'{model_name}_model.keras')
### -----------------------------------------------------------------------------------------------





### INFERENCE -------------------------------------------------------------------------------------
for i_sample in range(N_samples_test):
  input_seq = X_test[i_sample, ...].reshape(1, X_test.shape[1])
  target_seq = y_test_index[i_sample, 1:]

  decoded_seq = [soil_to_index['_START']]

  i = 0
  sampled_token_index = soil_to_index['_START']
  
  while sampled_token_index != soil_to_index['_END']:
    predictions = model.predict([input_seq, np.array(decoded_seq).reshape(1, len(decoded_seq))], verbose=0)

    sampled_token_index = np.argmax(predictions[0, i, :])
    decoded_seq.append(sampled_token_index)

    i += 1

  decoded_seq = decoded_seq[1:]



  m = Accuracy()
  accuracy = m(target_seq[1:-1], decoded_seq[1:-1]).numpy()


  print('\n\n----------------------------------------------')
  print("Decoded Sequence:\n", decoded_seq)

  print("\nTarget Sequence:\n", target_seq)

  print("\nTARGET | DECODED - O/X:")
  for i in range(0, len(target_seq)):
    if (target_seq[i] == decoded_seq[i]):
      symbol = "O"
    else:
      symbol = "X"
    print(f'{i+1}: {index_to_soil[target_seq[i]]} | {index_to_soil[decoded_seq[i]]} - {symbol}')

  print(f"\nAccuracy: {accuracy*100} %")
  print('----------------------------------------------\n\n')


  input("Press Enter to continue...")
### -----------------------------------------------------------------------------------------------