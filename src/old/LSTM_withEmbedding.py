"""
Author : José CUNHA TEIXEIRA
License : SNCF Réseau, UMR 7619 METIS
Date : April 30, 2024
"""



from keras import Model
from keras.layers import Input, LSTM, TimeDistributed, Dense, Attention, Concatenate, Embedding
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
from keras.metrics import Accuracy

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from folders import path_input, path_models
from misc import predict_sequence_LSTMwithEmbedding, load_X, load_y, index_representation





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
# X, _ = load_X(data_folder_train, N_samples)
# y_index, y_oneHot = load_y(data_folder_train, N_samples, soil_to_index)

# np.save(f'{data_folder_train}X_floats.npy', X)
# np.save(f'{data_folder_train}y_index.npy', y_index)
# np.save(f'{data_folder_train}y_oneHot.npy', y_oneHot)


X = np.load(f'{data_folder_train}X_floats.npy')
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




### TRAINING MODEL --------------------------------------------------------------------------------
# Parameters
LEN_INPUT_SEQUENCE = X_train.shape[1]
LEN_OUTPUT_SEQUENCE = 11

DEC_EMBED_DIM = 128

LATENT_DIM = 256 # Latent dimensionality of the encoding space : 2 4 8 16 32 64 128 256 512 1024


# Encoder ---
enc_inputs = Input(shape=(LEN_INPUT_SEQUENCE, 1))

#Layer 1
enc_back_lstm1 = LSTM(LATENT_DIM, return_sequences=True, return_state=True, go_backwards=True, dropout=0, recurrent_dropout=0)
enc_back_lstm1_outputs, enc_back_state_h1, enc_back_state_c1 = enc_back_lstm1(enc_inputs)

enc_frw_lstm1 = LSTM(LATENT_DIM, return_sequences=True, return_state=True, dropout=0, recurrent_dropout=0)
enc_frw_lstm1_outputs, enc_frw_state_h1, enc_frw_state_c1 = enc_frw_lstm1(enc_inputs)

enc_state_h1 = Concatenate()([enc_frw_state_h1, enc_back_state_h1])
enc_state_c1 = Concatenate()([enc_frw_state_c1, enc_back_state_c1])
enc_states1 = [enc_state_h1, enc_state_c1]

# Layer 2
enc_back_lstm2 = LSTM(LATENT_DIM, return_sequences=True, return_state=True, go_backwards=True, dropout=0, recurrent_dropout=0)
enc_back_lstm2_outputs, enc_back_state_h2, enc_back_state_c2 = enc_back_lstm2(enc_back_lstm1_outputs)

enc_frw_lstm2 = LSTM(LATENT_DIM, return_sequences=True, return_state=True, dropout=0, recurrent_dropout=0)
enc_frw_lstm2_outputs, enc_frw_state_h2, enc_frw_state_c2 = enc_frw_lstm2(enc_frw_lstm1_outputs)

enc_state_h2 = Concatenate()([enc_frw_state_h2, enc_back_state_h2])
enc_state_c2 = Concatenate()([enc_frw_state_c2, enc_back_state_c2])
enc_states2 = [enc_state_h2, enc_state_c2]

# Outputs
enc_outputs = Concatenate(axis=-1)([enc_back_lstm2_outputs, enc_frw_lstm2_outputs])
enc_states = [enc_state_h1, enc_state_c1, enc_state_h2, enc_state_c2]


# Decoder ---
dec_inputs = Input(shape=(None,))

# Add an Embedding layer
dec_emb_layer = Embedding(input_dim=SOIL_VOCAB_SIZE, output_dim=DEC_EMBED_DIM, input_length=LEN_OUTPUT_SEQUENCE)  # Define vocabulary_size based on your task
dec_emb_outputs = dec_emb_layer(dec_inputs)

#Layer 1
dec_lstm1 = LSTM(LATENT_DIM*2, return_sequences=True, return_state=True, dropout=0, recurrent_dropout=0)
dec_lstm1_outputs, _, _ = dec_lstm1(dec_emb_outputs, initial_state=enc_states1)

dec_lstm2 = LSTM(LATENT_DIM*2, return_sequences=True, return_state=True, dropout=0, recurrent_dropout=0)
dec_lstm2_outputs, _, _ = dec_lstm2(dec_lstm1_outputs, initial_state=enc_states2)

# Outputs
dec_outputs = dec_lstm2_outputs


# Attention ---
attention_layer = Attention(score_mode="concat")
attention_outputs = attention_layer([dec_outputs, enc_outputs])

concatenate_outputs = Concatenate(axis=-1)([dec_outputs, attention_outputs])


# Output dense layer ---
dense_layer = TimeDistributed(Dense(SOIL_VOCAB_SIZE, activation='softmax'))
outputs = dense_layer(concatenate_outputs)


# Model ---
model = Model([enc_inputs, dec_inputs], outputs)



# Print summary
model.summary()


# Compile model
model.compile(optimizer=RMSprop(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])


# Name model
model_name = f'{path_models}/[{datetime.now().strftime("%Y%m%d%H%M")}]_LSTM_withEmbedding'


# # Plot model
# plot_model(model, to_file=f'{model_name}_plot.png', show_shapes=True, show_layer_names=True)
### -----------------------------------------------------------------------------------------------






# TRAINING ----------------------------------------------------------------------------------------
history = model.fit([X_train, y_train_index[:, :-1]], y_train_oneHot[:, 1:],
                          epochs=100,
                          batch_size=32,
                          shuffle=True,
                          validation_data=([X_val, y_val_index[:, :-1]], y_val_oneHot[:, 1:]),
                          # validation_split=0.2,
                          callbacks=[EarlyStopping(monitor='val_loss', start_from_epoch=5, patience=5, restore_best_weights=True)],
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





### INFERENCE MODEL -------------------------------------------------------------------------------
# Encoder model for prediction
encoder_model = Model(enc_inputs, [enc_outputs] + enc_states)

# Decoder model for prediction
decoder_state_input_h1 = Input(shape=(LATENT_DIM*2,))
decoder_state_input_c1 = Input(shape=(LATENT_DIM*2,))
decoder_state_input_h2 = Input(shape=(LATENT_DIM*2,))
decoder_state_input_c2 = Input(shape=(LATENT_DIM*2,))

decoder_states_inputs = [decoder_state_input_h1, decoder_state_input_c1, decoder_state_input_h2, decoder_state_input_c2]

decoder_emb_outputs = dec_emb_layer(dec_inputs)

decoder_lstm1_outputs, state_h1, state_c1 = dec_lstm1(decoder_emb_outputs, initial_state=decoder_states_inputs[:2])

decoder_lstm2_outputs, state_h2, state_c2 = dec_lstm2(decoder_lstm1_outputs, initial_state=decoder_states_inputs[-2:])

decoder_states = [state_h1, state_c1, state_h2, state_c2]

encoder_outputs_as_input = Input(shape=(None, LATENT_DIM*2))
attention_outputs = attention_layer([decoder_lstm2_outputs, encoder_outputs_as_input])
concatenate_outputs = Concatenate(axis=-1)([decoder_lstm2_outputs, attention_outputs])

decoder_outputs = dense_layer(concatenate_outputs)

decoder_model = Model(
    [dec_inputs] + [encoder_outputs_as_input] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)
### -----------------------------------------------------------------------------------------------





### SAVE ------------------------------------------------------------------------------------------
with open(f'{model_name}_encoder.model.json', 'w', encoding='utf8') as f:
    f.write(encoder_model.to_json())
encoder_model.save_weights(f'{model_name}_encoder.weights.h5')

with open(f'{model_name}_decoder.model.json', 'w', encoding='utf8') as f:
    f.write(decoder_model.to_json())
decoder_model.save_weights(f'{model_name}_decoder.weights.h5')
### -----------------------------------------------------------------------------------------------





### INFERENCE -------------------------------------------------------------------------------------
for i_sample in range(N_samples_test):

    input_seq = X_test[i_sample, ...].reshape(1, X_test.shape[1])
    target_seq = y_test_index[i_sample, 1:]

    decoded_seq = predict_sequence_LSTMwithEmbedding(encoder_model, decoder_model, input_seq, soil_to_index)

    m = Accuracy()
    accuracy = m(target_seq, decoded_seq).numpy()


    print('\n\n----------------------------------------------')
    print("Decoded Sequence:\n", decoded_seq)

    print("\nTarget Sequence:\n", target_seq)

    print("\nTARGET | DECODED - O/X:")
    for i in range(0, len(target_seq)):
        if (target_seq[i] == decoded_seq[i]):
            symbol = "O"
        else:
            symbol = "X"
        print(f'{index_to_soil[target_seq[i]]} | {index_to_soil[decoded_seq[i]]} - {symbol}')

    print(f"\nAccuracy: {accuracy*100} %")
    print('----------------------------------------------\n\n')


    input("Press Enter to continue...")
### -----------------------------------------------------------------------------------------------