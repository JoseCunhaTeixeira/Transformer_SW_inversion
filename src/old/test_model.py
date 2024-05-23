"""
Author : José CUNHA TEIXEIRA
License : SNCF Réseau, UMR 7619 METIS
Date : April 30, 2024
"""



from keras.models import model_from_json
from keras.metrics import Accuracy

import numpy as np
from tqdm import tqdm

from misc import predict_sequence, load_data
from folders import path_input, path_models





### SOILS FORMAT ----------------------------------------------------------------------------------
one_hot_soils = {
                '_START': None,
                

                'clay_N8_frac0.3':   None,
                'silt_N8_frac0.3':   None,
                'loam_N8_frac0.3':   None,
                'sand_N8_frac0.3':   None,


                'WT1':            None,
                'WT2':            None,
                'WT3':            None,
                'WT4':            None,
                'WT5':            None,
                'WT6':            None,
                'WT7':            None,
                'WT8':            None,
                'WT9':            None,
                'WT10':           None,
                'WT11':           None,
                'WT12':           None,
                'WT13':           None,
                'WT14':           None,
                'WT15':           None,
                'WT16':           None,
                'WT17':           None,
                'WT18':           None,
                'WT19':           None,


                '_END':           None
                }


possible_soils = list(one_hot_soils.keys())
num_soil_types = len(possible_soils)
print(f'\n{num_soil_types = }')

for i, key in enumerate(possible_soils):
    one_hot_soils[key] = list(np.zeros(num_soil_types, dtype=int))
    one_hot_soils[key][i] = 1
### -----------------------------------------------------------------------------------------------





### LOAD TESTING DATA -----------------------------------------------------------------------------
data_folder_test = f'{path_input}training_data2/test/'
N_samples_test = 200000 # Number of samples to load

X_test, y_test = load_data(data_folder_test, N_samples_test, one_hot_soils)


# Split into validation and testing dataset
val_ratio = 0.25
idx = int(val_ratio*X_test.shape[0])

X_test = X_test[idx:, ...]
y_test = y_test[idx:, ...]

N_samples_test = len(X_test)


print(f'\n{X_test.shape = }')
print(f'{y_test.shape = }\n')



# # Test model with only one sample
# shape = X_test.shape
# first_sample = X_test[0, ...].reshape(1, X_test.shape[1], X_test.shape[2])
# X_test = np.full(shape, first_sample)

# shape_y = y_test.shape
# first_sample_y = y_test[0, ...].reshape(1, y_test.shape[1], y_test.shape[2])
# y_test = np.full(shape_y, first_sample_y)
### -----------------------------------------------------------------------------------------------






### LOAD INFERENCE MODEL --------------------------------------------------------------------------
model_name = f'{path_models}/[202405131427]_seq2seq'

def load_model(model_filename, model_weights_filename):
    with open(model_filename, 'r', encoding='utf8') as f:
        model = model_from_json(f.read())
    model.load_weights(model_weights_filename)
    return model

encoder_model = load_model(f'{model_name}_encoder.model.json', f'{model_name}_encoder.weights.h5')
decoder_model = load_model(f'{model_name}_decoder.model.json', f'{model_name}_decoder.weights.h5')
### -----------------------------------------------------------------------------------------------





### INFERENCE -------------------------------------------------------------------------------------
# print('\nComputing accuracy on testing data')
# accuracy_mean = 0
# for i_sample in tqdm(range(N_samples_test), leave=True):
#     input_seq_oneHot = X_test[i_sample, ...].reshape(1, X_test.shape[1], X_test.shape[2])
#     target_seq_oneHot = y_test[i_sample, 1:]

#     target_seq = []
#     for i in range(0, len(target_seq_oneHot)):
#         target_seq.append(np.argmax(target_seq_oneHot[i, :]))

#     decoded_seq = predict_sequence(encoder_model, decoder_model, input_seq_oneHot, num_soil_types, possible_soils)

#     m = Accuracy()
#     accuracy = m(target_seq, decoded_seq).numpy()

#     accuracy_mean += accuracy
# accuracy_mean /= N_samples_test
# print(f'Accuracy on testing data : {accuracy_mean*100} %')



for i_sample in range(N_samples_test):
    input_seq_oneHot = X_test[i_sample, ...].reshape(1, X_test.shape[1], X_test.shape[2])
    target_seq_oneHot = y_test[i_sample, 1:]

    target_seq = []
    for i in range(0, len(target_seq_oneHot)):
        target_seq.append(np.argmax(target_seq_oneHot[i, :]))

    decoded_seq = predict_sequence(encoder_model, decoder_model, input_seq_oneHot, num_soil_types, possible_soils)

    m = Accuracy()
    accuracy = m(target_seq, decoded_seq).numpy()


    print('\n\n----------------------------------------------')
    print("Decoded Sequence:\n", decoded_seq)

    print("\nTarget Sequence:\n", target_seq)

    print("\nTARGET | DECODED - O/X:")
    for i in range(0, len(target_seq_oneHot)):
        if (target_seq[i] == decoded_seq[i]):
            symbol = "O"
        else:
            symbol = "X"
        print(f'{possible_soils[target_seq[i]]} | {possible_soils[decoded_seq[i]]} - {symbol}')

    print(f"\nAccuracy: {accuracy*100} %")
    print('----------------------------------------------\n\n')


    input("Press Enter to continue...")
### -----------------------------------------------------------------------------------------------