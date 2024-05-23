"""
Author : José CUNHA TEIXEIRA
License : SNCF Réseau, UMR 7619 METIS
Date : April 30, 2024
"""





import random
from scipy.interpolate import interp1d
from numpy import geomspace, zeros, argmax, arange, min, max, arange





### -----------------------------------------------------------------------------------------------
def generate_numbers(N, min, max, step):

    numbers = random.choices(arange(min, max+step, step), k=N-1)
    
    # Calculate the fourth number so that the sum is exactly max
    total = sum(numbers)
    last_number = max - total
    
    # If the fourth number is out of range (min to max), start over
    if last_number <= 0 or last_number > max:
        return generate_numbers(N, min, max, step)
    
    numbers.append(last_number)
    random.shuffle(numbers)  # Shuffle to ensure random order
    return numbers
### -----------------------------------------------------------------------------------------------





### -----------------------------------------------------------------------------------------------
def resamp(f, v, axis_resamp=None, type="wavelength"):

    if axis_resamp is None :
        axis_resamp = arange(min(f), max(f), 0.1)


    if "wavelength" in type :
        w = v / f
        func_v = interp1d(w, v, fill_value='extrapolate')
        if type == "wavelength":
             w_resamp = axis_resamp
        elif type == "wavelength-log":
            w_resamp = geomspace(min(w), max(w), len(f))

        v_resamp = func_v(w_resamp)
        f_resamp = v_resamp/w_resamp

        return w_resamp, v_resamp


    elif "frequency" in type :
        func_v = interp1d(f, v)
        if type == "frequency":
            f_resamp = axis_resamp
        elif type == "frequency-log":
            f_resamp = geomspace(min(axis_resamp), max(axis_resamp), len(axis_resamp))

        v_resamp = func_v(f_resamp)
        
        return f_resamp, v_resamp
### -----------------------------------------------------------------------------------------------





### -----------------------------------------------------------------------------------------------
def predict_sequence_LSTMwithEmbedding(encoder_model, decoder_model, input_seq, soil_to_index):

    # Encode the input as state vectors.
    encoder_output_tokens, h1, c1, h2, c2 = encoder_model.predict(input_seq, verbose=0)

    # Initialize decoder input for inference
    target_seq = zeros((1, 1))

    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = soil_to_index['_START']  # Set the start token

    sampled_token_index = 0

    # Generate predictions recursively using the decoder
    decoded_seq = []
    while sampled_token_index != soil_to_index['_END']:
        # Predict the next token and the updated decoder states
        output_tokens, h1, c1, h2, c2 = decoder_model.predict([target_seq, encoder_output_tokens, h1, c1, h2, c2], verbose=0)

        # Sample a token with maximum probability
        sampled_token_index = argmax(output_tokens[0, -1, :])
        decoded_seq.append(sampled_token_index)

        # Update the target sequence (of length 1).
        target_seq = zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

    return decoded_seq
### -----------------------------------------------------------------------------------------------





### -----------------------------------------------------------------------------------------------
def predict_sequence_LSTMnoEmbedding(encoder_model, decoder_model, input_seq, soil_to_index):

    soil_vocab_size = len(soil_to_index.keys())

    # Encode the input as state vectors.
    encoder_output_tokens, h1, c1, h2, c2 = encoder_model.predict(input_seq, verbose=0)

    # Initialize decoder input for inference
    target_seq = zeros((1, 1, soil_vocab_size))

    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, soil_to_index['_START']] = 1  # Set the start token

    sampled_token_index = 0

    # Generate predictions recursively using the decoder
    decoded_seq = []
    while sampled_token_index != soil_to_index['_END']:
        # Predict the next token and the updated decoder states
        output_tokens, h1, c1, h2, c2 = decoder_model.predict([target_seq, encoder_output_tokens, h1, c1, h2, c2], verbose=0)

        # Sample a token with maximum probability
        sampled_token_index = argmax(output_tokens[0, -1, :])
        decoded_seq.append(sampled_token_index)

        print(decoded_seq)

        # Update the target sequence (of length 1).
        target_seq = zeros((1, 1, soil_vocab_size))
        target_seq[0, 0, sampled_token_index] = 1

    return decoded_seq
### -----------------------------------------------------------------------------------------------