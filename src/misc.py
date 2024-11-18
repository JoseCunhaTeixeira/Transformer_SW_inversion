"""
Author : José CUNHA TEIXEIRA
License : SNCF Réseau, UMR 7619 METIS, Sorbonne Université
Date : April 30, 2024
"""





import random
from scipy.interpolate import interp1d
import numpy as np
from json import load
import warnings





### -----------------------------------------------------------------------------------------------
def generate_numbers(N, min, max, step):

    numbers = random.choices(np.arange(min, max+step, step), k=N-1)
    
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
        axis_resamp = np.arange(min(f), max(f), 0.1)


    if "wavelength" in type :
        w = v / f
        func_v = interp1d(w, v, fill_value='extrapolate')
        if type == "wavelength":
             w_resamp = axis_resamp
        elif type == "wavelength-log":
            w_resamp = np.geomspace(min(w), max(w), len(f))

        v_resamp = func_v(w_resamp)
        f_resamp = v_resamp/w_resamp

        return w_resamp, v_resamp


    elif "frequency" in type :
        func_v = interp1d(f, v, fill_value='extrapolate')
        if type == "frequency":
            f_resamp = axis_resamp
        elif type == "frequency-log":
            f_resamp = np.geomspace(min(axis_resamp), max(axis_resamp), len(axis_resamp))

        v_resamp = func_v(f_resamp)
        
        return f_resamp, v_resamp
### -----------------------------------------------------------------------------------------------





### -----------------------------------------------------------------------------------------------
def load_X(data_folder, min_freq, max_freq, min_vel, max_vel, noise, N_samples):
  with open(f'{data_folder}/params.json', 'r') as f:
    data_params = load(f)
    fs = np.array(data_params['freqs'])
  X = np.loadtxt(f'{data_folder}DCs.txt')

  if N_samples != None:
    if N_samples < X.shape[0]:
      X = X[:N_samples]
      print(f'\033[92mINFO\033[0m: The number of samples in the dataset is reduced to {X.shape[0]}.')
    else:
      print(f'\033[93mWARNING\033[0m: {N_samples = } is greater than the number of samples in the dataset. The number of samples in the dataset will remain {X.shape[0]}.')

  i_min = np.where(fs == min_freq)[0][0]
  i_max = np.where(fs == max_freq)[0][0]
  fs = fs[i_min:i_max+1]
  X = X[:, i_min:i_max+1]

  print(f'\033[92mINFO\033[0m: {fs = }')
  print(f'\033[92mINFO\033[0m: {np.min(X) = } and {np.max(X) = }\n')

  if min_vel == None or max_vel == None:
    min_vel = np.min(X)
    max_vel = np.max(X)
  X = (X-min_vel) / (max_vel-min_vel)
  
  X = X.reshape(X.shape[0], X.shape[1], 1)

  if noise == True:
    noise = np.random.normal(loc=0, scale=0.01, size=X.shape)
    X += noise

  return X, min_vel, max_vel
### -----------------------------------------------------------------------------------------------





### -----------------------------------------------------------------------------------------------
def load_y(data_folder, soil_to_index, N_layers, N_samples):
  GMs = np.loadtxt(f'{data_folder}/GMs.txt', dtype=str)
  THKs = np.loadtxt(f'{data_folder}/THKs.txt', dtype=str)
  WTs = np.loadtxt(f'{data_folder}/WTs.txt', dtype=str)
  COORDs = np.loadtxt(f'{data_folder}/Ns.txt', dtype=str)

  if N_samples != None:
    if N_samples < GMs.shape[0]:
      GMs = GMs[:N_samples]
      THKs = THKs[:N_samples]
      WTs = WTs[:N_samples]
      COORDs = COORDs[:N_samples]
      print(f'\033[92mINFO\033[0m: The number of samples in the dataset is reduced to {GMs.shape[0]}.')
    else:
      print(f'\033[93mWARNING\033[0m: {N_samples = } is greater than the number of samples in the dataset. The number of samples in the dataset will remain {GMs.shape[0]}.')


  y_index = []
  for soils, thicknesses, WT, Ns in zip(GMs, THKs, WTs, COORDs):
    sequence_idx = [soil_to_index['[START]']]
    sequence_idx.append(soil_to_index['[WT]'])
    sequence_idx.append(soil_to_index[f'{float(WT):.2f}'])
    cpt_layers = 0
    for soil, thickness, N in zip(soils, thicknesses, Ns):
      if soil != 'None' and thickness != 'None':
        sequence_idx.append(soil_to_index[f'[SOIL{cpt_layers+1}]'])
        sequence_idx.append(soil_to_index[soil])
        sequence_idx.append(soil_to_index[f'[THICKNESS{cpt_layers+1}]'])
        sequence_idx.append(soil_to_index[f'{float(thickness):.2f}'])
        sequence_idx.append(soil_to_index[f'[N{cpt_layers+1}]'])
        sequence_idx.append(soil_to_index[f'{float(N):.2f}'])
        cpt_layers += 1
      else :
        break
    sequence_idx.append(soil_to_index['[END]'])
    while cpt_layers < N_layers:
      sequence_idx.append(soil_to_index[f'[PAD]'])
      sequence_idx.append(soil_to_index[f'[PAD]'])
      sequence_idx.append(soil_to_index[f'[PAD]'])
      sequence_idx.append(soil_to_index[f'[PAD]'])
      sequence_idx.append(soil_to_index[f'[PAD]'])
      sequence_idx.append(soil_to_index[f'[PAD]'])
      cpt_layers += 1
    sequence_idx.append(soil_to_index[f'[PAD]'])
    y_index.append(sequence_idx)
  return np.array(y_index)
### -----------------------------------------------------------------------------------------------





### -----------------------------------------------------------------------------------------------
def load_data(data_folder, soil_to_index, N_layers, min_freq, max_freq, min_vel=None, max_vel=None, N_samples=None, noise=False):

  print(f'\n\033[92mINFO\033[0m: Loading data from {data_folder}\033[0m')

  X, min_vel, max_vel = load_X(data_folder, min_freq, max_freq, min_vel=min_vel, max_vel=max_vel, noise=noise, N_samples=N_samples)
  y = load_y(data_folder, soil_to_index, N_layers, N_samples=N_samples)

  # _, unique_indices = np.unique(X, axis=0, return_index=True)
  # unique_indices = np.sort(unique_indices)
  # X = X[unique_indices]
  # y = y[unique_indices]

  if X.shape[0] != y.shape[0]:
    raise ValueError(f'\033[91mX and y have different number of samples: {X.shape[0]} and {y.shape[0]}.\033[0m')
  
  N_samples = X.shape[0]

  return X, min_vel, max_vel, y, N_samples
### -----------------------------------------------------------------------------------------------





### -----------------------------------------------------------------------------------------------
def mode_filter_count(values):
    counts = np.bincount(values.astype(int))
    if np.all(counts <= 1):
        return int(values[(len(values)//2)])
    else:
        return counts.argmax()
### -----------------------------------------------------------------------------------------------





### -----------------------------------------------------------------------------------------------
def mode_filter_mean(values):
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    mode = np.nanmean(values)
  return mode
### -----------------------------------------------------------------------------------------------





### -----------------------------------------------------------------------------------------------
def make_vocab(data_params):
    vocab = [
      '[PAD]',
      '[START]',
      '[END]',

      '[WT]',
    ]

    for layer in range(0, data_params['max_N_layers']):
        vocab.append(f'[SOIL{layer+1}]')
    for layer in range(0, data_params['max_N_layers']):
        vocab.append(f'[THICKNESS{layer+1}]')
    for layer in range(0, data_params['max_N_layers']):
        vocab.append(f'[N{layer+1}]')

    WTs = data_params['WTs']
    Ns = data_params['Ns']
    thicknesses = data_params['thicknesses']

    numbers = np.concatenate([WTs, Ns, thicknesses])
    numbers = np.unique(numbers)
    numbers = np.sort(numbers)

    for number in numbers:
        vocab.append(f'{number:.2f}')

    for soil in data_params['soils']:
        vocab.append(soil)

    return vocab
### -----------------------------------------------------------------------------------------------




### -----------------------------------------------------------------------------------------------
def make_index_representation(vocab):
  word_to_index = {}
  for i, word in enumerate(vocab):
      word_to_index[word] = i
  index_to_word = {index : word for word, index in word_to_index.items()}
  return word_to_index, index_to_word
### -----------------------------------------------------------------------------------------------





### -----------------------------------------------------------------------------------------------
def make_allowed_tokens(data_params, word_to_index):
  WTs = data_params['WTs']
  Ns = data_params['Ns']
  thicknesses = data_params['thicknesses']
  soils = data_params['soils']

  allowed_tokens = [
    [word_to_index['[WT]']],
    [word_to_index[f'{i:.2f}'] for i in WTs],
  ]

  for layer in range(0, data_params['max_N_layers']):
    if layer == 0:
      allowed_tokens.append([word_to_index[f'[SOIL{layer+1}]']])
    else:
      allowed_tokens.append([word_to_index[f'[SOIL{layer+1}]'], word_to_index['[END]']])
    allowed_tokens.append([word_to_index[f'{soil}'] for soil in soils])

    allowed_tokens.append([word_to_index[f'[THICKNESS{layer+1}]']])
    allowed_tokens.append([word_to_index[f'{i:.2f}'] for i in thicknesses])

    allowed_tokens.append([word_to_index[f'[N{layer+1}]']])
    allowed_tokens.append([word_to_index[f'{i:.2f}'] for i in Ns])

  allowed_tokens.append([word_to_index['[END]']])

  return allowed_tokens
### -----------------------------------------------------------------------------------------------





### -----------------------------------------------------------------------------------------------
def make_forbidden_tokens(data_params, word_to_index, length):
  allowed_tokens = make_allowed_tokens(data_params, word_to_index)
  tokens = list(word_to_index.values())
  forbidden_tokens = []
  for i in range(0, length-1):
    forbidden_tokens.append(tokens.copy())
  for i in range(0, length-1):
    for j in range(len(allowed_tokens[i])):
      forbidden_tokens[i].remove(allowed_tokens[i][j])
  return forbidden_tokens
### -----------------------------------------------------------------------------------------------