"""
Author : José CUNHA TEIXEIRA
License : SNCF Réseau, UMR 7619 METIS
Date : April 30, 2024
"""





import random
from tqdm import tqdm
from scipy.interpolate import interp1d
from numpy import geomspace, arange, min, max, arange, array, loadtxt, random





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
        func_v = interp1d(f, v, fill_value='extrapolate')
        if type == "frequency":
            f_resamp = axis_resamp
        elif type == "frequency-log":
            f_resamp = geomspace(min(axis_resamp), max(axis_resamp), len(axis_resamp))

        v_resamp = func_v(f_resamp)
        
        return f_resamp, v_resamp
### -----------------------------------------------------------------------------------------------





### -----------------------------------------------------------------------------------------------
def load_X(data_folder, N_samples, noise=False):
  X = loadtxt(f'{data_folder}DCs.txt')
  X = X[:N_samples, ...]

  X = X[:, 5:] # Only train with frequencies between 20 and 50 Hz, instead of 15 and 50 Hz

  print(f'\n{min(X) = }\n{max(X) = }')

  X = (X-150) / (400-150)
  X = X.reshape(X.shape[0], X.shape[1], 1)

  if noise == True:
    noise = random.normal(loc=0, scale=0.01, size=X.shape)
    X_noise = X + noise
    return X, X_noise

  return X
### -----------------------------------------------------------------------------------------------





### -----------------------------------------------------------------------------------------------
def load_y(data_folder, N_samples, soil_to_index, N_layers):
  GMs = loadtxt(f'{data_folder}/GMs.txt', dtype=str)
  THKs = loadtxt(f'{data_folder}/THKs.txt', dtype=str)
  WTs = loadtxt(f'{data_folder}/WTs.txt', dtype=str)
  COORDs = loadtxt(f'{data_folder}/Ns.txt', dtype=str)

  GMs = GMs[:N_samples, ...]
  THKs = THKs[:N_samples, ...]
  WTs = WTs[:N_samples]
  COORDs = COORDs[:N_samples, ...]

  y_index = []
  for soils, thicknesses, WT, Ns in tqdm(zip(GMs, THKs, WTs, COORDs), total=N_samples):
    sequence_idx = [soil_to_index['[START]']]
    sequence_idx.append(soil_to_index['[WT]'])
    sequence_idx.append(soil_to_index[WT])
    cpt_layers = 0
    for soil, thickness, N in zip(soils, thicknesses, Ns):
      if soil != 'None' and thickness != 'None':
        sequence_idx.append(soil_to_index[f'[SOIL{cpt_layers+1}]'])
        sequence_idx.append(soil_to_index[soil])
        sequence_idx.append(soil_to_index[f'[THICKNESS{cpt_layers+1}]'])
        sequence_idx.append(soil_to_index[f'{float(thickness):.1f}'])
        sequence_idx.append(soil_to_index[f'[N{cpt_layers+1}]'])
        sequence_idx.append(soil_to_index[f'{float(N):.1f}'])
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
  return array(y_index)
### -----------------------------------------------------------------------------------------------