"""
Author : José CUNHA TEIXEIRA
License : SNCF Réseau, UMR 7619 METIS
Date : April 30, 2024
"""




from keras.models import model_from_json

import matplotlib.pyplot as plt
import numpy as np
import sys
from io import StringIO
from subprocess import run, PIPE
from scipy.signal import savgol_filter

from misc import predict_sequence, resamp
from folders import path_input, path_models

# Import Santiludo functions
sys.path.append('/home/jteixeira/Documents/PhD_Monitoring/Work/Processing/Tools/Santiludo_layered/')
from lib.VGfunctions import vanGen
from lib.RPfunctions import hillsAverage, effFluid, hertzMindlin, biotGassmann
from lib.TTDSPfunctions import writeVelocityModel, readDispersion





### UNCOMMENT TO RUN ON CPUs ----------------------------------------------------------------------
# from tensorflow.config import set_visible_devices
# from tensorflow.config.threading import set_intra_op_parallelism_threads, set_inter_op_parallelism_threads

# set_visible_devices([], 'GPU')
# N_jobs = 16
# set_intra_op_parallelism_threads(N_jobs)
# set_inter_op_parallelism_threads(N_jobs)
### -----------------------------------------------------------------------------------------------





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





### LOAD REAL DATA --------------------------------------------------------------------------------
db = np.loadtxt(f'{path_input}/real_data/108.7500.M0.pvc')
fs_obs_raw, vs_obs_raw = db[:,0], db[:,1]


# if (len(vs_obs_raw)/2) % 2 == 0:
#     wl = len(vs_obs_raw)/2 + 1
# else:
#     wl = len(vs_obs_raw)/2
# vs_obs_raw = savgol_filter(vs_obs_raw, window_length=wl, polyorder=3, mode="nearest")


axis_resamp = np.arange(20, 50+1, 1)
fs_obs, vs_obs = resamp(fs_obs_raw, vs_obs_raw, axis_resamp=axis_resamp, type='frequency')

# axis_resamp = np.arange(4.5, 17.5+0.5, 0.5)
# fs_obs, vs_obs = resamp(fs_obs_raw, vs_obs_raw, axis_resamp=axis_resamp, type='wavelength')

vs_obs = (vs_obs-150) / (400-150)

print(f'\n{vs_obs.shape}')
### -----------------------------------------------------------------------------------------------





### LOAD INFERENCE MODEL --------------------------------------------------------------------------
model_name = f'{path_models}/[202405131012]_seq2seq'

def load_model(model_filename, model_weights_filename):
    with open(model_filename, 'r', encoding='utf8') as f:
        model = model_from_json(f.read())
    model.load_weights(model_weights_filename)
    return model

encoder_model = load_model(f'{model_name}_encoder.model.json', f'{model_name}_encoder.weights.h5')
decoder_model = load_model(f'{model_name}_decoder.model.json', f'{model_name}_decoder.weights.h5')
### -----------------------------------------------------------------------------------------------





### INFERENCE -------------------------------------------------------------------------------------
input_seq = vs_obs.reshape(1, vs_obs.shape[0], 1)
decoded_seq = predict_sequence(encoder_model, decoder_model, input_seq, num_soil_types, possible_soils)

decoded_GM = []

print('\n\n----------------------------------------------')
print("Decoded Sequence:\n", decoded_seq, '\n')
for i in range(0, len(decoded_seq)):
    decoded_GM.append(possible_soils[decoded_seq[i]])
    print(f'{i+1} : {possible_soils[decoded_seq[i]]}')
print('----------------------------------------------\n\n')

decoded_GM = decoded_GM[:-1]
soil_types = []
fracs = []
Ns = []
WT = None
for line in decoded_GM :
    if line == '_END':
        break
    elif line.startswith('WT'):
        WT = int(line[2:])
    else :
        soil_type, N, frac = line.split('_')
        soil_types.append(soil_type)
        fracs.append(float(frac[4:]))
        Ns.append(int(N[1:]))

depth = len(soil_types) * 2
GM_thicknesses = [2]*len(soil_types)

if np.sum(GM_thicknesses) != depth:
    print('Error: Sum of layer thicknesses is not equal to the total depth.')
    sys.exit()

print(f'\n{soil_types = }')
print(f'{Ns = }')
print(f'{fracs = }')
print(f'{WT = }')
print(f'{GM_thicknesses = }')
print(f'{depth = }')
### ------------------------------------------------------------------------------------------------





### ROCK PHYSICS CONSTANTS ------------------------------------------------------------------------
rhow = 1000.0 # Water density [Kg/m3]
rhoa = 1.0 # Air density [Kg/m3]
kw = 2.3e9 # Water bulk modulus [Pa]
ka = 1.01e5 # Air bulk modulus [Pa]
g = 9.82 # Gravity acceleration [m/s2]

# Grains/agregate mechanical properties
mu_clay = 6.8 # Shear moduli [GPa]
mu_silt = 45.0
mu_sand = 45.0
k_clay = 25.0 # Bulk moduli [GPa]
k_silt = 37.0
k_sand = 37.0
rho_clay = 2580.0 # Density [kg/m3]
rho_silt = 2600.0
rho_sand = 2600.0

# Geometry and discretisation of the medium
dz = 0.1 # Depth sample interval [m]
top_surface_level = dz # Altitude of the soil surface[m]
zs = -np.arange(top_surface_level, depth + dz, dz) # Depth positions (negative downward) [m]

NbCells = len(zs) - 1 # Number of exploration points in depth [#]

# Three possible RP models:
kk = 3 # Pe with suction (cf. Solazzi et al. 2021)
### -----------------------------------------------------------------------------------------------





### SEISMIC CONSTANTS -----------------------------------------------------------------------------
# Layers to put under the studied soil column on the velocity model
# In GPDC format : "thickness Vp Vs rho\n"
# Each layer is separated by \n | Only spaces between values | Last layer thickness must be 0)
# under_layers = "" # Empty string if no under layers
under_layers = "0 4000 2000 2500\n" # One substratum layer
n_under_layers = under_layers.count('\n') # Number of under layers

VM_thicknesses = np.diff(np.abs(zs)) # thickness vector [m]
# Save VM_thicknesses
np.savetxt(f'{path_input}/training_data2/test/VM_thicknesses.txt', VM_thicknesses, fmt='%f')

x0 = 1 # first geophone position [m]
Nx = 192 # number of geophones [m]
dx = 1 # geophone interval [m]
xs = np.arange(x0, Nx * dx + 1, dx)
trig  = 0 # data pretrig (if needed)


# Frequency domain and sampling setup to compute dispersion
nf = 46 # number of frequency samples [#]
df = 1 # frequency sample interval [Hz]
min_f = 5 # minimum frequency [Hz]
max_f = min_f + (nf - 1) * df

n_modes = 1 # Number of modes to compute
s = 'frequency' # Over frequencies mode
wave = 'R' # Rayleigh (PSV) fundamental mode
### -----------------------------------------------------------------------------------------------





#### ROCK PHYSICS ---------------------------------------------------------------------------------
# Saturation profile with depth
hs, Sws, Swes = vanGen(zs, WT, soil_types, GM_thicknesses)


# Effective Grain Properties (constant with depth)
mus, ks, rhos, nus = hillsAverage(mu_clay, mu_silt, mu_sand, rho_clay,
                                    rho_silt, rho_sand, k_clay, k_silt,
                                    k_sand, soil_types)

# Effective Fluid Properties
kfs, rhofs, rhobs = effFluid(Sws, kw, ka, rhow,
                                rhoa, rhos, soil_types, GM_thicknesses, dz)

# Hertz Mindlin Frame Properties
KHMs, muHMs = hertzMindlin(Swes, zs, hs, rhobs,
                            g, rhoa, rhow, Ns,
                            mus, nus, fracs, kk,
                            soil_types, GM_thicknesses)

# Saturated Properties
VPs, VSs = biotGassmann(KHMs, muHMs, ks, kfs,
                        rhobs, soil_types, GM_thicknesses, dz)
### -----------------------------------------------------------------------------------------------



#### SEISMIC FWD MODELING -------------------------------------------------------------------------
# Velocity model in string format for GPDC
velocity_model_string = writeVelocityModel(VM_thicknesses, VPs, VSs, rhobs, under_layers, n_under_layers)

# Dispersion curves computing with GPDC
velocity_model_RAMfile = StringIO(velocity_model_string) # Keep velocity model string in the RAM in a file format alike to trick GPDC which expects a file
gpdc_command = [f"gpdc -{wave} {n_modes} -n {nf} -min {min_f} -max {max_f} -s {s} -j 16"]
gpdc_output_string = run(gpdc_command, input=velocity_model_RAMfile.getvalue(), text=True, shell=True, stdout=PIPE).stdout # Raw output string from GPDC

dispersion_data, n_modes = readDispersion(gpdc_output_string) # Reads GPDC output and converts dispersion data to a list of numpy arrays for each mode
                                                            # Updates number of computed modes (can be lower than what was defined if frequency range too small)
### -----------------------------------------------------------------------------------------------




### PLOT ------------------------------------------------------------------------------------------
fig1, ax1 = plt.subplots()
ax1.plot(VSs[:-1], zs[:-1], linewidth=1.5)
ax1.axhline(-WT, linestyle='--', linewidth=0.5)
ax1.set_xlabel('$V_s$ [m/s]')
ax1.set_ylabel('$z$ [m]')

fig2, ax2 = plt.subplots()
ax2.plot(fs_obs_raw, vs_obs_raw)
ax2.plot(dispersion_data[0][:,0], dispersion_data[0][:,1])
ax2.set_xlabel('f [Hz]')
ax2.set_ylabel('$V_s$ [m/s]')

fig3, ax3 = plt.subplots()
ax3.plot(Sws, zs, linewidth=1.5)
ax3.axhline(-WT, linestyle='--', linewidth=0.5)
ax3.set_xlim(0, 1.1)
ax3.set_xlabel('$S_w$')
ax3.set_ylabel('$z$ [m]')

plt.show()
### -----------------------------------------------------------------------------------------------