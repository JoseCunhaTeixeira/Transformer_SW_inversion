"""
Author : José CUNHA TEIXEIRA
License : SNCF Réseau, UMR 7619 METIS
Date : April 30, 2024
"""





import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from io import StringIO
from pickle import load
from subprocess import run, PIPE
from scipy.signal import savgol_filter
from scipy.ndimage import generic_filter

from misc import resamp
from folders import PATH_INPUT, PATH_MODELS, PATH_OUTPUT

# Import Santiludo functions
sys.path.append('/home/jteixeira/Documents/PhD_Monitoring/Work/Processing/Tools/Santiludo_layered/')
from lib.VGfunctions import vanGen
from lib.RPfunctions import hillsAverage, effFluid, hertzMindlin, biotGassmann
from lib.TTDSPfunctions import writeVelocityModel, readDispersion

plt.rcParams.update({'font.size': 12})
cm = 1/2.54





### UNCOMMENT TO RUN ON CPUs ----------------------------------------------------------------------
from tensorflow.config import set_visible_devices
from tensorflow.config.threading import set_intra_op_parallelism_threads, set_inter_op_parallelism_threads

set_visible_devices([], 'GPU')
N_jobs = 16
set_intra_op_parallelism_threads(N_jobs)
set_inter_op_parallelism_threads(N_jobs)
### -----------------------------------------------------------------------------------------------





### LOAD MODEL ------------------------------------------------------------------------------------
# MODEL_NAME = '[]'
MODEL_NAME = sys.argv[1]

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





### FIELD DISPERSION DATA FILES -------------------------------------------------------------------
# PROFILE = 'PJ'
# dx = 1.5
PROFILE = sys.argv[2]
dx = float(sys.argv[3])


files = os.listdir(f'{PATH_INPUT}/real_data/{PROFILE}/')
files = sorted(files, key=lambda x: int(x.split('.')[0]))

xmid_min = 0
xmid_max = xmid_min + dx*(len(files)-1)
xmids = np.arange(xmid_min, xmid_max+dx, dx)

if not os.path.exists(f'{PATH_OUTPUT}/{MODEL_NAME}/{PROFILE}/'):
  os.makedirs(f'{PATH_OUTPUT}/{MODEL_NAME}/{PROFILE}/')
### -----------------------------------------------------------------------------------------------





### INVERSION -------------------------------------------------------------------------------------
GM_db = []
thicks_db = []
Vs_db = []
Vp_db = []
rho_db = []
Sw_db = []
disp_db = []
WT_db = []
depth_db = []
rms_db = []
Ns_db = []

num_plots = len(xmids)
subplot_width = int(np.ceil(np.sqrt(num_plots)))
subplot_height = int(np.ceil(num_plots / subplot_width))

fig, ax = plt.subplots(subplot_height, subplot_width, figsize=(16, 16), dpi=300)
plt_line = 0
plt_col = 0

for file in files:
    print('\n\n----------------------------------------------\n\n')
    print('\033[1;32mXmid: ' + file + '\033[0m')

    db = np.loadtxt(f'{PATH_INPUT}/real_data/{PROFILE}/{file}')
    fs_obs_raw, vs_obs_raw = db[:,0], db[:,1]


    # if (len(vs_obs_raw)/2) % 2 == 0:
    #     wl = len(vs_obs_raw)/2 + 1
    # else:
    #     wl = len(vs_obs_raw)/2
    # vs_obs_raw = savgol_filter(vs_obs_raw, window_length=wl, polyorder=3, mode="nearest")


    axis_resamp = np.arange(min_freq, max_freq+1, 1)
    fs_obs, vs_obs = resamp(fs_obs_raw, vs_obs_raw, axis_resamp=axis_resamp, type='frequency')

    vs_obs_comp = np.copy(vs_obs)

    vs_obs = (vs_obs-150) / (400-150)
    vs_obs = vs_obs.reshape(1, vs_obs.shape[0], 1)
    ### -----------------------------------------------------------------------------------------------





    ### INFERENCE -------------------------------------------------------------------------------------
    input_seq = vs_obs

    decoded_seq = model.decode_seq_restrictive(input_seq)

    decoded_GM = []

    print(f"\nDecoded Sequence: {decoded_seq}\n")
    for i in range(0, len(decoded_seq)):
        decoded_GM.append(index_to_word[decoded_seq[i]])
        print(f'{i+1} : {index_to_word[decoded_seq[i]]}')

    soil_types = decoded_GM[3::6]
    soil_types = [soil for soil in soil_types if soil not in ['[PAD]', '[END]']]

    GM_thicknesses = decoded_GM[5::6]
    GM_thicknesses = [float(thickness) for thickness in GM_thicknesses if thickness not in ['[PAD]', '[END]']]

    Ns = decoded_GM[7::6]
    Ns = [float(N) for N in Ns if N not in ['[PAD]', '[END]']]
    
    WT = float(decoded_GM[1])

    fracs = [0.3] * len(soil_types)

    depth = np.sum(GM_thicknesses)

    print(f'\n{soil_types = }')
    print(f'{GM_thicknesses = }', depth)
    print(f'{Ns = }')
    print(f'{fracs = }')
    print(f'{WT = }')
    print(f'{depth = }')
    if depth != output_sequence_format.vocab.max_depth:
        print(f'\033[1;33mWARNING: Sum of layer thicknesses {depth} is not equal to the expected total depth {output_sequence_format.vocab.max_depth}.\033[0m')


    for soil in soil_types:
        if soil not in output_sequence_format.vocab.words:
            print('Error: Soil type not in soil vocabulary.')
            sys.exit()

    thicks_db.append(GM_thicknesses)
    GM_db.append(soil_types)
    WT_db.append(WT)
    depth_db.append(depth)
    Ns_db.append(Ns)
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
    under_layers = "10 1500 750 2000\n0 4000 2000 2500\n" # One substratum layer
    n_under_layers = under_layers.count('\n') # Number of under layers

    VM_thicknesses = np.diff(np.abs(zs)) # thickness vector [m]

    x0 = 1 # first geophone position [m]
    Nx = 192 # number of geophones [m]
    dx = 1 # geophone interval [m]
    xs = np.arange(x0, Nx * dx + 1, dx)
    trig  = 0 # data pretrig (if needed)


    # Frequency domain and sampling setup to compute dispersion
    df = 1 # number of frequency samples [#]
    min_f = min_freq # minimum frequency [Hz]
    max_f = max_freq
    nf = int((max_f - min_f) / df) + 1 # number of frequency samples [#]

    n_modes = 1 # Number of modes to compute
    s = 'frequency' # Over frequencies mode
    wave = 'R' # Rayleigh (PSV) fundamental mode
    ### -----------------------------------------------------------------------------------------------





    #### ROCK PHYSICS ---------------------------------------------------------------------------------
    # Saturation profile with depth
    hs, Sws, Swes = vanGen(zs, WT, soil_types, GM_thicknesses)
    Sw_db.append(Sws)


    # Effective Grain Properties (constant with depth)
    mus, ks, rhos, nus = hillsAverage(mu_clay, mu_silt, mu_sand, rho_clay,
                                        rho_silt, rho_sand, k_clay, k_silt,
                                        k_sand, soil_types)
    rho_db.append(rhos)

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

    Vp_db.append(VPs)
    Vs_db.append(VSs)
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
    

    disp_db.append(dispersion_data)


    rms = np.sqrt(np.mean((vs_obs_comp - dispersion_data[0][:,1])**2))
    rms_db.append(rms)

    
    ax[plt_line, plt_col].plot(fs_obs_raw, vs_obs_raw, color='black')
    ax[plt_line, plt_col].plot(dispersion_data[0][:,0], dispersion_data[0][:,1], color='red', linestyle='--')
    ax[plt_line, plt_col].set_ylim([100, 500])
    ax[plt_line, plt_col].set_xlim([min_freq, max_freq])
    ax[plt_line, plt_col].set_title(f'{file} | RMS : {rms:.2f}', fontsize=6)


    if plt_line == subplot_height - 1:
        if plt_col == 0:
            ax[plt_line, plt_col].tick_params(top=False, labeltop=False, bottom=True, labelbottom=True, left=True, labelleft=True, right=False, labelright=False)
            ax[plt_line, plt_col].set_xlabel('Frequency\n[Hz]')
            ax[plt_line, plt_col].set_ylabel("${V_R}$ [m/s]")
        else :
            ax[plt_line, plt_col].tick_params(top=False, labeltop=False, bottom=True, labelbottom=True, left=False, labelleft=False, right=False, labelright=False)
            ax[plt_line, plt_col].set_xlabel('Frequency\n[Hz]')
            ax[plt_line, plt_col].set_ylabel('')
    else:
        if plt_col == 0:
            ax[plt_line, plt_col].tick_params(top=False, labeltop=False, bottom=False, labelbottom=False, left=True, labelleft=True, right=False, labelright=False)
            ax[plt_line, plt_col].set_xlabel('')
            ax[plt_line, plt_col].set_ylabel("${V_R}$ [m/s]")
        else :
            ax[plt_line, plt_col].tick_params(top=False, labeltop=False, bottom=False, labelbottom=False, left=False, labelleft=False, right=False, labelright=False)
            ax[plt_line, plt_col].set_xlabel('')
            ax[plt_line, plt_col].set_ylabel('')


    if plt_col == subplot_width - 1:
        plt_line += 1
        plt_col = 0
    else :
        plt_col +=1
    ### -----------------------------------------------------------------------------------------------





### -----------------------------------------------------------------------------------------------
while plt_col < subplot_width:
    ax[plt_line, plt_col].axis('off')
    plt_col += 1
fig.savefig(f'{PATH_OUTPUT}/{MODEL_NAME}/{PROFILE}/{MODEL_NAME}_{PROFILE}_disp-comp.png', bbox_inches='tight')
### -----------------------------------------------------------------------------------------------





### -----------------------------------------------------------------------------------------------
np.savetxt(f'{PATH_OUTPUT}/{MODEL_NAME}/{PROFILE}/{MODEL_NAME}_{PROFILE}_zs.txt', zs, fmt='%.2f')
np.savetxt(f'{PATH_OUTPUT}/{MODEL_NAME}/{PROFILE}/{MODEL_NAME}_{PROFILE}_xs.txt', xmids.reshape(1, len(xmids)), fmt='%.2f')
### -----------------------------------------------------------------------------------------------





### -----------------------------------------------------------------------------------------------
# Smooth water table profile
WT_db = np.array(WT_db)
if (len(WT_db)/2) % 2 == 0:
    wl = len(WT_db)/2 + 1
else:
    wl = len(WT_db)/2
WT_db_smooth = savgol_filter(WT_db, window_length=wl, polyorder=3, mode="nearest")
np.savetxt(f'{PATH_OUTPUT}/{MODEL_NAME}/{PROFILE}/{MODEL_NAME}_{PROFILE}_WT.txt', WT_db.reshape(1,len(WT_db)) , fmt='%.2f')
### -----------------------------------------------------------------------------------------------





### Vs --------------------------------------------------------------------------------------------
Vs_db = pd.DataFrame(Vs_db).to_numpy().T
fig, ax = plt.subplots(figsize=(19*cm, 6*cm), dpi=300)
extent = [xmid_min, xmid_max, -max(depth_db), -0.1]
im1 = ax.imshow(Vs_db, aspect='auto', cmap='terrain', extent=extent, vmin=200, vmax=700)
ax.plot(xmids, -WT_db, linestyle='--', color='k')
ax.plot(xmids, -WT_db_smooth, color='k')
ax.tick_params(which='both', labelbottom=False, labeltop=True, labelleft=True, labelright=False, bottom=True, top=True, left=True, right=True)
ax.set_xlabel('Position [m]')
ax.xaxis.set_label_position('top')
ax.set_ylabel('Depth [m]')
ax.set_ylim([-output_sequence_format.vocab.max_depth,0])
cb = fig.colorbar(im1, ax=ax, label="${V_S}$ [m/s]")
cb.minorticks_on()
ax.minorticks_on()
fig.savefig(f'{PATH_OUTPUT}/{MODEL_NAME}/{PROFILE}/{MODEL_NAME}_{PROFILE}_Vs.png', bbox_inches='tight')
np.savetxt(f'{PATH_OUTPUT}/{MODEL_NAME}/{PROFILE}/{MODEL_NAME}_{PROFILE}_Vs.txt', Vs_db, fmt='%.2f')


def mode_filter(values):
    mode = np.nanmean(values)
    return mode

smoothed_map = generic_filter(Vs_db, mode_filter, size=(1,3))

fig, ax = plt.subplots(figsize=(19*cm, 6*cm), dpi=300)
extent = [xmid_min, xmid_max, -max(depth_db), -0.1]
im1 = ax.imshow(smoothed_map, aspect='auto', cmap='terrain', extent=extent, vmin=200, vmax=700)
ax.plot(xmids, -WT_db, linestyle='--', color='k')
ax.plot(xmids, -WT_db_smooth, color='k')
ax.set_xlabel('Position [m]')
ax.xaxis.set_label_position('top')
ax.set_ylabel('Depth [m]')
ax.tick_params(which='both', labelbottom=False, labeltop=True, labelleft=True, labelright=False, bottom=True, top=True, left=True, right=True)
ax.set_ylim([-output_sequence_format.vocab.max_depth,0])
cb = fig.colorbar(im1, ax=ax, label="${V_S}$ [m/s]")
cb.minorticks_on()
ax.minorticks_on()
fig.savefig(f'{PATH_OUTPUT}/{MODEL_NAME}/{PROFILE}/{MODEL_NAME}_{PROFILE}_Vs-smooth.png', bbox_inches='tight')
### -----------------------------------------------------------------------------------------------





### Vp --------------------------------------------------------------------------------------------
Vp_db = pd.DataFrame(Vp_db).to_numpy().T
fig, ax = plt.subplots(figsize=(19*cm, 6*cm), dpi=300)
extent = [xmid_min, xmid_max, -max(depth_db), -0.1]
im2 = ax.imshow(Vp_db, aspect='auto', cmap='terrain', extent=extent)
ax.plot(xmids, -WT_db, linestyle='--', color='k')
ax.plot(xmids, -WT_db_smooth, color='k')
ax.set_xlabel('Position [m]')
ax.xaxis.set_label_position('top')
ax.set_ylabel('Depth [m]')
ax.set_ylim([-output_sequence_format.vocab.max_depth,0])
ax.tick_params(which='both', labelbottom=False, labeltop=True, labelleft=True, labelright=False, bottom=True, top=True, left=True, right=True)
cb = fig.colorbar(im2, ax=ax, label="${V_P}$ [m/s]")
cb.minorticks_on()
ax.minorticks_on()
fig.savefig(f'{PATH_OUTPUT}/{MODEL_NAME}/{PROFILE}/{MODEL_NAME}_{PROFILE}_Vp.png', bbox_inches='tight')
np.savetxt(f'{PATH_OUTPUT}/{MODEL_NAME}/{PROFILE}/{MODEL_NAME}_{PROFILE}_Vp.txt', Vp_db, fmt='%.2f')


def mode_filter(values):
    mode = np.nanmean(values)
    return mode

smoothed_map = generic_filter(Vp_db, mode_filter, size=(1,3))

fig, ax = plt.subplots(figsize=(19*cm, 6*cm), dpi=300)
extent = [xmid_min, xmid_max, -max(depth_db), -0.1]
im2 = ax.imshow(smoothed_map, aspect='auto', cmap='terrain', extent=extent)
ax.plot(xmids, -WT_db, linestyle='--', color='k')
ax.plot(xmids, -WT_db_smooth, color='k')
ax.tick_params(which='both', labelbottom=False, labeltop=True, labelleft=True, labelright=False, bottom=True, top=True, left=True, right=True)
ax.set_xlabel('Position [m]')
ax.xaxis.set_label_position('top')
ax.set_ylabel('Depth [m]')
ax.set_ylim([-output_sequence_format.vocab.max_depth,0])
cb = fig.colorbar(im2, ax=ax, label="${V_P}$ [m/s]")
cb.minorticks_on()
ax.minorticks_on()
fig.savefig(f'{PATH_OUTPUT}/{MODEL_NAME}/{PROFILE}/{MODEL_NAME}_{PROFILE}_Vp-smooth.png', bbox_inches='tight')
### -----------------------------------------------------------------------------------------------





### SOILS -----------------------------------------------------------------------------------------
GM_int_db = []
soil_to_int = {'None':0, 'clay':1, 'silt':2, 'loam':3, 'sand':4}
int_to_soil = {0:'None', 1:'clay', 2:'silt', 3:'loam', 4:'sand'}
for i, (soils, thicks) in enumerate(zip(GM_db, thicks_db)):
    log = []
    for soil, thick in zip(soils, thicks):
        log.extend([soil_to_int[soil]]*int(thick))
    GM_int_db.append(log)
GM_int_db = pd.DataFrame(GM_int_db)
GM_int_db = GM_int_db.fillna(0)
GM_int_db = GM_int_db.to_numpy().T

cmap = ListedColormap(["white", "mediumblue", "dodgerblue", "limegreen", "yellow"])


fig, ax = plt.subplots(figsize=(19*cm, 6*cm), dpi=300)
extent = [xmid_min, xmid_max, -max(depth_db), 0]
im3 = ax.imshow(GM_int_db, aspect='auto', cmap=cmap, extent=extent, vmin=0, vmax=4, alpha=0.5)
ax.plot(xmids, -WT_db, linestyle='--', color='k')
ax.plot(xmids, -WT_db_smooth, color='k')
ax.tick_params(which='both', labelbottom=False, labeltop=True, labelleft=True, labelright=False, bottom=True, top=True, left=True, right=True)
ax.set_xlabel('Position [m]')
ax.xaxis.set_label_position('top')
ax.set_ylabel('Depth [m]')
ax.set_ylim([-output_sequence_format.vocab.max_depth,0])
colorbar = fig.colorbar(im3, ax=ax, label='Soil type')
colorbar.set_ticks(list(soil_to_int.values()))
colorbar.set_ticklabels(list(soil_to_int.keys()))
ax.minorticks_on()
fig.savefig(f'{PATH_OUTPUT}/{MODEL_NAME}/{PROFILE}/{MODEL_NAME}_{PROFILE}_GM.png', bbox_inches='tight')

# Save GM_int_db as text file
GM_int_db_toSave = np.full(GM_int_db.shape, 'NaN', dtype=object)
for i in range(GM_int_db_toSave.shape[0]):
    for j in range(GM_int_db_toSave.shape[1]):
        GM_int_db_toSave[i,j] = int_to_soil[GM_int_db[i,j]]
np.savetxt(f'{PATH_OUTPUT}/{MODEL_NAME}/{PROFILE}/{MODEL_NAME}_{PROFILE}_GM.txt', GM_int_db_toSave, fmt='%s')


def mode_filter(values):
    counts = np.bincount(values.astype(int))
    if np.all(counts <= 1):
        return int(values[(len(values)//2)])
    else:
        return counts.argmax()

smoothed_map = generic_filter(GM_int_db, mode_filter, size=(1,3))
smoothed_map = generic_filter(smoothed_map, mode_filter, size=(1,3))

fig, ax = plt.subplots(figsize=(19*cm, 6*cm), dpi=300)
extent = [xmid_min, xmid_max, -max(depth_db), 0]
im3 = ax.imshow(smoothed_map, aspect='auto', cmap=cmap, extent=extent, vmin=0, vmax=4, alpha=0.5)
ax.plot(xmids, -WT_db, linestyle='--', color='k')
ax.plot(xmids, -WT_db_smooth, color='k')
ax.tick_params(which='both', labelbottom=False, labeltop=True, labelleft=True, labelright=False, bottom=True, top=True, left=True, right=True)
ax.set_xlabel('Position [m]')
ax.xaxis.set_label_position('top')
ax.set_ylabel('Depth [m]')
ax.set_ylim([-output_sequence_format.vocab.max_depth,0])
colorbar = fig.colorbar(im3, ax=ax, label='Soil type')
colorbar.set_ticks(list(soil_to_int.values()))
colorbar.set_ticklabels(list(soil_to_int.keys()))
ax.minorticks_on()
fig.savefig(f'{PATH_OUTPUT}/{MODEL_NAME}/{PROFILE}/{MODEL_NAME}_{PROFILE}_GM-smooth.png', bbox_inches='tight')
### -----------------------------------------------------------------------------------------------





### N ---------------------------------------------------------------------------------------------
Ns_int_db = []
for i, (Ns, thicks) in enumerate(zip(Ns_db, thicks_db)):
    log = []
    for N, thick in zip(Ns, thicks):
        log.extend([N]*int(thick))
    Ns_int_db.append(log)
Ns_int_db = pd.DataFrame(Ns_int_db)
Ns_int_db = Ns_int_db.to_numpy().T

fig, ax = plt.subplots(figsize=(19*cm, 6*cm), dpi=300)
extent = [xmid_min, xmid_max, -max(depth_db), -0.1]
cmap = ListedColormap(["tab:blue", "tab:cyan", "tab:green", "tab:orange", "tab:red"])
im4 = ax.imshow(Ns_int_db, aspect='auto', cmap=cmap, extent=extent, vmin=6, vmax=10)
ax.plot(xmids, -WT_db, linestyle='--', color='k')
ax.plot(xmids, -WT_db_smooth, color='k')
ax.tick_params(which='both', labelbottom=False, labeltop=True, labelleft=True, labelright=False, bottom=True, top=True, left=True, right=True)
ax.set_xlabel('Position [m]')
ax.xaxis.set_label_position('top')
ax.set_ylabel('Depth [m]')
ax.set_ylim([-output_sequence_format.vocab.max_depth,0])
cb = fig.colorbar(im4, ax=ax, label='N')
ax.minorticks_on()
fig.savefig(f'{PATH_OUTPUT}/{MODEL_NAME}/{PROFILE}/{MODEL_NAME}_{PROFILE}_N.png', bbox_inches='tight')
np.savetxt(f'{PATH_OUTPUT}/{MODEL_NAME}/{PROFILE}/{MODEL_NAME}_{PROFILE}_N.txt', Ns_int_db, fmt='%s')
### -----------------------------------------------------------------------------------------------





### DISP ------------------------------------------------------------------------------------------
cmap = plt.get_cmap('nipy_spectral')
colors = [cmap(i / (len(disp_db) - 1)) for i in range(len(disp_db))]
fig, ax = plt.subplots(figsize=(19*cm, 6*cm), dpi=300)
for i, disp in enumerate(disp_db):
    ax.plot(disp[0][:,0], disp[0][:,1], color=colors[i])
ax.set_ylim([100, 500])
ax.set_xlim([min_freq, max_freq])
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel("${V_R}$ [m/s]")
ax.minorticks_on()
fig.savefig(f'{PATH_OUTPUT}/{MODEL_NAME}/{PROFILE}/{MODEL_NAME}_{PROFILE}_disp.png', bbox_inches='tight')

disp_toSave = []
for disp in disp_db:
    disp_toSave.append(disp[0][:,1])
disp_toSave = np.array(disp_toSave).T
np.savetxt(f'{PATH_OUTPUT}/{MODEL_NAME}/{PROFILE}/{MODEL_NAME}_{PROFILE}_disp.txt', disp_toSave, fmt='%.2f')
np.savetxt(f'{PATH_OUTPUT}/{MODEL_NAME}/{PROFILE}/{MODEL_NAME}_{PROFILE}_freqs.txt', disp_db[0][0][:,0], fmt='%.2f')
### -----------------------------------------------------------------------------------------------





### RMS -------------------------------------------------------------------------------------------
with open(f'{PATH_OUTPUT}/{MODEL_NAME}/{PROFILE}/{MODEL_NAME}_{PROFILE}_disp-rms.txt', 'w') as f:
    f.write(f'Model ID: {model.id}\n\n')
    f.write(f'Mean RMS: {np.mean(rms_db)} m/s')
### -----------------------------------------------------------------------------------------------