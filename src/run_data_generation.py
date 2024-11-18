"""
Author : José CUNHA TEIXEIRA
License : SNCF Réseau, UMR 7619 METIS, Sorbonne Université
Date : April 30, 2024
"""





import sys
import numpy as np
from io import StringIO
from subprocess import run, PIPE, CalledProcessError
import random
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

from folders import PATH_INPUT, PATH_RPMODEL
from misc import generate_numbers

# Import Santiludo functions
sys.path.append(PATH_RPMODEL)
from lib.VGfunctions import vanGen
from lib.RPfunctions import hillsAverage, effFluid, hertzMindlin, biotGassmann
from lib.TTDSPfunctions import writeVelocityModel, readDispersion
import json





### SITE NAME -------------------------------------------------------------------------------------
site = 'Grand_Est'
### -----------------------------------------------------------------------------------------------




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

# # Geometry and discretisation of the medium
# depth = 20 # Depth of the soil column [m]
dz = 0.1 # Depth sample interval [m]
top_surface_level = dz # Altitude of the soil surface[m]
# zs = -np.arange(top_surface_level, depth + dz, dz) # Depth positions (negative downward) [m]

# NbCells = len(zs) - 1 # Number of exploration points in depth [#]

# Three possible RP models:
kk = 3 # Pe with suction (cf. Solazzi et al. 2021)
### -----------------------------------------------------------------------------------------------





### SEISMIC CONSTANTS -----------------------------------------------------------------------------
# Layers to put under the studied soil column on the velocity model
# In GPDC format : "thickness Vp Vs rho\n"
# Each layer is separated by \n | Only spaces between values | Last layer thickness must be 0)
# under_layers = "" # Empty string if no under layers
under_layers = "5 2000 1000 2000\n0 4000 2000 2500\n" # One substratum layer
N_under_layers = under_layers.count('\n') # Number of under layers


# Frequency domain and sampling setup to compute dispersion
d_freq = 1 # number of frequency samples [#]
min_freq = 5 # minimum frequency [Hz]
max_freq = 100 # maximum frequency [Hz]
nf = int((max_freq - min_freq) / d_freq) + 1 # number of frequency samples [#]

n_modes = 1 # Number of modes to compute
s = 'frequency' # Over frequencies mode
wave = 'R' # Rayleigh (PSV) fundamental mode
### -----------------------------------------------------------------------------------------------





### GENERATION PARAMETERS -------------------------------------------------------------------------
possible_soils = np.array(['clay', 'loam', 'silt', 'sand'])

max_N_layers = 4

d_WT = 1
min_WT = 1
max_WT = 10
possible_WTs = np.arange(min_WT, max_WT+d_WT, d_WT)

d_thickness = 0.5
min_thickness = 0.5
max_depth = 20

possible_Ns = np.array([6, 7, 8, 9, 10])

frac = 0.3

N_models = None
print(f'\nGeneating {N_models} models')
### -----------------------------------------------------------------------------------------------




### GENERATE MODELS -------------------------------------------------------------------------------
DCs = []
GMs = []
THKS = []
WTs = []
COORDs = []


pbar = tqdm(total=N_models)
i=0
redo = False
while i < N_models:

    ### PARAMETERS --------------------------------------------------------------------------------
    if redo == False:
        N_layers = random.choices(range(1, max_N_layers+1), weights=[0.01, 0.05, 0.25, 0.69], k=1)[0]

        GM_thicknesses = generate_numbers(N_layers, min_thickness, max_depth, d_thickness)
        if np.sum(GM_thicknesses) != max_depth:
            raise ValueError('Sum of thicknesses is not equal to max_depth')

        soil_types = []
        Ns = []
        tmp = list(possible_soils)
        for j in range(N_layers):
            N = random.choice(possible_Ns)
            Ns.append(N)
            if j > 0:
                tmp.remove(previous_soil_type)
                soil_type = random.choice(tmp)
                soil_types.append(soil_type)
                tmp.append(previous_soil_type)
                previous_soil_type = soil_type
            else :
                soil_type = random.choice(tmp)
                soil_types.append(soil_type)
                previous_soil_type = soil_type

        if N_layers < 4:
            soil_types_to_save = soil_types + [None] * (max_N_layers - N_layers)
            GM_thicknesses_to_save = GM_thicknesses + [None] * (max_N_layers - N_layers)
            Ns_to_save = Ns + [None] * (max_N_layers - N_layers)
        else : 
            soil_types_to_save = soil_types
            GM_thicknesses_to_save = GM_thicknesses
            Ns_to_save = Ns

        WT = random.choice(possible_WTs)

        fracs = frac*N_layers
    

    elif redo == True:
        dz /= 10
        top_surface_level = dz

        if dz < 0.01:
            print(f'{dz = } too small')
            redo = False
            dz = 0.1
            top_surface_level = dz
            print(f'Back to normal computation with {dz = }\n')
            continue

        print(f'Redoing computation with {dz = }\n')
    ### -------------------------------------------------------------------------------------------


    # Geometry and discretisation of the medium ---------------------------------------------------
    depth = np.sum(GM_thicknesses) # Depth of the soil column [m]
    zs = -np.arange(top_surface_level, depth + dz, dz) # Depth positions (negative downward) [m]

    NbCells = len(zs) - 1 # Number of exploration points in depth [#]

    VM_thicknesses = np.diff(np.abs(zs)) # thickness vector [m]
    ### -------------------------------------------------------------------------------------------


    #### ROCK PHYSICS -----------------------------------------------------------------------------
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
    ### -------------------------------------------------------------------------------------------



    #### SEISMIC FWD MODELING ---------------------------------------------------------------------
    # Velocity model in string format for GPDC
    velocity_model_string = writeVelocityModel(VM_thicknesses, VPs, VSs, rhobs, under_layers, N_under_layers)

    # Dispersion curves computing with GPDC
    velocity_model_RAMfile = StringIO(velocity_model_string) # Keep velocity model string in the RAM in a file format alike to trick GPDC which expects a file
    gpdc_command = [f"gpdc -{wave} {n_modes} -n {nf} -min {min_freq} -max {max_freq} -s {s} -j 1"]
    
    try :
        process = run(gpdc_command, input=velocity_model_RAMfile.getvalue(), text=True, shell=True, stdout=PIPE, stderr=PIPE, check=True) # Raw output string from GPDC
    except CalledProcessError as e:
        print("\n\nNo dispersion data could be computed with the following parameters:")
        print(f'{soil_types = }')
        print(f'{GM_thicknesses = }')
        print(f'{Ns = }')
        print(f'{fracs = }')
        print(f'{WT = }')
        print(f'{dz = }\n')
        redo = True
        continue

    if redo == True:
        redo = False
        dz = 0.1
        top_surface_level = dz
        print(f'Back to normal computation with {dz = }\n')

    gpdc_output_string = process.stdout # Raw output string from GPDC
    dispersion_data, n_modes = readDispersion(gpdc_output_string) # Reads GPDC output and converts dispersion data to a list of numpy arrays for each mode
                                                                # Updates number of computed modes (can be lower than what was defined if frequency range too small)
    ### -------------------------------------------------------------------------------------------



    ### APPEND DATA -------------------------------------------------------------------------------
    DCs.append(dispersion_data[0][:,1])
    GMs.append(soil_types_to_save)
    THKS.append(GM_thicknesses_to_save)
    WTs.append(WT)
    COORDs.append(Ns_to_save)
    ### -------------------------------------------------------------------------------------------



    pbar.update(1)
    i += 1



    ### PLOT DISPERSION DATA -----------------------------------------------------------------------
    # print(f'{Ns_to_save = }')
    # print(f'{soil_types_to_save = }')
    # print(f'{GM_thicknesses_to_save = }')
    # print(f'{WT = }')
    # fig, ax = plt.subplots(1, 1, figsize=(16,9))
    # ax.plot(dispersion_data[0][:,0], dispersion_data[0][:,1], color='r')
    # plt.show()
    ### -------------------------------------------------------------------------------------------



pbar.close()
### -----------------------------------------------------------------------------------------------





### SAVE DATA -------------------------------------------------------------------------------------
DCs = np.array(DCs)
GMs = np.array(GMs)
THKS = np.array(THKS)
WTs = np.array(WTs)
COORDs = np.array(COORDs)

_, unique_indices = np.unique(DCs, axis=0, return_index=True)
unique_indices = np.sort(unique_indices)

print(f'\nRemoving duplicates from {len(DCs)} models to {len(unique_indices)} models.\n')

DCs = DCs[unique_indices]
GMs = GMs[unique_indices]
THKS = THKS[unique_indices]
WTs = WTs[unique_indices]
COORDs = COORDs[unique_indices]

name = f'training_data_{site}'
np.savetxt(f'{PATH_INPUT}/training_data/{site}/DCs.txt', np.array(DCs), fmt='%.3f')
np.savetxt(f'{PATH_INPUT}/training_data/{site}/GMs.txt', np.array(GMs), fmt='%s')
np.savetxt(f'{PATH_INPUT}/training_data/{site}/THKs.txt', np.array(THKS), fmt='%s')
np.savetxt(f'{PATH_INPUT}/training_data/{site}/WTs.txt', np.array(WTs), fmt='%s')
np.savetxt(f'{PATH_INPUT}/training_data/{site}/Ns.txt', np.array(COORDs), fmt='%s')
fs = np.arange(min_freq, max_freq+d_freq, d_freq)
np.savetxt(f'{PATH_INPUT}/training_data/{site}/fs.txt', np.array(fs), fmt='%s')


params = {
    "name": name,

    "N_samples": len(WTs),

    "soils": possible_soils,
    "max_N_layers": max_N_layers,

    "d_thickness": d_thickness,
    "min_thickness": min_thickness,
    "max_thickness": max_depth,
    "thicknesses": np.arange(min_thickness, max_depth+d_thickness, d_thickness).tolist(),

    "max_depth": max_depth,

    "d_WT": d_WT,
    "min_WT": min_WT,
    "max_WT": max_WT,
    "WTs": np.arange(min_WT, max_WT+d_WT, d_WT).tolist(),

    "Ns": possible_Ns,

    "frac": frac,

    "d_freq": d_freq,
    "min_freq": min_freq,
    "max_freq": max_freq,
    "freqs": np.arange(min_freq, max_freq+d_freq, d_freq).tolist(),
    "N_freqs" : DCs.shape[1],

    "min_vel" : np.min(DCs),
    "max_vel" : np.max(DCs),

    "n_modes": n_modes,

    "under_layers": under_layers,
    'N_under_layers': N_under_layers,

    "dz": dz,
    "top_surface_level": dz
}

with open(f'{PATH_INPUT}/training_data/{site}/params.json', 'w') as file:
    json.dump(params, file, indent=2)
### -----------------------------------------------------------------------------------------------