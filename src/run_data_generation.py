"""
Author : José CUNHA TEIXEIRA
License : SNCF Réseau, UMR 7619 METIS
Date : April 30, 2024
"""





import sys
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from subprocess import run, PIPE
import random
from tqdm import tqdm

from folders import PATH_INPUT
from misc import generate_numbers

# Import Santiludo functions
sys.path.append('/home/jteixeira/Documents/PhD_Monitoring/Work/Processing/Tools/Santiludo_layered/')
from lib.VGfunctions import vanGen
from lib.RPfunctions import hillsAverage, effFluid, hertzMindlin, biotGassmann
from lib.TTDSPfunctions import writeVelocityModel, readDispersion





### -----------------------------------------------------------------------------------------------
files = [
        "0.7500.M0.pvc",
        "3.7500.M0.pvc",
        "6.7500.M0.pvc",
        "9.7500.M0.pvc",
        "12.7500.M0.pvc",
        "15.7500.M0.pvc",
        "18.7500.M0.pvc",
        "21.7500.M0.pvc",
        "24.7500.M0.pvc",
        "27.7500.M0.pvc",
        "30.7500.M0.pvc",
        "33.7500.M0.pvc",
        "36.7500.M0.pvc",
        "39.7500.M0.pvc",
        "42.7500.M0.pvc",
        "45.7500.M0.pvc",
        "48.7500.M0.pvc",
        "51.7500.M0.pvc",
        "54.7500.M0.pvc",
        "57.7500.M0.pvc",
        "60.7500.M0.pvc",
        "63.7500.M0.pvc",
        "66.7500.M0.pvc",
        "69.7500.M0.pvc",
        "72.7500.M0.pvc",
        "75.7500.M0.pvc",
        "78.7500.M0.pvc",
        "81.7500.M0.pvc",
        "84.7500.M0.pvc",
        "87.7500.M0.pvc",
        "90.7500.M0.pvc",
        "93.7500.M0.pvc",
        "96.7500.M0.pvc",
        "99.7500.M0.pvc",
        "102.7500.M0.pvc",
        "105.7500.M0.pvc",
        "108.7500.M0.pvc",
        "111.7500.M0.pvc",
        "114.7500.M0.pvc",
        "117.7500.M0.pvc",
        "120.7500.M0.pvc",
        "123.7500.M0.pvc",
        "126.7500.M0.pvc",
        ]
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
under_layers = "10 1500 750 2000\n0 4000 2000 2500\n" # One substratum layer
n_under_layers = under_layers.count('\n') # Number of under layers

# VM_thicknesses = np.diff(np.abs(zs)) # thickness vector [m]

x0 = 1 # first geophone position [m]
Nx = 192 # number of geophones [m]
dx = 1 # geophone interval [m]
xs = np.arange(x0, Nx * dx + 1, dx)
trig  = 0 # data pretrig (if needed)


# Frequency domain and sampling setup to compute dispersion
df = 1 # number of frequency samples [#]
min_f = 15 # minimum frequency [Hz]
max_f = 50
nf = int((max_f - min_f) / df) + 1 # number of frequency samples [#]

n_modes = 1 # Number of modes to compute
s = 'frequency' # Over frequencies mode
wave = 'R' # Rayleigh (PSV) fundamental mode
### -----------------------------------------------------------------------------------------------





### GENERATION PARAMETERS -------------------------------------------------------------------------
possible_soils = np.array(['clay', 'silt', 'loam', 'sand'])

max_layers = 4

d_WT = 0.5
possible_WTs = np.arange(d_WT, 10, d_WT)

max_depth = 20

N_models = 400000
print(f'\nGeneating {N_models} models')
### -----------------------------------------------------------------------------------------------




### GENERATE MODELS -------------------------------------------------------------------------------
DCs = []
GMs = []
THKS = []
WTs = []

for i in tqdm(range(N_models)):


    ### PARAMETERS --------------------------------------------------------------------------------
    N_layers = random.choice(range(1, max_layers+1))

    GM_thicknesses = generate_numbers(N_layers, 1, max_depth, 1)
    if np.sum(GM_thicknesses) != max_depth:
        raise ValueError('Sum of thicknesses is not equal to max_depth')

    soil_types = []
    tmp = list(possible_soils)
    for j in range(N_layers):
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
        soil_types_to_save = soil_types + [None] * (max_layers - N_layers)
        GM_thicknesses_to_save = GM_thicknesses + [None] * (max_layers - N_layers)
    else : 
        soil_types_to_save = soil_types
        GM_thicknesses_to_save = GM_thicknesses

    WT = random.choice(possible_WTs)

    Ns = [8] * N_layers

    fracs = [0.3] * N_layers
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
    velocity_model_string = writeVelocityModel(VM_thicknesses, VPs, VSs, rhobs, under_layers, n_under_layers)

    # Dispersion curves computing with GPDC
    velocity_model_RAMfile = StringIO(velocity_model_string) # Keep velocity model string in the RAM in a file format alike to trick GPDC which expects a file
    gpdc_command = [f"gpdc -{wave} {n_modes} -n {nf} -min {min_f} -max {max_f} -s {s} -j 16"]
    gpdc_output_string = run(gpdc_command, input=velocity_model_RAMfile.getvalue(), text=True, shell=True, stdout=PIPE).stdout # Raw output string from GPDC

    dispersion_data, n_modes = readDispersion(gpdc_output_string) # Reads GPDC output and converts dispersion data to a list of numpy arrays for each mode
                                                                # Updates number of computed modes (can be lower than what was defined if frequency range too small)
    ### -------------------------------------------------------------------------------------------



    ### APPEND DATA -------------------------------------------------------------------------------
    # Check if dispersion data is empty
    if not dispersion_data:
        print("No dispersion data could be computed with the following parameters:\n")
        print(f'{soil_types = }\n')
        print(f'{Ns = }\n')
        print(f'{fracs = }\n')
        print(f'{WT = }\n')
        print('Skipping to next iteration.\n')
        continue
    

    DCs.append(dispersion_data[0][:,1])
    GMs.append(soil_types_to_save)
    THKS.append(GM_thicknesses_to_save)
    WTs.append(WT)
    ### -------------------------------------------------------------------------------------------



    # ### PLOTS -------------------------------------------------------------------------------------
    # fig, ax = plt.subplots()

    # for file in tqdm(files):
    #     db = np.loadtxt(f'{PATH_INPUT}/real_data/{file}')
    #     fs_obs_raw, vs_obs_raw = db[:,0], db[:,1]
    #     ax.plot(fs_obs_raw, vs_obs_raw, linewidth=0.5, color='k')
    
    # # Plot simulated dispersion curve
    # max_vr = np.max(dispersion_data[0][:,1])
    # min_vr = np.min(dispersion_data[0][:,1])
    # for mode in range(n_modes):
    #     ax.plot(dispersion_data[mode][:,0], dispersion_data[mode][:,1], linewidth=1.5, color='red')
    #     if np.max(dispersion_data[mode][:,1]) > max_vr:
    #         max_vr = np.max(dispersion_data[mode][:,1])
    #     if np.min(dispersion_data[mode][:,1]) < min_vr:
    #         min_vr = np.min(dispersion_data[mode][:,1])
    # ax.set_xlim([min_f-5, max_f+5])
    # ax.set_ylim([100, 1100])
    # ax.set_xlabel('Frequency [Hz]')
    # ax.set_ylabel('P-SV phase vel. [m/s]')
    # plt.show()
    # ### ------------------------------------------------------------------------------------------

### -----------------------------------------------------------------------------------------------





### SAVE DATA -------------------------------------------------------------------------------------
np.savetxt(f'{PATH_INPUT}training_data6/DCs_part2.txt', np.array(DCs), fmt='%.3f')
np.savetxt(f'{PATH_INPUT}training_data6/GMs_part2.txt', np.array(GMs), fmt='%s')
np.savetxt(f'{PATH_INPUT}training_data6/THKs_part2.txt', np.array(THKS), fmt='%s')
np.savetxt(f'{PATH_INPUT}training_data6/WTs_part2.txt', np.array(WTs), fmt='%s')
### -----------------------------------------------------------------------------------------------