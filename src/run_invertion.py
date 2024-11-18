"""
Author : José CUNHA TEIXEIRA
License : SNCF Réseau, UMR 7619 METIS, Sorbonne Université
Date : April 30, 2024
"""





import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from io import StringIO
from pickle import load
from subprocess import run, PIPE, CalledProcessError
from tqdm import tqdm

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
N_jobs = 1
set_intra_op_parallelism_threads(N_jobs)
set_inter_op_parallelism_threads(N_jobs)
### -----------------------------------------------------------------------------------------------





### PARAMS ----------------------------------------------------------------------------------------
model_id = '[202407170928]'
site = 'Grand_Est'
dx = 3 # Distance between the dispersion curves [m]
### -----------------------------------------------------------------------------------------------





### LOAD MODEL ------------------------------------------------------------------------------------
path_model = f'{PATH_MODELS}/{model_id}/{model_id}_model'
print(f'\nLoading model : {path_model}')
with open(f'{path_model}.pkl', 'rb') as f:
    transformer = load(f)
### -----------------------------------------------------------------------------------------------





### PROFILES TO INVERT ----------------------------------------------------------------------------
dates = os.listdir(f'{PATH_INPUT}/real_data/{site}/')
dates = sorted(dates)
profiles = []
for date in dates:
    year, month, day = date.split('-')
    if year in ['2022', '2023'] and month in ['07']: ### <- Change this line to select the desired dates
        list_profiles = os.listdir(f'{PATH_INPUT}/real_data/{site}/{date}/')
        list_profiles = sorted(list_profiles)
        profiles += [f'{site}/{date}/{profile}' for profile in list_profiles]
dxs = [dx]*len(profiles)
### -----------------------------------------------------------------------------------------------






### FORMATS ---------------------------------------------------------------------------------------
params = transformer.params

if params['model_params']['trained'] == False:
  print(f'\033[1;33mERROR: Model {model_id} was not trained yet.\033[0m')
  sys.exit

data_params = params['data_params']

max_N_layers = params['data_params']['max_N_layers']

min_freq = params['input_seq_format']['min_freq']
max_freq = params['input_seq_format']['max_freq']
min_vel = params['input_seq_format']['min_vel']
max_vel = params['input_seq_format']['max_vel']
d_freq = params['input_seq_format']['d_freq']
N_freqs = params['input_seq_format']['N_freqs']
print(f'\n{min_freq = }, {max_freq = }, {min_vel = }, {max_vel = }')

word_to_index = params['output_seq_format']['word_to_index']
index_to_word = params['output_seq_format']['index_to_word']
len_output_seq = params['output_seq_format']['length']

# Under layers
under_layers = data_params['under_layers']
N_under_layers = data_params['N_under_layers']

# Geometry and discretisation of the medium
dz_origin = data_params['dz'] # Depth sample interval [m]
dz = dz_origin
top_surface_level_origin = data_params['top_surface_level'] # Altitude of the soil surface[m]
top_surface_level = top_surface_level_origin

n_modes = data_params['n_modes'] # Number of modes to compute
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

# Three possible RP models:
kk = 3 # Pe with suction (cf. Solazzi et al. 2021)
### -----------------------------------------------------------------------------------------------





### SEISMIC CONSTANTS -----------------------------------------------------------------------------
s = 'frequency' # Over frequencies mode
wave = 'R' # Rayleigh (PSV) fundamental mode
### -----------------------------------------------------------------------------------------------





for profile, dx in tqdm(zip(profiles, dxs), total=len(profiles), desc='Profiles', colour='green'):
    ### FIELD DISPERSION DATA FILES ---------------------------------------------------------------
    PROFILE_NAME = profile.split('/')[-1]

    files = os.listdir(f'{PATH_INPUT}/real_data/{profile}/')
    files = sorted(files, key=lambda x: float(x.split('_')[0]))

    xmids = [float(x.split('_')[0]) for x in files]
    xmids = sorted(xmids)
    
    if not os.path.exists(f'{PATH_OUTPUT}/{model_id}/{profile}/'):
        os.makedirs(f'{PATH_OUTPUT}/{model_id}/{profile}/')
    ### -------------------------------------------------------------------------------------------





    ### INVERSION ---------------------------------------------------------------------------------
    z_zx = []

    soil_zx = []
    thick_zx = []
    N_zx = []
    WT_x = []

    h_zx = []
    Sw_zx = []
    Swe_zx = []

    mus_zx = []
    Ks_zx = []
    rhos_zx = []
    nus_zx = []

    Kf_zx = []
    rhof_zx = []
    rhob_zx = []

    Km_zx = []
    mum_zx = []

    Vs_zx = []
    Vp_zx = []
    disp_db = []
    
    rms_x = []


    for file in tqdm(files, total=len(files), leave=False, desc='Xmids'):
        computed  = False
        flag = False
        if dz != dz_origin:
            dz = dz_origin
            top_surface_level = top_surface_level_origin
            print(f'INFO : dz reset at {dz_origin}\n')
            
            
        while computed == False:
            ### OBSERVED DATA -------------------------------------------------------------------------
            db = np.loadtxt(f'{PATH_INPUT}/real_data/{profile}/{file}')
            fs_obs_raw, Vr_obs_raw = db[:,0], db[:,1]

            if (len(Vr_obs_raw)/4) % 2 == 0:
                wl = len(Vr_obs_raw)/4 + 1
            else:
                wl = len(Vr_obs_raw)/4
            Vr_obs_raw = savgol_filter(Vr_obs_raw, window_length=wl, polyorder=2, mode="nearest")

            axis_resamp = np.arange(min_freq, max_freq+1, 1)
            fs_obs, Vr_obs = resamp(fs_obs_raw, Vr_obs_raw, axis_resamp=axis_resamp, type='frequency')

            Vr_obs_comp = np.copy(Vr_obs)

            Vr_obs = (Vr_obs-min_vel) / (max_vel-min_vel)

            Vr_obs = Vr_obs.reshape(1, Vr_obs.shape[0], 1)
            ### ---------------------------------------------------------------------------------------





            ### INFERENCE -----------------------------------------------------------------------------
            input_seq = Vr_obs

            decoded_seq = transformer.decode_seq_restrictive(input_seq)

            decoded_GM = []

            # print(f"\nDecoded Sequence: {decoded_seq}\n")
            for i in range(0, len(decoded_seq)):
                decoded_GM.append(index_to_word[decoded_seq[i]])
                # print(f'{i+1} : {index_to_word[decoded_seq[i]]}')

            soil_types = decoded_GM[3::6]
            soil_types = [soil for soil in soil_types if soil not in ['[PAD]', '[END]']]

            GM_thicknesses = decoded_GM[5::6]
            GM_thicknesses = [float(thickness) for thickness in GM_thicknesses if thickness not in ['[PAD]', '[END]']]

            Ns = decoded_GM[7::6]
            Ns = [float(N) for N in Ns if N not in ['[PAD]', '[END]']]
            
            WT = float(decoded_GM[1])

            fracs = [0.3] * len(soil_types)

            depth = np.sum(GM_thicknesses)

            # print(f'\n{soil_types = }')
            # print(f'{GM_thicknesses = }', depth)
            # print(f'{Ns = }')
            # print(f'{fracs = }')
            # print(f'{WT = }')
            # print(f'{depth = }')
            # print(f'{dz = }')
            # if depth != params['data_params']['max_depth']:
            #     print(f"\033[1;33mWARNING: Sum of layer thicknesses {depth} is not equal to the expected total depth {params['data_params']['max_depth']}.\033[0m")


            for soil in soil_types:
                if soil not in params['data_params']['soils']:
                    print('Error: Soil type not in soil vocabulary.')
                    sys.exit()
            ### ---------------------------------------------------------------------------------------





            ### ROCK PHYSICS CONSTANTS ----------------------------------------------------------------
            zs = -np.arange(top_surface_level, depth + dz, dz) # Depth positions (negative downward) [m]

            NbCells = len(zs) - 1 # Number of exploration points in depth [#]
            ### ---------------------------------------------------------------------------------------





            ### SEISMIC CONSTANTS ---------------------------------------------------------------------
            VM_thicknesses = np.diff(np.abs(zs)) # thickness vector [m]
            ### ---------------------------------------------------------------------------------------





            #### ROCK PHYSICS -------------------------------------------------------------------------
            # Saturation profile with depth
            h_z, Sw_z, Swe_z = vanGen(zs, WT, soil_types, GM_thicknesses)


            # Effective Grain Properties (constant with depth)
            mus_z, Ks_z, rhos_z, nus_z = hillsAverage(mu_clay, mu_silt, mu_sand, rho_clay,
                                                rho_silt, rho_sand, k_clay, k_silt,
                                                k_sand, soil_types)
       

            # Effective Fluid Properties
            Kf_z, rhof_z, rhob_z = effFluid(Sw_z, kw, ka, rhow,
                                            rhoa, rhos_z, soil_types, GM_thicknesses, dz)


            # Hertz Mindlin Frame Properties
            Km_z, mum_z = hertzMindlin(Swe_z, zs, h_z, rhob_z,
                                        g, rhoa, rhow, Ns,
                                        mus_z, nus_z, fracs, kk,
                                        soil_types, GM_thicknesses)


            # Saturated Properties
            Vp_z, Vs_z = biotGassmann(Km_z, mum_z, Ks_z, Kf_z,
                                    rhob_z, soil_types, GM_thicknesses, dz)
            ### ---------------------------------------------------------------------------------------



            #### SEISMIC FWD MODELING -----------------------------------------------------------------
            # Velocity model in string format for GPDC
            velocity_model_string = writeVelocityModel(VM_thicknesses, Vp_z, Vs_z, rhob_z, under_layers, N_under_layers)

            # Dispersion curves computing with GPDC
            velocity_model_RAMfile = StringIO(velocity_model_string) # Keep velocity model string in the RAM in a file format alike to trick GPDC which expects a file
            gpdc_command = [f"gpdc -{wave} {n_modes} -n {N_freqs} -min {min_freq} -max {max_freq} -s {s}"]

            try:
                process = run(gpdc_command, input=velocity_model_RAMfile.getvalue(), text=True, shell=True, stdout=PIPE, stderr=PIPE, check=True) # Raw output string from GPDC
            except CalledProcessError as e:
                print(f"\nERROR during GPDC computation. Returned:\n{e.stdout}")
                print("Used parameters:")
                print(f'{soil_types = }')
                print(f'{GM_thicknesses = }')
                print(f'{Ns = }')
                print(f'{fracs = }')
                print(f'{WT = }')
                print(f'{dz = }\n')
                dz /= 10
                top_surface_level /= 10
                print(f'INFO : dz reduced at {dz}\n')
                if dz > 0.001:
                    continue
                else:
                    dispersion_data = disp_db[-1]
                    rms = rms_x[-1]
                    flag = True
            
            computed = True       

            if not flag:
                gpdc_output_string = process.stdout # Raw output string from GPDC
                dispersion_data, n_modes = readDispersion(gpdc_output_string) # Reads GPDC output and converts dispersion data to a list of numpy arrays for each mode
                                                                        # Updates number of computed modes (can be lower than what was defined if frequency range too small)
            flag = False

            rms = np.sqrt(np.mean((Vr_obs_comp - dispersion_data[0][:,1])**2))
            nrms = rms / (np.max(Vr_obs_comp) - np.min(Vr_obs_comp))
            
            factor = int(dz_origin/dz)
            zs = zs[::factor]
            h_z = h_z[::factor]
            Sw_z = Sw_z[::factor]
            Swe_z = Swe_z[::factor]
            Kf_z = Kf_z[::factor]
            rhof_z = rhof_z[::factor]
            rhob_z
            Km_z = Km_z[::factor]
            mum_z = mum_z[::factor]
            Vp_z = Vp_z[::factor]
            Vs_z = Vs_z[::factor]
            
            thick_zx.append(GM_thicknesses)
            soil_zx.append(soil_types)
            WT_x.append(WT)
            N_zx.append(Ns)
            z_zx.append(zs)
            h_zx.append(h_z)
            Sw_zx.append(Sw_z)
            Swe_zx.append(Swe_z)
            mus_zx.append(mus_z)
            Ks_zx.append(Ks_z)
            rhos_zx.append(rhos_z)
            nus_zx.append(nus_z)
            Kf_zx.append(Kf_z)
            rhof_zx.append(rhof_z)
            rhob_zx.append(rhob_z)
            Km_zx.append(Km_z)
            mum_zx.append(mum_z)
            Vp_zx.append(Vp_z)
            Vs_zx.append(Vs_z)
            disp_db.append(dispersion_data)
            rms_x.append(rms)
            ### ---------------------------------------------------------------------------------------




    ### SAVE DATA ---------------------------------------------------------------------------------
    z_zx = pd.DataFrame(z_zx).to_numpy().T
    xmids = np.array(xmids)
    xs = xmids

    soil_zx = pd.DataFrame(soil_zx).to_numpy().T
    if soil_zx.shape[0] < 4:
        soil_zx = np.pad(soil_zx, ((0, 4-soil_zx.shape[0]), (0, 0)), 'constant', constant_values='None')
    thick_zx = pd.DataFrame(thick_zx).to_numpy().T
    if thick_zx.shape[0] < 4:
        thick_zx = np.pad(thick_zx, ((0, 4-thick_zx.shape[0]), (0, 0)), 'constant', constant_values=np.nan)
    N_zx = pd.DataFrame(N_zx).to_numpy().T
    if N_zx.shape[0] < 4:
        N_zx = np.pad(N_zx, ((0, 4-N_zx.shape[0]), (0, 0)), 'constant', constant_values=np.nan)
    WT_x = pd.array(WT_x).reshape(len(WT_x))

    h_zx = pd.DataFrame(h_zx).to_numpy().T
    Sw_zx = pd.DataFrame(Sw_zx).to_numpy().T
    Swe_zx = pd.DataFrame(Swe_zx).to_numpy().T

    mus_zx = pd.DataFrame(mus_zx).to_numpy().T
    if mus_zx.shape[0] < 4:
        mus_zx = np.pad(mus_zx, ((0, 4-mus_zx.shape[0]), (0, 0)), 'constant', constant_values=np.nan)
    Ks_zx = pd.DataFrame(Ks_zx).to_numpy().T
    if Ks_zx.shape[0] < 4:
        Ks_zx = np.pad(Ks_zx, ((0, 4-Ks_zx.shape[0]), (0, 0)), 'constant', constant_values=np.nan)
    rhos_zx = pd.DataFrame(rhos_zx).to_numpy().T
    if rhos_zx.shape[0] < 4:
        rhos_zx = np.pad(rhos_zx, ((0, 4-rhos_zx.shape[0]), (0, 0)), 'constant', constant_values=np.nan)
    nus_zx = pd.DataFrame(nus_zx).to_numpy().T
    if nus_zx.shape[0] < 4:
        nus_zx = np.pad(nus_zx, ((0, 4-nus_zx.shape[0]), (0, 0)), 'constant', constant_values=np.nan)

    Kf_zx = pd.DataFrame(Kf_zx).to_numpy().T
    rhof_zx = pd.DataFrame(rhof_zx).to_numpy().T
    rhob_zx = pd.DataFrame(rhob_zx).to_numpy().T

    Km_zx = pd.DataFrame(Km_zx).to_numpy().T
    mum_zx = pd.DataFrame(mum_zx).to_numpy().T

    Vs_zx = pd.DataFrame(Vs_zx).to_numpy().T
    Vp_zx = pd.DataFrame(Vp_zx).to_numpy().T
    disp_db = pd.DataFrame(disp_db).to_numpy()
      


    # zs
    np.savetxt(f'{PATH_OUTPUT}/{model_id}/{profile}/z_zx.txt', z_zx, fmt='%.2f')

    # max_depth
    with open(f'{PATH_OUTPUT}/{model_id}/{profile}/max_z.txt', 'w') as f:
        f.write(f"{-params['data_params']['max_depth']}")

    # xs
    np.savetxt(f'{PATH_OUTPUT}/{model_id}/{profile}/xs.txt', xs.reshape(1, len(xs)), fmt='%.3f')

    #xmids
    np.savetxt(f'{PATH_OUTPUT}/{model_id}/{profile}/xmids.txt', xmids.reshape(1, len(xmids)), fmt='%.3f')

    # freqs
    np.savetxt(f'{PATH_OUTPUT}/{model_id}/{profile}/fs.txt', disp_db[0][0][:,0], fmt='%.2f')



    # GMs
    np.savetxt(f'{PATH_OUTPUT}/{model_id}/{profile}/soil_zx.txt', soil_zx, fmt='%s')

    # thicks
    np.savetxt(f'{PATH_OUTPUT}/{model_id}/{profile}/thick_zx.txt', thick_zx, fmt='%.2f')

    # Ns
    np.savetxt(f'{PATH_OUTPUT}/{model_id}/{profile}/N_zx.txt', N_zx, fmt='%.2f')

    # WT
    np.savetxt(f'{PATH_OUTPUT}/{model_id}/{profile}/WT_x.txt', WT_x, fmt='%.2f')



    # h
    np.savetxt(f'{PATH_OUTPUT}/{model_id}/{profile}/h_zx.txt', h_zx, fmt='%.2f')

    # Sw
    np.savetxt(f'{PATH_OUTPUT}/{model_id}/{profile}/Sw_zx.txt', Sw_zx, fmt='%.2f')

    # Swe
    np.savetxt(f'{PATH_OUTPUT}/{model_id}/{profile}/Swe_zx.txt', Swe_zx, fmt='%.2f')



    # mus
    np.savetxt(f'{PATH_OUTPUT}/{model_id}/{profile}/mus_zx.txt', mus_zx, fmt='%.2f')

    # Ks
    np.savetxt(f'{PATH_OUTPUT}/{model_id}/{profile}/Ks_zx.txt', Ks_zx, fmt='%.2f')

    # rhos
    np.savetxt(f'{PATH_OUTPUT}/{model_id}/{profile}/rhos_zx.txt', rhos_zx, fmt='%.2f')

    # nus
    np.savetxt(f'{PATH_OUTPUT}/{model_id}/{profile}/nus_zx.txt', nus_zx, fmt='%.2f')



    # Kf
    np.savetxt(f'{PATH_OUTPUT}/{model_id}/{profile}/Kf_zx.txt', Kf_zx, fmt='%.2f')

    # rhof
    np.savetxt(f'{PATH_OUTPUT}/{model_id}/{profile}/rhof_zx.txt', rhof_zx, fmt='%.2f')

    # rhob
    np.savetxt(f'{PATH_OUTPUT}/{model_id}/{profile}/rhob_zx.txt', rhob_zx, fmt='%.2f')



    # Km
    np.savetxt(f'{PATH_OUTPUT}/{model_id}/{profile}/Km_zx.txt', Km_zx, fmt='%.2f')

    # mum
    np.savetxt(f'{PATH_OUTPUT}/{model_id}/{profile}/mum_zx.txt', mum_zx, fmt='%.2f')



    # Vs
    np.savetxt(f'{PATH_OUTPUT}/{model_id}/{profile}/Vs_zx.txt', Vs_zx, fmt='%.2f')

    # Vp
    np.savetxt(f'{PATH_OUTPUT}/{model_id}/{profile}/Vp_zx.txt', Vp_zx, fmt='%.2f')

    # disp
    Vr_fx = []
    for disp in disp_db:
        Vr_fx.append(disp[0][:,1])
    Vr_fx = np.array(Vr_fx).T
    np.savetxt(f'{PATH_OUTPUT}/{model_id}/{profile}/Vr_fx.txt', Vr_fx, fmt='%.2f')
    ### -------------------------------------------------------------------------------------------





    ### RMS ---------------------------------------------------------------------------------------
    with open(f'{PATH_OUTPUT}/{model_id}/{profile}/DCs-rms.txt', 'w') as f:
        f.write(f'Model ID: {model_id}\n\n')
        f.write(f'Profile: {profile}\n\n')
        f.write('Average root mean square error on the dispersion curves\n\n')
        f.write(f'RMS: {np.mean(rms_x)} m/s\n')
        f.write(f'NRMS: {np.mean(rms_x)/(np.nanmax(Vr_fx)-np.nanmin(Vr_fx))*100} %\n')
    ### -------------------------------------------------------------------------------------------