"""
Author : José CUNHA TEIXEIRA
License : SNCF Réseau, UMR 7619 METIS, Sorbonne Université
Date : April 30, 2024
"""





import numpy as np
import os
import pandas as pd
from datetime import datetime
import re

from folders import PATH_OUTPUT

### PARAMS ----------------------------------------------------------------------------------------
model_id = '[202407170928]'
periods = [['2022-07-01', '2022-07-31']]
site = 'Grand_Est'
### -----------------------------------------------------------------------------------------------


RMSs = []
NRMSs = []

rms_pattern = r"RMS: ([\d.]+) m/s"
nrms_pattern = r"NRMS: ([\d.]+) %"

existing_results = sorted(os.listdir(f'{PATH_OUTPUT}/{model_id}/{site}/'))
if 'results' in existing_results:
    existing_results.remove('results')
        
profiles = ['P1', 'P2', 'P3', 'P4', 'P5']
Nprofiles = len(profiles)

for period in periods:
    dates = pd.date_range(start=datetime.strptime(period[0], '%Y-%m-%d'), end=datetime.strptime(period[1], '%Y-%m-%d'), freq='D')
    dates = dates[dates.isin(existing_results)]
    N_dates = len(dates)
    
    for date in dates:
        
        for profile in profiles:
            path = f"{PATH_OUTPUT}/{model_id}/{site}/{date.strftime('%Y-%m-%d')}/{profile}/DCs-rms.txt"
            with open (path, 'r') as file:
                string = file.read()
                
                # Extract values using regular expressions
                rms_match = re.search(rms_pattern, string)
                nrms_match = re.search(nrms_pattern, string)

                if rms_match:
                    rms = float(rms_match.group(1))
                else:
                    rms = np.nan

                if nrms_match:
                    nrms = float(nrms_match.group(1))
                else:
                    nrms = np.nan
        
            RMSs.append(rms)
            NRMSs.append(nrms)


print(f'RMS = {np.median(RMSs)}')
print(f'NRMS = {np.median(NRMSs)}')