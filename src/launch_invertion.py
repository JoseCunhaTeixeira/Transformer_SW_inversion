"""
Author : José CUNHA TEIXEIRA
License : SNCF Réseau, UMR 7619 METIS
Date : April 30, 2024
"""





import subprocess





model = ['[202405241321]']


profiles = [
            'PJ',
            'P1_LW',
            'P1_HW',
            'P2_LW',
            'P2_HW',
            'P3_LW',
            'P3_HW',
            'P4_LW',
            'P4_HW',
            'P5_LW',
            'P5_HW'
            # 'PSR2',
            ]


dxs = [
       1.5,
       3.0,
       3.0,
       3.0,
       3.0,
       3.0,
       3.0, 
       3.0,
       3.0,
       3.0,
       3.0,
      #  0.4,
       ]


for model_name in model:
    for profile, dx in zip(profiles, dxs):
        print(model_name, profile, dx)
        subprocess.run(['python3.10', '/home/jteixeira/Documents/PhD_Monitoring/Work/Processing/Tools/Santiludo_DL_Inversion/src/run_invertion.py', model_name, profile, str(dx)])