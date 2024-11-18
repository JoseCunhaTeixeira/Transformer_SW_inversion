"""
Author : José CUNHA TEIXEIRA
License : SNCF Réseau, UMR 7619 METIS, Sorbonne Université
Date : April 30, 2024
"""





import os
import sys
import numpy as np
import colormaps as cmaps
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import interp2d
import datetime
import csv

from folders import PATH_INPUT, PATH_OUTPUT
import pandas as pd


plt.rcParams.update({'font.size': 12})
CM = 1/2.54





### PARAMS ----------------------------------------------------------------------------------------
model_id = '[202407170928]'
site = 'Grand_Est'
### -----------------------------------------------------------------------------------------------





### DAYS ------------------------------------------------------------------------------------------
date_start = datetime.datetime.strptime('2020-09-04 00:00:00', '%Y-%m-%d %H:%M:%S')
date_end = datetime.datetime.strptime('2023-09-03 00:00:00', '%Y-%m-%d %H:%M:%S')
days = []
day = datetime.timedelta(days=1)
date = date_start
i = 0
while date <= date_end:
    days.append(date)
    date += day
    i += 1
days = np.array(days)
N_samples = len(days)

print(f'\nFrom {date_start} to {date_end} -> {N_samples} days')
### -----------------------------------------------------------------------------------------------





### LOAD WATER TABLE DATA -------------------------------------------------------------------------
PZ1 = np.full((N_samples), np.nan)
file_path = f"{PATH_INPUT}/water_table/{site}/PZ1.csv"
with open(file_path, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        if row[0] != '' and row[1] != '':
            date = datetime.datetime.strptime(f"{row[0]}", "%Y-%m-%d %H:%M:%S")
            if date >= date_start and date <= date_end and date.hour == 0 and date.minute == 0 and date.second == 0:
                i = np.where(days == date)[0][0] - 2
                PZ1[i] = float(row[1])
PZ1 = abs(PZ1)

PZ2 = np.full((N_samples), np.nan)
file_path = f"{PATH_INPUT}/water_table/{site}/PZ2.csv"
with open(file_path, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        if row[0] != '' and row[1] != '':
            date = datetime.datetime.strptime(f"{row[0]}", "%Y-%m-%d %H:%M:%S")
            if date >= date_start and date <= date_end and date.hour == 0 and date.minute == 0 and date.second == 0:
                i = np.where(days == date)[0][0]
                PZ2[i] = float(row[1])
PZ2 = abs(PZ2)


pluvio = np.zeros((len(days)))
file_path = f"{PATH_INPUT}/rain/{site}/rain.csv"
with open(file_path, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        if row[0] != '' and row[1] != '':
            date = datetime.datetime.strptime(row[0], '%d/%m/%Y %H:%M:%S')
            if '00:00:00' in row[0]:
                if date >= date_start and date <= date_end:
                    i = np.where(days == date)[0][0]
                    pluvio[i] = float(row[2])
### -----------------------------------------------------------------------------------------------





### -----------------------------------------------------------------------------------------------
if not os.path.exists(f"{PATH_OUTPUT}/{model_id}/{site}/"):
    print(f'\033[1;31mERROR: Model {model_id} has no inversion data for site {site}.\033[0m')
    sys.exit()

if not os.path.exists(f"{PATH_OUTPUT}/{model_id}/{site}/results/water_table/"):
    os.makedirs(f"{PATH_OUTPUT}/{model_id}/{site}/results/water_table/")
### -----------------------------------------------------------------------------------------------





### -----------------------------------------------------------------------------------------------
dates = sorted(os.listdir(f"{PATH_OUTPUT}/{model_id}/{site}"))
if 'results' in dates:
    dates.remove('results')


profiles = ['P1', 'P2', 'P3', 'P4', 'P5']

xs = np.loadtxt(f'{PATH_OUTPUT}/{model_id}/{site}/{dates[-1]}/P1/xs.txt')
ys = np.arange(0, 23.75, 4.75)


piezo_db = np.full((len(days), len(profiles), len(xs)), np.nan)

start = None
for date in dates:
    day = datetime.datetime.strptime(date, '%Y-%m-%d')
    if day in days:
        i_date = np.where(days == day)[0][0]
        if start is None:
            start = i_date
        piezo_db[i_date, 0, :] = np.loadtxt(f"{PATH_OUTPUT}/{model_id}/{site}/{date}/P1/WT_x.txt")
        piezo_db[i_date, 1, :] = np.loadtxt(f"{PATH_OUTPUT}/{model_id}/{site}/{date}/P2/WT_x.txt")
        piezo_db[i_date, 2, :] = np.loadtxt(f"{PATH_OUTPUT}/{model_id}/{site}/{date}/P3/WT_x.txt")
        piezo_db[i_date, 3, :] = np.loadtxt(f"{PATH_OUTPUT}/{model_id}/{site}/{date}/P4/WT_x.txt")
        piezo_db[i_date, 4, :] = np.loadtxt(f"{PATH_OUTPUT}/{model_id}/{site}/{date}/P5/WT_x.txt")
end = i_date
        
        
tmp = piezo_db[start : end+1, :, :]
nans = np.isnan(tmp)

# Interpolate NaN values
for y in range(len(profiles)):
    for x in range(len(xs)):
        if np.any(nans[:, y, x]):
            df = pd.DataFrame({'days': days, 'values': tmp[:, y, x]})
            df['values'] = df['values'].interpolate(method='linear')
            tmp[:, y, x] = df['values'].values


# Temporal smoothing
if (len(tmp[:, 0, 0])/8) % 2 == 0:
    wl = len(tmp[:, 0, 0])/8 + 1
else:
    wl = len(tmp[:, 0, 0])/8
for y in range(len(profiles)):
    for x in range(len(xs)):
        print(wl)
        tmp[:, y, x] = savgol_filter(tmp[:, y, x], window_length=wl, polyorder=2, mode="nearest")


# Spatial smoothing over x
if (len(xs)/4) % 2 == 0:
    wl = len(xs)/4 + 1
else:
    wl = len(xs)/4
for i_date in range(len(days)):
    for y in range(len(profiles)):
        # print(wl)
        tmp[i_date, y, :] = savgol_filter(tmp[i_date, y, :], window_length=wl, polyorder=2, mode="nearest")
        
tmp[nans] = np.nan
piezo_db[start : end+1, :, :] = tmp
        
    



# plt.rcParams.update({'font.size': 20})
# for date_i, date in enumerate(days):
#     xgrid, ygrid = np.meshgrid(xs, ys)

#     zi = -piezo_db[date_i,:,:].T
#     xi = xs
#     yi = ys
#     xigrid = xgrid
#     yigrid = ygrid

#     f = interp2d(ys, xs, zi, kind='cubic')
#     xi, yi = np.linspace(xs[0], xs[-1], len(xs)*100), np.linspace(ys[0], ys[-1], len(ys)*100)
#     xigrid, yigrid = np.meshgrid(xi, yi)
#     zi = f(yi, xi)

#     fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(16,9), dpi=600)
#     ax.set_proj_type('persp')
#     ax.set_box_aspect((np.ptp(xs), np.ptp(ys)*3, 40))
#     ax.xaxis.labelpad = 20
#     ax.yaxis.labelpad = 10
#     ax.zaxis.labelpad = 10
#     ax.set_zticks([-7, -6, -5,-4, -3, -2, -1, 0])
#     ax.set_zticklabels(['', '-6', '', '-4', '', '-2', '', '0'])
#     ax.set_yticks(np.arange(0, 23.75, 4.75))
#     ax.set_xticks(np.arange(0, 123, 15))
#     ax.set_zlim([-7, 0])
#     ax.set_title(f'{date}')
#     surf = ax.plot_surface(xigrid, yigrid, zi.T, cmap=cmaps.deep, linewidth=0, antialiased=True, zorder=0, vmin=-5.5, vmax=-1.5)
#     fig.colorbar(surf, shrink=0.4, pad=0.1, aspect=15, label="WT level [m]")
    
    
#     drills_x = [20, 36, 104]
#     drills_y = [13, 8.9, 2.5]
#     piezos_x = [66, 96]
#     piezos_y = [3, 17]
    
#     # colors = ["orangered", "darksalmon", "tab:orange"]
#     # for i, (x,y) in enumerate(zip(drills_x, drills_y)):
#     #     ax.plot(x, y, 0.01, markerfacecolor=colors[i], markeredgecolor='black', markeredgewidth=0.7, marker='^', markersize=10, zorder=10)
#     #     ax.plot([x,x], [y,y], [-0.17, zi[np.where(xs>=x)[0][0], np.where(ys>=y)[0][0]]], color=colors[i], linewidth=2, zorder=10)

#     colors = ["royalblue", "deepskyblue"]
#     for i, (x,y) in enumerate(zip(piezos_x, piezos_y)):
#         ax.plot(x, y, 0.01, markerfacecolor=colors[i], markeredgecolor='black', markeredgewidth=0.7, marker='^', markersize=10, zorder=20)
#         ax.plot([x,x], [y,y], [-0.35, zi[np.where(xs>=x)[0][0], np.where(ys>=y)[0][0]]], color=colors[i], linewidth=2, zorder=20)

#     for y in ys:
#         if y == 0 or y == 9.5 or y == 19 :
#             for x in xs[:len(xs)-1]:
#                 ax.plot(x, y, 0.01, markerfacecolor='gold', markeredgecolor='darkblue', markeredgewidth=0.7, marker='o', markersize=5, zorder=30)
#         elif y == 4.75 or y == 14.25 :
#             for x in xs[1:]:
#                 ax.plot(x, y, 0.01, markerfacecolor='gold', markeredgecolor='darkblue', markeredgewidth=0.7, marker='o', markersize=5, zorder=30)


#     fig.savefig(f'{PATH_OUTPUT}/{model_id}/{site}/results/water_table/{date}_piezo_map.svg', dpi=600, bbox_inches='tight')




ticks = [day for day in days if day.day == 1]
labels = [tick.strftime('%Y') if tick.month == 7 else '' for tick in ticks]

plt.rcParams.update({'font.size': 8})

fig2, ax2 = plt.subplots(3, 1 ,figsize=(12.1*CM, 8*CM), dpi=600)

wd = 45
pluvio_smooth = np.zeros((len(pluvio)))
for i in range(wd//2, len(pluvio)-wd//2):
    pluvio_smooth[i] = np.nanmean(pluvio[i-wd//2:i+wd//2])
    
ax2[2].vlines([datetime.datetime.strptime('2021-01-01', '%Y-%m-%d'), datetime.datetime.strptime('2022-01-01', '%Y-%m-%d'), datetime.datetime.strptime('2023-01-01', '%Y-%m-%d')], 0, 50, colors='grey', linewidth=1, linestyle='--')
ax2[2].bar(days, pluvio, color='grey', width=1.2)
ax2[2].plot(days, pluvio_smooth, color='black', linewidth=1.2)
ax2[2].set_ylim([0, 20])
ax2[2].set_xticks(ticks)
ax2[2].set_xticklabels([label for label in labels], ha='center')
# ax2[2].set_yticks([0, 25, 50])
ax2[2].set_xlabel('Time')
ax2[2].set_ylabel('Rainfall [mm]')
ax2[2].tick_params(top=True, labeltop=False, bottom=True, labelbottom=True, left=True, labelleft=True, right=True, labelright=False)
ax2[2].grid(lw=0.2)
# ax2[2].legend(["Daily rainfall","45 day running mean rainfall"], fontsize=10, loc="lower left")

ax2[0].vlines([datetime.datetime.strptime('2021-01-01', '%Y-%m-%d'), datetime.datetime.strptime('2022-01-01', '%Y-%m-%d'), datetime.datetime.strptime('2023-01-01', '%Y-%m-%d')], -5, -1, colors='grey', linewidth=1, linestyle='--')
ax2[0].plot(days, -PZ1, color='royalblue', linewidth=0.9)
# ax2[0].set_title("${PZ_1}$", weight='bold')
ax2[0].plot(days, -piezo_db[:, 0, np.where(xs == 63.0)[0][0]], color='orange', linewidth=1.2, markersize=3)
ax2[0].set_ylim([-4, -1])
ax2[0].set_xticks(ticks)
ax2[0].set_xticklabels([label for label in labels], ha='center')
ax2[0].set_yticks([-1, -2, -3, -4])
ax2[0].set_xlabel('')
ax2[0].set_ylabel('WT level [m]')
ax2[0].tick_params(top=True, labeltop=False, bottom=True, labelbottom=False, left=True, labelleft=True, right=True, labelright=False)
ax2[0].grid(lw=0.2)
# ax2[0].legend(["Observed WT","Inferred WT"], fontsize=10, loc="lower left")

ax2[1].vlines([datetime.datetime.strptime('2021-01-01', '%Y-%m-%d'), datetime.datetime.strptime('2022-01-01', '%Y-%m-%d'), datetime.datetime.strptime('2023-01-01', '%Y-%m-%d')], -5, -1, colors='grey', linewidth=1, linestyle='--')
ax2[1].plot(days, -PZ2, color='royalblue', linewidth=0.9)
# ax2[1].set_title("${PZ_2}$", weight='bold')
ax2[1].plot(days, -piezo_db[:, 3, np.where(xs == 90.0)[0][0]], color='orange', linewidth=1.2, markersize=3)
ax2[1].set_ylim([-4, -1])
ax2[1].set_xticks(ticks)
ax2[1].set_xticklabels([label for label in labels], ha='center')
ax2[1].set_yticks([-1, -2, -3, -4])
ax2[1].set_xlabel('')
ax2[1].set_ylabel('WT level [m]')
ax2[1].tick_params(top=True, labeltop=False, bottom=True, labelbottom=False, left=True, labelleft=True, right=True, labelright=False)
ax2[1].grid(lw=0.2)
# ax2[1].legend(["Observed","Inferred"], fontsize=10, loc="lower left")

fig2.savefig(f'{PATH_OUTPUT}/{model_id}/{site}/results/water_table/PZ1_PZ2.svg', dpi=600, bbox_inches='tight')


