import numpy as np


DCs1 = np.loadtxt('/home/jteixeira/Documents/PhD_Monitoring/Work/Processing/Tools/Santiludo_DL_Inversion/input/training_data6/DCs_part1.txt', dtype=float)
GMS1 = np.loadtxt('/home/jteixeira/Documents/PhD_Monitoring/Work/Processing/Tools/Santiludo_DL_Inversion/input/training_data6/GMs_part1.txt', dtype=str)
THKs1 = np.loadtxt('/home/jteixeira/Documents/PhD_Monitoring/Work/Processing/Tools/Santiludo_DL_Inversion/input/training_data6/THKs_part1.txt', dtype=str)
WTs1 = np.loadtxt('/home/jteixeira/Documents/PhD_Monitoring/Work/Processing/Tools/Santiludo_DL_Inversion/input/training_data6/WTs_part1.txt', dtype=str)

DCs2 = np.loadtxt('/home/jteixeira/Documents/PhD_Monitoring/Work/Processing/Tools/Santiludo_DL_Inversion/input/training_data6/DCs_part2.txt', dtype=float)
GMS2 = np.loadtxt('/home/jteixeira/Documents/PhD_Monitoring/Work/Processing/Tools/Santiludo_DL_Inversion/input/training_data6/GMs_part2.txt', dtype=str)
THKs2 = np.loadtxt('/home/jteixeira/Documents/PhD_Monitoring/Work/Processing/Tools/Santiludo_DL_Inversion/input/training_data6/THKs_part2.txt', dtype=str)
WTs2 = np.loadtxt('/home/jteixeira/Documents/PhD_Monitoring/Work/Processing/Tools/Santiludo_DL_Inversion/input/training_data6/WTs_part2.txt', dtype=str)

DCs = np.concatenate((DCs1, DCs2), axis=0)
GMS = np.concatenate((GMS1, GMS2), axis=0)
THKs = np.concatenate((THKs1, THKs2), axis=0)
WTs = np.concatenate((WTs1, WTs2), axis=0)

print(DCs.shape)
print(GMS.shape)
print(THKs.shape)
print(WTs.shape)

np.savetxt('/home/jteixeira/Documents/PhD_Monitoring/Work/Processing/Tools/Santiludo_DL_Inversion/input/training_data6/DCs.txt', DCs, fmt='%.3f')
np.savetxt('/home/jteixeira/Documents/PhD_Monitoring/Work/Processing/Tools/Santiludo_DL_Inversion/input/training_data6/GMs.txt', GMS, fmt='%s')
np.savetxt('/home/jteixeira/Documents/PhD_Monitoring/Work/Processing/Tools/Santiludo_DL_Inversion/input/training_data6/THKs.txt', THKs, fmt='%s')
np.savetxt('/home/jteixeira/Documents/PhD_Monitoring/Work/Processing/Tools/Santiludo_DL_Inversion/input/training_data6/WTs.txt', WTs, fmt='%s')