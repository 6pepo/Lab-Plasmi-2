import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter
import os

def gaussian(x,a,mean,sigma):
    return a*np.exp(-((x-mean)**2/(sigma**2))/2)

def line(x,m,q):
    return m*x+q

file = f'Dati Bromuro/fondo1_1000V/RAW/CH7@DT5730SB_2289_EspectrumR_fondo1_1000V_20250528_094644.txt3'

#load data
data = np.genfromtxt(file, delimiter=' ', skip_header=2, usecols=(0,1), unpack=True)
channels = data[0]
counts = data[1]


fig, ax = plt.subplots()

mask_ec = (channels >= 1080) & (channels <= 1180) #cattura elettronica
ax.plot(channels,counts, label='Data')
ax.set_xlabel('Channels')
ax.set_ylabel('Counts')

mask1 = (channels >= 1080) & (channels <= 1120)
mask2 = (channels > 1120) & (channels <= 1180)

par1 = [1300., 1114., 5.]
popt1, pcov1 = curve_fit(gaussian, channels[mask1], counts[mask1], p0=par1)

par2 = [2100., 1133., 5.]
popt2, pcov2 = curve_fit(gaussian, channels[mask2], counts[mask2], p0=par2)

x = np.linspace(np.min(channels[mask_ec]), np.max(channels[mask_ec]), 100)
ax.plot(x, gaussian(x,*popt1), 'r--', label=f'Peak-1440: A = {popt1[0]:.2f}, $\mu$ = {popt1[1]:.2f}, $\sigma$ = {popt1[2]:.2f}')
ax.plot(x, gaussian(x,*popt2), 'g--', label=f'Peak-1472: A = {popt2[0]:.2f}, $\mu$ = {popt2[1]:.2f}, $\sigma$ = {popt2[2]:.2f}')

mask_beta = (channels >= 600) & (channels <= 850)
filt_counts = savgol_filter(counts[mask_beta],20,1)
ax.plot(channels[mask_beta], filt_counts, 'y--', label='Sagov')

d2 = np.gradient(np.gradient(filt_counts))
i = 0
index_flex = 0
while True:
    if np.sign(d2[i]) * np.sign(d2[i+1]) < 0:
        index_flex = i
        break
    else:
        i = i+1

index_flex = 600 + index_flex
ax.plot(channels[mask_beta], d2, label='Derivata')

ax.grid(True)
ax.legend()

peaks0 = 789 #kev
peak1 = 1440 #kev
peak2 = 1472 #kev

energy_peaks = [peaks0,peak1,peak2]
channels_peaks = [index_flex,popt1[1], popt2[1]]

fig1, ax1 = plt.subplots()
popt_energy, pcov_energy = curve_fit(line, channels_peaks, energy_peaks)

ax1.plot(channels_peaks, energy_peaks, 'rx', label='Peaks')
x = np.linspace(np.min(channels_peaks), np.max(channels_peaks), 100)
ax1.plot(x, line(x,*popt_energy), 'b--', label=f'm={popt_energy[0]:.2f}, q={popt_energy[1]:.2f}')
ax1.set_xlabel('Channels')
ax1.set_ylabel('Energy [keV]')
ax1.grid(True)
ax1.legend()

# Conversione post calibrazione
energy = channels * popt_energy[0] + popt_energy[1]
fig2, ax2 = plt.subplots()
ax2.plot(energy, counts, 'rx', label='Data')
plt.title('Conversione in energia))
ax2.set_xlabel('Energy [keV]')
ax2.set_ylabel('Counts')
ax2.grid('--')
ax2.legend()

plt.show()
