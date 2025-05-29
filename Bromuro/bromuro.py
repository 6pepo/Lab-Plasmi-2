import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter
import os

def gaussian(x,a,mean,sigma):
    return a*np.exp(-((x-mean)**2/(sigma**2))/2)

def line(x,m,q):
    return m*x+q

def find_FWHM(a):
    x_idx = [i for i, x in enumerate(a) if x in a] # Indicizza il vettore
    peak = max(a) # Trova il valore del picco
    idx_peak = [i for i, x in enumerate(a) if x == max(a)] 
    idx_FWHM_min = [idx for idx in x_idx if idx < idx_peak[0]]
    idx_FWHM_max = [idx for idx in x_idx if idx > idx_peak[0]]
    FWHM_min = min(idx_FWHM_min, key=lambda idx: abs(a[idx] - peak/2))
    FWHM_max = min(idx_FWHM_max, key=lambda idx: abs(a[idx] - peak/2))
    return FWHM_max-FWHM_min

file_fondo = r'Dati Bromuro/fondo1_1000V/RAW/CH7@DT5730SB_2289_EspectrumR_fondo1_1000V_20250528_094644.txt3'
file_cesio = r'Dati Bromuro/cesio1_1000V/RAW/CH7@DT5730SB_2289_EspectrumR_cesio1_1000V_20250528_102235.txt3'
file_sodio = r'Dati Bromuro/sodio22_1000V/RAW/CH7@DT5730SB_2289_EspectrumR_sodio22_1000V_20250528_104413.txt3'
file_ignoto = r'Dati Bromuro/ignota_1000V/RAW/CH7@DT5730SB_2289_EspectrumR_ignota_1000V_20250528_110606.txt3'

#bromuro di lantanio
peak0_la = 789 #kev   #beta
peak1_la = 1440 #kev   #cattura elettronica 
peak2_la = 1472 #kev   #cattura elettronica

#cesio
peak_ce = 662 #kev

#sodio
peak_na = 511 #kev

energy_peaks = [peak0_la, peak1_la, peak2_la, peak_ce, peak_na]
channels_peaks = []

#file fondo (Bromuro di Lantanio)
channels, counts = np.genfromtxt(file_fondo, delimiter=' ', skip_header=2, usecols=(0,1), unpack=True)

fig, ax = plt.subplots()

mask_ec = (channels >= 1080) & (channels <= 1180) #cattura elettronica
ax.plot(channels,counts, label='Data')
ax.set_xlabel('Channels')
ax.set_ylabel('Counts')
ax.set_title('$LaBr_{3}$ Spectrum')

mask1 = (channels >= 1080) & (channels <= 1120)
mask2 = (channels > 1120) & (channels <= 1180)

par1 = [1300., 1114., 5.]
popt1, pcov1 = curve_fit(gaussian, channels[mask1], counts[mask1], p0=par1)

par2 = [2100., 1133., 5.]
popt2, pcov2 = curve_fit(gaussian, channels[mask2], counts[mask2], p0=par2)

x = np.linspace(np.min(channels[mask_ec]), np.max(channels[mask_ec]), 100)
ax.plot(x, gaussian(x,*popt1), 'r--', label=f'EC Peak-1440: A = {popt1[0]:.2f}, $\mu$ = {popt1[1]:.2f}, $\sigma$ = {popt1[2]:.2f}')
ax.plot(x, gaussian(x,*popt2), 'g--', label=f'EC Peak-1472: A = {popt2[0]:.2f}, $\mu$ = {popt2[1]:.2f}, $\sigma$ = {popt2[2]:.2f}')

print('Bromuro 1:', find_FWHM(gaussian(x,*popt1))/popt1[1] * 100)
print('Bromuro 2:', find_FWHM(gaussian(x,*popt2))/popt2[1] * 100)

mask_beta = (channels >= 600) & (channels <= 850)
filt_counts = savgol_filter(counts[mask_beta],20,1)

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
ax.vlines(index_flex, 0, counts[index_flex-1]+10, colors='m', label=f'Beta Peak: {index_flex}')

ax.grid(True)
ax.legend()

channels_peaks.extend([index_flex,popt1[1], popt2[1]])


#file cesio
channels, counts = np.genfromtxt(file_cesio, delimiter=' ', skip_header=2, usecols=(0,1), unpack=True)

fig1, ax1 = plt.subplots()

ax1.plot(channels,counts, label='Data')
ax1.set_xlabel('Channels')
ax1.set_ylabel('Counts')
ax1.set_title('Ce-133 Spectrum')

mask = (channels >= 480) & (channels <= 540)

par = [50000., 510., 10]
popt, pcov = curve_fit(gaussian, channels[mask], counts[mask], p0=par)

x = np.linspace(np.min(channels[mask]), np.max(channels[mask]), 100)
ax1.plot(x, gaussian(x,*popt), 'r--', label=f'Peak: A = {popt[0]:.2f}, $\mu$ = {popt[1]:.2f}, $\sigma$ = {popt[2]:.2f}')

print('Cesio :', find_FWHM(gaussian(x,*popt))/popt[1] * 100)

ax1.grid(True)
ax1.legend()

channels_peaks.extend([popt[1]])


#file sodio
channels, counts = np.genfromtxt(file_sodio, delimiter=' ', skip_header=2, usecols=(0,1), unpack=True)

fig2, ax2 = plt.subplots()

ax2.plot(channels,counts, label='Data')
ax2.set_xlabel('Channels')
ax2.set_ylabel('Counts')
ax2.set_title('Na-22 Spectrum')

mask = (channels >= 380) & (channels <= 410)

par = [5000., 395., 10]
popt, pcov = curve_fit(gaussian, channels[mask], counts[mask], p0=par)

x = np.linspace(np.min(channels[mask]), np.max(channels[mask]), 100)
ax2.plot(x, gaussian(x,*popt), 'r--', label=f'Peak: A = {popt[0]:.2f}, $\mu$ = {popt[1]:.2f}, $\sigma$ = {popt[2]:.2f}')

print('Sodio :', find_FWHM(gaussian(x,*popt))/popt[1] * 100)

ax2.grid(True)
ax2.legend()

channels_peaks.extend([popt[1]])


fig_ene, ax_ene = plt.subplots()
popt_energy, pcov_energy = curve_fit(line, channels_peaks, energy_peaks)

ax_ene.plot(channels_peaks, energy_peaks, 'rx', label='Peaks')
x = np.linspace(np.min(channels_peaks), np.max(channels_peaks), 100)
ax_ene.plot(x, line(x,*popt_energy), 'b--', label=f'm={popt_energy[0]:.2f}, q={popt_energy[1]:.2f}')
ax_ene.set_xlabel('Channels')
ax_ene.set_ylabel('Energy [keV]')
ax_ene.grid(True)
ax_ene.legend()

m = popt_energy[0]
q = popt_energy[1]

#sostanza ignota
channels, counts = np.genfromtxt(file_ignoto, delimiter=' ', skip_header=2, usecols=(0,1), unpack=True)
energy = channels*m+q

fig3, ax3 = plt.subplots()

ax3.plot(energy,counts, label='Data')
ax3.set_xlabel('Energy [keV]')
ax3.set_ylabel('Counts')
ax3.set_title('The Substance Spectrum')
ax3.grid(True)
ax3.legend()

peaks, info = find_peaks(counts, height=2e3, prominence=2e3)
peaks = np.delete(peaks, [0,-1])
peaks_markers = ['r*', 'g^']

for i,p in enumerate(peaks):
    ax3.plot(energy[p], counts[p],peaks_markers[i], label=f'Peak Energy: {energy[p]:.2f} keV')

ax3.grid(True)
ax3.legend()

plt.show()
