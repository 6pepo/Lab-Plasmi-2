import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
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

mask = (channels >= 1080) & (channels <= 1180)
ax.plot(channels[mask],counts[mask], label='Data')
ax.grid(True)

mask1 = (channels >= 1080) & (channels <= 1120)
mask2 = (channels > 1120) & (channels <= 1180)

par1 = [1300., 1114., 5.]
popt1, pcov1 = curve_fit(gaussian, channels[mask1], counts[mask1], p0=par1)

par2 = [2100., 1133., 5.]
popt2, pcov2 = curve_fit(gaussian, channels[mask2], counts[mask2], p0=par2)

x = np.linspace(np.min(channels[mask]), np.max(channels[mask]), 100)
ax.plot(x, gaussian(x,*popt1), 'r--', label=f'Peak-1440: A = {popt1[0]:.2f}, $\mu$ = {popt1[1]:.2f}, $\sigma$ = {popt1[2]:.2f}')
ax.plot(x, gaussian(x,*popt2), 'g--', label=f'Peak-1472: A = {popt2[0]:.2f}, $\mu$ = {popt2[1]:.2f}, $\sigma$ = {popt2[2]:.2f}')

ax.legend()

plt.show()