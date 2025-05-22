import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import os

def gaussian(x,a,mean,sigma):
    return a*np.exp(-((x-mean)**2/(sigma**2))/2)

def line(x,m,q):
    return m*x+q

file = 'S4.mca'

#load data
data = np.genfromtxt(file, delimiter=None, skip_header=12, skip_footer=72, unpack=True)

data_info = np.genfromtxt(file, delimiter=';', dtype=str, skip_header=4110, skip_footer=17, unpack=True)
shaping_time = data_info[0,2]
shaping_time = float(shaping_time.split('=')[1].strip())
gain = data_info[0,4]
gain = float(gain.split('=')[1].strip())
print(f'Shaping time: {shaping_time} mus\tgain: {gain}\tConteggi: {np.sum(data)}')

peaks,info = find_peaks(data, threshold=2, prominence=3, width=2, distance=None)
# print(info)

alpha_peak = 5.9 #KeV
beta_peak = 6.5 #KeV
energy_peaks = [alpha_peak, beta_peak]

fig, ax = plt.subplots()
ax.plot(data, color='blue', label='Conteggi')
ax.plot(peaks, data[peaks], 'rx', label='peaks')
ax.set_xlim(0,1000)
ax.legend()

fig1, ax1 = plt.subplots()
popt,pcov = curve_fit(line, peaks, energy_peaks) #, maxfev=10000)
perr = np.sqrt(np.diag(pcov))

m = popt[0]
err_m = perr[0]
q = popt[1]
err_q = perr[1]

ax1.plot(peaks, energy_peaks, 'rx', linestyle='dashed', label=f'm:{m:.4f}, q: {q:.4f}')
ax1.set_xlabel('Conteggi')
ax1.set_ylabel('Energia (KeV)')
ax1.legend()
print(f'm:{m:.4f}, q: {q:.4f}')

max_channel = 4096
print(f'E_max: {m*max_channel+q:4f} KeV')

plt.show()