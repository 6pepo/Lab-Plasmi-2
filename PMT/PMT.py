
from read_bin import read_agilent_binary
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import numpy as np

# Lettura dati
filename = './PMT/Data/200verde255.bin'
data = read_agilent_binary(filename, include_time_vector=True)

x = data['channel_2']['x_data']
y = data['channel_2']['y_data']
y_neg = np.where(y<0, y, 0)
y_pos = np.where(y>0, y, 0)

height_tresh = 0.007
width_scale = 0.000000002

peaks_neg, _ = find_peaks(abs(y_neg), height=height_tresh)
peaks_pos, _ = find_peaks(abs(y_pos), height=height_tresh)

# Numero di picchi
print('Numero di picchi negativi:', len(peaks_neg))
print('Numero di picchi positivi:', len(peaks_pos))

# Intensità
intensity_neg = np.array(y[peaks_neg])
avg_intensity_neg = np.mean(abs(intensity_neg))
intensity_pos = np.array(y[peaks_pos])
avg_intensity_pos = np.mean(abs(intensity_pos))
print('Intensità media dei picchi negativi:', avg_intensity_neg)
print('Intensità media dei picchi positivi:', avg_intensity_pos)


fig, ax = plt.subplots()
ax.plot(x, y, zorder=0)
ax.scatter(x[peaks_neg], y[peaks_neg], color='yellow', zorder=1)
ax.scatter(x[peaks_pos], y[peaks_pos], color='red', zorder=1)

plt.show()

