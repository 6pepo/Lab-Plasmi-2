
from read_bin import read_agilent_binary
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import numpy as np

# Lettura dati
filename = './PMT/Data/200verde255.bin'
data = read_agilent_binary(filename, include_time_vector=True)

x = data['channel_2']['x_data']
y = data['channel_2']['y_data']

height_tresh = 0.007
width_scale = 0.000000002

peaks, _ = find_peaks(abs(y), height=height_tresh)

# Numero di picchi
print('Numero di picchi:', len(peaks))
# DA AGGIUNGRE: numero di picchi NEGATIVI

# Intensità
intensity = np.array(y[peaks])
avg_intensity = np.mean(abs(intensity)) # DA SISTEMARE, STO PRENDENDO SIA NEG CHE POS!
print('Intensità media dei picchi:', avg_intensity)


# fig, ax = plt.subplots()
# ax.plot(x, y, zorder=0)

# plt.show()

