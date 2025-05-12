from read_bin import read_agilent_binary
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from pprint import pprint
from collections import Counter  # Per contare le occorrenze dei valori
import numpy as np  # Per operazioni numeriche
import os
import scipy.optimize as opt
from sklearn.metrics import r2_score
from matplotlib import cm

def fattoriale(n):
    if n == 1:
        return 1
    else:
        return n * fattoriale(n - 1)

def retta(x, m, q):
    return x * m + q

def gaussian(x, a, mean, sigma):
    return a * np.exp(-((x - mean) ** 2 / (2 * sigma ** 2)))

def poisson(x, a, mean):
    return a * np.exp(-mean) * mean ** x / fattoriale(x)

var = str(input('Quale file vuoi caricare: '))
fl = var + '.bin'

# variabili
cwd = os.getcwd()

# load data
os.chdir(cwd + '/Dati')
data = read_agilent_binary(fl, include_time_vector=True)
pprint(data)
x, y = data['channel_2']['x_data'], data['channel_2']['y_data']

# plot data
popt, pcov = opt.curve_fit(retta, x, y)
perr = np.sqrt(np.diag(pcov))
baseline = retta(x, *popt)
y_min_mask = y < -0.0035
x_min, y_min = x[y_min_mask], y[y_min_mask]

# Trova i massimi locali (in valore assoluto)
abs_y = np.abs(y_min)
mask = (abs_y[1:-1] >= abs_y[:-2]) & (abs_y[1:-1] >= abs_y[2:])
x_loc_max = x_min[1:-1][mask]
y_loc_max = y_min[1:-1][mask]

# Creazione del grafico
fig = plt.figure()
ax1 = fig.add_subplot(311)
plt.grid(linestyle='--')
ax1.plot(x, y, label='Data')
ax1.plot(x, baseline, label='Baseline')
ax1.scatter(x_loc_max, y_loc_max, marker='x', color='red')
plt.xlabel('Time (s)')
plt.ylabel('Signals')
plt.legend()

ax2 = fig.add_subplot(312)
plt.grid(linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Peaks')

window_size = 6
width = []
x_idx_map = np.searchsorted(x, x_loc_max)
lines = []
peak_points = []

for i, idx in enumerate(x_idx_map):
    start = max(0, idx - window_size)
    end = min(len(x), idx + window_size)

    if end <= start:
        continue

    y_segment = y[start:end]
    base_segment = baseline[start:end]

    left = y[start:idx]
    left_base = baseline[start:idx]
    right = y[idx:end]
    right_base = baseline[idx:end]

    if len(left) == 0 or len(right) == 0:
        continue

    idx_start = np.argmin(np.abs(left - left_base))
    idx_end = np.argmin(np.abs(right - right_base))

    width.append(np.abs(x[idx + idx_end] - x[start + idx_start]) * 13560 * 6.28 / 360)

    # Costruzione segmenti per LineCollection
    segment_x = x[start:end] - x[start]
    segment_y = y[start:end]
    lines.append(np.column_stack([segment_x, segment_y]))
    peak_points.append([x[idx] - x[start], y[idx]])

# Genera una lista di colori da una colormap
cmap = cm.get_cmap('jet', len(lines))
colors = [cmap(i) for i in range(len(lines))]

# Disegna tutte le finestre con LineCollection
line_collection = LineCollection(lines, colors=colors, linewidths=0.8)
ax2.add_collection(line_collection)

# Aggiungi i picchi come scatter
peak_points = np.array(peak_points)
if peak_points.size > 0:
    ax2.scatter(peak_points[:, 0], peak_points[:, 1], color='red', marker='x')

ax3 = fig.add_subplot(313)
plt.grid(linestyle='--')
plt.xlabel('Counts')
plt.ylabel('Time (s)')
# Calcolo l'istogramma
n, bins = np.histogram(width, 10)
print(n, bins)
ax3.hist(width, 10, orientation='horizontal')

plt.tight_layout()
plt.show()
