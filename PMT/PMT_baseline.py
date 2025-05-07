
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

baseline_x = []
baseline_y = []
for i in range(0, len(x), 100):
# for i in range(0, 1000, 10):

    # indice che avrebbe il float x[i] se fosse un elemento del vettore x[peaks]
    idx = np.searchsorted(x[peaks], x[i])
    # print(idx)

    # Gestisci i casi limite
    if idx == 0 and x[i] < x[peaks[0]]-width_scale:
        baseline_x.append(x[i])
        baseline_y.append(y[i])
        # print('a')
    elif idx == len(x[peaks]) and x[i] > x[peaks[-1]]+width_scale:
        baseline_x.append(x[i])
        baseline_y.append(y[i])
        # print('b')
    elif x[i] < x[peaks[idx]]-width_scale and x[i] > x[peaks[idx-1]]+width_scale:
        baseline_x.append(x[i])
        baseline_y.append(y[i])
        # print('c')
    # else:
    #     print('NO')

# print(x)
# print(len(x), len(y))
# print(peaks)
# print(x[peaks[0]])

# print(x[peaks[-1]])

# print(len(baseline_x), len(baseline_y))
# print(baseline_x)

# Linear regression
m, b = np.polyfit(baseline_x, baseline_y, 1)
print('Linear regression: m={}, b={}'.format(m, b))
# Valori ottenuti con step=10:
# Linear regression: m=0.050192949958465474, b=0.0011869688474004795


fig, ax = plt.subplots()
ax.plot(x, y, zorder=0)
ax.scatter(baseline_x, baseline_y, color='red', zorder=1)

#ax.hlines(0, x[peaks[0]]-width_scale, x[peaks[0]]+width_scale, colors='red')

# Plot regressione lineare
xreg = np.linspace(np.min(baseline_x), np.max(baseline_x), 100)
yreg = m*xreg + b
ax.plot(xreg, yreg, label="fit", color='orange', zorder=2)


plt.show()

