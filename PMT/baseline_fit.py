import numpy as np
import matplotlib.pyplot as plt

from read_bin import read_agilent_binary
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter

def gaussian(x,a,mean,sigma):
    return a*np.exp(-(x-mean)/(sigma**2))/2

def h_line(x,c):
    return np.ones(len(x))*c

def get_baseline(time,signal,window,poly):
    filt_sign = savgol_filter(signal, window, poly)

    popt,pcov = curve_fit(h_line,time,filt_sign,0,maxfev=10000)

    return popt[0]

def get_intersections(signal,baseline,peaks,tolerance):
    intersections = []
    for p in peaks:
        i_left = p
        found_left = False
        while found_left==False:
            if np.abs(signal[i_left] - baseline) < tolerance:
                left = i_left
                found_left = True
            else:
                i_left -= 1
        i_right = p
        found_right = False
        while found_right == False:
            if np.abs(signal[i_right] - baseline) < tolerance:
                right = i_right
                found_right = True
            else:
                i_right += 1
        
        intersections.append([left,right])

    return np.asarray(intersections)


file_path = 'Dati/50verde5.bin'

data = read_agilent_binary(file_path, include_time_vector=True)

time = data['channel_2']['x_data']
signal = data['channel_2']['y_data']

peaks,_ = find_peaks(-signal, height=5e-3, prominence=1e-3)
print(len(peaks))

b_line = get_baseline(time,signal, 1000, 5)

intersections = get_intersections(signal, b_line, peaks, tolerance=1e-3)

fig, ax = plt.subplots()
ax.plot(time, signal)
ax.plot(time[peaks], signal[peaks], 'rx', label='peaks')
ax.plot(time, h_line(time,b_line), 'g--', label=f'baseline:{b_line}')

for i in range(len(intersections)):
    ax.plot(time[intersections[i,0]], signal[intersections[i,0]], 'yD')
    ax.plot(time[intersections[i,1]], signal[intersections[i,1]], 'mp')

ax.plot([],[], 'yD', label='left')
ax.plot([],[], 'mp', label='right')

# for i, peak in enumerate(peaks):
#     #interval around the peak
#     left = intersections[i,0]
#     right = intersections[i,1]
#     time_local = time[left:right]
#     signal_local = signal[left:right]

#     #initial fit parameters estimate
#     par = [signal[peak], time[peak], 0.25]

#     #fit the peak with a gaussian
#     popt, pcov = curve_fit(gaussian, time_local, signal_local, par, maxfev=8000)
#     # print('\np-->', peak , '\tm-->', m[peak], '\tp-->', p[peak],'\tAmpiezza-->', popt[0], '\tMedia-->', popt[1], '\tSigma-->', popt[2])

#     #increase the interval for a better drawing
#     pad = 100
#     time_local = time[left-pad:right+pad]

#     #plot the gaussian
#     plt.plot(time_local, gaussian(time_local, *popt), 'r--') 

plt.legend()
plt.grid()
plt.show()