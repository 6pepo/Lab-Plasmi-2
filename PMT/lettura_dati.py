from read_bin import read_agilent_binary
import matplotlib.pyplot as plt
from pprint import pprint
from collections import Counter  # Per contare le occorrenze dei valori
import numpy as np  # Per operazioni numeriche
import os
import scipy.optimize as opt
from sklearn.metrics import r2_score

def fattoriale(n):
    if n==1:
        return 1
    else:
        return n*fattoriale(n-1)
    
def retta(x,m,q):
    return x*m+q
def gaussian(x,a,mean,sigma):
    return a*np.exp(-((x-mean)**2/(2*sigma**2)))
def poisson(x,a,mean):
    return a*np.exp(-mean)*mean**x/fattoriale(x)

var=str(input('Quale file vuoi caricare: '))
fl=var+'.bin'

#variabili
cwd=os.getcwd()

#load data
os.chdir(cwd+'/Dati')
data = read_agilent_binary(fl, include_time_vector=True)
pprint(data)
x, y = data['channel_2']['x_data'], data['channel_2']['y_data']

#plot data
popt, pcov = opt.curve_fit(retta, x, y)
perr=np.sqrt(np.diag(pcov))
#baseline=retta(x, *popt)
y_min_mask = y < -0.0025
x_min, y_min = x[y_min_mask], y[y_min_mask]

# Trova i massimi locali (in valore assoluto)
# Confronta elemento i con i-1 e i+1
abs_y = np.abs(y_min)
mask = (abs_y[1:-1] >= abs_y[:-2]) & (abs_y[1:-1] >= abs_y[2:])

# Aggiusta gli indici perché il confronto salta il primo e l'ultimo
x_loc_max = x_min[1:-1][mask]
y_loc_max = y_min[1:-1][mask]

# Creazione del grafico
fig=plt.figure()
ax1=fig.add_subplot(211)
plt.grid(linestyle='--')
ax1.plot(x, y, label='data')
ax1.plot(x, retta(x, *popt), label='baseline')
#ax1.scatter(x_min, y_min, marker='x')
ax1.scatter(x_loc_max, y_loc_max, marker='x', color='red')
plt.xlabel('Time (s)')
plt.ylabel('Counts')

ax2=fig.add_subplot(212)
plt.grid(linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Counts')
window_size=5
for xp, yp in zip(x_loc_max, y_loc_max):
    idx = np.argmin(np.abs(x - xp))

    # Definisci gli estremi della finestra
    start = idx - window_size
    end = idx + window_size

    # Plot della finestra locale
    #popt, pcov = opt.curve_fit(poisson, x[start:end]-x[start], y[start:end])
    #perr=np.sqrt(np.diag(pcov))
    #ax2.plot(x[start:end]-x[start], poisson(x[start:end]-x[start], *popt))
    ax2.plot(x[start:end]-x[start], y[start:end])
    ax2.scatter(x[idx]-x[start], y[idx], marker='x', color='red')

'''
# Secondo asse
ax2 = ax1.twiny()  # Secondo asse x
ax2.plot(counts, y_value, color='orange', label='Media')

# Legende separate per ciascun asse
ax1.legend(loc='upper right')
ax2.legend(loc='upper left')

os.chdir(cwd+'/Immagini')
plt.savefig(var+'.png')
'''
plt.tight_layout()
plt.show()
