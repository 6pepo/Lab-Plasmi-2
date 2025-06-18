import numpy as np
import os
import matplotlib.pyplot as plt
#import scipy.optimize as opt
#import scipy.signal as sig
#import fnmatch
#from sklearn.metrics import r2_score
#from funct_utils import find_peaks

def signal_lenght(x,y):
	return y - x
	
# Load Data
cwd = os.getcwd()
os.chdir(cwd + '/Parametri')
files = os.listdir()
print(files)
var = str(input('Quale file vuoi caricare: ')) 
x, y = np.loadtxt(var + '.txt', usecols = (0,1), unpack = True, delimiter = ',')
z = signal_lenght(x, y)

# Creazione del grafico
fig = plt.figure()
ax = fig.add_subplot(111)
plt.grid(linestyle='--')
# Calcolo l'istogramma
n, bins = np.histogram(z, 1000)
print(n, bins)
ax.hist(z, 1000, color='skyblue', edgecolor='black')
plt.xlabel('Signal lenght [ns]')
plt.ylabel('Counts')
plt.legend()

plt.show()