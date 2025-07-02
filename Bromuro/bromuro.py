import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter
import os
import fnmatch
from sklearn.metrics import r2_score

def gaussian(x,a,mean,sigma):
    return a*np.exp(-((x-mean)**2/(sigma**2))/2)

def line(x,m,q):
    return m*x+q
	
def curve(x,a,b,c):
    return c+(b/x)+(a/(x**2))

# funzione che restituisce FWHM per un picco
def find_FWHM(a):
    x_idx = [i for i, x in enumerate(a) if x in a] # Indicizza il vettore
    peak = max(a) # Trova il valore del picco
    idx_peak = [i for i, x in enumerate(a) if x == peak] # Trova indice del picco
    idx_FWHM_min = [idx for idx in x_idx if idx < idx_peak[0]]
    idx_FWHM_max = [idx for idx in x_idx if idx > idx_peak[0]]
    FWHM_min = min(idx_FWHM_min, key=lambda idx: abs(a[idx] - peak/2))
    FWHM_max = min(idx_FWHM_max, key=lambda idx: abs(a[idx] - peak/2))
    return FWHM_max-FWHM_min
    
def option0():# Calibrazione
	# Load data
	cwd = os.getcwd()
	os.chdir(cwd + '/Dati')
	files = os.listdir()
	print(files)
	var = input('Quale file vuoi caricare: ')
	path = os.path.join(cwd, 'Dati', var, 'RAW')
	os.chdir(path)
	file = fnmatch.filter(os.listdir(path), '*txt3').pop(0)
	
	channels, counts = np.loadtxt(file, delimiter=' ', skiprows=3, usecols=(0,1), unpack=True)
	
	fig = plt.figure()
	
	mask_ec = (channels >= 1080) & (channels <= 1180) # cattura elettronica
	plt.plot(channels,counts, label='Data')
	plt.title('Ricerca dei picchi')
	plt.xlabel('Channels')
	plt.ylabel('Counts')
	plt.grid(linestyle='--')
	
	mask1 = (channels >= 1080) & (channels <= 1120)
	mask2 = (channels > 1120) & (channels <= 1180)
	
	par1 = [1300., 1114., 5.]
	popt1, pcov1 = curve_fit(gaussian, channels[mask1], counts[mask1], p0=par1)
	
	par2 = [2100., 1133., 5.]
	popt2, pcov2 = curve_fit(gaussian, channels[mask2], counts[mask2], p0=par2)
	
	x = np.linspace(np.min(channels[mask_ec]), np.max(channels[mask_ec]), 100)
	plt.plot(x, gaussian(x,*popt1), 'r--', label=f'Peak-1440: A = {popt1[0]:.2f}, $\mu$ = {popt1[1]:.2f}, $\sigma$ = {popt1[2]:.2f}')
	plt.plot(x, gaussian(x,*popt2), 'g--', label=f'Peak-1472: A = {popt2[0]:.2f}, $\mu$ = {popt2[1]:.2f}, $\sigma$ = {popt2[2]:.2f}')
	
	mask_beta = (channels >= 600) & (channels <= 850)
	filt_counts = savgol_filter(counts[mask_beta],20,1)
	plt.plot(channels[mask_beta], filt_counts, 'y--', label='Sagov')
	
	# Ricerca del punto di flesso nel decadimento beta
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
	plt.plot(channels[mask_beta], d2, label='Derivata')
	
	plt.grid(linestyle='--')
	plt.legend()
	
	peaks0 = 789 #kev
	peak1 = 1440 #kev
	peak2 = 1472 #kev
	
	energy_peaks = [peaks0,peak1,peak2]
	channels_peaks = [index_flex,popt1[1], popt2[1]]
	
	fig1 = plt.figure()
	popt_energy, pcov_energy = curve_fit(line, channels_peaks, energy_peaks)
	
	plt.plot(channels_peaks, energy_peaks, 'rx', label='Peaks')
	x = np.linspace(np.min(channels_peaks), np.max(channels_peaks), 100)
	plt.plot(x, line(x,*popt_energy), 'b--', label=f'm={popt_energy[0]:.2f}, q={popt_energy[1]:.2f}')
	plt.title('Calibrazione')
	plt.xlabel('Channels')
	plt.ylabel('Energy [KeV]')
	plt.grid(linestyle='--')
	plt.legend()
	
	# Conversione post calibrazione --> Alberto
	energy = channels * popt_energy[0] + popt_energy[1]
	
	# Scrittura dati sul file ausiliario
	#os.chdir(cwd + '/Parametri')
	#fl=open('Parametri_calibrazione.txt', 'a')
	#fl.write(str(popt_energy[0])+', '+str(popt_energy[1])+'\n')
	#fl.close()
	fig2 = plt.figure()
	plt.plot(energy, counts, label='Data')
	plt.title('Conversione in energia')
	plt.xlabel('Energy [KeV]')
	plt.ylabel('Counts')
	plt.grid(linestyle='--')
	plt.legend()
	
	plt.show()

def option1():# Plot dei grafici per trovare i picchi
	# Load data
	cwd = os.getcwd()
	os.chdir(cwd + '/Dati')
	files = os.listdir()
	print(files)
	var = input('Quale file vuoi caricare: ')
	path = os.path.join(cwd, 'Dati', var, 'RAW')
	os.chdir(path)
	file = fnmatch.filter(os.listdir(path), '*txt3').pop(0)
	
	channels, counts = np.loadtxt(file, delimiter=' ', skiprows=3, usecols=(0,1), unpack=True)
	
	# Lettura parametri da file
	os.chdir(cwd + '/Parametri')
	m, q = np.loadtxt('Parametri_calibrazione.txt', delimiter=',', skiprows=1, usecols=(0,1), unpack=True)
	energy = channels * m + q
	
	fig = plt.figure()
	plt.plot(energy, counts, label='Data')
	plt.xlabel('Energy [KeV]')
	plt.ylabel('Counts')
	plt.grid(linestyle='--')
	if fnmatch.fnmatchcase(file, '*cesio*'):
		plt.title('Picchi di Cesio-137')
		
		par1 = [48000., 650., 5.]
		par2 = [3000., 1450., 5.]
		mask1 = (energy>=625) & (energy<=700)
		mask2 = (energy>=1350) & (energy<=1550)
		popt1, pcov1 = curve_fit(gaussian, energy[mask1], counts[mask1], par1)
		popt2, pcov2 = curve_fit(gaussian, energy[mask2], counts[mask2], par2)
		plt.plot(energy[mask1], gaussian(energy[mask1],*popt1), label=f'Peak: A = {popt1[0]:.2f}, $\mu$ = {popt1[1]:.2f}, $\sigma$ = {popt1[2]:.2f}', color='red')
		plt.plot(energy[mask2], gaussian(energy[mask2],*popt2), label=f'Peak: A = {popt2[0]:.2f}, $\mu$ = {popt2[1]:.2f}, $\sigma$ = {popt2[2]:.2f}', color='red')
		
	elif fnmatch.fnmatchcase(file, '*sodio*'):
		plt.title('Picchi di Sodio-22')
		
		par1 = [5000., 515., 5.]
		par2 = [3000., 1450., 5.]
		par3 = [1000., 1260., 5.]
		mask1 = (energy>=475) & (energy<=550)
		mask2 = (energy>=1350) & (energy<=1550)
		mask3 = (energy>=1200) & (energy<=1340)
		popt1, pcov1 = curve_fit(gaussian, energy[mask1], counts[mask1], par1)
		popt2, pcov2 = curve_fit(gaussian, energy[mask2], counts[mask2], par2)
		popt3, pcov3 = curve_fit(gaussian, energy[mask3], counts[mask3], par3)
		plt.plot(energy[mask1], gaussian(energy[mask1],*popt1), label=f'Peak: A = {popt1[0]:.2f}, $\mu$ = {popt1[1]:.2f}, $\sigma$ = {popt1[2]:.2f}', color='red')
		plt.plot(energy[mask2], gaussian(energy[mask2],*popt2), label=f'Peak: A = {popt2[0]:.2f}, $\mu$ = {popt2[1]:.2f}, $\sigma$ = {popt2[2]:.2f}', color='red')
		plt.plot(energy[mask3], gaussian(energy[mask3],*popt3), label=f'Peak: A = {popt3[0]:.2f}, $\mu$ = {popt3[1]:.2f}, $\sigma$ = {popt3[2]:.2f}', color='red')
		
	elif fnmatch.fnmatchcase(file, '*ignota*'):
		plt.title('Picchi di Ignota')
		
		par1 = [6000., 1160., 5.]
		par2 = [5000., 1320., 5.]
		par3 = [2000., 1450., 5.]
		mask1 = (energy>=1100) & (energy<=1220)
		mask2 = (energy>=1280) & (energy<=1370)
		mask3 = (energy>=1370) & (energy<=1520)
		popt1, pcov1 = curve_fit(gaussian, energy[mask1], counts[mask1], par1)
		popt2, pcov2 = curve_fit(gaussian, energy[mask2], counts[mask2], par2)
		popt3, pcov3 = curve_fit(gaussian, energy[mask3], counts[mask3], par3)
		plt.plot(energy[mask1], gaussian(energy[mask1],*popt1), label=f'Peak: A = {popt1[0]:.2f}, $\mu$ = {popt1[1]:.2f}, $\sigma$ = {popt1[2]:.2f}', color='red')
		plt.plot(energy[mask2], gaussian(energy[mask2],*popt2), label=f'Peak: A = {popt2[0]:.2f}, $\mu$ = {popt2[1]:.2f}, $\sigma$ = {popt2[2]:.2f}', color='red')
		plt.plot(energy[mask3], gaussian(energy[mask3],*popt3), label=f'Peak: A = {popt3[0]:.2f}, $\mu$ = {popt3[1]:.2f}, $\sigma$ = {popt3[2]:.2f}', color='red')
		
	
	plt.legend()
	os.chdir(cwd + '/Immagini')
	#plt.savefig(var+'.png')
	plt.show()

def option2():# Rilevazione della sorgente esterna di cesio al variare delle distanze
	# Load data
	cwd = os.getcwd()
	os.chdir(cwd + '/Dati')
	files = os.listdir()
	print(files)
	var = input('Quale file vuoi caricare: ')
	path = os.path.join(cwd, 'Dati', var, 'RAW')
	os.chdir(path)
	file = fnmatch.filter(os.listdir(path), '*txt3').pop(0)
	
	channels, counts = np.loadtxt(file, delimiter=' ', skiprows=3, usecols=(0,1), unpack=True)
	
	# Lettura parametri da file
	os.chdir(cwd + '/Parametri')
	m, q = np.loadtxt('Parametri_calibrazione.txt', delimiter=',', skiprows=1, usecols=(0,1), unpack=True)
	energy = channels * m + q
	
	par1 = [48000., 650., 5.]
	par2 = [5000., 1450., 5.]
	mask1 = (energy>=625) & (energy<=700)
	mask2 = (energy>=1370) & (energy<=1510)
	popt1, pcov1 = curve_fit(gaussian, energy[mask1], counts[mask1], par1)
	popt2, pcov2 = curve_fit(gaussian, energy[mask2], counts[mask2], par2)
	
	os.chdir(cwd + '/Immagini')
	plt.figure()
	plt.plot(energy, counts, label='Data')
	plt.plot(energy[mask1], gaussian(energy[mask1], *popt1), label=f'Peak: A = {popt1[0]:.2f}, $\mu$ = {popt1[1]:.2f}, $\sigma$ = {popt1[2]:.2f}', color='red')
	plt.plot(energy[mask2], gaussian(energy[mask2], *popt2), label=f'Peak: A = {popt2[0]:.2f}, $\mu$ = {popt2[1]:.2f}, $\sigma$ = {popt2[2]:.2f}', color='red')
	plt.grid(linestyle='--')
	plt.legend()
	plt.savefig(var+'.png')
	plt.show()
	
	os.chdir(cwd + '/Parametri')
	fl=open('Cesio.txt', 'a')
	fl.write(str(var[9:11])+','+str(max(counts[mask1]))+'\n')
	fl.close()
	
def option3():# Simulazione
	# Load data
	cwd = os.getcwd()
	os.chdir(cwd + '/Parametri')
	dist, eff = np.loadtxt('Sim_Eps_Diss_137Cs_LaBr3Nuovo.txt', usecols=(0,1), delimiter=' ', unpack=True)
	eff = eff * 1000
	
	popt, pcov = curve_fit(curve, dist, eff)
	perr=np.sqrt(np.diag(pcov))
	R2 = r2_score(curve(dist, *popt), eff)
	
	fig=plt.figure()
	plt.plot(dist, eff, label='Data')
	plt.scatter(dist, eff, marker='x')
	plt.plot(np.linspace(min(dist),max(dist)), curve(np.linspace(min(dist),max(dist)), *popt), linestyle='--', color='red')
	plt.grid(linestyle='--')
	os.chdir(cwd + '/Immagini')
	plt.savefig('Dati_sim.png')
	plt.show()
	
	os.chdir(cwd + '/Parametri')
	fl=open('Dati_sim.txt', 'a')
	fl.write(str(popt[0])+','+str(perr[0])+','+str(popt[1])+','+str(perr[1])+','+str(popt[2])+','+str(perr[2])+','+str(R2)+'\n')
	fl.close()
	
def option4(): # Calcolo attività del Cs-137
	# Load data
	cwd = os.getcwd()
	os.chdir(cwd + '/Parametri')
	dist, eff = np.loadtxt('Sim_Eps_Diss_137Cs_LaBr3Nuovo.txt', usecols=(0,1), delimiter=' ', unpack=True)
	distance, counts, mini, maxi = np.loadtxt('Cesio.txt', usecols=(0,1,2,3), delimiter=',', unpack=True)
	
	efficenza = 0.2085
	tempo = 1200
	common_elements = np.intersect1d(dist, distance)
	eff = np.array([b for a, b in zip(dist, eff) if a in common_elements])
	counts = np.array([b for a, b in zip(distance, counts) if a in common_elements])
	mini = np.array([b for a, b in zip(distance, mini) if a in common_elements])
	maxi = np.array([b for a, b in zip(distance, maxi) if a in common_elements])
	
	activity = np.array([((maxi[i]-mini[i])*counts[i]/2)/(eff[i]*efficenza*tempo) for i in range(len(common_elements))])
	
	fig=plt.figure()
	plt.title('Activity')
	plt.errorbar(common_elements, activity, label='Data', uplims=True, lolims=True)
	plt.hlines(np.mean(activity), xmin=5, xmax=45, linestyle='--', color='red')
	plt.grid(linestyle='--')
	os.chdir(cwd + '/Immagini')
	plt.savefig('Attivita.png')
	plt.show()
	

#program
OPTIONS = {
    0: option0,
    1: option1,
    2: option2,
	3: option3,
	4: option4,
}

print("0 - Calibrazione")
print("1 - Plot dei grafici per trovare i picchi")
print("2 - Rilevazione della sorgente esterna di Cesio al variare delle distanze")
print("3 - Simulazione")
print("4 - Calcolo attività del Cesio")
print("\n")

while True:
    try:
        option=int(input("Quale opzione vuoi eseguire:"))
        if option in OPTIONS:
            OPTIONS[option]()
            break
        else:
            print("Errore: inserire un'opzione valida.")
    except ValueError:
        pass
    finally:
        True