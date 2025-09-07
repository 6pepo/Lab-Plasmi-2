import matplotlib.pyplot as plt
import numpy as np
import os
import fnmatch
import scipy.optimize as opt
import scipy.signal as sig
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from sklearn.metrics import r2_score, mean_squared_error
import uncertainties as unc
import uncertainties.unumpy as unp
import uncertainties.umath as umath
import funct_utils as fun

def gauss(x,A,mu,sigma):
	exp=-(x-mu)**2/(2*sigma**2)
	norm=A/(sigma*np.sqrt(2*np.pi))
	return norm*np.exp(exp)
def retta(x,m,q):
	return x*m+q

# Funzione per gestire numeri senza incertezza
def safe_ufloat(x):
	try:
		return unc.ufloat_fromstr(x)
	except Exception:
		return unc.ufloat(float(x), 0)

def option0(): # Calibrazione
	cwd = os.getcwd()
	os.chdir(cwd + '/Dati')
	files = os.listdir()
	print(files)
	var = input('Quale file vuoi caricare: ')
	file = fnmatch.filter(os.listdir(), var+'*').pop(0)
	
	# Load data + find peaks
	x = np.loadtxt(file, usecols=(0), unpack=True, skiprows=12, max_rows=4096)
	x_smooth = savgol_filter(x, 21, 3)
	max_x = fun.find_peaks(x_smooth, trigger_max=2, mode='max')
	
	# Find local max 
	energy_peaks = np.array([5.899, 6.489], dtype=float)
	x_peaks = max_x['max']['indices']
	
	# Gaussian fit
	par1 = [40, 338, 5]
	par2 = [10, 372, 5]
	mask1 = np.loadtxt(file, usecols=(0), unpack=True, skiprows=337, max_rows=25)	#(x >= 320) & (x <= 355)
	mask2 = np.loadtxt(file, usecols=(0), unpack=True, skiprows=367, max_rows=30)	#(x >= 350) & (x <= 390)
	popt1, pcov1 = curve_fit(gauss, np.linspace(325, 350, len(mask1)), mask1, p0=par1)
	popt2, pcov2 = curve_fit(gauss, np.linspace(355, 385, len(mask2)), mask2, p0=par2)
	perr1 = np.sqrt(np.diag(pcov1))
	perr2 = np.sqrt(np.diag(pcov2))
	chi1 = r2_score(gauss(np.linspace(325, 350, len(mask1)), *popt1), mask1)
	chi2 = r2_score(gauss(np.linspace(355, 385, len(mask2)), *popt2), mask2)
	print(chi1)
	print(chi2)
	
	# Plot data + Calibration
	fig=plt.figure()
	ax1=fig.add_subplot(111)
	ax1.scatter(x_peaks, energy_peaks, label='Data', color='blue')
	popt, pcov = curve_fit(retta, x_peaks, energy_peaks)
	perr = np.sqrt(np.diag(pcov))
	ax1.plot(x_peaks, retta(x_peaks, *popt), label='y={:.3e}*x + {:.3e}'.format(popt[0],popt[1]), color='red')
	plt.text(355, 6.0, f'Shaping time $= 32 \mu s$\nGain $= 16 dB$', fontsize=12, color='black')
	plt.title('Calibration')
	plt.xlabel('Channel [Ch]')
	plt.ylabel('Energy [KeV]')
	plt.legend()
	plt.grid(linestyle='--')
	
	fig2 = plt.figure()
	ax2=fig2.add_subplot(121)
	ax2.plot(np.linspace(0,4096,len(x))*popt[0]+popt[1], x, label='Raw data')
	ax2.plot(np.linspace(0,4096,len(x))*popt[0]+popt[1], x_smooth, label='Smooth data')
	plt.title('Energy conversion')
	plt.xlabel('Energy [KeV]')
	plt.ylabel('Signal [#]')
	plt.legend()
	plt.grid(linestyle='--')
	
	ax3=fig2.add_subplot(122)
	#ax3.plot(x, label='Raw data') #senza la conversione
	ax3.plot(np.linspace(0,4096,len(x))*popt[0]+popt[1], x, label='Raw data')
	ax3.plot(np.linspace(325, 350, len(mask1))*popt[0]+popt[1], gauss(np.linspace(325, 350, len(mask1)), *popt1), label=rf'Peak-5.899 KeV: A = {popt1[0]:.2f}, $\mu$ = {popt1[1]:.2f}, $\sigma$ = {popt1[2]:.2f}')
	ax3.plot(np.linspace(355, 385, len(mask2))*popt[0]+popt[1], gauss(np.linspace(355, 385, len(mask2)), *popt2), label=rf'Peak-6.489 KeV: A = {popt2[0]:.2f}, $\mu$ = {popt2[1]:.2f}, $\sigma$ = {popt2[2]:.2f}')
	plt.title('Zoom on peaks')
	#plt.xlabel('Channel [Ch]') #senza la conversione
	plt.xlabel('Energy [KeV]')
	plt.ylabel('Signal [#]')
	#plt.xlim(320, 390) #senza la conversione
	plt.xlim(5.5, 6.8)
	plt.legend()
	plt.grid(linestyle='--')
	
	os.chdir(cwd + '/Immagini')
	#plt.savefig('Calibrazione.png')
	
	os.chdir(cwd + '/Parametri')
	fl1=open('Param_calibration.txt', 'a')
	fl1.write(str(popt[0])+', '+str(popt[1])+'\n')
	fl1.close()
	
	plt.show()

def option1(): # Plot dei dati + salvataggio FWHM, H, fattore di Fano, ecc.
	cwd = os.getcwd()
	os.chdir(cwd + '/Dati')
	files = os.listdir()
	print(files)
	var = input('Quale file vuoi caricare: ')
	fl = fnmatch.filter(os.listdir(), var+'*').pop(0)
	exception_files = ('G1','G2')
	
	# Load data
	x = np.loadtxt(fl, usecols=(0), unpack=True, skiprows=12, max_rows=4096)
	if fl.startswith(exception_files):
		x_smooth = savgol_filter(x, 11, 3)
	elif fl.startswith('G5'):
		x_smooth = savgol_filter(x, 155, 3)
	else:
		x_smooth = savgol_filter(x, 61, 2)
	
	max_x = fun.find_peaks(x_smooth, trigger_max=0.5, mode='max')
	if fl.startswith('G5'):
		mask_G5 = max_x['max']['indices'] > 1000
		max_x['max']['values'] = max_x['max']['values'][mask_G5]
		max_x['max']['indices'] = max_x['max']['indices'][mask_G5]
	print(max_x['max'])
	print(x_smooth)
	
	# Mostra il grafico con indice di picco e segnale smussato
	fig = plt.figure()
	plt.plot(x, label = 'Raw data')
	plt.plot(x_smooth, label = 'Smoothed data')
	for i in range(len(max_x['max']['indices'])):
		
		# Ricerca vettore per il calcolo della FWHM
		left_branch = [j for j, x_zero in enumerate(x_smooth) if (x_zero<=0 and j<max_x['max']['indices'][i])]
		left_lim = max(left_branch)
		right_branch = [j for j, x_zero in enumerate(x_smooth) if (x_zero<=0 and j>max_x['max']['indices'][i])]
		right_lim = min(right_branch)
		FWHM = fun.find_FWHM(x[left_lim:right_lim+1])
		print(FWHM)
		
		# Ottimizzazione per il calcolo della gaussiana
		idx = np.linspace(left_lim, right_lim, right_lim-left_lim)
		par = [100, max_x['max']['indices'][i], FWHM]
		popt, pcov = curve_fit(gauss, idx, x[left_lim:right_lim], p0 = par)
		perr = np.sqrt(np.diag(pcov))
		print(perr)
		plt.plot(idx, gauss(idx, *popt), label=rf'Peak: A = {popt[0]:.2f}, $\mu$ = {popt[1]:.2f}, $\sigma$ = {popt[2]:.2f}')
		
		# Disegno la centroide
		plt.vlines(max_x['max']['indices'][i], 0, max_x['max']['values'][i], color='red')
		
		# Calcolo del fattore di fano
		uFWHM = unc.ufloat(FWHM, 1)
		peak = unc.ufloat(popt[1], popt[2])
		centroide = unc.ufloat(popt[1], perr[1])
		sigma = unc.ufloat(popt[2], perr[2])
		#nominal_resolution = unc.ufloat(0.0236, 0.001)
		nominal_resolution = uFWHM/peak
		statistical_resolution = 2.35*sigma/centroide
		poisson_resolution = 2.35/umath.sqrt(peak)
		
		fano = (statistical_resolution/poisson_resolution)**2 #più il fattore di fano è piccolo più si è precisi
		print(uFWHM, peak, centroide, sigma, nominal_resolution, statistical_resolution, poisson_resolution, fano)
		
		# Save results
		os.chdir(cwd + '/Parametri')
		param = {
					'G1': 4.986, 'G2': 9.803, 'G3': 29.938, 'G4': 46.61, 'G5': 72.829, 'G6': 1.161, 'G7': 46.61,
					'S1': 3.2, 'S2': 5.6, 'S3': 9.6, 'S4': 16, 'S5': 25.6, 'S6': 44.8, 'live_data': 32,
				}
		if fnmatch.fnmatchcase(fl, 'G*.mca'):
			fl1=open('Results_gain_stat.txt', 'a')
			fl1.write(str(var)+', '+str(param[var])+', '+str(i+1)+', '+str(peak)+', '+str(x[max_x['max']['indices'][i]])+', '+str(uFWHM)+', '+str(fano)+', '+str(statistical_resolution)+'\n')
			fl1.close()
		elif fnmatch.fnmatchcase(fl, 'S*.mca') or fnmatch.fnmatchcase(fl, 'live_data.mca'):
			fl1=open('Results_shaping_stat.txt', 'a')
			fl1.write(str(var)+', '+str(param[var])+', '+str(i+1)+', '+str(peak)+', '+str(x[max_x['max']['indices'][i]])+', '+str(uFWHM)+', '+str(fano)+', '+str(statistical_resolution)+'\n')
			fl1.close
	
	plt.legend()
	plt.grid(linestyle='--')
	
	os.chdir(cwd + '/Immagini')
	#plt.savefig(var+'.png')
	
	#plt.show()
	
def option2(): # Plot dei dati salvati 
	cwd = os.getcwd()
	os.chdir(cwd + '/Parametri')
	files = os.listdir()
	print(files)
	var = input('Quale file vuoi caricare: ')
	fl = fnmatch.filter(os.listdir(), var+'*').pop(0)
	
	# Load data with uncertainties format
	# Specifica i converter solo per le colonne con incertezza
	converters = {
		3: safe_ufloat,  # peak_channel
		5: safe_ufloat,  # FWHM
		6: safe_ufloat,  # fano_factor
		7: safe_ufloat	 # resolution
	}
	file_name, param, peak, peak_channel, peak_value, FWHM, fano_factor, resolution = np.loadtxt(fl, usecols=(0,1,2,3,4,5,6,7), delimiter=',', unpack=True, skiprows=1, converters=converters, dtype=object)
	param, peak, peak_value = param.astype(float), peak.astype(float), peak_value.astype(float)
	
	fig = plt.figure()
	ax = fig.add_subplot()
	plt.title('Fano vs Gain')
	plt.xlabel('Gain')
	plt.ylabel('Fano')
	plt.grid(linestyle='--')
	for i in range(1,3):
		param_aux, peak_channel_aux = param[[j for j, x in enumerate(peak) if peak[j] == i]], peak_channel[[j for j, x in enumerate(peak) if peak[j] == i]]
		fano_factor_aux = fano_factor[[j for j, x in enumerate(peak) if peak[j] == i]]
		
		mask = np.ones(param_aux.size, dtype=bool) # genera un array di boolean tutti veri
		if fl.startswith('Results_gain_3') or fl.startswith('Results_shaping_3'):
			if fl.startswith('Results_gain_3'):
				mask[5] = False 
			popt, pcov = curve_fit(retta, param_aux[mask], unp.nominal_values(fano_factor_aux[mask]))
			perr = np.sqrt(np.diag(pcov))
			R2 = r2_score(retta(param_aux[mask], *popt), unp.nominal_values(fano_factor_aux[mask]))
			print(R2)
			ax.plot(param_aux[mask], retta(param_aux[mask], *popt))
		
		ax.errorbar(param_aux, unp.nominal_values(fano_factor_aux), unp.std_devs(fano_factor_aux), fmt='o', label=f'Peak {i}')
	plt.grid(linestyle='--')
	plt.legend()
	
	fig1 = plt.figure()
	ax1 = fig1.add_subplot()
	plt.title('Resolution vs Gain')
	plt.xlabel('Gain')
	plt.ylabel('Resolution')
	plt.grid(linestyle='--')
	for i in range(1,3):
		param_aux, peak_channel_aux = param[[j for j, x in enumerate(peak) if peak[j] == i]], peak_channel[[j for j, x in enumerate(peak) if peak[j] == i]]
		resultion_aux = resolution[[j for j, x in enumerate(peak) if peak[j] == i]]
		
		#mask = np.ones(param_aux.size, dtype=bool) # genera un array di boolean tutti veri
		#if fl.startswith('Results_gain'):
		#	mask[5] = False 
		#popt, pcov = curve_fit(retta, param_aux[mask], unp.nominal_values(resultion_aux[mask]))
		#perr = np.sqrt(np.diag(pcov))
		#R2 = r2_score(retta(param_aux[mask], *popt), unp.nominal_values(resultion_aux[mask]))
		#print(R2)
		
		ax1.errorbar(param_aux, unp.nominal_values(resultion_aux), unp.std_devs(resultion_aux), fmt='o', label=f'Peak {i}')
		#ax1.plot(param_aux[mask], retta(param_aux[mask], *popt))
	plt.grid(linestyle='--')
	plt.legend()
	plt.show()
	
def option3(): # analisi americio 241
	cwd = os.getcwd()
	os.chdir(cwd + '/Dati')
	files = os.listdir()
	print(files)
	var = input('Quale file vuoi caricare: ')
	fl = fnmatch.filter(os.listdir(), var+'*').pop(0)
	
	x = np.loadtxt(fl, usecols=(0), unpack=True, skiprows=12, max_rows=4096)
	os.chdir(cwd + '/Parametri')
	m, q = np.loadtxt('Param_calibration.txt', usecols=(0, 1), unpack=True, skiprows=1, max_rows=1, delimiter=',')
	real_peak = np.array([11.87, 13.9, 16.9, 17.7, 20.7, 26.3, 59.5]) # KeV
	x_smooth = savgol_filter(x, 21, 3)
	
	max_x = fun.find_peaks(x_smooth, trigger_max=41, mode='max')
	
	# si assumuma che sulla calibrazione siano stati fatti degli errori di 0.0001 
	err_peak = (max_x['max']['indices']*m+q)*np.sqrt((0.0001/m)**2+(0.0001/q)**2)
	print('real peak: ' + str(real_peak))
	print('Peak: ' + str(max_x['max']['indices']*m+q))
	print('Errore sul picco: ' + str(err_peak))
	
	fig1 = plt.figure()
	plt.plot(np.linspace(0,4096,len(x))*m+q, x, label='Raw data')
	#plt.plot(np.linspace(0,4096,len(x))*m+q, x_smooth, label='Smooth data')
	for i in range(len(real_peak)):
		plt.vlines(real_peak[i], 0, max_x['max']['values'][i], color='red')
		plt.vlines(max_x['max']['indices'][i]*m+q, 0, max_x['max']['values'][i], color='orange')
		plt.text(max_x['max']['indices'][i]*m+q, max_x['max']['values'][i], rf'{real_peak[i]:.2f} keV', fontsize=10, color='black')
	plt.xlabel('Energy [KeV]')
	plt.ylabel('Signal [#]')
	plt.grid(linestyle='--')
	plt.legend()
	
	fig2 = plt.figure()
	plt.errorbar(max_x['max']['indices']*m+q, max_x['max']['values'], xerr=err_peak, fmt='x', label='Observed Peak')
	plt.errorbar(real_peak, max_x['max']['values'], fmt='*', label='Real Peak')
	for i in range(len(real_peak)):
		plt.text(max_x['max']['indices'][i]*m+q, max_x['max']['values'][i], rf'{real_peak[i]:.2f} keV', fontsize=10, color='black')
	plt.xlabel('Energy [KeV]')
	plt.ylabel('Signal [#]')
	plt.grid(linestyle='--')
	plt.legend()
	
	plt.show()
	
	os.chdir(cwd + '/Parametri')
	fl=open('Results_peak_Americio241.txt', 'a')
	obs_peak = unp.uarray(max_x['max']['indices']*m+q, err_peak)
	for i in range(len(real_peak)):
		fl.write(str(real_peak[i])+', '+str(obs_peak[i])+'\n')
	fl.close
	
#program
OPTIONS = {
	0: option0,
	1: option1,
	2: option2,
	3: option3,
}

print("0 - calibrazione")
print("1 - plot grafico + scrive su file")
print("2 - plot grafici guadagno/shaping time + salvataggio immagine")
print("3 - analisi americio")
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