import numpy as np
import scipy.optimize as opt
import scipy.signal as sig
import scipy.interpolate as inter
import uncertainties as unc
import uncertainties.unumpy as unp
import uncertainties.umath as umath
#import sklearn.metrics # funzioni più utilizzate: r2_score, mean_squared_error

def find_FWHM(a, peak=None, idx_peak=None):
	'''
	Calcola la larghezza a metà altezza (FWHM) di un segnale
	
	Args:
		a: array dei dati (1D)
		peak: piccco dell'array (facoltativo)
		idx_peak: indice del picco (facoltativo)

	Returns:
		FWHM: larghezza a metà altezza
	'''
	a = np.abs(np.asarray(a))
	x_idx = [i for i, x in enumerate(a) if x in a]
	
	if peak is None or idx_peak is None:
		peak = max(a)
		idx_peak = min([i for i, x in enumerate(a) if x == peak])
	
	if not idx_peak:
		return None

	#idx_peak = np.asarray(idx_peak)
	idx_FWHM_min = [idx for idx in x_idx if idx < idx_peak] #[0]
	idx_FWHM_max = [idx for idx in x_idx if idx > idx_peak] #[0]

	if not idx_FWHM_min or not idx_FWHM_max:
		return None

	FWHM_min = min(idx_FWHM_min, key=lambda idx: abs(a[idx] - peak / 2))
	FWHM_max = min(idx_FWHM_max, key=lambda idx: abs(a[idx] - peak / 2))
	return FWHM_max - FWHM_min

def find_FWFM(x, peak=None, idx_peak=None, window=None, poly=None):
	'''
	Calcola la larghezza a piena altezza (FWFM) di un segnale
	
	Args:
		x: array dei dati (1D)
		peak: piccco dell'array (facoltativo)
		idx_peak: indice del picco (facoltativo)

	Returns:
		FWLM: larghezza a piena altezza
	'''
	x = np.abs(np.asarray(x))
	
	if window is None or poly is None:
		x = sig.savgol_filter(x, 21, 3)
	else:
		x = sig.savgol_filter(x, window, poly)
	
	if peak is None or idx_peak is None:
		peak = max(x)
		idx_peak = min([i for i, x in enumerate(x) if x == peak])
	
	if not idx_peak:
		return None
	
	left_lim = max([j for j, x_zero in enumerate(x) if (x_zero <= 0 and j < idx_peak)])
	right_lim = min([j for j, x_zero in enumerate(x) if (x_zero <= 0 and j > idx_peak)])
	return right_lim - left_lim, find_FWHM(x[left_lim:right_lim])

def find_peaks(x, trigger_max=None, trigger_min=None, mode='both'):
	'''
	Trova i massimi e/o minimi locali in un array.

	Args:
		x: array dei dati (1D)
		trigger_max: soglia minima per accettare un massimo (facoltativa)
		trigger_min: soglia massima per accettare un minimo (facoltativa, deve essere negativa o zero)
		mode: 'max', 'min', o 'both'

	Returns:
		dict con chiavi: 'max' e/o 'min', ognuna contenente:
			- 'values': np.array dei valori dei picchi
			- 'indices': np.array degli indici originali
	'''
	x = np.asarray(x)
	#x = sig.savgol_filter(x, 21, 3)
	result = {}

	if mode in ['max', 'both']:
		mask_max = (x[1:-1] > x[:-2]) & (x[1:-1] > x[2:])
		peak_indices_max = np.where(mask_max)[0] + 1
		peak_values_max = x[peak_indices_max]  # usa x originale!

		if trigger_max is not None:
			valid = peak_values_max > float(trigger_max)
			peak_indices_max = peak_indices_max[valid]
			peak_values_max = peak_values_max[valid]

		result['max'] = {'values': peak_values_max, 'indices': peak_indices_max}

	if mode in ['min', 'both']:
		mask_min = (x[1:-1] < x[:-2]) & (x[1:-1] < x[2:])
		peak_indices_min = np.where(mask_min)[0] + 1
		peak_values_min = x[peak_indices_min]  # usa x originale!

		if trigger_min is not None:
			valid = peak_values_min < float(trigger_min)
			peak_indices_min = peak_indices_min[valid]
			peak_values_min = peak_values_min[valid]

		result['min'] = {'values': peak_values_min, 'indices': peak_indices_min}

	return result
	
def find_max(x, y=None):
	'''
	Trova i massimi locali assoluti in un array
	
	Args:
		x: array dei dati (1D)
		y: array dei dati (1D) (facoltativo)

	Returns:
		x_peaks: vettore dei picchi (1D)
		x_idx_peaks: vettore indice dei picchi (1D)
	'''
	x = np.asarray(x)
	#x = sig.savgol_filter(x, 21, 3)
	
	if y is None:
		original_indices = np.arange(len(x))
	else:
		original_indices = np.asarray(y)
		
	mask = (x[1:-1] > x[:-2]) & (x[1:-1] > x[2:])
	x_peaks = x[1:-1][mask]
	local_peak_indices = np.array([i + 1 for i, x_max in enumerate(mask) if x_max])
	x_idx_peaks = original_indices[local_peak_indices]
	return x_peaks, x_idx_peaks

def find_min(x, y=None):
	'''
	Trova i minimi locali assoluti in un array
	
	Args:
		x: array dei dati (1D)
		y: array dei dati (1D) (facoltativo)

	Returns:
		x_peaks: vettore dei picchi (1D)
		x_idx_peaks: vettore indice dei picchi (1D)
	'''
	x = np.asarray(x)
	#x = sig.savgol_filter(x, 21, 3)
		
	if y is None:
		original_indices = np.arange(len(x))
	else:
		original_indices = np.asarray(y)
		
	original_indices = np.arange(len(x))
	mask = (x[1:-1] < x[:-2]) & (x[1:-1] < x[2:])
	x_peaks = x[1:-1][mask]
	local_peak_indices = np.array([i + 1 for i, x_min in enumerate(mask) if x_min])
	x_idx_peaks = original_indices[local_peak_indices]
	return x_peaks, x_idx_peaks

def fattoriale(n):
	'''Calcolo del fattoriale'''
	if n == 1:
		return 1
	else:
		return n * fattoriale(n - 1)

def common_el(x, y, u=None, v=None, w=None, z=None):
	'''
	Ricerca degli elementi in comune tra due arrray
	
	Args:
		x: array dei dati (1D)
		y: array dei dati (1D)
		tutti gli altri array sono facoltativi

	Returns:
		common_elements: elementi in comune tra i due vettori
		se fati gli altri vettori ritorna gli elementi corrispondenti alla comunanza
	'''
	x, y = np.asarray(x), np.asarray(y)
	
	if u is None and v is None and w is None and z is None:
		common_elements = np.intersect1d(x, y)
		return common_elements
	else:
		u, v, w, z = np.asarray(u), np.asarray(v), np.asarray(w), np.asarray(z)
		common_elements = np.intersect1d(x, y)
		u = np.array([b for a, b in zip(x, u) if a in common_elements])
		v = np.array([b for a, b in zip(y, v) if a in common_elements])
		w = np.array([b for a, b in zip(y, w) if a in common_elements])
		z = np.array([b for a, b in zip(y, z) if a in common_elements])
		return common_elements, u, v, w, z

def bisection(f, a, b, tol=1e-6, max_iter=100):
	'''
	Trova uno zero della funzione f nell'intervallo [a, b] usando il metodo della bisezione.
	
	Args:
		f: La funzione di cui trovare lo zero.
		a: Estremo sinistro dell'intervallo.
		b: Estremo destro dell'intervallo.
		tol: Precisione desiderata (tolleranza).
		max_iter: Numero massimo di iterazioni.
	
	Returns:
		Uno zero della funzione o None se non trovato.
	'''
	if f(a) * f(b) >= 0:
		print('La funzione non cambia segno nell''intervallo.')
		return None
	
	for _ in range(max_iter):
		c = (a + b) / 2
		
		if f(c) == 0 or (b - a) / 2 < tol:
			return c
		
		if f(c) * f(a) < 0:
			b = c
		else:
			a = c
		
	print('Tolleranza non raggiunta dopo il numero massimo di iterazioni.')
	return None
	
def bisection_graph(y, peak, idx_peak, tol=1e-6, max_iter=100000):
	'''
	Trova uno zero del grafico, usando il metodo della bisezione.
	
	Args:
		y: ordinate del gafico
		x: ascisse del grafico (necessario?)
		peak: lista dei picchi
		idx_peak: lista delgi indici dei picchi

	Returns:
		Lista degli zeri di un grafico
	'''
	y = np.asarray(y)
	
	# Pair and sort the peak
	peak, idx_peak = np.asarray(peak), np.asarray(idx_peak) 
	matrix_peak = np.column_stack([idx_peak, peak])
	sorted_matrix = matrix_peak[matrix_peak[:,0].argsort()]
	idx_zero = []
	
	for i in range(len(sorted_matrix) - 1):
		if sorted_matrix[i, 1] * sorted_matrix[i+1, 1] <= 0:
			a, b = int(sorted_matrix[i, 0]), int(sorted_matrix[i+1, 0])
			
			for _ in range(max_iter):
				c = int(round((a + b) / 2, 0))
				if y[c] == 0 or (b - a) / 2 < tol:
					idx_zero.append(c)
					break
				
				if y[c] * y[a] < 0:
					b = c
				else:
					a = c
			else:
				print('Tolleranza non raggiunta dopo il numero massimo di iterazioni.')
				idx_zero.append(None)
		else:
			print('La funzione non cambia segno.')
			idx_zero.append(None)
	
	return idx_zero

def bisection_graph1(y, x=None, tol=1e-6, max_iter=100000):
	'''
	Trova gli zeri del grafico (interpolato) usando il metodo della bisezione.

	Args:
		y: array-like, ordinate del grafico.
		x: array-like, ascisse del grafico. Se None, assume np.arange(len(y)).
		tol: tolleranza per l'arresto.
		max_iter: massimo numero di iterazioni.

	Returns:
		Lista di zeri trovati.
	'''
	if x is None:
		x = np.arange(len(y))
	
	x = np.asarray(x)
	y = np.asarray(y)

	# Crea interpolazione continua della funzione
	f_interp = inter.interp1d(x, y, kind='linear', bounds_error=False, fill_value='extrapolate')

	zeros = {}

	for i in range(len(y) - 1):
		if y[i] * y[i+1] < 0:
			# C'è uno zero nell'intervallo [x[i], x[i+1]]
			a, b = x[i], x[i+1]

			for _ in range(max_iter):
				c = (a + b) / 2
				fc = f_interp(c)

				if abs(fc) < tol or (b - a) / 2 < tol:
					#zeros.append((c, fc))
					zeros = {'values': fc, 'indices': c}
					break

				if f_interp(a) * fc < 0:
					b = c
				else:
					a = c
		else:
			continue  # Nessun cambio di segno

	return zeros

def loadufile(file, num_cols, delimiter=',', skiprows=0, max_rows=None, ufloat_cols=()):
	# Funzione per gestire numeri con o senza incertezza
	def safe_ufloat(x):
		if isinstance(x, bytes):
			x = x.decode('utf-8')
		x = x.strip()
		try:
			return unc.ufloat_fromstr(x)
		except Exception:
			try:
				return unc.ufloat(float(x), 0)
			except Exception:
				return x  # Se non è nemmeno un numero
	
	# Specifica i converter solo per le colonne con incertezza
	#converters = {i: safe_ufloat for i in ufloat_cols}
	#result = np.loadtxt(file, 
	#					usecols=range(num_cols), 
	#					delimiter=delimiter, 
	#					unpack=True, 
	#					skiprows=skiprows, 
	#					max_rows=max_rows, 
	#					converters=converters, 
	#					dtype=object)
	raw_data = np.loadtxt(file, 
						usecols=range(num_cols), 
						delimiter=delimiter, 
						unpack=True, 
						skiprows=skiprows, 
						max_rows=max_rows, 
						dtype=str)
	
	# Conversione colonna per colonna
	result = {}
	for i, col in enumerate(raw_data):
		if i in ufloat_cols:
			result[i] = np.array([safe_ufloat(v) for v in col], dtype=object)
		else:
			try:
				result[i] = np.array(col, dtype=float)
			except ValueError:
				result[i] = np.array(col, dtype=type(raw_data[i]))
	
	return result