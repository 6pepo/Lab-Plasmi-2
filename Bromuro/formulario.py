import uncertainties as unc
from uncertainties.umath import *
import math
from funct_utils import common_el
import os
import numpy as np

# Conteggio
maxi = unc.ufloat(1170, 10)
mini = unc.ufloat(1080, 10)
altezza = unc.ufloat(2152, 40)
conteggio = (maxi-mini)*altezza/2
print('conteggio: ')
print(conteggio)
print()

# Volume cristallo
t_La = unc.ufloat(1.02*10**11, 0.01*10**11) # years
rel_eff = 19/100
BR = 65.2/100
AI = 0.08881/100
N_a = 6.023*10**23 # numero avogadro
rho_La = unc.ufloat(5.06, 0.01) # g/cm^3
rho_Ce = unc.ufloat(5.1, 0.1) # g/cm^3
Br = unc.ufloat(79.9, 0.1)
La = unc.ufloat(138.91, 0.01)
Ce = unc.ufloat(140.12, 0.01)
mm_LaBr3 = Br*3+La
mm_CeBr3 = Br*3+Ce
time = unc.ufloat(1200, 1) # s
volume = (((t_La*365.25*24*60*60)*conteggio)/(math.log(2)*time*rel_eff*BR*AI*N_a))*(mm_LaBr3/rho_La*95/100+mm_CeBr3/rho_Ce*5/100)
print('volume: ')
print(volume)
print()

# Attività intrinseca La-138
intrinsic_activity = (N_a*(rho_La*95/100+rho_Ce*5/100)*volume*AI*math.log(2))/((mm_LaBr3*95/100+mm_CeBr3*5/100)*t_La*365*24*60*60)
print('attivita intrinseca: ')
print(intrinsic_activity)
print()

# Efficenza intrinseca LaBr3
intrinsic_eff = conteggio/(intrinsic_activity*time*BR)
print('efficenza intrinseca: ')
print(intrinsic_eff)
print()

cwd = os.getcwd()
os.chdir(cwd + '/Parametri')
distance, ratio_eff = np.loadtxt('Sim_Eps_Diss_137Cs_LaBr3Nuovo.txt', usecols=(0,1), delimiter=' ', unpack=True)
distance2, counts, maxi, mini = np.loadtxt('Cesio.txt', usecols=(0,1,2,3), delimiter=',', unpack=True)
distance, ratio_eff, counts, maxi, mini = common_el(distance, distance2, ratio_eff, counts, maxi, mini)
for i in range(len(distance)):
	# Attività Cesio-137 misurata
	maxi_Cs = unc.ufloat(maxi[i], 10)
	mini_Cs = unc.ufloat(mini[i], 10)
	altezza_Cs = unc.ufloat(counts[i], 50)
	conteggio_Cs = (maxi_Cs-mini_Cs)*altezza_Cs/2
	print('conteggio cesio: ')
	print(conteggio_Cs)
	print()

	# Caricamento efficenze
	#ratio_eff = 1.932/100 # valido a 5 cm
	measured_activity = conteggio_Cs/(ratio_eff[i]*intrinsic_eff*time) # manca un fatttore 9 al dividendo
	print('attività cesio misurata: ')
	print(measured_activity)
	print()

# Attività Cesio-137 stimata
A_0 = unc.ufloat(41.3*10**3, 0.1*10**3)
t_Cs = unc.ufloat(30.08, 0.01) # years
dt = unc.ufloat(15, 0.1) # years
exp = math.exp(1)
estimated_activity = A_0*exp**(-math.log(2)*dt/t_Cs)
print('attività cesio stimata: ')
print(estimated_activity)
print()