import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.signal as sig
import os
import fnmatch

def gauss(x,A,mu,sigma):
    exp=-(x-mu)**2/(2*sigma**2)
    norm=A/(sigma*np.sqrt(2*np.pi))
    return norm*np.exp(exp)
def retta(x,m,q):
    return x*m+q

def option0(): # Calibrazione
    var = str(input('Quale file vuoi caricare: '))
    fl = var + '.mca'
    cwd = os.getcwd()
    
    # Load data
    os.chdir(cwd + '/Dati')
    x=np.loadtxt(fl, usecols=(0), unpack=True, skiprows=12, max_rows=4096)
    
    # Find local max 
    energy_peaks = np.array([6.9, 2.5], dtype=float)
    x_peaks = np.array([42,8], dtype=float)
    
    # Plot data
    fig=plt.figure()
    ax1=fig.add_subplot(111)
    ax1.scatter(x_peaks, energy_peaks, label='Data', color='blue')
    popt, pcov = opt.curve_fit(retta, x_peaks, energy_peaks)
    perr = np.sqrt(np.diag(pcov))
    ax1.plot(x_peaks, retta(x_peaks, *popt), label='y={:.3e}*x + {:.3e}'.format(popt[0],popt[0]), color='red')

    plt.xlabel('Channel (Ch)')
    plt.ylabel('Energy (KeV)')
    plt.legend()
    plt.grid(linestyle='--')
    #plt.savefig(var+'.png')
    plt.show()

def option1(): # Plot dei dati + salvataggio FWHM, H
    var = str(input('Quale file vuoi caricare: '))
    fl = var + '.mca'
    cwd = os.getcwd()
    
    # Load data
    os.chdir(cwd + '/Dati')
    x=np.loadtxt(fl, usecols=(0), unpack=True, skiprows=12, max_rows=4096)
    
    # Find local max 
    mask = (x[1:-1] >= x[:-2]) & (x[1:-1] >= x[2:]) & (x[1:-1] > 2)
    x_peaks = x[1:-1][mask] # Picchi
    x_idx_peaks = [i for i, x_max in enumerate(x) if x_max in x_peaks] # Indici dei picchi
    mask1 = x > 2
    x_mask = x[mask1]
    x_idx_mask = [i for i, x_max in enumerate(x) if x_max in x_mask] # Indici della mashera
    idx_left = min(x_idx_mask)
    
    # Plot data
    fig=plt.figure()
    ax1=fig.add_subplot(111)
    ax1.plot(x, label='Data')
    for i in range(len(x_peaks)):
        result = [x_idx_mask[i] for i in range(len(x_idx_mask) - 1) if (x_idx_mask[i+1] - x_idx_mask[i] > 1) and (x_idx_mask[i] > idx_left)]
        result1 = [x_idx_mask[i+1] for i in range(len(x_idx_mask) - 1) if (x_idx_mask[i+1] - x_idx_mask[i] > 1) and (x_idx_mask[i] > idx_left)]
        
        # Set idx_right
        if not result:
            idx_right = max(x_idx_mask)
        else:
            idx_right = result[0]
        
        if idx_left >= max(x_idx_mask):
            pass
        else:
            x_fit = x[idx_left:idx_right]
            x_range = np.linspace(idx_left, idx_right, idx_right - idx_left)

            popt, pcov = opt.curve_fit(gauss, x_range, x_fit)
            perr = np.sqrt(np.diag(pcov))

            ax1.plot(x_range, gauss(x_range, *popt), label='Fit')
            
            idx_FWHM_min = [idx for idx in x_idx_mask if idx < x_idx_peaks[i]]
            idx_FWHM_max = [idx for idx in x_idx_mask if idx > x_idx_peaks[i]]
            FWHM_min = min(idx_FWHM_min, key=lambda idx: abs(x[idx] - x_peaks[i]/2))
            FWHM_max = min(idx_FWHM_max, key=lambda idx: abs(x[idx] - x_peaks[i]/2))
            
            # Save results
            for file in os.listdir(cwd + '/Dati'):
                if fnmatch.fnmatchcase(file, 'G*.mca'):
                    fl1=open('Results_gain.txt', 'a')
                    fl1.write(str(var)+', '+str(i)+', '+str(x_peaks[i])+', '+str(FWHM_max-FWHM_min)+'\n')
                    fl1.close()
                elif fnmatch.fnmatchcase(file, 'S*.mca'):
                    fl1=open('Results_shaping.txt', 'a')
                    fl1.write(str(var)+', '+str(i)+', '+str(x_peaks[i])+', '+str(FWHM_max-FWHM_min)+'\n')
                    fl1.close
    
            # Update idx_left if possible
            if result1:
                idx_left = result1[0]
            else:
                break  # No other segment to analyze
            
    #plt.title('')
    plt.xlabel('Channel (Ch)')
    plt.ylabel('Counts')
    plt.legend()
    plt.grid(linestyle='--')
    #plt.savefig(var+'.png')
    plt.show()
    
def option2(): # Plot dei dati salvati 
    var = str(input('Quale file vuoi caricare: '))
    fl = var + '.txt'
    cwd = os.getcwd()
    
    # Load dati
    os.chdir(cwd + '/Dati')
    file_name, num_peak, peak, FWHM, gain = np.loadtxt(fl, usecols=(0,1,2,3,4), unpack=True, skiprows=1)
    
    fig = plt.figure()
    ax=fig.add_subplot(111)
    
    # Set collection
    gain0 = [gain[i] for i in range(len(file_name)) if num_peak[i] == 0]
    gain1 = [gain[i] for i in range(len(file_name)) if num_peak[i] == 1]
    peak0 = [peak[i] for i in range(len(file_name)) if num_peak[i] == 0]
    peak1 = [peak[i] for i in range(len(file_name)) if num_peak[i] == 1]
    FWHM0 = [FWHM[i] for i in range(len(file_name)) if num_peak[i] == 0]
    FWHM1 = [FWHM[i] for i in range(len(file_name)) if num_peak[i] == 1]
    
    for i in range(len(file_name)):
        if num_peak[i] == 0:
            aux0_res.append(peak[i])
        elif num_peak[i] == 1:
            aux1_res.append(peak[i])
            
    # Plot data
    ax.errorbar(gain, FWHM/peak, label='Data')
    ax.errorbar(gain0, FWHM0/peak0 , label='First Peak')
    ax.errorbar(gain1, FWHM1/peak1 , label='Second Peak')
    if var == 'Results_gain':
        plt.xlabel('Gain')
    else:
        plt.xlabel('Shaping')
    plt.ylabel('Resolution')
    plt.legend()
    plt.grid(linestyle='--')
    #plt.savefig(var+'.png')
    plt.show()
    
#program
OPTIONS = {
    0: option0,
    1: option1,
    2: option2,
}

print("0 - calibrazione")
print("1 - plot grafico + scrive su file")
print("2 - plot grafici guadagno/shaping time + salvataggio immagine")
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