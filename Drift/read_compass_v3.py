import numpy as np
import os
import struct
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import time
import scipy.optimize as opt
import scipy.signal as sig
from funct_utils import find_peaks

def retta(x,q):
    return x*0+q

class compass_reader:
    def __init__(self, filename_w_relative_path) -> None:
        ''' Requires filename to read with relative path'''
        self.path = os.path.split(os.path.abspath(filename_w_relative_path))[0]
        self.filename = os.path.basename(filename_w_relative_path)
    
    def readfile(self):
        ''' Read data file'''
        start = time.time()
        board = []
        channel = []
        timestamp = []
        Elong = []
        Eshort = []
        flags = []
        nsamples = []
        wave = []
        dt = np.dtype('short') 
        dt = dt.newbyteorder('<')
        with open(os.path.join(self.path, self.filename), 'rb') as file:
            buf = file.read(2)
            self.header = struct.unpack_from(">H", buf)[0]
            while True:
                try:
                    buf = file.read(25)
                    line = struct.unpack_from("<HHQHHIBI", buf)
                    board.append(line[0])
                    channel.append(line[1])
                    timestamp.append(line[2])
                    Elong.append(line[3])
                    Eshort.append(line[4])
                    flags.append(line[5])
                    samples = line[7]
                    nsamples.append(samples)
                    buf = file.read(2*samples)
                    wave.append(np.frombuffer(buf, dtype=dt))
                except(struct.error):
                    break
            self.board =  np.array(board)
            self.channel = np.array(channel)
            self.timestamp = np.array(timestamp)
            self.Elong = np.array(Elong)
            self.Eshort = np.array(Eshort)
            self.flags = np.array(flags)
            self.nsamples = np.array(nsamples)
            self.wave = np.array(wave)
            print(f'{len(wave)} wavefroms imported in {time.time()-start:.4f} s')
    
    def plot_all_waves_1000(self):
        '''Plots waveforms in batches of 1000 per figure'''
        batch_size = 1000
        total_waves = len(self.wave)
        #print (total_waves)
    
        for batch_start in range(0, total_waves, batch_size):
            batch_end = min(batch_start + batch_size, total_waves)
            lines = []
            baseline_lines = []
    
            for wave in self.wave[batch_start:batch_end]:
                t = np.arange(0, 2*len(wave), 2)
                popt, pcov = opt.curve_fit(retta, t, wave)
                perr = np.sqrt(np.diag(pcov))
                baseline = retta(t, *popt)
                
                lines.append(np.column_stack([t, wave]))
                baseline_lines.append(baseline)
                #baseline_lines.append(np.column_stack([t, baseline]))
            
            baseline_mean = np.mean(baseline_lines)
            baseline_std = np.std(baseline_lines)
            print(baseline_mean, baseline_std)
            
            fig, ax = plt.subplots()
            ax.add_collection(LineCollection(lines, colors='blue', linewidths=0.8))
            #ax.add_collection(LineCollection(baseline_lines, colors='red', linewidths=0.5, linestyles='--'))
            plt.fill_between(t, baseline_mean - baseline_std, baseline_mean + baseline_std, facecolor='none', edgecolor='red', hatch='///', label=f'Baseline: {baseline_mean:.3e} ± {baseline_std:.3e}')
            plt.autoscale()
            plt.xlabel('Time [ns]')
            plt.ylabel('Amplitude')
            plt.title(f'Waveforms {batch_start + 1}–{batch_end}')
            plt.grid(linestyle='--')
            plt.legend()
            plt.show()
        
    def plot_all_waves_tot (self):
        '''plots all waveforms in same windows'''
        lines = []
        for i in self.wave:
            t = np.arange(0, 2*len(i), 2)
            popt, pcov = opt.curve_fit(retta, t, i)
            perr = np.sqrt(np.diag(pcov))
            baseline = retta(t, *popt)
            
            lines.append(np.column_stack([t, i]))
            
        line_collection = LineCollection(lines, linewidths=0.8)

        fig, ax = plt.subplots()
        ax.add_collection(line_collection)
        #ax.autoscale()
        #plt.plot(t, i)
        plt.plot(t, baseline, label='Baseline')
        plt.xlabel('Time [ns]')
        plt.title('All Waveforms')
        plt.legend()
        plt.show()
        
    def plot_all_waves (self):
        '''plots all waveforms in separate windows (Be careful)'''
        for i in self.wave:
            t = np.arange(0, 2*len(i), 2)
            res = find_peaks(i, mode='min')
            print(len(res['min']['values']), len(res['min']['indices']))
            popt, pcov = opt.curve_fit(retta, t, i)
            baseline_mean = np.mean(retta(t, *popt))
            #baseline_std = float(np.mean(np.sqrt(np.diag(pcov))))
            plt.figure()
            plt.plot(t, i)
            plt.plot(t, retta(t, *popt), '--') #-50*baseline_std
            plt.scatter(res['min']['indices']*2, res['min']['values'], color='red', marker='x')
            plt.xlabel('Time [ns]')
            plt.grid(linestyle='--')
            plt.show()
    
    def plot_spectrum(self, bins=1000):
        ''' Plot elong spectrum (default bin number 1000)'''
        plt.figure()
        plt.hist(self.Elong, bins)
        plt.xlabel('Channel')
        plt.grid(linestyle='--')
        plt.show()

    def shaping_thickness(self):
        '''
        ottimizzazione della soglia
        1- calcola baseline e deviazione standard --> OK
        2- prendi i picchi che sono fuori 10*sigma (in negativo) --> OK
        3- trovati i picchi valuta se sono singoli oppure in gruppo per raggio di 10
            3.1- se in gruppo prendi min/max per ogni picco misurato (good count)
            3.2- se singolo scarta (bad count)
        '''
        # Carico il parametro iniziale
        os.chdir(cwd + '/Parametri')
        #fl=open('Parametri.txt', 'r+')
        #threshold_upl = np.loadtxt('Parametri.txt', usecols=(0), unpack=True)
        fl1=open(var + '.txt', 'a')
        
        for i in self.wave:
            flag_shape = True # Imposto il flag per ottimizzazione della soglia
            
            # Calcolo della baseline e della deviazione standard
            t = np.arange(0, 2*len(i), 2)
            popt, pcov = opt.curve_fit(retta, t, i)
            baseline_mean = np.mean(retta(t, *popt))
            
            '''funzionano entrambe le righe ma la seconda da un DeprecationWarning'''
            baseline_std = float(np.mean(np.sqrt(np.diag(pcov))))
            #baseline_std = float(np.sqrt(np.diag(pcov)))
            
            threshold_calc = baseline_mean - 50*baseline_std
            threshold_upl = 13000 #sembra funzionare bene 3/10 vengono sbagliate
            
            # Cerco nell'intorno se ci sono altri picchi
            while flag_shape:
                mask = i < threshold_upl
                res = find_peaks(i[mask], trigger_min=threshold_upl, mode='min')
                n = 100
                if i[mask].size == 0:
                    threshold_upl = threshold_calc
                    print(threshold_upl)
                    #fl.write(str(threshold_upl))
                else:
                    min_snip, max_snip = 50000000000, 0.000000000000001
                    for j in range(len(res['min']['indices'])):
                        large_snippet = np.arange(res['min']['indices'][j] - n, res['min']['indices'][j] + n)
                        # Verifica che j-1 e j+1 siano indici validi
                        if j > 0 and j+2 < len(res['min']['indices']):
                            # Rimozione dell'elemento centrale in numpy usa arr = np.delete(arr,indice riga,indice colonna)
                            small_snippet = np.delete(res['min']['indices'][j-1:j+2], 1)
                            common_elements = [element for element in small_snippet if element in large_snippet]
                            if any(elem in small_snippet for elem in large_snippet):
                                min_snip = min(min_snip, small_snippet[0])
                                max_snip = max(max_snip, small_snippet[1])
                            elif len(common_elements) == 1 and j == len(res['min']['indices']) - 1:
                                flag_shape = False
                            elif len(common_elements) == 1:
                                pass
                    
                    #fl.seek(0) #imposta il cursore all'inizio del file
                    #fl.write(str((threshold_upl + threshold_calc) / 2))
                    flag_shape = False
                    
            if min_snip == 50000000000 and max_snip == 0.000000000000001:
                fl1.write(str(0) + ', ' + str(0) + '\n')
            else:    
                fl1.write(str(t[mask][min_snip]) + ', ' + str(t[mask][max_snip]) + '\n')
                    
        #fl.close()
        fl1.close()



if __name__ == '__main__':
    # nomefile = 'DataF_CH0@DT5730SB_2289_drift_scan_0.BIN'
    cwd = os.getcwd()
    os.chdir(cwd + '/Dati')
    files = os.listdir()
    print(files)
    var = input('Quale file vuoi caricare: ')
    
    if var in files:
        path = os.path.join(cwd, 'Dati', var, 'RAW')
        os.chdir(path)
        nomefile = 'DataR_CH1@DT5730_641_' + var + '.BIN'
        
    reader = compass_reader(os.path.join(path, nomefile))    
    reader.readfile()
    #reader.plot_spectrum()
    #reader.plot_all_waves_1000()
    #reader.plot_all_waves()
    reader.shaping_thickness()
