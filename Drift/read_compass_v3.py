import numpy as np
import os
import struct
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import time
import scipy.optimize as opt

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
        '''Plots waveforms in batches of 100 per figure'''
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
            plt.fill_between(t, baseline_mean - baseline_std, baseline_mean + baseline_std, facecolor='none', edgecolor='red', hatch='///', label=f'Baseline: {baseline_mean:.3E} ± {baseline_std:.3E}')
            plt.autoscale()
            plt.xlabel('Time [ns]')
            plt.ylabel('Amplitude')
            plt.title(f'Waveforms {batch_start + 1}–{batch_end}')
            plt.grid('--')
            plt.legend()
            plt.show()
        
    def plot_all_waves (self):
        '''plots all waveforms in separate windows (Be careful)'''
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
    
    def plot_spectrum(self, bins=1000):
        ''' Plot elong spectrum (default bin number 1000)'''
        plt.figure()
        plt.hist(self.Elong, bins)
        plt.xlabel('Channel')
        plt.grid('--')
        plt.show()



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
    reader.plot_spectrum()
    reader.plot_all_waves_1000()
