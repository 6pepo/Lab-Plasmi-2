import numpy as np
import os
import struct
import matplotlib.pyplot as plt
import time

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
            self.header = struct.unpack_from("<H", buf)[0]
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
    
    def plot_all_waves (self):
        '''plots all waveforms in separate windows (Be careful)'''
        for i in self.wave:
            t = np.arange(0, 2*len(i), 2)
            plt.figure()
            plt.plot(t, i)
            plt.xlabel('Time [ns]')
            plt.show()
    
    def plot_spectrum(self, bins=1000):
        ''' Plot elong spectrum (default bin number 1000)'''
        plt.figure()
        plt.hist(self.Elong, bins)
        plt.xlabel('Channel')
        plt.show()



if __name__ == '__main__':
    nomefile = 'DataF_CH0@DT5730SB_2289_drift_scan_0.BIN'
    path = 'example/drift_scan_0/FILTERED/'
    reader = compass_reader(os.path.join(path, nomefile))
    reader.readfile()
    reader.plot_spectrum()