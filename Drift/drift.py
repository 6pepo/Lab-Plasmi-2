from compass_reader.read_compass import compass_reader

import os


# nomefile = 'DataR_CH0@DT5730SB_2289_drift_scan_0.BIN'
# path = 'Drift/compass_reader/example/drift_scan_0/RAW/'

nomefile = 'DataR_CH1@DT5730_641_test1.BIN'
path = 'Drift/Dati/test1/RAW/'

reader = compass_reader(os.path.join(path, nomefile))
reader.readfile()
reader.plot_spectrum()
reader.plot_all_waves()