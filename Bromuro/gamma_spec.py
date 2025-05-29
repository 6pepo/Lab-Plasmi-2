import numpy as np

file = 'gamma_database.txt'

energies, isotopes = np.genfromtxt(
    file,
    skip_header=1,
    usecols=(1, 2),
    delimiter=None,
    dtype=['f8', 'U50'],
    unpack=True,
)
sort_index = np.argsort(isotopes)
energies = energies[sort_index]
isotopes = isotopes[sort_index]

for i in range(len(energies)):
    print(energies[i], isotopes[i])
