import numpy as np
import matplotlib.pylab as plt
import multiprocessing
from multiprocessing import Pool
import time
import os

def compute(pair):
    T, rho = pair
    OutputTXT = 'EW_model_%.0f_%.2e.txt' % (T,rho)
    with open('test_output/'+OutputTXT, 'w') as f_out:

        f_out.write('%.0f\t%.2e\n' % (T,rho))

    for t in range(10):
        for r in range(50):
            x = t**5*np.log(rho)
    return T + 10*np.log10(rho)


jet_temperatures = np.arange(4000,6001,100)
jet_densities = 10**np.arange(14,18.01,0.1)

n_t = len(jet_temperatures)
n_d = len(jet_densities)


grid_temperatures_densities = np.zeros((n_t*n_d, 2))
grid_number = 0
grid_list = []
for T in range(n_t):
    for d in range(n_d):
        grid_list.append([jet_temperatures[T],jet_densities[d]])
        grid_temperatures_densities[grid_number][0] = jet_temperatures[T]
        grid_temperatures_densities[grid_number][1] = jet_densities[d]
        grid_number += 1

pool = Pool(processes=4)


start = time.time()
pool.map(compute, grid_list)
end = time.time()
print(end - start)




with open('test_output/'+'EW_model.txt', 'a') as f_out:

    for jet_temperature in jet_temperatures:

        for jet_density_max in jet_densities:

            f_out.write('\n%.0f\t%.2e\t' % (jet_temperature,jet_density_max))
            OutputTXT = 'EW_model_%.0f_%.2e.txt' % (jet_temperature,jet_density_max)
            with open('test_output/'+OutputTXT, 'r') as f_single:
                for line in f_single.readlines():
                    f_out.write(line)
            os.remove('test_output/'+OutputTXT)
