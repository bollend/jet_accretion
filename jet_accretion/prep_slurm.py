import numpy as np
import sys
import os
import argparse

"""
==================================================
Command line input
==================================================
"""
args = sys.argv

parser = argparse.ArgumentParser()

parser.add_argument('-o', dest='object_id',
                    help='Object identifier')

parser.add_argument('-d', dest='datafile',
                    help='data file with all the input parameters and specifics')

args          = parser.parse_args()
object_id     = args.object_id
datafile      = str(args.datafile)
parameters = {}
InputDir   = 'input_data/'+str(object_id)+'/'
InputFile = datafile
with open(InputDir+InputFile) as f:
    lines  = f.readlines()[2:]

for l in lines:
    split_lines       = l.split()
    title             = split_lines[0]
    value             = split_lines[1]
    parameters[title] = value

###### Temperature and density grid ############################################
T_min               = eval(parameters['T_min'])                 # Min temperature (K)
T_max               = eval(parameters['T_max'])                 # Max temperature (K)
T_step              = eval(parameters['T_step'])                # step temperature (K)
density_log10_min   = eval(parameters['rho_min'])               # Minimum density (log10 m^-3)
density_log10_max   = eval(parameters['rho_max'])               # Maximum density (log10 m^-3)
density_log10_step  = eval(parameters['rho_step'])              # Step density (log10 m^-3)

jet_temperatures = np.arange(T_min, T_max+1, T_step)
jet_density_log  = np.arange(density_log10_min, density_log10_max+0.001, density_log10_step)
jet_densities    = 10**(jet_density_log)

n_t = len(jet_temperatures)
n_d = len(jet_densities)


grid_number = 0
grid_list = []
for T in range(n_t):
    for d in range(n_d):
        grid_list.append([jet_temperatures[T],jet_densities[d]])
        grid_number += 1
ff = open('list_T_rho.txt', 'w')
ff.close()

with open('list_T_rho.txt', 'a') as f:
    for line in grid_list:
        TandRho = '-t %.0f -d %.2e\n' % (line[0],line[1])
        f.write(TandRho)
