import numpy as np
import sys
sys.path.append('./tools')
import os
import argparse
import parameters_DICT
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


parameters = parameters_DICT.read_parameters(InputDir+InputFile)

jet_temperatures = np.arange(parameters['OTHER']['T_min'], parameters['OTHER']['T_max']+1, parameters['OTHER']['T_step'])
jet_density_log  = np.arange(parameters['OTHER']['density_log10_min'], parameters['OTHER']['density_log10_max']+0.001, parameters['OTHER']['density_log10_step'])
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
