import sys
sys.path.append('/lhome/dylanb/astronomy/jet_accretion/jet_accretion')
import os
import shutil
import argparse
import numpy as np
import matplotlib.pylab as plt
import scipy
from scipy.constants import *
from scipy import integrate
from sympy import mpmath as mp
import pickle
import scale_intensity
from astropy import units as u
import datetime
import EW

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    ============================================================================
"
"   We calculate the EW of the models and the observations and fit the former to
"   the latter.
"
    ============================================================================
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""
==================================================
Command line input
==================================================
"""
args = sys.argv

parser = argparse.ArgumentParser()

parser.add_argument('-o', dest='object_id',
                    help='Object identifier')
parser.add_argument('-of', dest='DataDir', help='The file with all the output data')

args       = parser.parse_args()
object_id  = args.object_id
DataDir    = args.DataDir

parameters = {}

Path       = "../../jet_accretion_output/"
OutputDir  = Path+str(object_id)+'/'+DataDir+'/'
InputFile  = str(object_id)+'.dat'
with open(OutputDir+InputFile) as f:
    lines  = f.readlines()[2:]

for l in lines:
    split_lines       = l.split()
    title             = split_lines[0]
    value             = split_lines[1]
    parameters[title] = value

"""
===================================================
Balmer line properties and temperature-density grid
===================================================
"""
###### Hydrogen properties #####################################################
E_ionisation_H      = np.array([13.6, 0]) # ionisation energy in eV
E_levels_H          = {1: np.array([0, 10.2, 12.1, 12.76]), 2: np.array([0])} # (eV) energy levels of all excitation states for each ionisation level
degeneracy_H        = {1: np.array([2, 8, 18, 32]), 2: np.array([1])} # Degeneracy of the excitation states

###### Balmer properties, i.e., einstein coefficients for Halpha, Hbeta, #######
###### Hgamma, and Hdelta ######################################################

balmer_properties = {'wavelength': {'halpha': 6562.8e-10, 'hbeta': 4861.35e-10, 'hgamma': 4340.47e-10, 'hdelta': 4101.73e-10},
                     'frequency': {'halpha': 4.5681e14, 'hbeta': 6.1669e14, 'hgamma': 6.9069e14, 'hdelta': 7.3089e14},
                     'f_osc': {'halpha': 6.407e-1, 'hbeta': 1.1938e-1, 'hgamma': 4.4694e-2, 'hdelta': 2.2105e-2},
                     'Aul' : {'halpha': 4.4101e7, 'hbeta': 8.4193e6, 'hgamma' : 2.530e6, 'hdelta': 9.732e5}}

balmer_lines = ['halpha', 'hbeta', 'hgamma', 'hdelta']

###### Temperature and density grid ############################################
T_min               = eval(parameters['T_min'])                 # Min temperature (K)
T_max               = eval(parameters['T_max'])                 # Max temperature (K)
T_step              = eval(parameters['T_step'])                # step temperature (K)
density_log10_min   = eval(parameters['rho_min'])               # Minimum density (log10 m^-3)
density_log10_max   = eval(parameters['rho_max'])               # Maximum density (log10 m^-3)
density_log10_step  = eval(parameters['rho_step'])              # Step density (log10 m^-3)


"""
===============
Stellar spectra
===============
"""

###### Observed spectra, background spectra, and wavelength region #############

spectra_observed    = {}
spectra_wavelengths = {}
spectra_frequencies = {}
spectra_background  = {}
for line in balmer_lines:
    with open('../jet_accretion/input_data/'+object_id+'/'+line+'/'+object_id+'_observed_'+line+'.txt', 'rb') as f:
        spectra_observed[line]    = pickle.load(f)
    with open('../jet_accretion/input_data/'+object_id+'/'+line+'/'+object_id+'_wavelength_'+line+'.txt', 'rb') as f:
        spectra_wavelengths[line] = pickle.load(f) * 1e-10
        spectra_frequencies[line] = constants.c / spectra_wavelengths[line]

    with open('../jet_accretion/input_data/'+object_id+'/'+line+'/'+object_id+'_init_'+line+'.txt', 'rb') as f:
        spectra_background[line]  = pickle.load(f)

phases  = list()
spectra = list()
for ph in spectra_observed['halpha'].keys():
    phases.append(ph)
    for spec in spectra_observed['halpha'][ph].keys():
        spectra.append(spec)

###### The correct intensity level of the spectra from the       ###############
###### synthetic spectra. We fit a straight line to the relevant ###############
###### part of the synthetic spectra to determine the correct    ###############
###### intensity and scale the other spectra accordingly.        ###############

synthetic           = parameters['synthetic']
spectra_synth_wavelengths, spectra_synth_I = np.loadtxt(
               '../jet_accretion/input_data/'+object_id+'/synthetic/'+synthetic)
spectra_synth_wavelengths *= 1e-9 # m
spectra_synth_I           *= 1e-7*1e10*1e4 # W m-2 m-1 sr-1

spectra_background_I = {}   # The background spectra with the correct intensity (W m-2 m-1 sr-1)
spectra_observed_I   = {}

created_interpol_normalisation = {line:False for line in balmer_lines}
scaling_interpolation          = {}

EW_observations                = {line:{} for line in balmer_lines}
EW_model                       = {line:{} for line in balmer_lines}
for line in balmer_lines:

    spectra_background_I[line] = {}
    spectra_observed_I[line]   = {}
    EW_observations[line]      = {}
    EW_model[line]             = {}

    for ph in phases:

        spectra_background_I[line][ph] = {}
        spectra_observed_I[line][ph]   = {}
        EW_observations[line][ph]      = {}
        EW_model[line][ph]             = {}

        for spec in spectra_observed[line][ph]:

            EW_observations[line][ph][spec] = {}
            EW_model[line][ph][spec]        = {}

            # Correction in the normalisation of the spectra
            # if line=='hbeta':
            #     spectra_observed[line][ph][spec] *= 0.94
            # if line=='hgamma':
            #     spectra_observed[line][ph][spec] *= 1.00
            # if line=='hdelta':
            #     spectra_observed[line][ph][spec] *= 0.9
            if created_interpol_normalisation[line]==True:
                spectra_background_I[line][ph][spec] = spectra_background[line][ph][spec] * scaling_interpolation[line](spectra_wavelengths[line])
                spectra_observed_I[line][ph][spec]   = spectra_observed[line][ph][spec] * scaling_interpolation[line](spectra_wavelengths[line])

            else:
                spectra_background_I[line][ph][spec], scaling_interpolation[line] = scale_intensity.scale_intensity(balmer_properties['wavelength'][line],
                                                 spectra_synth_wavelengths,
                                                 spectra_synth_I, spectra_wavelengths[line],
                                                 spectra_background[line][ph][spec])
                spectra_observed_I[line][ph][spec], scaling_interpolation[line]   = scale_intensity.scale_intensity(balmer_properties['wavelength'][line],
                                                 spectra_synth_wavelengths,
                                                 spectra_synth_I, spectra_wavelengths[line],
                                                 spectra_observed[line][ph][spec])
                created_interpol_normalisation[line] = True
# for line in balmer_lines:
#     with open('input_data/IRAS19135+3937/'+line+'/IRAS19135+3937_observed_'+line+'_corrected.txt', 'wb') as f:
#         pickle.dump(spectra_observed[line], f)

"""
=======================
Uncertainty of the data
=======================
"""

standard_deviation = {}
signal_to_noise    = {}
for line in balmer_lines:
    standard_deviation[line] = {}
    f_snr = open('../jet_accretion/input_data/'+object_id+'/'+line+'/'+object_id+'_signal_to_noise_'+str(line)+'.txt', 'rb')
    lines = f_snr.readlines()[:]
    for l in lines:
        l = l.decode('utf-8')
        title = l[:7].strip()
        value = eval(l[7:].split()[0])
        signal_to_noise[title] = value
    f_snr.close()

    if line == 'halpha':
        with open('../jet_accretion/input_data/'+object_id+'/'+line+'/'+object_id+'_stdev_init_'+str(line)+'.txt', 'rb') as f:
            uncertainty_background = pickle.load(f)

        for ph in uncertainty_background:
            standard_deviation[line][ph] = {}
            for spectrum in uncertainty_background[ph]:
                standard_deviation[line][ph][spectrum] = \
                            2./signal_to_noise[spectrum] #+ uncertainty_background[ph][spectrum]
                            # Twice the uncertainty from S/N because the input spectrum is a subtraction between two spectra
                            # --> spec_tot = spec_1 - spec_2
                            # ----> delta_tot = delta_1 + delta_2

    else:
        for ph in phases:
            standard_deviation[line][ph] = {}
            for spec in spectra_observed[line][ph]:
                standard_deviation[line][ph][spec] = \
                            1./signal_to_noise[spectrum]


"""
===========================================
Create the grid for temperature and density
===========================================
"""

jet_temperatures = np.arange(T_min, T_max+1, T_step)
jet_density_log  = np.arange(density_log10_min, density_log10_max+0.001, density_log10_step)
jet_densities    = 10**(jet_density_log)

jet_temperatures = np.arange(4000, 7000+1, T_step)
jet_density_log  = np.arange(14, 17+0.001, density_log10_step)
jet_densities    = 10**(jet_density_log)

"""
==========================================================================
Calculate the equivalent width of the spectral lines for several intervals
==========================================================================
"""

range_EW_calculation_km_per_s = 750
range_EW_calculation_angstrom = {line:range_EW_calculation_km_per_s * 1000 / constants.c * balmer_properties['wavelength'][line] for line in balmer_lines}

OutputEW  = open(OutputDir+'EW_fit.txt', 'w')
OutputEW.write('The equivalent width (Angstrom) for the Balmer lines of the observations over the whole spectrum\n')
OutputEW.write('Phase\tspectrum\tEW %s \n' % str(balmer_lines))

fig, axes = plt.subplots(2,2)
colors = {'halpha': 'black','hbeta': 'darkblue','hgamma': 'blue', 'hdelta': 'lightblue'}
numbers = {'halpha': [0,0],'hbeta': [0,1],'hgamma': [1,0], 'hdelta': [1,1]}
for phase in phases:

    for spectrum in spectra_observed['halpha'][phase].keys():

            OutputEW.write('\n%s\t%s\t' % (str(phase), spectrum))

            for line in balmer_lines:

                spectrum_absorption = spectra_observed[line][phase][spectrum] - spectra_background[line][phase][spectrum] + 1
                EW_line, uncertainty_EW = EW.equivalent_width_uncertainty(spectra_wavelengths[line]*1e10, spectrum_absorption, standard_deviation[line][phase][spectrum],
                                            cut=True,
                                            wave_min=1e10*(balmer_properties['wavelength'][line] - range_EW_calculation_angstrom[line]),
                                            wave_max=1e10*(balmer_properties['wavelength'][line] + range_EW_calculation_angstrom[line]))

                EW_observations[line][phase][spectrum]['values']  = EW_line
                EW_observations[line][phase][spectrum]['uncertainty'] = uncertainty_EW

                OutputEW.write('%.4f\t' % EW_line)

                axes[numbers[line][0], numbers[line][1]].scatter(phase,EW_line,color=colors[line])
                axes[numbers[line][0], numbers[line][1]].errorbar(phase,EW_line, yerr=uncertainty_EW,color=colors[line])
OutputEW.write('\n')
OutputEW.close()
plt.show()

"""
===============================
Normalise the equivalent widths
===============================
"""

# The phases for which the EW should be zero
phases_zero = [11, 14, 22, 74, 82, 94, 98, 99]
# phases_zero = [74, 82, 94, 98, 99]

EW_zero = {line: 0 for line in balmer_lines}
for line in balmer_lines:
    for phase in phases_zero:
        for spectrum in spectra_observed['halpha'][phase].keys():
            EW_zero[line] += EW_observations[line][phase][spectrum]['values']
    EW_zero[line] /= len(phases_zero)

fig, axes = plt.subplots(2,2)

for line in balmer_lines:

    for phase in phases:

        for spectrum in spectra_observed['halpha'][phase].keys():

            EW_observations[line][phase][spectrum]['values'] -= EW_zero[line]

            axes[numbers[line][0], numbers[line][1]].scatter(phase,EW_observations[line][phase][spectrum]['values'],color=colors[line])
            axes[numbers[line][0], numbers[line][1]].errorbar(phase,EW_observations[line][phase][spectrum]['values'], yerr=EW_observations[line][phase][spectrum]['uncertainty'],color=colors[line])

axes[0,0].axhline(0)
axes[0,0].axhline(1)
axes[0,1].axhline(0)
axes[0,1].axhline(1)
axes[1,0].axhline(0)
axes[1,0].axhline(1)
axes[1,1].axhline(0)
axes[1,1].axhline(1)
plt.show()


"""
==========================================================
Fit the equivalent width of each model to the observations
==========================================================
"""


for jet_temperature in jet_temperatures:
    ###### The jet temperature (K)

    for jet_density_max in jet_densities:
        ###### The jet number density at its outer edge (m^-3)

        print('temperature %.0fK and density 10^%.2em^-3'% (jet_temperature,jet_density_max) )
        OutputDirTempRho = '%.0f_%.2e' % (jet_temperature,jet_density_max)

        fig, axes = plt.subplots(2,2)
        EW_model = {line:[] for line in balmer_lines}

        for line in balmer_lines:

            for phase in phases:

                for spectrum in spectra_observed['halpha'][phase].keys():

                    intensity = np.loadtxt(OutputDir+OutputDirTempRho+'/'+line+'/'+spectrum+'_'+str(phase)+'_'+line+'.txt', unpack=True, skiprows=1)

                    ###### Normalise the spectrum to compute the equivalent width
                    spectrum_absorption = np.array(intensity) / scaling_interpolation[line](spectra_wavelengths[line]) - spectra_background[line][phase][spectrum] + 1
                    EW_line = EW.equivalent_width(spectra_wavelengths[line]*1e10, spectrum_absorption,
                                                    cut=True,
                                                    wave_min=1e10*(balmer_properties['wavelength'][line] - 1*range_EW_calculation_angstrom[line]),
                                                    wave_max=1e10*(balmer_properties['wavelength'][line] + 1*range_EW_calculation_angstrom[line]))

                    EW_model[line].append(EW_line)
                    axes[numbers[line][0], numbers[line][1]].scatter(phase,EW_observations[line][phase][spectrum]['values'],color=colors[line])
                    axes[numbers[line][0], numbers[line][1]].errorbar(phase,EW_observations[line][phase][spectrum]['values'], yerr=EW_observations[line][phase][spectrum]['uncertainty'],color=colors[line])

            axes[numbers[line][0], numbers[line][1]].plot(phases,EW_model[line], color=colors[line])
        axes[0,0].axhline(0)
        axes[0,0].axhline(1)
        axes[0,1].axhline(0)
        axes[0,1].axhline(1)
        axes[1,0].axhline(0)
        axes[1,0].axhline(1)
        axes[1,1].axhline(0)
        axes[1,1].axhline(1)
        plt.show()












print('done')
