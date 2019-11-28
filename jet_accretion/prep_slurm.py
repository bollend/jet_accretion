import numpy as np
import sys
sys.path.append('./tools')
sys.path.append('/home/dbollen/MCMC_main/MCMC_main')
sys.path.append('/home/dbollen/jet_accretion/jet_accretion')
import os
import argparse
import parameters_DICT
import shutil
import scipy
from scipy.constants import *
from scipy import integrate
import pickle
import MCMC
import eval_type
import create_jet
import geometry_binary
import scale_intensity
from astropy import units as u
import datetime
import EW
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
"""
==================================================
Balmer line properties (ionisation and excitation)
==================================================
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


"""
================================
Binary system and jet properties
================================
"""

AU              = 1.496e+11     # 1AU in m
AU_to_km        = 1.496e+08     # 1AU in km
days_to_sec     = 24*60*60      # 1day in seconds
degr_to_rad     = np.pi/180.    # Degrees to radians

###### Read in the object specific and model parameters ########################

InputDir   = 'input_data/'+str(object_id)+'/'
InputFile  = datafile
###### Create the parameter dictionary with all jet, binary, and model parameters

parameters = parameters_DICT.read_parameters(InputDir+InputFile)
parameters['BINARY']['T_inf'] = geometry_binary.T0_to_IC(parameters['BINARY']['omega'],
                                                         parameters['BINARY']['ecc'],
                                                         parameters['BINARY']['period'],
                                                         parameters['BINARY']['T0'])

pars_model = parameters_DICT.read_model_parameters(InputDir+InputFile)

pars_model_array = np.zeros( len(pars_model.keys()) )

for n,param in enumerate(parameters['MODEL'].keys()):

    parameters['MODEL'][param]['id'] = n
    pars_model_array[n] = pars_model[param]

pars_add   = MCMC.calc_additional_par(parameters, pars_model_array)


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
phases_dict = {}
for phase in spectra_observed['halpha'].keys():
    phases.append(phase)
    phases_dict[phase] = []
    for spec in spectra_observed['halpha'][phase].keys():
        spectra.append(spec)
        phases_dict[phase].append(spec)

###### The correct intensity level of the spectra from the       ###############
###### synthetic spectra. We fit a straight line to the relevant ###############
###### part of the synthetic spectra to determine the correct    ###############
###### intensity and scale the other spectra accordingly.        ###############

spectra_synth_wavelengths, spectra_synth_I = np.loadtxt(
               '../jet_accretion/input_data/'+object_id+'/synthetic/'+parameters['OTHER']['synthetic'])
spectra_synth_wavelengths *= 1e-9 # m
spectra_synth_I           *= 1e-7*1e10*1e4 # W m-2 m-1 sr-1

spectra_background_I = {}   # The background spectra with the correct intensity (W m-2 m-1 sr-1)
spectra_observed_I   = {}

created_interpol_normalisation = {line:False for line in balmer_lines}
scaling_interpolation          = {}

for line in balmer_lines:

    spectra_background_I[line] = {}
    spectra_observed_I[line]   = {}

    for ph in phases:

        spectra_background_I[line][ph] = {}
        spectra_observed_I[line][ph]   = {}

        for spec in spectra_observed[line][ph]:

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

    if line == 'halpha' and object_id=='IRAS19135+3937':
        with open('../jet_accretion/input_data/'+object_id+'/'+line+'/'+object_id+'_stdev_init_'+str(line)+'.txt', 'rb') as f:
            uncertainty_background = pickle.load(f)

        for ph in uncertainty_background:
            standard_deviation[line][ph] = {}
            for spectrum in uncertainty_background[ph]:
                standard_deviation[line][ph][spectrum] = \
                            2./signal_to_noise[spectrum] + uncertainty_background[ph][spectrum]
                            # Twice the uncertainty from S/N because the input spectrum is a subtraction between two spectra
                            # --> spec_tot = spec_1 - spec_2
                            # ----> delta_tot = delta_1 + delta_2

    else:
        for ph in phases:
            standard_deviation[line][ph] = {}
            for spectrum in spectra_observed[line][ph]:
                standard_deviation[line][ph][spectrum] = \
                            1./signal_to_noise[spectrum]

"""
======================================
Cut the wavelength region if necessary
======================================
"""

# wavmin = min(range(len(spectra_wavelengths)), key = lambda j: abs(spectra_wavelengths[j]- w_begin))
# wavmax = min(range(len(spectra_wavelengths)), key = lambda j: abs(spectra_wavelengths[j]- w_end))
# spectra_wavelengths = spectra_wavelengths[wavmin:wavmax]
#
# for ph in spectra_observed:
#     for spectrum in spectra_observed[ph]:
#         spectra_observed[ph][spectrum]   = spectra_observed[ph][spectrum][wavmin:wavmax]
#         spectra_background[ph][spectrum] = spectra_background[ph][spectrum][wavmin:wavmax]
#         standard_deviation[ph][spectrum] = standard_deviation[ph][spectrum][wavmin:wavmax]

wavelength_bins     = {}
wavelength_bin_size = {}
for line in balmer_lines:
    wavelength_bins[line] = len(spectra_wavelengths[line])
    wavelength_bin_size[line] = \
            (spectra_wavelengths[line][-1] - spectra_wavelengths[line][0]) / (wavelength_bins[line] - 1)

"""
===========================================
Create the grid for temperature and density
===========================================
"""

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
        TandRho = 'python main_slurm.py -o %s -dat %s -t %.0f -d %.2e\n' % (str(object_id), str(datafile), line[0],line[1])
        f.write(TandRho)


"""
===================================
Create the output folders and files
===================================
"""

###### Create the output directories ###########################################

# Path              = "/fred/oz061/jet_accretion_output/"
Path              = "/lhome/dylanb/astronomy/jet_accretion_output/"
OutputDirObjectID = Path+object_id
OutputDir         = OutputDirObjectID+'/'+object_id+'_'+parameters['OTHER']['jet_type']\
                    +'_T'+str(parameters['OTHER']['T_min'])+'_'+str(parameters['OTHER']['T_max'])\
                    +'_'+str(parameters['OTHER']['T_step'])+'_rho_'\
                    +str(parameters['OTHER']['density_log10_min'])+'_'\
                    +str(parameters['OTHER']['density_log10_max'])\
                    +'_'+str(parameters['OTHER']['density_log10_step'])+'/'

if not os.path.exists(OutputDirObjectID):

    os.makedirs(OutputDirObjectID)

if not os.path.exists(OutputDir):

    os.makedirs(OutputDir)

    ###### Create the log file and copy the inputfile to the output directory ########

    shutil.copyfile(InputDir+InputFile, OutputDir+InputFile)

    OutputLog = open(OutputDir+'output.log', 'w')

    OutputLog.write('This file contains the most important information of the run.\n')
    OutputLog.write('\n')
    OutputLog.write('Date \t %s \n' % datetime.datetime.now())
    OutputLog.write('Object: \t %s \n' % object_id)
    OutputLog.write('Balmer lines: \t %s \n' % str(balmer_lines))
    OutputLog.write('The jet type is %s \n' % parameters['OTHER']['jet_type'])
    OutputLog.write('The inclination angle is %f degrees \n' % (pars_model['inclination']*180./np.pi))
    OutputLog.write('The jet angle is %f degrees\n' % (pars_model['jet_angle']*180./np.pi))
    OutputLog.write('The velocity at the centre of the jet is %f km/s \n' % pars_model['velocity_max'])
    OutputLog.write('The velocity at the edge of the jet is %f km/s \n' % pars_model['velocity_edge'])
    OutputLog.write('The model is computed for jet temperatures between %dK and %dK in steps of %dK. \n' % (parameters['OTHER']['T_min'], parameters['OTHER']['T_max'], parameters['OTHER']['T_step']))
    OutputLog.write('The model is computed for jet densities between 1e%dm^-3 and 1e%dm^-3 in increasing order of magnitudes of %d. \n' % (parameters['OTHER']['density_log10_min'], parameters['OTHER']['density_log10_max'], parameters['OTHER']['density_log10_step']))
    OutputLog.write('\n')
    OutputLog.write('\n')
    # OutputLog.write(' \n' % )
    # OutputLog.write(' \n' % )
    OutputLog.close()

    OutputEWModel = open(OutputDir+'EW_model.txt', 'w')
    OutputEWModel.write('The equivalent width (Angstrom) for the Balmer lines of the models over the whole spectrum\n')
    OutputEWModel.write('Temperature (K) and density\n')
    OutputEWModel.write('Phase\tspectrum\tEW %s \n' % str(balmer_lines))
    OutputEWModel.close()


    """
    ==========================================================================
    Calculate the equivalent width of the spectral lines for several intervals
    ==========================================================================
    """

    range_EW_calculation_km_per_s = 750
    range_EW_calculation_angstrom = {line:range_EW_calculation_km_per_s * 1000 / constants.c * balmer_properties['wavelength'][line] for line in balmer_lines}

    OutputLog = open(OutputDir+'output.log', 'a')
    OutputEW  = open(OutputDir+'EW_observations.txt', 'w')
    OutputLog.write('The equivalent width (Angstrom) for the Balmer lines of the observations over the whole spectrum\n')
    OutputLog.write('Phase\tspectrum\tEW %s \n' % str(balmer_lines))
    OutputEW.write('The equivalent width (Angstrom) for the Balmer lines of the observations over the whole spectrum\n')
    OutputEW.write('Phase\tspectrum\tEW %s \n' % str(balmer_lines))

    for phase in phases:
        for spectrum in spectra_observed['halpha'][phase].keys():

                OutputLog.write('\n%s\t%s\t' % (str(phase), spectrum))
                OutputEW.write('\n%s\t%s\t' % (str(phase), spectrum))

                for line in balmer_lines:

                    spectrum_absorption = spectra_observed[line][phase][spectrum] - spectra_background[line][phase][spectrum] + 1
                    EW_line = EW.equivalent_width(spectra_wavelengths[line]*1e10, spectrum_absorption)
                    OutputLog.write('%.4f\t' % EW_line)
                    OutputEW.write('%.4f\t' % EW_line)

    OutputLog.write('\nThe equivalent width (Angstrom) for the Balmer lines of the observations over a total width of %.3f km/s\n' % (4.*range_EW_calculation_km_per_s) )
    OutputLog.write('Phase\tspectrum\tEW %s \n' % str(balmer_lines))
    OutputEW.write('\nThe equivalent width (Angstrom) for the Balmer lines of the observations over a total width of %.3f km/s\n' % (4.*range_EW_calculation_km_per_s) )
    OutputEW.write('Phase\tspectrum\tEW %s \n' % str(balmer_lines))


    for phase in phases:

        for spectrum in spectra_observed['halpha'][phase].keys():

                OutputLog.write('\n%s\t%s\t' % (str(phase), spectrum))
                OutputEW.write('\n%s\t%s\t' % (str(phase), spectrum))

                for line in balmer_lines:

                    spectrum_absorption = spectra_observed[line][phase][spectrum] - spectra_background[line][phase][spectrum] + 1
                    EW_line = EW.equivalent_width(spectra_wavelengths[line]*1e10, spectrum_absorption,
                                                cut=True,
                                                wave_min=1e10*(balmer_properties['wavelength'][line] - 2*range_EW_calculation_angstrom[line]),
                                                wave_max=1e10*(balmer_properties['wavelength'][line] + 2*range_EW_calculation_angstrom[line]))
                    OutputLog.write('%.4f\t' % EW_line)
                    OutputEW.write('%.4f\t' % EW_line)

    OutputLog.write('\nThe equivalent width (Angstrom) for the Balmer lines of the observations over a total width of %.3f km/s\n' % (2.*range_EW_calculation_km_per_s))
    OutputLog.write('Phase\tspectrum\tEW %s \n' % str(balmer_lines))
    OutputEW.write('\nThe equivalent width (Angstrom) for the Balmer lines of the observations over a total width of %.3f km/s\n' % (2.*range_EW_calculation_km_per_s))
    OutputEW.write('Phase\tspectrum\tEW %s \n' % str(balmer_lines))

    for phase in phases:

        for spectrum in spectra_observed['halpha'][phase].keys():

                OutputLog.write('\n%s\t%s\t' % (str(phase), spectrum))
                OutputEW.write('\n%s\t%s\t' % (str(phase), spectrum))

                for line in balmer_lines:

                    spectrum_absorption = spectra_observed[line][phase][spectrum] - spectra_background[line][phase][spectrum] + 1
                    EW_line = EW.equivalent_width(spectra_wavelengths[line]*1e10, spectrum_absorption,
                                                cut=True,
                                                wave_min=1e10*(balmer_properties['wavelength'][line] - range_EW_calculation_angstrom[line]),
                                                wave_max=1e10*(balmer_properties['wavelength'][line] + range_EW_calculation_angstrom[line]))
                    OutputLog.write('%.4f\t' % EW_line)
                    OutputEW.write('%.4f\t' % EW_line)

    OutputLog.write('\n')
    OutputEW.write('\n')
    OutputLog.close()
    OutputEW.close()


else:

    print('THE DIRECTORY ALREADY EXISTS. CLEAN UP FIRST BEFORE STARTING A NEW JOB, DYLAN!')
