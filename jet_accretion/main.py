import sys
sys.path.append('/lhome/dylanb/astronomy/MCMC_main/MCMC_main')
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
import ionisation_excitation as ie
import radiative_transfer as rt
import pickle
import Cone
import geometry_binary
import scale_intensity
from radiative_transfer import *
from astropy import units as u
import datetime
import EW

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    ============================================================================
"   We first initialise the spectral line properties, input data (spectra),
"   orbital parameters, and jet parameters that we need to calculate the
"   absorption by the jet.
"   Next, create the post-AGB star (as a Fibonacci-grid), the binary system and
"   the jet configuration.
"   We then calculate the amount of absorption by the jet in the spectral line,
"   and its equivalent width and compare it with the observations.
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
parser.add_argument('-l', dest='line', help='The spectral line')

args          = parser.parse_args()
object_id     = args.object_id
spectral_line = args.line

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
parameters = {}
InputDir   = 'input_data/'+str(object_id)+'/'
InputFile = str(object_id)+'.dat'
with open(InputDir+InputFile) as f:
    lines  = f.readlines()[2:]

for l in lines:
    split_lines       = l.split()
    title             = split_lines[0]
    value             = split_lines[1]
    parameters[title] = value

###### Wavelength ##############################################################
# central_wavelength  = eval(parameters['w_c_'+str(line)])            # (angstrom)
# w_begin             = eval(parameters['w_begin_'+str(line)])*1e-10  # (angstrom)
# w_end               = eval(parameters['w_end_'+str(line)])*1e-10    # (angstrom)
###### Binary system and stellar parameters ####################################
omega               = eval(parameters['omega'])    # Argument of periastron (degrees)
ecc                 = eval(parameters['ecc'])      # Eccentricity
T0                  = eval(parameters['T0'])       # Time of periastron (days)
period              = eval(parameters['period'])   # period (days)
primary_asini       = eval(parameters['asini'])    # asini of the primary (AU)
primary_rad_vel     = eval(parameters['K_p'])      # Radial velocity primary (km s^-1)
primary_sma_a1      = eval(parameters['R_p'])      # SMA of the primary (a1)
primary_mass        = eval(parameters['m_p'])	   # mass primary (M_sol)
primary_Teff        = eval(parameters['T_eff'])    # Surface temperature of the primary (K)
mass_function       = eval(parameters['fm'])       # mass function (AU)
angular_frequency   = 2. * np.pi / period          # angular frequency (days^-1)
gridpoints_LOS      = eval(parameters['points_pathlength']) # number of points along the path length trough the jet
gridpoints_primary  = eval(parameters['points_primary'])    # number of points on the primary star
synthetic           = parameters['synthetic']
###### Jet model solution parameters ###########################################
jet_type            = parameters['jet_type']                     # None
inclination         = eval(parameters['incl']) * degr_to_rad     # radians
jet_angle           = eval(parameters['jet_angle']) * degr_to_rad# radians
const_optical_depth = eval(parameters['const_optical_depth'])    # None
velocity_centre     = eval(parameters['velocity_centre'])        # km s^-1
velocity_edge       = eval(parameters['velocity_edge'])          # km s^-1
primary_radius_a1   = eval(parameters['radius_primary_a1'])      # a1
primary_radius_au   = eval(parameters['radius_primary_au'])      # AU
power_density       = eval(parameters['c_den'])                  # None
power_velocity      = eval(parameters['c_vel'])                  # None
###### Binary system and stellar parameters from jet solution ##################
primary_sma_AU      = primary_asini / np.sin(inclination) # SMA of the primary (AU)
primary_max_vel     = primary_rad_vel / np.sin(inclination) # Orbital velocity (km/s)
secondary_mass      = geometry_binary.calc_mass_sec(primary_mass, inclination, mass_function) # (AU)
mass_ratio          = primary_mass / secondary_mass       # None
secondary_sma_AU    = primary_sma_AU * mass_ratio         # SMA of the secondary (AU)
secondary_rad_vel   = primary_rad_vel * mass_ratio        # Radial velocity secondary (km/s)
secondary_max_vel   = primary_max_vel * mass_ratio        # Orbital velocity secondary (km/s)
T_inf               = geometry_binary.T0_to_IC(omega, ecc, period, T0)
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

spectra_synth_wavelengths, spectra_synth_I = np.loadtxt(
               '../jet_accretion/input_data/'+object_id+'/synthetic/'+synthetic)
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

    if line == 'halpha':
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
            for spec in spectra_observed[line][ph]:
                standard_deviation[line][ph][spec] = \
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
=======================
Create the binary orbit
=======================
"""

primary_orbit = {}
secondary_orbit = {}

for ph in phases:
    prim_pos, sec_pos, prim_vel, sec_vel = geometry_binary.pos_vel_primary_secondary(
                                           ph, period, omega, ecc, primary_sma_AU,
                                           secondary_sma_AU, T_inf, T0)
    primary_orbit[ph]               = {}
    secondary_orbit[ph]             = {}
    primary_orbit[ph]['position']   = prim_pos
    primary_orbit[ph]['velocity']   = prim_vel
    secondary_orbit[ph]['position'] = sec_pos
    secondary_orbit[ph]['velocity'] = sec_vel

"""
==========================================
Create post-AGB star with a Fibonacci grid
==========================================
"""

import Star
postAGB = Star.Star(primary_radius_au, np.array([0,0,0]), inclination, gridpoints_primary)
postAGB._set_grid()
postAGB._set_grid_location()

"""
==============
Create the jet
==============
"""

jet = Cone.Stellar_jet_simple(inclination, jet_angle,
                              velocity_centre, velocity_edge,
                              jet_type)

"""
===========================================
Create the grid for temperature and density
===========================================
"""

jet_temperatures = np.arange(T_min, T_max+1, T_step)
jet_density_log  = np.arange(density_log10_min, density_log10_max+0.001, density_log10_step)
jet_densities    = 10**(jet_density_log)

"""
===================================
Create the output folders and files
===================================
"""

###### Create the output directories ###########################################

Path              = "../../jet_accretion_output/"
OutputDirObjectID = Path+object_id
OutputDir         = OutputDirObjectID+'/'+object_id+'_'+jet_type+'_T'+str(T_min)+'_'+str(T_max)+'_'+str(T_step)+'_rho_'+str(density_log10_min)+'_'+str(density_log10_max)+'_'+str(density_log10_step)

if not os.path.exists(OutputDirObjectID):

    os.makedirs(OutputDirObjectID)

NewDirNumber = 0
NewDirCreated = False

while NewDirCreated==False:

    if not os.path.exists(OutputDir+'_'+str(NewDirNumber)):

        os.makedirs(OutputDir+'_'+str(NewDirNumber))
        NewDirCreated = True
    else:

        NewDirNumber += 1

OutputDir = OutputDir+'_'+str(NewDirNumber)+'/'

###### Create the log file and copy the inputfile to the output directory ########

shutil.copyfile(InputDir+InputFile, OutputDir+InputFile)

OutputLog = open(OutputDir+'output.log', 'w')

OutputLog.write('This file contains the most important information of the run.\n')
OutputLog.write('\n')
OutputLog.write('Date \t %s \n' % datetime.datetime.now())
OutputLog.write('Object: \t %s \n' % object_id)
OutputLog.write('Balmer lines: \t %s \n' % str(balmer_lines))
OutputLog.write('The jet type is %s \n' % jet_type)
OutputLog.write('The inclination angle is %f degrees \n' % (inclination*180./np.pi))
OutputLog.write('The jet angle is %f degrees\n' % (jet_angle*180./np.pi))
OutputLog.write('The velocity at the centre of the jet is %f km/s \n' % velocity_centre)
OutputLog.write('The velocity at the edge of the jet is %f km/s \n' % velocity_edge)
OutputLog.write('The model is computed for jet temperatures between %dK and %dK in steps of %dK. \n' % (T_min, T_max, T_step))
OutputLog.write('The model is computed for jet densities between 1e%dm^-3 and 1e%dm^-3 in increasing order of magnitudes of %d. \n' % (density_log10_min, density_log10_max, density_log10_step))
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

                # if line=='halpha':
                # plt.plot(spectra_wavelengths[line]*1e10,spectra_background[line][phase][spectrum])
                # plt.plot(spectra_wavelengths[line]*1e10, 1-spectrum_absorption )
                # plt.plot(spectra_wavelengths[line]*1e10,spectra_observed[line][phase][spectrum])
                # plt.axvline(1e10*(balmer_properties['wavelength'][line] - 2*range_EW_calculation_angstrom[line]))
                # plt.axvline(1e10*(balmer_properties['wavelength'][line] - 1*range_EW_calculation_angstrom[line]))
                # plt.axvline(1e10*(balmer_properties['wavelength'][line] + 2*range_EW_calculation_angstrom[line]))
                # plt.axvline(1e10*(balmer_properties['wavelength'][line] + 1*range_EW_calculation_angstrom[line]))
                # plt.axvline(1e10*balmer_properties['wavelength'][line], color='k')
                # plt.axhline(1., color='k')
                # plt.axhline(0., color='k')
                # plt.show()

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


"""
===========================================================================
Compute the spectral lines for the whole grid of temperatures and densities
===========================================================================
"""

for jet_temperature in jet_temperatures:
    ###### The jet temperature (K)

    jet_thermal_velocity = ( 2 * constants.k * jet_temperature / constants.m_p)**.5 # The jet thermal velocity (m/s)

    for jet_density_max in jet_densities:
        ###### The jet number density at its outer edge (m^-3)

        with open(OutputDir+'EW_model.txt', 'a') as f_out:

            f_out.write('\n')
            f_out.write('\n%.0f\t%.2e\t' % (jet_temperature,jet_density_max))

        OutputDirTempRho = '%.0f_%.2e' % (jet_temperature,jet_density_max)
        os.makedirs(OutputDir+OutputDirTempRho)

        for line in balmer_lines:

            os.makedirs(OutputDir+OutputDirTempRho+'/'+line)

        for phase in phases:
            ###### Calculate the radiative transfer through the jet for each
            ###### orbital phase

            postAGB.centre      = primary_orbit[phase]['position']
            jet.jet_centre      = secondary_orbit[phase]['position']
            postAGB._set_grid_location()

            for spectrum in spectra_observed['halpha'][phase].keys():
                ###### Iterate over all spectra with this phase

                intensity = {line:np.zeros(wavelength_bins[line]) for line in balmer_lines}

                for (pointAGB, coordAGB) in enumerate(postAGB.grid_location):
                    ###### For each ray of light from a gridpoint on the
                    ###### post-AGB star, we calculate the absorption by the jet

                    jet._set_gridpoints(coordAGB, gridpoints_LOS)

                    if jet.gridpoints is None:
                        ###### The ray does not pass through the jet

                        # intensity_point = list(0.*spectra_background_I[phase][spectrum])
                        intensity_point = {line:list(spectra_background_I[line][phase][spectrum]) for line in balmer_lines}

                    if jet.gridpoints is not None:
                        ###### The ray passes through the jet

                        jet._set_gridpoints_unit_vector()
                        jet._set_gridpoints_polar_angle()

                        ###### Jet velocity and density ########################
                        jet_density_scaled      = jet.density(gridpoints_LOS, power_density)   # The scaled number density of the jet
                        jet_density             = jet_density_scaled*jet_density_max   # The number density of the jet at each gridpoint (m^-3)
                        jet_velocity            = jet.poloidal_velocity(gridpoints_LOS, power_velocity) # The velocity of the jet at each gridpoint (km/s)
                        jet_radvel_km_per_s     = jet.radial_velocity(jet_velocity, secondary_rad_vel) # Radial velocity of each gridpoint (km/s)
                        jet_radvel_m_per_s      = jet_radvel_km_per_s * 1000 # Radial velocity of each gridpoint (m/s)
                        jet_delta_gridpoints_AU = np.linalg.norm(jet.gridpoints[0,:] - jet.gridpoints[1,:]) # The length of each gridpoint (AU)
                        jet_delta_gridpoints_m  = jet_delta_gridpoints_AU * AU  # The length of each gridpoint (m)
                        jet_radvel_gradient     = jet.radial_velocity_gradient(jet_radvel_m_per_s, jet_delta_gridpoints_m) # Radial velocity gradient of each gridpoint (s^-1)
                        # The shifted central frequency of the line
                        jet_frequency_0_rv      = {line:balmer_properties['frequency'][line] * (1. - jet_radvel_m_per_s / constants.c) for line in balmer_lines}
                        # The frequency width due to the thermal velocity
                        jet_delta_nu_thermal    = {line:jet_thermal_velocity * jet_frequency_0_rv[line] / constants.c for line in balmer_lines}

                        """
                        Synthetic line profile and EW for a specific object given a temperature and density
                        """

                        jet_n_e    = np.zeros(len(jet_density))     # Jet electron number density (m^-3)
                        jet_n_HI   = np.zeros(len(jet_density))     # Jet neutral H number density (m^-3)
                        jet_n_HI_2 = np.zeros(len(jet_density))     # Jet neutral H in the first excited state (m^-3)

                        for point, d in enumerate(jet_density):
                            jet_n_e[point]       = ie.n_electron_for_hydrogen(E_ionisation_H, E_levels_H, degeneracy_H, jet_temperature, jet_density[point])
                            jet_n_HI[point]      = jet_density[point] * ie.saha_E(E_ionisation_H, E_levels_H, degeneracy_H, jet_temperature, 1, n_e=jet_n_e[point])
                            jet_n_HI_2[point]    = jet_density[point] * ie.saha_boltz_E(E_ionisation_H, E_levels_H, degeneracy_H, jet_temperature, 1, 2, n=jet_n_e[point]) # HI in energy level n=2

                        # intensity_point = 0.*np.copy(spectra_background_I[phase][spectrum])
                        intensity_point = {line:np.copy(spectra_background_I[line][phase][spectrum]) for line in balmer_lines}

                        for pointLOS in range(gridpoints_LOS-1):
                            # We first select the frequencies for which the current point in the jet
                            # will cause absorption

                            for line in balmer_lines:

                                diff_nu = np.abs(jet_frequency_0_rv[line][pointLOS+1] - spectra_frequencies[line])
                                indices_frequencies = np.where(diff_nu < 2.2 * jet_delta_nu_thermal[line][pointLOS+1])

                                for index in indices_frequencies:

                                    delta_tau = jet_delta_gridpoints_m \
                                                      * opacity(spectra_frequencies[line][index],
                                                                jet_temperature, jet_n_HI[pointLOS+1], jet_n_e[pointLOS+1],
                                                                jet_n_HI_2[pointLOS+1],
                                                                jet_radvel_m_per_s[pointLOS+1], line=line)
                                    intensity_point[line][index] = rt_isothermal(spectra_wavelengths[line][index], jet_temperature, intensity_point[line][index], delta_tau)

                    for line in balmer_lines:
                        ###### add the spectrum for this ray to the lines

                        intensity[line] += gridpoints_primary**-1 * np.array(intensity_point[line])

                with open(OutputDir+'EW_model.txt', 'a') as f_out:

                    f_out.write('\n%s\t%s\t' % (str(phase), spectrum))

                for line in balmer_lines:
                    ###### write the output for this phase to the output directory

                    Header = 'Synthetic %s line at phase %d' % (line, phase)
                    np.savetxt(OutputDir+OutputDirTempRho+'/'+line+'/'+spectrum+'_'+str(phase)+'_'+line+'.txt', np.array(intensity[line]), header=Header)

                    ###### Normalise the spectrum to compute the equivalent width
                    spectrum_absorption = np.array(intensity[line]) / scaling_interpolation[line](spectra_wavelengths[line]) - spectra_background[line][phase][spectrum] + 1
                    EW_line = EW.equivalent_width(spectra_wavelengths[line]*1e10, spectrum_absorption,
                                                    cut=True,
                                                    wave_min=1e10*(balmer_properties['wavelength'][line] - 1*range_EW_calculation_angstrom[line]),
                                                    wave_max=1e10*(balmer_properties['wavelength'][line] + 1*range_EW_calculation_angstrom[line]))

                    with open(OutputDir+'EW_model.txt', 'a') as f_out:

                        f_out.write('%.4f\t' % EW_line)









print('done')
