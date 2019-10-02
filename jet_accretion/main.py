import sys
sys.path.append('/lhome/dylanb/astronomy/MCMC_main/MCMC_main')
sys.path.append('/lhome/dylanb/astronomy/jet_accretion/jet_accretion')
import os
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
with open('input_data/'+str(object_id)+'/'+str(object_id)+'.dat') as f:
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
spectra_background  = {}
for line in balmer_lines:
    with open('../jet_accretion/input_data/'+object_id+'/'+line+'/'+object_id+'_observed_'+line+'.txt', 'rb') as f:
        spectra_observed[line]    = pickle.load(f)
    with open('../jet_accretion/input_data/'+object_id+'/'+line+'/'+object_id+'_wavelength_'+line+'.txt', 'rb') as f:
        spectra_wavelengths[line] = pickle.load(f) * 1e-10
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


for line in balmer_lines:
    spectra_background_I[line] = {}
    spectra_observed_I[line]   = {}
    for ph in phases:
        spectra_background_I[line][ph] = {}
        spectra_observed_I[line][ph]   = {}
        for spec in spectra_observed[line][ph]:
            spectra_background_I[line][ph][spec] = scale_intensity.scale_intensity(balmer_properties['wavelength'][line],
                                             spectra_synth_wavelengths,
                                             spectra_synth_I, spectra_wavelengths[line],
                                             spectra_background[line][ph][spec])
            spectra_observed_I[line][ph][spec]   = scale_intensity.scale_intensity(balmer_properties['wavelength'][line],
                                             spectra_synth_wavelengths,
                                             spectra_synth_I, spectra_wavelengths[line],
                                             spectra_observed[line][ph][spec])

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
postAGB = Star.Star(primary_radius_au, primary_orbit[phase_test]['position'], inclination, gridpoints_primary)
postAGB._set_grid()
postAGB._set_grid_location()

"""
===========================================
Create the grid for temperature and density
===========================================
"""

jet_temperature = np.arange(T_min, T_max+1, T_step)
jet_density_log = np.arange(density_log10_min, density_log10_max+0.001, density_log10_step)
jet_density     = 10**(jet_density_log)


"""
=========================
Create the output folders
=========================
"""



"""
==============
Create the jet
==============
"""

jet = Cone.Stellar_jet_simple(inclination, jet_angle,
                              velocity_centre, velocity_edge,
                              jet_type,
                              jet_centre=secondary_orbit[phase_test]['position'])


jet_temperature         = 5000      # The jet temperature (K)
jet_density_max         = 1.e15      # The jet number density at its outer edge (m^-3)

jet_thermal_velocity    = ( 2 * constants.k * jet_temperature / constants.m_p)**.5 # The jet thermal velocity (m/s)
jet_frequency_0         = constants.c / balmer_properties['wavelength'][line]
spectra_frequencies = constants.c / spectra_wavelengths
