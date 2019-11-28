import sys
sys.path.append('/lhome/dylanb/astronomy/MCMC_main/MCMC_main')
sys.path.append('/lhome/dylanb/astronomy/jet_accretion/jet_accretion')
import argparse
import numpy as np
import matplotlib.pylab as plt
import scipy
from scipy.constants import *
from scipy import integrate
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
                     'f_osc': {'halpha': 6.407e-1, 'hbeta': 1.1938e-1, 'hgamma': 4.4694e-2, 'hdelta': 2.2105e-2},
                     'Aul' : {'halpha': 4.4101e7, 'hbeta': 8.4193e6, 'hgamma' : 2.530e6, 'hdelta': 9.732e5}}
wave_0 = {'halpha': 6562.8e-10, 'hbeta': 4861.35e-10, 'hgamma': 4340.47e-10,\
         'hdelta': 4101.73e-10}
# B_lu = np.array([4.568e14, 6.167e14, 6.907e14, 7.309e14])
B_lu = np.array([1.6842e+21]) # from wikipedia
line = 'halpha'


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
with open('../jet_accretion/input_data/'+str(object_id)+'/'+str(object_id)+'.dat') as f:
    lines  = f.readlines()[2:]

for l in lines:
    split_lines       = l.split()
    title             = split_lines[0]
    value             = split_lines[1]
    parameters[title] = value

###### Wavelength ##############################################################
central_wavelength  = eval(parameters['w_c_'+str(line)])            # (angstrom)
w_begin             = eval(parameters['w_begin_'+str(line)])*1e-10  # (angstrom)
w_end               = eval(parameters['w_end_'+str(line)])*1e-10    # (angstrom)
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

"""
===============
Stellar spectra
===============
"""

###### Observed spectra, background spectra, and wavelength region #############
spectrum_test = '416105'
phase_test    = 30
with open('../jet_accretion/input_data/'+object_id+'/halpha/'+object_id+'_observed_'+line+'.txt', 'rb') as f:
    spectra_observed    = pickle.load(f)
with open('../jet_accretion/input_data/'+object_id+'/halpha/'+object_id+'_wavelength_'+line+'.txt', 'rb') as f:
    spectra_wavelengths = pickle.load(f) * 1e-10
with open('../jet_accretion/input_data/'+object_id+'/halpha/'+object_id+'_init_'+line+'.txt', 'rb') as f:
    spectra_background  = pickle.load(f)
phases = list()
spectra = list()
for ph in spectra_observed.keys():
    phases.append(ph)
    for spec in spectra_observed[ph].keys():
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

for ph in phases:
    spectra_background_I[ph] = {}
    spectra_observed_I[ph]   = {}
    for spec in spectra_observed[ph].keys():
        spectra_background_I[ph][spec], scaling_interpolation = scale_intensity.scale_intensity(wave_0[line],
                                         spectra_synth_wavelengths,
                                         spectra_synth_I, spectra_wavelengths,
                                         spectra_background[ph][spec])
        spectra_observed_I[ph][spec], scaling_interpolation   = scale_intensity.scale_intensity(wave_0[line],
                                         spectra_synth_wavelengths,
                                         spectra_synth_I, spectra_wavelengths,
                                         spectra_observed[ph][spec])
###### uncertainty on the data #################################################

# standard_deviation = {}
# with open('../jet_accretion/input_data/'+object_id+'/halpha/'+object_id+'_signal_to_noise_'+str(line)+'.txt', 'rb') as f:
#     signal_to_noise = pickle.load(f)
# with open('../jet_accretion/input_data/'+object_id+'/halpha/'+object_id+'_stdev_init_'+str(line)+'.txt', 'rb') as f:
#     uncertainty_background = pickle.load(f)
#
# for ph in uncertainty_background:
#     standard_deviation[ph] = {}
#     for spectrum in uncertainty_background[ph]:
#         standard_deviation[ph][spectrum] = \
#                     2./signal_to_noise[spectrum] + uncertainty_background[ph][spectrum]

###### Cut the wavelength region if necessary ##################################

# wavmin = min(range(len(spectra_wavelengths)), key = lambda j: abs(spectra_wavelengths[j]- w_begin))
# wavmax = min(range(len(spectra_wavelengths)), key = lambda j: abs(spectra_wavelengths[j]- w_end))
# spectra_wavelengths = spectra_wavelengths[wavmin:wavmax]
#
# for ph in spectra_observed:
#     for spectrum in spectra_observed[ph]:
#         spectra_observed[ph][spectrum]   = spectra_observed[ph][spectrum][wavmin:wavmax]
#         spectra_background[ph][spectrum] = spectra_background[ph][spectrum][wavmin:wavmax]
#         standard_deviation[ph][spectrum] = standard_deviation[ph][spectrum][wavmin:wavmax]

wavelength_bins = len(spectra_wavelengths)
wavelength_bin_size = \
            (spectra_wavelengths[-1] - spectra_wavelengths[0]) / (wavelength_bins - 1)

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
==============
Create the jet
==============
"""

jet = Cone.Stellar_jet_simple(inclination, jet_angle,
                              velocity_centre, velocity_edge,
                              jet_type,
                              jet_centre=secondary_orbit[phase_test]['position'])
jet_temperature         = 5400      # The jet temperature (K)
jet_density_max         = 2.e15      # The jet number density at its outer edge (m^-3)

jet_thermal_velocity    = ( 2 * constants.k * jet_temperature / constants.m_p)**.5 # The jet thermal velocity (m/s)
jet_frequency_0         = constants.c / balmer_properties['wavelength'][line]
spectra_frequencies = constants.c / spectra_wavelengths


"""
=======================================================================
Calculate the radiative transfer through the jet for each orbital phase
=======================================================================
"""

for phase in phases:
    for spectrum in spectra_observed[phase].keys():
        """
        =============================================================
        Calculate the radiative transfer through the jet for each LOS
        =============================================================
        """
        print('current phase is ', phase)
        postAGB.centre      = primary_orbit[phase]['position']
        jet.jet_centre      = secondary_orbit[phase]['position']
        postAGB._set_grid_location()

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        intensity = np.zeros(wavelength_bins)
        for (pointAGB, coordAGB) in enumerate(postAGB.grid_location):

            ### For each ray of light from a gridpoint on the post-AGB star, we calculate
            ### the absorption by the jet.
            jet._set_gridpoints(coordAGB, gridpoints_LOS)

            if jet.gridpoints is None:
                # intensity_point = list(0.*spectra_background_I[phase][spectrum])
                intensity_point = list(spectra_background_I[phase][spectrum])

            if jet.gridpoints is not None:

                jet._set_gridpoints_unit_vector()
                jet._set_gridpoints_polar_angle()

                ###### Jet velocity and density ###################################
                jet_density_scaled      = jet.density(gridpoints_LOS, power_density)   # The scaled number density of the jet
                jet_density             = jet_density_scaled*jet_density_max   # The number density of the jet at each gridpoint (m^-3)
                jet_velocity            = jet.poloidal_velocity(gridpoints_LOS, power_velocity) # The velocity of the jet at each gridpoint (km/s)
                jet_radvel_km_per_s     = jet.radial_velocity(jet_velocity, secondary_rad_vel) # Radial velocity of each gridpoint (km/s)
                jet_radvel_m_per_s      = jet_radvel_km_per_s * 1000 # Radial velocity of each gridpoint (m/s)
                jet_delta_gridpoints_AU = np.linalg.norm(jet.gridpoints[0,:] - jet.gridpoints[1,:]) # The length of each gridpoint (AU)
                jet_delta_gridpoints_m  = jet_delta_gridpoints_AU * AU  # The length of each gridpoint (m)
                jet_radvel_gradient     = jet.radial_velocity_gradient(jet_radvel_m_per_s, jet_delta_gridpoints_m) # Radial velocity gradient of each gridpoint (s^-1)
                jet_frequency_0_rv      = jet_frequency_0 * (1. - jet_radvel_m_per_s / constants.c) # The shifted central frequency of the line
                jet_delta_nu_thermal    = jet_thermal_velocity * jet_frequency_0_rv / constants.c # The frequency width due to the thermal velocity

                """
                Synthetic line profile and EW for a specific object given a temperature and density
                """

                jet_n_e    = np.zeros(len(jet_density))     # Jet electron number density (m^-3)
                jet_n_HI   = np.zeros(len(jet_density))     # Jet neutral H number density (m^-3)
                jet_n_HI_1 = np.zeros(len(jet_density))     # Jet neutral H in the groundstate (m^-3)
                jet_n_HI_2 = np.zeros(len(jet_density))     # Jet neutral H in the first excited state (m^-3)

                for point, d in enumerate(jet_density):
                    jet_n_e[point]       = ie.n_electron_for_hydrogen(E_ionisation_H, E_levels_H, degeneracy_H, jet_temperature, jet_density[point])
                    jet_n_HI[point]      = jet_density[point] * ie.saha_E(E_ionisation_H, E_levels_H, degeneracy_H, jet_temperature, 1, n_e=jet_n_e[point])
                    jet_n_HI_1[point]    = jet_density[point] * ie.saha_boltz_E(E_ionisation_H, E_levels_H, degeneracy_H, jet_temperature, 1, 1, n=jet_n_e[point]) # HI in energy level n=1
                    jet_n_HI_2[point]    = jet_density[point] * ie.saha_boltz_E(E_ionisation_H, E_levels_H, degeneracy_H, jet_temperature, 1, 2, n=jet_n_e[point]) # HI in energy level n=2

                # plt.semilogy(jet.gridpoints[:,1], jet_density/np.max(jet_density), label='scaled density')
                # plt.semilogy(jet.gridpoints[:,1], jet_n_HI_2/np.max(jet_n_HI_2), label='density HI_2')
                # plt.semilogy(jet.gridpoints[:,1], jet_radvel_km_per_s/np.max(jet_radvel_km_per_s), label='radial velocity')
                # plt.semilogy(jet.gridpoints[:,1], jet.gridpoints[:,2], label='height in jet')
                # plt.semilogy(jet.gridpoints[:,1], jet_radvel_gradient/np.max(jet_radvel_gradient), label='gradient radial velocity')
                # plt.legend()
                # plt.xlabel('distance through jet')
                # plt.ylabel('normalised quantity')
                # plt.show()
                # plt.plot(jet.gridpoints[:,1], jet_radvel_gradient)
                # plt.plot(jet.gridpoints[:,1], jet_radvel_km_per_s)
                # plt.show()

                # intensity_point = 0.*np.copy(spectra_background_I[phase][spectrum])
                intensity_point = np.copy(spectra_background_I[phase][spectrum])

                for pointLOS in range(gridpoints_LOS-1):
                    # We first select the frequencies for which the current point in the jet
                    # will cause absorption
                    diff_nu = np.abs(jet_frequency_0_rv[pointLOS+1] - spectra_frequencies)
                    indices_frequencies = np.where(diff_nu < 2. * jet_delta_nu_thermal[pointLOS+1])

                    for index in indices_frequencies:
                        delta_tau = jet_delta_gridpoints_m \
                                          * opacity(spectra_frequencies[index],
                                                    jet_temperature, jet_n_HI[pointLOS+1], jet_n_e[pointLOS+1],
                                                    jet_n_HI_2[pointLOS+1],
                                                    jet_radvel_m_per_s[pointLOS+1], line=line)
                        intensity_point[index] = rt_isothermal(spectra_wavelengths[index], jet_temperature, intensity_point[index], delta_tau)


            #     intensity_point = []
            #     for wavebin, wave in enumerate(spectra_wavelengths):
            #         # intensity_point.append(spectra_background_I[phase][spectrum][wavebin])
            #         intensity_point.append(0)
            #         if wave > 6540e-10 and wave < 6580e-10:
            #             frequency       = constants.c / wave
            #             delta_tau       = jet_delta_gridpoints_m \
            #                               * opacity(frequency, jet_temperature, jet_n_HI[1:], jet_n_e[1:],
            #                                         jet_n_HI_2[1:], B_lu[0],
            #                                         jet_radvel_m_per_s[1:], line=line)
            #             # delta_tau       = jet_delta_gridpoints_m \
            #             #                   * opacity_rectangular(frequency, jet_temperature,
            #             #                             jet_n_HI_2[1:], wavelength_bin_size, B_lu[0],
            #             #                             jet_radvel_m_per_s[1:],
            #             #                             jet_radvel_gradient[1:],
            #             #                             gridpoints_LOS,
            #             #                             line=line)
            #             # delta_tau       = jet_delta_gridpoints_m \
            #             #                   * opacity_sobolev(frequency, jet_temperature,
            #             #                             jet_n_HI_2[1:],
            #             #                             jet_radvel_m_per_s[1:],
            #             #                             jet_radvel_gradient[1:],
            #             #                             wavelength_bin_size,
            #             #                             gridpoints_LOS,
            #             #                             line=line)
            #             # delta_tau       = jet_delta_gridpoints_m \
            #             #                   * opacity_both(frequency, jet_temperature, jet_n_HI[1:], jet_n_e[1:],
            #             #                             jet_n_HI_2[1:],
            #             #                             jet_delta_gridpoints_m,
            #             #                             B_lu[0],
            #             #                             jet_radvel_m_per_s[1:],
            #             #                             jet_radvel_gradient[1:],
            #             #                             gridpoints_LOS,
            #             #                             line=line)
            #             for pointLOS in range(gridpoints_LOS-1):
            #                 intensity_point[wavebin] = rt_isothermal(wave, jet_temperature, intensity_point[wavebin], delta_tau[pointLOS])
            #

            intensity += gridpoints_primary**-1 * np.array(intensity_point)
            # # plt.plot(spectra_wavelengths, intensity_point)
            # # plt.show()


        ax.plot(spectra_wavelengths*1e10, np.array(intensity), label="absorbed spectrum, n=%.1e m^-3"%(jet_density_max))
        ax.fill_between(spectra_wavelengths*1e10, 0, np.array(intensity), alpha=0.1)
        ax.plot(spectra_wavelengths*1e10, spectra_observed_I[phase][spectrum], label="synthetic photospheric line")
        # ax.plot(wave_range, np.array(I_wrong), label="wrong")
        # ax.fill_between(wave_range, 0, I_0, alpha=0.1) #And fill beneath it with a light shade of the same colour
        ax.set_title(r"Absorption of H$\alpha$ lines by jet", size=16)
        # ax.set_xticklabels(ax.get_xticks()*1e9) #Change x-ticks to be in nm
        # ax.set_yticklabels(ax.get_yticks()*1e-9) #Change y-ticks to be in nm^-1
        ax.grid(lw=0.5)
        ax.plot(spectra_wavelengths*1e10, spectra_background_I[phase][spectrum], label="synthetic, T = 6250K", color = 'green')
        # ax.fill_between(wave_range*1e10, 0, I_0, alpha=0.1, color='green')
        ax.legend()
        ax.set_xlabel(r"Wavelength ($\AA$)")
        ax.set_ylabel("Intensity W m^-2 m^-1 sr^-1")
        ax.set_xlim([6550,6575])

        ax.set_ylim([0, 5.1e13])
        plt.savefig('../docs/output_tests/'+object_id+'/'+object_id+'_'+str(phase)+'_'+spectrum+'.png')




# """
# Example for Halpha
# """
#
#
# lambdas   = np.arange(1e-7,1e-5,1e-8)
# Temp      = np.array([2000,3000,4000,5000]) # jet temperature
# # Temp      = np.array([4000]) # jet temperature
# lams      = np.arange(50, 2000)*1.e-9 #m
# Ts        = np.arange(5e3, 11e3, 1e3) #K
# Temp_star = 6250 #K
#
# ###### Hydrogen properties
# E_ionisation_H      = np.array([13.6, 0]) # ionisation energy in eV
# E_levels_H          = {1: np.array([0, 10.2, 12.1, 12.76]), 2: np.array([0])} # (eV) energy levels of all excitation states for each ionisation level
# degeneracy_H        = {1: np.array([2, 8, 18, 32]), 2: np.array([1])} # Degeneracy of the excitation states
#
# ###### Balmer properties, i.e., einstein coefficients for Halpha, Hbeta, Hgamma, and Hdelta
#
# wave_0 = {'halpha': 6562.8e-10, 'hbeta': 4861.35e-10, 'hgamma': 4340.47e-10,\
#          'hdelta': 4101.73e-10}
# # B_lu = np.array([4.568e14, 6.167e14, 6.907e14, 7.309e14])
# B_lu = np.array([1.6842e+21]) # from wikipedia
# line = 'hbeta'
#
# ###### jet properties
#
# jet_velocity        = 100. * 1e3 #m/s
# jet_velocity_axis   = 800. * 1e3
# jet_velocity_edge   = 10.  * 1e3
# # jet_n               = np.array([1e16,1e18,1e20]) # m-3
# jet_n               = np.array([1e14,1e16,1e18,1e20,1e22,1e24,1e26, 1e28,1e30]) # m-3
#                                # 6.25e12 cm^-3 is based on a mass outflow rate of 10^-8Mdot/yr for
#                                # a jet with radius of 2AU and outflow velocity of 200km/s
# jet_gridpoints      = 100
# jet_pathlength      = (2.*u.au).to(u.m).value # in meters
# jet_angle_out       = 70.*np.pi/180. # outer jet angle
# jet_positions       = np.linspace(0,jet_pathlength, jet_gridpoints) # from 0 to pathlength_jet
# jet_positions_relto = jet_positions - jet_pathlength / 2. # from -0.5*pathlength_jet to 0.5*pathlength_jet
# jet_height          = jet_pathlength / (2. * np.tan(jet_angle_out))
# jet_angles          = np.arctan(jet_positions_relto / jet_height) # jet angles
# # jet_velocities      = jet_velocity + 0 * jet_angles
# jet_velocities      = jet_velocity * np.ones(jet_gridpoints) #jet_velocity_axis + (jet_velocity_edge - jet_velocity_axis)*(jet_angles/jet_angle_out)**2
# jet_radial_velocity = -1. * jet_velocities * np.sin(jet_angles)
#
#
#
# ###### Synthetic spectra as stellar spectra
#
# # wave_range_IRAS, I_IRAS = np.loadtxt('IRAS19135+3937/halpha/IRAS19135+3937_415971.txt')
# # wave_range_beta, I_beta = np.loadtxt('IRAS19135+3937/hbeta/IRAS19135+3937_415971.txt')
# # wave_range_gamma, I_gamma = np.loadtxt('IRAS19135+3937/hgamma/IRAS19135+3937_415971.txt')
# wave_range_IRAS, I_IRAS = np.loadtxt('IRAS19135+3937/halpha/IRAS19135+3937_416105.txt')
# wave_range_beta, I_beta = np.loadtxt('IRAS19135+3937/hbeta/IRAS19135+3937_416105.txt')
# wave_range_gamma, I_gamma = np.loadtxt('IRAS19135+3937/hgamma/IRAS19135+3937_416105.txt')
# # wave_range_IRAS, I_IRAS = np.loadtxt('IRAS19135+3937/halpha/IRAS19135+3937_399553.txt')
# # wave_range_beta, I_beta = np.loadtxt('IRAS19135+3937/hbeta/IRAS19135+3937_399553.txt')
# # wave_range_gamma, I_gamma = np.loadtxt('IRAS19135+3937/hgamma/IRAS19135+3937_399553.txt')
# # wave_range_IRAS = wave_range_IRAS + (6562.8) * 1300/constants.c
# wave_range, I_0    = np.loadtxt('IRAS19135+3937/synthetic/06250_g+1.0_m10p00_hr.txt')
# wave_range        *= 1e-9 #m
# wave_range_IRAS   *= 1e-10
# I_0               *= 1e-7*1e10*1e4 #W m-2 m-1 sr-1
# wave_gridpoints    = len(wave_range)
#
# ###### H_alpha central wavelength and frequency
# wave_0_halpha = 6562.8e-10 #Halpha wavelength in m
# nu_0_halpha   = constants.c / wave_0_halpha
#
# ###### blackbody background: wavelength range and corresponding intensities
# # wave_gridpoints = 200
# # wave_range      = np.linspace(6540., 6580., wave_gridpoints)*1e-10
# # freq_range      = constants.c / wave_range
# # I_0             = planck_w(wave_range, Temp_star)
#
# ###### Equivalent width of the line
#
# import EW
#
# EW_background  = EW.equivalent_width(wave_range, I_0, True, wave_0[line] - 50e-10, wave_0[line] + 50e-10)
#
# EW_background *= 1e10
#
# fig, ax = plt.subplots(1, 1, figsize=(12, 8))
# colors  = ['lightblue', 'blue', 'darkblue']
# EW_all = {}
#
# for T in Temp:
#     EW_all[T] = {}
#     for (n,jet) in enumerate(jet_n):
#         jet_densities = jet + jet_angles*0# * jet_angles**8 / jet_angle_out
#         # jet_densities = jet + jet_angles**8 / jet_angle_out
#
#         jet_n_e    = np.zeros(len(jet_densities))
#         jet_n_HI   = np.zeros(len(jet_densities))
#         jet_n_HI_1 = np.zeros(len(jet_densities))
#         jet_n_HI_2 = np.zeros(len(jet_densities))
#
#         for point, d in enumerate(jet_densities):
#             jet_n_e[point]       = ie.n_electron_for_hydrogen(E_ionisation_H, E_levels_H, degeneracy_H, T, jet_densities[point])
#             jet_n_HI[point]      = jet_densities[point] * ie.saha_E(E_ionisation_H, E_levels_H, degeneracy_H, T, 1, n_e=jet_n_e[point])
#             jet_n_HI_1[point]    = jet_densities[point] * ie.saha_boltz_E(E_ionisation_H, E_levels_H, degeneracy_H, T, 1, 1, n=jet_n_e[point]) # HI in energy level n=1
#             jet_n_HI_2[point]    = jet_densities[point] * ie.saha_boltz_E(E_ionisation_H, E_levels_H, degeneracy_H, T, 1, 2, n=jet_n_e[point]) # HI in energy level n=2
#             # jet_n_HI_
#
#
#         I = []
#         for i, wave in enumerate(wave_range):
#             I.append(I_0[i])
#
#             # if wave > 6500e-10 and wave < 6620e-10:
#             if wave > (wave_0[line] - 60*1e-10) and wave < (wave_0[line] + 60*1e-10):
#                 nu_test   = constants.c / wave
#                 delta_s   = np.abs( jet_positions[1:] - jet_positions[0:-1] )
#                 delta_tau = delta_s * opacity(nu_test, T, jet_n_HI[1:], jet_n_e[1:], jet_n_HI_2[1:], B_lu[0], jet_radial_velocity[1:], line=line)
#                 for point in range(jet_gridpoints-1):
#                     I[i]    = rt_isothermal(wave, T, I[i], delta_tau[point])
#                 # I[i] = rt_num(wave, Temp, I[i], delta_tau)
#
#         EW_line        = EW.equivalent_width(wave_range, I, True, wave_0[line] - 50e-10, wave_0[line] + 50e-10)
#         EW_line       *= 1e10
#         EW_difference  = EW_line - EW_background
#         EW_all[T][jet] = EW_difference
#         print('The difference in equivalenth width for %s with n=%.1e m^-3 and temperature T = %.f is %f' % (line, jet, T, EW_difference))
#         # ax.plot(wave_range*1e10, np.array(I), label="absorbed spectrum, n=%.1e m^-3"%(jet), color=colors[n])
#         # ax.fill_between(wave_range*1e10, 0, np.array(I), alpha=0.1, color=colors[n])
#         # # # ax.fill_between(wave_range, 0, I_0, alpha=0.1) #And fill beneath it with a light shade of the same colour
#         # ax.set_title(r"Absorption of %s lines by jet" % line, size=16)
#         # # # ax.set_xticklabels(ax.get_xticks()*1e9) #Change x-ticks to be in nm
#         # # # ax.set_yticklabels(ax.get_yticks()*1e-9) #Change y-ticks to be in nm^-1
#         # ax.grid(lw=0.5)
#
#
# ax.plot(wave_range*1e10, I_0, label="synthetic, T = 6250K", color = 'green')
# # ax.fill_between(wave_range*1e10, 0, I_0, alpha=0.1, color='green')
# ax.legend(fontsize=12)
# ax.set_xlabel(r"Wavelength ($\AA$)", size=12)
# ax.set_ylabel(r"Intensity ($W m^{-2} m^{-1} sr^{-1}$)", size=12)
# # ax.set_xlim([4850,4870])
# # ax.set_xlim([6555,6570])
# ax.set_xlim([(wave_0[line]*1e10 - 10),(wave_0[line]*1e10 + 10)])
# ax.set_ylim([0,5.5e13])
# ax.tick_params(labelsize=12)
# plt.show()
#
#
# fig, ax = plt.subplots(1, 1, figsize=(12,8))
# import matplotlib.cm as cm
# dT = np.max(EW_all.keys()) - np.min(EW_all.keys())
# colors = (- (np.array([EW_all.keys()]) - min(EW_all.keys()) ) + dT) / dT
# ccolors = cm.jet(colors)
# for (t,key) in enumerate(EW_all):
#     N = []
#     ew = []
#
#     for n in EW_all[key]:
#         N.append(n)
#         ew.append(EW_all[key][n])
#
#     EW = [x for _, x in sorted(zip(N,ew))]
#     N.sort()
#     print(key)
#     ax.semilogx(N,EW, label='%s K' % str(key), color=ccolors[0,t,:])#, alpha=(key/5001.)**2)
#     ax.fill_between(N, 0, EW, color=ccolors[0,t,:], alpha=0.1)
#
# ax.legend()
# ax.set_title(r'EW(H$\alpha$) for different jet temperatures and densities')
# ax.set_xlabel(r'n ($m^{-3}$)')
# ax.set_ylabel(r'EW(H$\alpha$)/EW(H$\beta$)')
# ax.grid(lw=0.5)
# plt.show()
#
#
# '''
# Calculate the ratio of equivalent widths for IRAS19135+3937
# '''
