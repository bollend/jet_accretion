import numpy as np
import matplotlib.pylab as plt
import scipy
from scipy.constants import *
from scipy import integrate
from sympy import mpmath as mp
import ionisation_excitation as ie
import radiative_transfer as rt
import pickle
import sys
sys.path.append('/lhome/dylanb/astronomy/MCMC_main/MCMC_main')
import Cone
from radiative_transfer import *
from astropy import units as u

"""
Balmer line properties (ionisation and excitation)
"""

###### Hydrogen properties
E_ionisation_H      = np.array([13.6, 0]) # ionisation energy in eV
E_levels_H          = {1: np.array([0, 10.2, 12.1, 12.76]), 2: np.array([0])} # (eV) energy levels of all excitation states for each ionisation level
degeneracy_H        = {1: np.array([2, 8, 18, 32]), 2: np.array([1])} # Degeneracy of the excitation states

###### Balmer properties, i.e., einstein coefficients for Halpha, Hbeta, Hgamma, and Hdelta

wave_0 = {'halpha': 6562.8e-10, 'hbeta': 4861.35e-10, 'hgamma': 4340.47e-10,\
         'hdelta': 4101.73e-10}
# B_lu = np.array([4.568e14, 6.167e14, 6.907e14, 7.309e14])
B_lu = np.array([1.6842e+21]) # from wikipedia
line = 'hbeta'

"""
Synthetic spectra as stellar spectra
"""

star                    = 'IRAS19135+3937'
spectrum                = '416105'
wave_range_IRAS, I_IRAS_obs = np.loadtxt(star+'/halpha/'+star+'_'+spectrum+'.txt')

"""
Jet properties
"""

jet = Cone.Stellar_jet_simple(78.8/180.*np.pi, 75.9/180.*np.pi, np.array([0,1,0]), \
                            1210, 11, 'simple_stellar_jet')
print(jet)


"""
Synthetic line profile and EW for a specific object given a temperature and density
"""




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
