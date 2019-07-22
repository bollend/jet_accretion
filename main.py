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
sys.path.append('lhome/dylanb/astronomy/MCMC_main/MCM_main')
import Cone

from astropy import units as u
lambdas   = np.arange(1e-7,1e-5,1e-8)
Temp      = 9000 # jet temperature
lams      = np.arange(50, 2000)*1.e-9 #m
Ts        = np.arange(5e3, 11e3, 1e3) #K
Temp_star = 6250 #K

###### Hydrogen properties
E_ionisation_H      = np.array([13.6, 0]) # ionisation energy in eV
E_levels_H          = {1: np.array([0, 10.2, 12.1, 12.76]), 2: np.array([0])} # (eV) energy levels of all excitation states for each ionisation level
degeneracy_H        = {1: np.array([2, 8, 18, 32]), 2: np.array([1])} # Degeneracy of the excitation states

###### einstein coefficients for Halpha, Hbeta, Hgamma, and Hdelta
B_lu = np.array([4.568e14, 6.167e14, 6.907e14, 7.309e14])

###### jet properties
jet_velocity        = 200. * 1e3 #m/s
jet_velocity_axis   = 800. * 1e3
jet_velocity_edge   = 10.  * 1e3
jet_n               = 3.   * 1e10 # m-3
                               # 6.25e12 cm^-3 is based on a mass outflow rate of 10^-8Mdot/yr for
                               # a jet with radius of 2AU and outflow velocity of 200km/s
jet_gridpoints      = 100
jet_pathlength      = (2.*u.au).to(u.m).value # in meters
jet_angle_out       = 75.*np.pi/180 # outer jet angle

###### jet number densities

# jet_densities = jet_n
# jet_n_e       = ie.n_electron_for_hydrogen(E_ionisation_H, E_levels_H, degeneracy_H, Temp, jet_densities)
# jet_n_HI      = jet_densities * ie.saha_E(E_ionisation_H, E_levels_H, degeneracy_H, Temp, 1, n_e=jet_n_e)
# jet_n_HI_1    = jet_densities * ie.saha_boltz_E(E_ionisation_H, E_levels_H, degeneracy_H, Temp, 1, 1, n=jet_n_e) # HI in energy level n=1
# jet_n_HI_2    = jet_densities * ie.saha_boltz_E(E_ionisation_H, E_levels_H, degeneracy_H, Temp, 1, 2, n=jet_n_e) # HI in energy level n=2

# jet_densities = jet_n * jet_angles**8 / jet_angle_out
# jet_densities = jet_n + 0 * jet_angles**8 / jet_angle_out

# jet_n_e    = np.zeros(len(jet_densities))
# jet_n_HI   = np.zeros(len(jet_densities))
# jet_n_HI_1 = np.zeros(len(jet_densities))
# jet_n_HI_2 = np.zeros(len(jet_densities))
#
# for point, d in enumerate(jet_densities):
#     jet_n_e[point]       = ie.n_electron_for_hydrogen(E_ionisation_H, E_levels_H, degeneracy_H, Temp, jet_densities[point])
#     jet_n_HI[point]      = jet_densities[point] * ie.saha_E(E_ionisation_H, E_levels_H, degeneracy_H, Temp, 1, n_e=jet_n_e[point])
#     jet_n_HI_1[point]    = jet_densities[point] * ie.saha_boltz_E(E_ionisation_H, E_levels_H, degeneracy_H, Temp, 1, 1, n=jet_n_e[point]) # HI in energy level n=1
#     jet_n_HI_2[point]    = jet_densities[point] * ie.saha_boltz_E(E_ionisation_H, E_levels_H, degeneracy_H, Temp, 1, 2, n=jet_n_e[point]) # HI in energy level n=2
#     # jet_n_HI_

###### Synthetic spectra as stellar spectra

wave_range, I_0    = np.loadtxt('IRAS19135+3937/synthetic/06250_g+1.0_m10p00_hr.txt')

###### Observed spectra

wave_range_IRAS, I_IRAS = np.loadtxt('IRAS19135+3937/halpha/IRAS19135+3937_416105.txt')
wave_range_beta, I_beta = np.loadtxt('IRAS19135+3937/hbeta/IRAS19135+3937_416105.txt')
wave_range_gamma, I_gamma = np.loadtxt('IRAS19135+3937/hgamma/IRAS19135+3937_416105.txt')

###### Initial background spectra

f = open('IRAS19135+3937/IRAS19135+3937_var2_init_interp2.txt', 'rb')
I0_background_all = pickle.load(f)
I0_background = I0_background_all[45]['416105']
f.close()
f = open('IRAS19135+3937/IRAS19135+3937_var2_wavelength_halpha_abs.txt', 'rb')
wave_range_background = pickle.load(f, encoding='latin1')
f.close()

###### Wavelength range

wave_range_background *= 1e-10 #m
wave_range            *= 1e-9 #m
wave_range_IRAS       *= 1e-10
I_0                   *= 1e-7*1e10*1e4 #W m-2 m-1 sr-1
wave_gridpoints        = len(wave_range)

###### Change intensity of background spectrum according to the synthetic spectrum

value_low  = 6520e-10
value_high = 6600e-10
print(wave_range)
idx_low    = (np.abs(wave_range - value_low)).argmin()
idx_high   = (np.abs(wave_range - value_high)).argmin()
I_low      = I_0[idx_low]
I_high     = I_0[idx_high]
print(idx_low, idx_high, I_low, I_high)
m                 = (I_high - I_low) / (value_high - value_low)
b                 = I_low - m * value_low
wavelength_factor = b + m * wave_range_background
I0_background     = I0_background * wavelength_factor


plt.plot(wave_range, I_0)
plt.plot(wave_range_background, I0_background)
plt.plot(wave_range_background, wavelength_factor)
plt.show()

###### H_alpha central wavelength and frequency
wave_0_halpha = 6562.8e-10 #Halpha wavelength in m
nu_0_halpha   = constants.c / wave_0_halpha

###### blackbody background: wavelength range and corresponding intensities
# wave_gridpoints = 200
# wave_range      = np.linspace(6540., 6580., wave_gridpoints)*1e-10
# freq_range      = constants.c / wave_range
# I_0             = planck_w(wave_range, Temp_star)


I = []
for i, wave in enumerate(wave_range):
    # I.append(I_0[i])
    I.append(0)
    if wave > 6520e-10 and wave < 6600e-10:
        nu_test         = constants.c / wave
        delta_s         = np.abs( jet_positions[1:] - jet_positions[0:-1] )
        delta_tau = delta_s * opacity(nu_test, Temp, jet_n_HI[1:], jet_n_e[1:], jet_n_HI_2[1:], B_lu[0], jet_radial_velocity[1:])
        for point in range(jet_gridpoints-1):
            I[i]    = rt_isothermal(wave, Temp, I[i], delta_tau[point])

I_wrong = []
for i, wave in enumerate(wave_range):
    I_wrong.append(I_0[i])
    if wave > 6520e-10 and wave < 6600e-10:
        nu_test         = constants.c / wave
        delta_s         = np.abs( jet_positions[1:] - jet_positions[0:-1] )
        delta_tau = delta_s * opacity(nu_test, Temp, jet_densities[1:], jet_n_e[1:], jet_n_HI_1[1:], B_lu[0], jet_radial_velocity[1:], line='halpha')
        for point in range(jet_gridpoints-1):
            I_wrong[i]    = rt_isothermal(wave, Temp, I_wrong[i], delta_tau[point])



fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.plot(wave_range, np.array(I), label="absorbed synthetic spectrum")
ax.plot(wave_range, I_0, label="synthetic, T = 6250K")
ax.fill_between(wave_range, 0, I_0, alpha=0.1)
ax.fill_between(wave_range_IRAS, 0, I_IRAS, alpha=0.1, color="darkblue")
ax.plot(wave_range, np.array(I_wrong), label="wrong")
ax.fill_between(wave_range, 0, I_0, alpha=0.1) #And fill beneath it with a light shade of the same colour
ax.set_title("Absorption of Balmer lines by jet", size=16)
# ax.set_xticklabels(ax.get_xticks()*1e9) #Change x-ticks to be in nm
# ax.set_yticklabels(ax.get_yticks()*1e-9) #Change y-ticks to be in nm^-1
ax.grid(lw=0.5)
ax.legend()

plt.show()
