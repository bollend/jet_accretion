import numpy as np
import matplotlib.pylab as plt
import scipy
import astropy.constants as consts
from scipy.constants import *
from scipy import integrate
from sympy import mpmath as mp
import ionisation_excitation as ie

def planck_w(lam, T):
    """
    Calculate the Planck function for given wavelength range and temperature
    in units of W * sr^-1 * m^-2 * m^-1

    Parameters
    ==========
    lam : np.array
        Wavelength range in units of m^-1
    T : float
        Temperature in units of Kelvin
    """
    return ((2*h*c**2)/(lam**5))*(1./(np.exp((h*c)/(lam*k*T))-1))

def planck_f(nu, T):
    """
    Calculate the Planck function for given frequency range and temperature
    in units of W * sr^-1 * m^-2 * Hz^-1

    Parameters
    ==========
    nu : np.array
        Frequency range in units of Hz
    T : float
        Temperature in units of Kelvin
    """
    return ((2*h*nu**3)/(c**2))*(1./(np.exp((h*nu)/(k*T))-1))

def rt_general(lam, T, I_0, tau):
    """
    Calculate the intensity through a layer

    Parameters
    ==========
    lam : np.array
        Wavelenght range in units of m^-1
    T : np.array
        Temperature at each postition in the layer in units of Kelvin
    I_0 : float
        Initial intensity in units of W * sr^-1 * m^-2 * m^-1
    tau : np.array
        Optical depth at each position in the layer
    """

    B_lam               = planck_w(lam, T)
    tau_total           = tau[-1]
    delta_tau           = tau[1:] - tau[0:-1]
    integrate_radiation = B_lam[1:] * np.exp(- (tau_total - tau[:])) * delta_tau
    return I_0 * np.exp(-tau) + B_lam * (1 - np.exp(-tau))


def rt_isothermal(lam, T, I_0, tau):
    """
    Calculate the intensity through an isothermal layer (T and B(T) are independent of x)

    Parameters
    ==========
    lam : np.array
        Wavelength range in units of m^-1
    T : float
        Temperature in units of Kelvin
    I_0 : float
        Initial intensity in units of W * sr^-1 * m^-2 * m^-1
    tau : float
        Optical depth
    """
    B_lam = planck_w(lam, T)
    return I_0 * np.exp(-tau) + B_lam * (1 - np.exp(-tau))

def rt_layers(lam, T, T_0, tau):
    """
    Calculate the intensity through an isothermal layer (T and B(T) are independent of x)
    where the surface layer radiates as a blackbody

    Parameters
    ==========
    lam : np.array
        Wavelenght range in units of m^-1
    T : float
        Temperature in units of Kelvin
    T_0 : float
        Temperature of surface layer in units of Kelvin
    I_0 : float
        Initial intensity in units of W * sr^-1 * m^-2 * m^-1
    tau : float
        Optical depth
    """
    B_lam = planck_w(lam, T)
    B_surf = planck_w(lam, T_0)
    return B_surf * np.exp(-tau) + B_lam * (1 - np.exp(-tau))

def rt_num(lam, T, I_0, tau):
    """
    Calculate the intensity through several layers (numerically)

    Parameters
    ==========
    lam : np.array
        Wavelength range in units of m^-1
    T : float
        Temperature in units of Kelvin
    I_0 : float
        Initial intensity in units of W * sr^-1 * m^-2 * m^-1
    tau : float
        Optical depth
    """
    B_lam = planck_w(lam, T)
    exp_tau_i_n = np.zeros(len(tau)-2)
    for i in range(len(exp_tau_i_n)):
        exp_tau_i_n[i] = np.exp(-(np.sum(tau[i+1:])))

    I = I_0 * np.exp(-(np.sum(tau))) + B_lam * (1 - np.exp(-tau[-1]))\
        + B_lam * np.sum( (1 - np.exp(-tau[1:-1])) * exp_tau_i_n)

    return I


def u_a(nu, nu_0, n_HI, n_e, Temp, rv, line='halpha'):
    broadening_constants = {'rad' : {'halpha' : 6.5e-14, 'hbeta' : 1, 'hgamma' : 1, 'hdelta' : 1},
                            'vdw' : {'halpha' : 4.4e-14, 'hbeta' : 1, 'hgamma' : 1, 'hdelta' : 1},
                            'stark' : {'halpha' : 1.17e-13, 'hbeta' : 1, 'hgamma' : 1, 'hdelta' : 1}}

    if line=='halpha':

        nu_0       = constants.c / ( 6562.8 * 1e-10 )
        nu_0_rv    = nu_0 * (1. - rv / constants.c)
        C_rad      = 8.2 * 1e-3 * 1e-10
        C_vdw      = 5.5 * 1e-3 * 1e-10
        C_stark    = 1.47 * 1e-2 * 1e-10
        delta_nu   = ( 2 * constants.k * Temp / constants.m_p)**.5 * nu_0_rv/constants.c
        gamma_damp = C_rad + C_vdw * ( n_HI / 10**22 ) * ( Temp / 5000 )**0.3 + C_stark * ( n_e / 10**18 )**(2./3.)
        a          = gamma_damp / ( 4. * np.pi * delta_nu )
        u          = ( nu - nu_0_rv ) / delta_nu

    if line=='hbeta':
        nu_0       = constants.c / ( 4861.3 * 1e-10 )
        nu_0_rv    = nu_0 * (1. - rv / constants.c)
        C_rad      = 1e6 * 1e-10
        C_vdw      = 5.5 * 1e-3 * 1e-20
        C_stark    = 1.47 * 1e-2 * 1e-20
        delta_nu   = ( 2 * constants.k * Temp / constants.m_p)**.5 * nu_0_rv/constants.c
        gamma_damp = C_rad + C_vdw * ( n_HI / 10**22 ) * ( Temp / 5000 )**0.3 + C_stark * ( n_e / 10**18 )**(2./3.)
        a          =1. / ( 4. * np.pi )
        u          = ( nu - nu_0_rv ) / delta_nu

    return u, a, delta_nu

def voigt(u, a):
    """
    Calculate the voigt profile H(a,u)
    """
    with mp.workdps(20):
        # z = mp.mpc(u, a)
        z      = u + a*1j
        result = np.exp(-z.astype(complex)*z.astype(complex)) * scipy.special.erfc(-1j*z.astype(complex))

    where_are_NaNs              = np.isnan(result.real)
    result.real[where_are_NaNs] = 0

    return result.real

def opacity_sobolev(nu, T, n_HI, n_e, n_l, Blu, rv, line='halpha'):
    """
    Calculate the absorption coefficient alpha

    Parameters
    ==========
    nu : float
        The wavelength in units of Hz
    T : float
        Temperature in units of Kelvin
    n_HI : float
        The number density of HI in units of m^-3
    n_e : float
        The electron number density in units of m^-3
    n_l : float
        The number density of the lower energy level in units of m^-3
    Blu : float
        The einstein coefficient for excitation
    rv : float
        Radial velocity of the gridpoint in m^1 s^-1

    returns
    =======
    abs_coeff_si : float
        The absorption coefficient in units of m^-1
    """
    n_l_cgs = n_l * 1e-6
    balmer_properties = {'wavelength': {'halpha': 6562.8e-10, 'hbeta': 4861.35e-10, 'hgamma': 4340.47e-10, 'hdelta': 4101.73e-10},
                         'f_osc': {'halpha': 6.407e-1, 'hbeta': 1.1938e-1, 'hgamma': 4.4694e-2, 'hdelta': 2.2105e-2},
                         'Aul' : {'halpha': 4.4101e7, 'hbeta': 8.4193e6, 'hgamma' : 2.530e6, 'hdelta': 9.732e5}}
    nu_0              = constants.c / balmer_properties['wavelength'][line]
    f_line            = balmer_properties['f_osc'][line]

    u, a, delta_nu = u_a(nu, nu_0, n_HI, n_e, T, rv, line=line)
    phi_nu = voigt(u, a) / ( np.pi**.5 * delta_nu )
    C = ( consts.h.cgs.value * Blu / (4. * np.pi) )
    C_osc = ( np.pi * consts.e.esu.value**2 / (consts.m_e.cgs.value * consts.c.cgs.value ) )

    abs_coeff_cgs = C_osc * f_line * phi_nu * n_l_cgs * ( 1 - np.exp(-constants.h * nu_0 / (constants.k * T )))
    abs_coeff_si = abs_coeff_cgs * 1e2

    return abs_coeff_si

def opacity(nu, T, n_HI, n_e, n_l, Blu, rv, line='halpha'):
    """
    Calculate the absorption coefficient alpha

    Parameters
    ==========
    nu : float
        The wavelength in units of Hz
    T : float
        Temperature in units of Kelvin
    n_HI : float
        The number density of HI in units of m^-3
    n_e : float
        The electron number density in units of m^-3
    n_l : float
        The number density of the lower energy level in units of m^-3
    Blu : float
        The einstein coefficient for excitation
    rv : float
        Radial velocity of the gridpoint in m^1 s^-1

    returns
    =======
    abs_coeff_si : float
        The absorption coefficient in units of m^-1
    """
    n_l_cgs = n_l * 1e-6
    balmer_properties = {'wavelength': {'halpha': 6562.8e-10, 'hbeta': 4861.35e-10, 'hgamma': 4340.47e-10, 'hdelta': 4101.73e-10},
                         'f_osc': {'halpha': 6.407e-1, 'hbeta': 1.1938e-1, 'hgamma': 4.4694e-2, 'hdelta': 2.2105e-2},
                         'Aul' : {'halpha': 4.4101e7, 'hbeta': 8.4193e6, 'hgamma' : 2.530e6, 'hdelta': 9.732e5}}
    nu_0              = constants.c / balmer_properties['wavelength'][line]
    f_line            = balmer_properties['f_osc'][line]

    u, a, delta_nu = u_a(nu, nu_0, n_HI, n_e, T, rv, line=line)
    phi_nu = voigt(u, a) / ( np.pi**.5 * delta_nu )
    C = ( consts.h.cgs.value * Blu / (4. * np.pi) )
    C_osc = ( np.pi * consts.e.esu.value**2 / (consts.m_e.cgs.value * consts.c.cgs.value ) )

    # return C * nu_0 * phi_nu * n_l * ( 1 - np.exp(-constants.h * nu / (constants.k * T )))
    # return C_osc * f_line * phi_nu * n_l * ( 1 - np.exp(-constants.h * nu / (constants.k * T )))
    # return (6.626e-27 * Blu /4./np.pi) * nu_0 * phi_nu * n_l * ( 1 - np.exp(-constants.h * nu / (constants.k * T )))
    abs_coeff_cgs = C_osc * f_line * phi_nu * n_l_cgs * ( 1 - np.exp(-constants.h * nu_0 / (constants.k * T )))
    abs_coeff_si = abs_coeff_cgs * 1e2

    return abs_coeff_si



if __name__=='__main__':
    """
    Example for Halpha
    """
    from astropy import units as u
    lambdas   = np.arange(1e-7,1e-5,1e-8)
    Temp      = 4000 # jet temperature
    lams      = np.arange(50, 2000)*1.e-9 #m
    Ts        = np.arange(5e3, 11e3, 1e3) #K
    Temp_star = 6250 #K

    ###### Hydrogen properties
    E_ionisation_H      = np.array([13.6, 0]) # ionisation energy in eV
    E_levels_H          = {1: np.array([0, 10.2, 12.1, 12.76]), 2: np.array([0])} # (eV) energy levels of all excitation states for each ionisation level
    degeneracy_H        = {1: np.array([2, 8, 18, 32]), 2: np.array([1])} # Degeneracy of the excitation states

    ###### einstein coefficients for Halpha, Hbeta, Hgamma, and Hdelta

    # B_lu = np.array([4.568e14, 6.167e14, 6.907e14, 7.309e14])
    B_lu = np.array([1.6842e+21]) # from wikipedia

    ###### jet properties

    jet_velocity        = 100. * 1e3 #m/s
    jet_velocity_axis   = 800. * 1e3
    jet_velocity_edge   = 10.  * 1e3
    jet_n               = np.array([1e20,1e18,1e16]) # m-3
                                   # 6.25e12 cm^-3 is based on a mass outflow rate of 10^-8Mdot/yr for
                                   # a jet with radius of 2AU and outflow velocity of 200km/s
    jet_gridpoints      = 100
    jet_pathlength      = (2.*u.au).to(u.m).value # in meters
    jet_angle_out       = 70.*np.pi/180 # outer jet angle
    jet_positions       = np.linspace(0,jet_pathlength, jet_gridpoints) # from 0 to pathlength_jet
    jet_positions_relto = jet_positions - jet_pathlength / 2. # from -0.5*pathlength_jet to 0.5*pathlength_jet
    jet_height          = jet_pathlength / (2. * np.tan(jet_angle_out))
    jet_angles          = np.arctan(jet_positions_relto / jet_height) # jet angles
    # jet_velocities      = jet_velocity + 0 * jet_angles
    jet_velocities      = jet_velocity * np.ones(jet_gridpoints) #jet_velocity_axis + (jet_velocity_edge - jet_velocity_axis)*(jet_angles/jet_angle_out)**2
    jet_radial_velocity = -1. * jet_velocities * np.sin(jet_angles)



    ###### Synthetic spectra as stellar spectra

    # wave_range_IRAS, I_IRAS = np.loadtxt('IRAS19135+3937/halpha/IRAS19135+3937_415971.txt')
    # wave_range_beta, I_beta = np.loadtxt('IRAS19135+3937/hbeta/IRAS19135+3937_415971.txt')
    # wave_range_gamma, I_gamma = np.loadtxt('IRAS19135+3937/hgamma/IRAS19135+3937_415971.txt')
    wave_range_IRAS, I_IRAS = np.loadtxt('IRAS19135+3937/halpha/IRAS19135+3937_416105.txt')
    wave_range_beta, I_beta = np.loadtxt('IRAS19135+3937/hbeta/IRAS19135+3937_416105.txt')
    wave_range_gamma, I_gamma = np.loadtxt('IRAS19135+3937/hgamma/IRAS19135+3937_416105.txt')
    # wave_range_IRAS, I_IRAS = np.loadtxt('IRAS19135+3937/halpha/IRAS19135+3937_399553.txt')
    # wave_range_beta, I_beta = np.loadtxt('IRAS19135+3937/hbeta/IRAS19135+3937_399553.txt')
    # wave_range_gamma, I_gamma = np.loadtxt('IRAS19135+3937/hgamma/IRAS19135+3937_399553.txt')
    # wave_range_IRAS = wave_range_IRAS + (6562.8) * 1300/constants.c
    wave_range, I_0    = np.loadtxt('IRAS19135+3937/synthetic/06250_g+1.0_m10p00_hr.txt')
    wave_range        *= 1e-9 #m
    wave_range_IRAS   *= 1e-10
    I_0               *= 1e-7*1e10*1e4 #W m-2 m-1 sr-1
    wave_gridpoints    = len(wave_range)

    ###### H_alpha central wavelength and frequency
    wave_0_halpha = 6562.8e-10 #Halpha wavelength in m
    nu_0_halpha   = constants.c / wave_0_halpha

    ###### blackbody background: wavelength range and corresponding intensities
    # wave_gridpoints = 200
    # wave_range      = np.linspace(6540., 6580., wave_gridpoints)*1e-10
    # freq_range      = constants.c / wave_range
    # I_0             = planck_w(wave_range, Temp_star)

    ###### jet number densities

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    colors  = ['darkblue', 'blue', 'lightblue']

    for (n,jet) in enumerate(jet_n):
        jet_densities = jet * jet_angles**8 / jet_angle_out
        # jet_densities = jet + 0 * jet_angles**8 / jet_angle_out

        jet_n_e    = np.zeros(len(jet_densities))
        jet_n_HI   = np.zeros(len(jet_densities))
        jet_n_HI_1 = np.zeros(len(jet_densities))
        jet_n_HI_2 = np.zeros(len(jet_densities))

        for point, d in enumerate(jet_densities):
            jet_n_e[point]       = ie.n_electron_for_hydrogen(E_ionisation_H, E_levels_H, degeneracy_H, Temp, jet_densities[point])
            jet_n_HI[point]      = jet_densities[point] * ie.saha_E(E_ionisation_H, E_levels_H, degeneracy_H, Temp, 1, n_e=jet_n_e[point])
            jet_n_HI_1[point]    = jet_densities[point] * ie.saha_boltz_E(E_ionisation_H, E_levels_H, degeneracy_H, Temp, 1, 1, n=jet_n_e[point]) # HI in energy level n=1
            jet_n_HI_2[point]    = jet_densities[point] * ie.saha_boltz_E(E_ionisation_H, E_levels_H, degeneracy_H, Temp, 1, 2, n=jet_n_e[point]) # HI in energy level n=2
            # jet_n_HI_




        I = []
        for i, wave in enumerate(wave_range):
            I.append(I_0[i])
            # I.append(0)
            if wave > 6520e-10 and wave < 6600e-10:
            # if wave > 4840e-10 and wave < 4880e-10:
                nu_test         = constants.c / wave
                delta_s         = np.abs( jet_positions[1:] - jet_positions[0:-1] )
                delta_tau = delta_s * opacity(nu_test, Temp, jet_n_HI[1:], jet_n_e[1:], jet_n_HI_2[1:], B_lu[0], jet_radial_velocity[1:], line='halpha')
                for point in range(jet_gridpoints-1):
                    I[i]    = rt_isothermal(wave, Temp, I[i], delta_tau[point])


        ax.plot(wave_range*1e10, np.array(I), label="absorbed spectrum, n=%.1e m^-3"%(jet), color=colors[n])
        ax.fill_between(wave_range*1e10, 0, np.array(I), alpha=0.1, color=colors[n])
        # ax.plot(wave_range, np.array(I_wrong), label="wrong")
        # ax.fill_between(wave_range, 0, I_0, alpha=0.1) #And fill beneath it with a light shade of the same colour
        ax.set_title(r"Absorption of H$\alpha$ lines by jet", size=16)
        # ax.set_xticklabels(ax.get_xticks()*1e9) #Change x-ticks to be in nm
        # ax.set_yticklabels(ax.get_yticks()*1e-9) #Change y-ticks to be in nm^-1
        ax.grid(lw=0.5)


    ax.plot(wave_range*1e10, I_0, label="synthetic, T = 6250K", color = 'green')
    # ax.fill_between(wave_range*1e10, 0, I_0, alpha=0.1, color='green')
    ax.legend()
    ax.set_xlabel(r"Wavelength ($\AA$)")
    ax.set_ylabel("Intensity W m^-2 m^-1 sr^-1")
    # ax.set_xlim([4850,4870])
    ax.set_xlim([6550,6575])

    ax.set_ylim([0, 3.1e13])
    plt.show()


    # wav_synthetic, flux_synthetic = np.loadtxt('IRAS19135+3937/synthetic/06250_g+1.0_m10p00_hr.txt')
    # wav_synthetic   *= 1e-9 #m
    # flux_synthetic  *= 1e-7*1e10*1e4*1e-2 #W m-2 m-1 sr-1
    #
    # wav_synthetic, flux_synthetic = np.loadtxt('IRAS19135+3937/synthetic/06250_g+1.0_m10p00_hr.txt')
    # wav_synthetic_sed, flux_synthetic_sed = np.loadtxt('IRAS19135+3937/synthetic/06250_g+1.0_m10p00_sed.txt')
    # wav_synthetic_marcs, flux_synthetic_marcs = np.loadtxt('IRAS19135+3937/synthetic/marcs_6250_m10_10/06250_g+1.0_m10p00_marcs_angstrom_ergscm-2s-1A-1.txt')
    # flux_synthetic       *= 1e-7*1e10*1e4
    # flux_synthetic_sed   *= 1e-7*1e10*1e4
    # flux_synthetic_marcs *= 1e-7*1e10*1e4/np.pi
    # fig, ax = plt.subplots(1, 1, figsize=(12,8))
    # ax.plot(wav_synthetic*10**-9, flux_synthetic, label="synthetic, T = 6250K")
    # ax.plot(lams, planck_w(lams, Temp_star), label="T = %.0fK" % Temp_star) #Plot a curve for each temperature
    # ax.plot(wav_synthetic_sed, flux_synthetic_sed, label="syntheticSED, T = 6250K")
    # ax.plot(wav_synthetic_marcs*1e-10, flux_synthetic_marcs, label="synthetic marcs T=6250")
    # ax.fill_between(lams, 0, planck_w(lams, Temp_star), alpha=0.1) #And fill beneath it with a light shade of the same colour
    # ax.set_title("planck_w function ", size=16)
    # # ax.set_xticklabels(ax.get_xticks()*1e9) #Change x-ticks to be in nm
    # ax.set_xlabel("Wavelength $\lambda$ ")
    # # ax.set_yticklabels(ax.get_yticks()*1e-9) #Change y-ticks to be in nm^-1
    # ax.set_ylabel("Spectral radiance $B$ ")
    # ax.grid(lw=0.5)
    # ax.legend()
    # plt.show()


    # fig, ax = plt.subplots(1, 1, figsize=(12,8))
    #
    # for T in Ts: #Step through the range of temperatures
    #     ax.plot(lams, planck_w(lams, T), label="T = %.0fK" % T) #Plot a curve for each temperature
    #     ax.fill_between(lams, 0, planck_w(lams, T), alpha=0.1) #And fill beneath it with a light shade of the same colour
    # ax.set_title("Planck function for a range of temperatures", size=16)
    # ax.set_xticklabels(ax.get_xticks()*1e9) #Change x-ticks to be in nm
    # ax.set_xlabel("Wavelength $\lambda$ ($nm$)")
    # ax.set_yticklabels(ax.get_yticks()*1e-9*1e-3) #Change y-ticks to be in nm^-1 and kW
    # ax.set_ylabel("Spectral radiance $B$ ($kW \cdot m^{-2} \cdot nm^{-1} \cdot sr^{-1}$)")
    # ax.grid(lw=0.5)
    # ax.legend()
    # plt.show()

    B_lam = 2.0
    taus = np.arange(0.01, 10, 0.01)
    I_lam_zeros = [4] #np.arange(1e12, 1e13, 2e12)

    # fig, ax = plt.subplots(1, 1, figsize=(12,8))
    #
    # for I_lam_zero in I_lam_zeros:
    #     ax.plot(taus, rt_isothermal(656.2e-9, Temp, I_lam_zero, taus), label="Initial beam intensity $I_{\lambda}(0)$ = %.0f" % I_lam_zero)
    # ax.set_title("Emergent intensity $I_{\lambda}$ as a function of optical depth $\\tau$", size=16)
    # ax.set_xlabel("Optical thickness of isothermal layer $\\tau$")
    # ax.set_ylabel("Emergent intensity $I_{\lambda}$")
    # ax.grid(lw=0.5)
    # ax.legend()
    # plt.show()

    ''' CAN BE REMOVED
    B_lam = 2.0
    taus = np.arange(0.01, 10, 0.001)
    I_lam_zeros = [0]
    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    for I_lam_zero in I_lam_zeros:
        ax.plot(np.log10(taus), np.log10(rt_isothermal(6562.8, Temp, I_lam_zero, taus)), label="Initial beam intensity $I_{\lambda}(0)$ = %.0f" % I_lam_zero)
    xfit = np.log10(taus)
    cut = 10
    m, b = np.polyfit(xfit[:cut], np.log10(rt_isothermal(6562.8, Temp, I_lam_zero, taus[:cut])), deg=1)
    yfit = m*xfit + b
    ax.plot(xfit, yfit, ls="--", color="k", label="Linear fit for $\tau \ll 1$: y=(%.3f)x+(%.3f)" % (m, b))
    ax.set_title("Emergent intensity $I_{\lambda}$ as a function of optical depth $\\tau$ for initial beam intensity $I_{\lambda}(0) = 0$", size=16)
    ax.set_xlabel("Log of optical thickness of isothermal layer $\\tau$")
    ax.set_ylabel("Log of emergent intensity $I_{\lambda}$")
    ax.grid(lw=0.5)
    ax.legend()
    plt.show()


    B_lam = 2.0
    taus = np.arange(0.01, 10, 0.001)
    I_lam_zeros = [4]
    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    for I_lam_zero in I_lam_zeros:
        ax.plot(np.log10(taus), np.log10(I_lam_zero*np.exp(-taus) + B_lam*(1-np.exp(-taus))), label="Initial beam intensity $I_{\lambda}(0)$ = %.0f" % I_lam_zero)
    xfit = np.log10(taus)
    cut = 10
    m, b = np.polyfit(xfit[:cut], np.log10(I_lam_zero*np.exp(-taus[:cut]) + B_lam*(1-np.exp(-taus[:cut]))), deg=1)
    yfit = m*xfit + b
    ax.plot(xfit, yfit, ls="--", color="k", label="Linear fit for $\tau \ll 1$: y=(%.3f)x+(%.3f)" % (m, b))
    ax.set_title("Emergent intensity $I_{\lambda}$ as a function of optical depth $\\tau$ for initial beam intensity $I_{\lambda}(0)$ = %.0f" % I_lam_zeros[0], size=16)
    ax.set_xlabel("Log of optical thickness of isothermal layer $\\tau$")
    ax.set_ylabel("Log of emergent intensity $I_{\lambda}$")
    ax.grid(lw=0.5)
    ax.legend()
    plt.show()
    '''


    # lams = np.arange(300, 1100, 100)*1.e-9
    # T_surface = 5772
    # T_layer = 5400
    # B_lam_T = planck_w(T_surface, lams)
    # B_lam_T_layer = planck_w(T_layer, lams)
    # taus = np.arange(0.01, 10, 0.01)
    # fig, ax = plt.subplots(1, 1, figsize=(12,8))
    # for lam, I_lam_zero, B_lam in zip(lams, B_lam_T, B_lam_T_layer):
    #     ax.plot(taus, I_lam_zero*np.exp(-taus) + B_lam*(1-np.exp(-taus)), label="Wavelength $\lambda$ = %.1f $\mu m$" % (lam*1e6))
    # ax.set_title("Emergent intensity $I_{\lambda}$ as a function of constant optical depth $\\tau$ for\n$T_{surface}$ = %.0fK and $T_{layer}$ = %.0fK for a range of wavelengths" % (T_surface, T_layer), size=16)
    # ax.set_xlabel("Optical thickness of isothermal layer $\\tau$")
    # ax.set_ylabel("Emergent intensity $I_{\lambda}$")
    # ax.grid(lw=0.5)
    # ax.legend()
    # plt.show()



    '''
    Voigt profile
    '''

    # u_array  = np.arange(-10., 10.1, 0.1)
    # lambdas  = np.arange(6550, 6580, .01)*1e-10
    # freqs    = constants.c / lambdas
    # C_rad    = 8.2*1e-3*1e-10
    # C_vdw    = 5.5*1e-3*1e-10
    # C_stark  = 1.47*1e-2*1e-10
    # n_H      = 10e20
    # n_e      = 1e-10
    # a        = 0.1
    # lambda_0 = 6562.8e-10
    # nu_0     = constants.c / lambda_0
    #
    # fig, ax  = plt.subplots(1, 1, figsize=(12,8))
    #
    # # y = []
    # # for u in u_array:
    # #     y.append(voigt(u, a))
    # # ax.plot(u_array, y, label="Voigt profile for $a$ = %.1f" % a)
    # # ax.set_title("Voigt profile for $a=0.1$ as a function of $u$", size=16)
    # # ax.set_xlabel("Relative wavelengths around centre of line $u$")
    # # ax.set_ylabel("Voight line intensity")
    # # ax.grid(lw=0.5)
    # # plt.show()
    # # print(np.trapz(np.array(y), u_array))
    #
    #
    # delta_nu     = ( 2 * constants.k * Temp / constants.m_p)**.5 * nu_0/constants.c
    # freqs        = np.arange(nu_0 - 10*delta_nu, nu_0 + 10*delta_nu, delta_nu/10)
    # lambdas      = constants.c / freqs
    # gamma_damp   = C_rad + C_vdw*(n_H/10**22) * (Temp/5000)**0.3 + C_stark * (n_e/10**18)**(2./3.)
    # a            = gamma_damp / (4.*np.pi*delta_nu)
    # u            = ( freqs - nu_0 ) / delta_nu
    # voigt_result = voigt(u, a)
    # delta_lambda = delta_nu * lambda_0**2/constants.c
    #
    #
    # plt.plot(freqs, voigt_result/np.pi**.5/delta_nu)
    # plt.show()
