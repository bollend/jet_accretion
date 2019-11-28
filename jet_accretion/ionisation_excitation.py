import numpy as np
import matplotlib.pylab as plt
import scipy
from scipy.constants import *

def partfunc_E(E_levels, degeneracy, T):
    """
    Calculate the partition function of an ion.

    Parameters
    ==========
    E_levels : np.array
        The energy at each level in units of eV
    degeneracy : np.array
        The degeneracy of each energy level
    T : float
        Temperature in units of Kelvin
    """
    # if isinstance(E_ionisation, float):
    part_func = 0
    can_part  = 0
    for (d,E) in zip(degeneracy, E_levels):

        can_part += d * np.exp(-(E)*e/(k*T))

    part_func += can_part

    # else:
    # N = len(chiI)
    # part_func = np.zeros(len(chiI))
    # for I in range(N):
    #     can_part = 0
    #
    #     for i in range(int(chiI[I]) + 1):
    #
    #         can_part += np.exp(-(i)*e/(k*T))
    #
    #     part_func[I] = can_part
    return part_func

def boltz_E(E_l, deg, T, i):
    """
    Calculate the fraction of particles in a given energy state i,
    of ion I over the total number of particles in ion I.

    Parameters
    ==========
    E_l : np.array
        The energy at each state in units of eV
    deg : np.array
        The degeneracy of each energy state
    T : float
        Temperature in units of Kelvin
    i : integer
        The energy state

    """
    Z   = partfunc_E(E_l, deg, T)
    g   = deg[i-1]
    E_i = E_l[i-1]

    return g/Z * np.exp(-(E_i)*e/(k*T))

def saha_E(chiI_arr, E_l, deg, T, I, P_e=0, n_e=0):
    """
    Calculate the fraction of particles in a given ion state I,
    over the total number of particles in all ion states.

    Parameters
    ==========
    chiI_arr : np.array
        The ionisation energy of the ions in units of eV.
    E_l : dictionary
        The energy at each state for each ion in units of eV
    deg : dictionary
        The degeneracy of each energy state for each ion
    T : float
        Temperature in units of Kelvin
    I : integer
        The ion state
    P_e : float
        The electron pressure in units of Pa
    n_e : float
        The electron number density in units of m**-3
    """
    ion_levels = len(chiI_arr)
    N          = np.zeros(ion_levels)
    N[0]       = 1
    Z_i        = partfunc_E(E_l[1], deg[1], T)

    if not P_e and not n_e:
        N[I-1] = 0

    else:
        if P_e:
            "using electron pressure"
            consts = (2.*k*T)/(h**3*P_e) * (2.*np.pi*m_e*k*T)**1.5

        elif n_e:
            "using electron number density"
            consts = 2./(h**3*n_e) * (2.*np.pi*m_e*k*T)**1.5

        for ionstate in range(1, ion_levels):
            Z_ii = partfunc_E(E_l[ionstate+1], deg[ionstate+1], T)
            N[ionstate] = N[ionstate-1] * Z_ii/Z_i * consts * np.exp(-chiI_arr[ionstate-1]*e/(k*T))
            Z_i = np.copy(Z_ii)

    N_sum = np.sum(N)
    N /= N_sum

    return N[I-1]

def saha_boltz_E(chiI_arr, E_l, deg, T, I, i, p=0, n=0):
    """
    Calculate the Saha-Boltzmann population n_(r,s)/N for level r,s of E
    Gives the fraction of ions in a given energy level and ionization state for
    a given temperature and electron pressure or electron number density.

    Parameters
    ==========
    chiI_arr : np.array
        The ionisation energy of the ions in units of eV.
    E_l : dictionary
        The energy at each state for each ion in units of eV
    deg : dictionary
        The degeneracy of each energy state for each ion
    T : float
        Temperature in units of Kelvin
    I : integer
        The ion state
    i : integer
        The energy state
    P_e : float
        The electron pressure in units of Pa
    n_e : float
        The electron number density in units of m**-3
    """
    if not p and not n:
        Saha_Boltz = 0
    elif p:
        Saha_Boltz = saha_E(chiI_arr, E_l, deg, T, I, P_e=p) * boltz_E(E_l[I], deg[I], T, i)
    elif n:
        Saha_Boltz = saha_E(chiI_arr, E_l, deg, T, I, n_e=n) * boltz_E(E_l[I], deg[I], T, i)

    return Saha_Boltz

def n_electron_for_hydrogen(chiI_arr, E_l, deg, T, n):
    """
    Calculate the electron number density for a pure hydrogen gas
    from the total number density, using the Saha equation

    Parameters
    ==========
    n : float
        The total number density in m^-3
    T : float
        Temperature in K
    """
    ion_levels = len(chiI_arr)
    N          = np.zeros(ion_levels)
    N[0]       = 1
    Z_i        = partfunc_E(E_l[1], deg[1], T)
    Z_ii       = partfunc_E(E_l[2], deg[2], T)
    consts = 2./(h**3) * (2.*np.pi*m_e*k*T)**1.5
    x = Z_ii/Z_i * consts * np.exp(-chiI_arr[0]*e/(k*T))
    n_e = (-x + (x**2 + 4. * x * n)**.5) / 2.
    return n_e

if __name__=='__main__':
    """
    Example for hydrogen
    """
    E_ionisation_H = 13.6
    E_levels_H     = np.array([0, 10.2, 12.1, 12.76])
    degeneracy_H   = np.array([2, 8, 18, 32])
    Temp           = 5400
    Temp_range     = np.arange(1e2, 1e4, 1e2)

    part_func_H = partfunc_E(E_levels_H, degeneracy_H, Temp)
    print('The partition function of neutral Hydrogen at %.fK is ' % Temp, part_func_H)

    fraction_n1 = boltz_E(E_levels_H, degeneracy_H, Temp, 1)
    print('The fraction of hydrogen in energy state n=1 at %.fK is ' % Temp, fraction_n1)

    fraction_n2 = boltz_E(E_levels_H, degeneracy_H, Temp, 2)
    print('The fraction of hydrogen in energy state n=2 at %.fK is ' % Temp, fraction_n2)

    fraction_n2_over_n1 =\
        boltz_E(E_levels_H, degeneracy_H, Temp, 2) /\
        boltz_E(E_levels_H, degeneracy_H, Temp, 1)
    print('The fraction of hydrogen in energy state n=2 over n=1 at %.fK is ' % Temp,\
                fraction_n2_over_n1)

    ### Plot the population of energy states in neutral hydrogen as a function of Temperature
    # fig, ax = plt.subplots(1,1, figsize=(12,8))
    # energy_states = np.arange(1,5)
    # for e_state in energy_states:
    #     fraction_n2_range = boltz_E(E_levels_H, degeneracy_H, Temp_range, e_state)
    #     ax.plot(Temp_range, fraction_n2_range, label="Energy level: %.f" % e_state)
    #     ax.fill_between(Temp_range, 0, fraction_n2_range, alpha=0.1)
    # ax.set_title("Boltzmann Equation for $H\ I$ as a function of temperature \
    #              for a range of energy levels", size=16)
    # ax.set_xlabel("Temperature $T$ ($K$)", size=12)
    # ax.set_ylabel("Fractional concentration in the given energy level $\\frac{N_i^I}{N_I}$", size=12)
    # ax.legend()
    # ax.set_xscale("log")
    # extratick = 1./len(energy_states) #Create an exta tickmark for the y-axis at an even split between all levels
    # ax.set_yticks(np.append(ax.get_yticks(), extratick))
    # # ax.axhline(y=extratick, lw=2, ls="--", color="k", alpha=0.5) #Draw a dashed line at this level
    # ax.set_ylim(-0.05, 1.05)
    # ax.grid(lw=0.5)
    # plt.show()

    " Saha equation for Hydrogen "

    E_ionisation_H      = np.array([13.6, 0])
    E_levels_H          = {1: np.array([0, 10.2, 12.1, 12.76]), 2: np.array([0])}
    degeneracy_H        = {1: np.array([2, 8, 18, 32]), 2: np.array([1])}
    P                   = 1.5
    n_el                = 5.3e+18
    Temp_arr            = np.arange(1e2,2e4, 1e2)
    n                   = 2e14
    print('test', n_electron_for_hydrogen(E_ionisation_H, E_levels_H, degeneracy_H, Temp, n))
    fraction_HII_over_H = saha_E(E_ionisation_H, E_levels_H, degeneracy_H, Temp, 2, n_e=n_el)
    print("The fraction of ionised hydrogen for T = %.f and Pe = %.3f is" \
                % (Temp, P), fraction_HII_over_H)

    fraction_array = np.zeros(len(Temp_arr))

    # fig, ax = plt.subplots(1,1, figsize=(12,8))
    # ionstates = np.array([1,2])
    # for ion in ionstates:
    #     for index,t in enumerate(Temp_arr):
    #         fraction_array[index] = saha_E(E_ionisation_H, E_levels_H, degeneracy_H, t, ion, P_e=P)
    #     ax.plot(Temp_arr, fraction_array, label="ionstate: H%.f" % ion)
    #     ax.fill_between(Temp_arr, 0, fraction_array, alpha=0.1)
    # ax.set_title("fraction of HII over H as a function of T for HI (neutral) and HII (ionised))", size=16)
    # ax.set_xlabel("Temperature $T$ ($K$)", size=12)
    # ax.set_ylabel("Fractional concentration in the given ion state $\\frac{HII}{H}$", size=12)
    # ax.legend()
    # ax.set_xscale("log")
    # ax.set_ylim(-0.05, 1.05)
    # ax.grid(lw=0.5)
    # plt.show()

    " Saha-Boltzmann equation"

    ion         = 1
    level       = 2
    fraction_SB = saha_boltz_E(E_ionisation_H, E_levels_H, degeneracy_H, Temp, ion, level, n=n_el)
    print("The fraction of hydrogen in ion stage %.f in level %.f for T = %.f and Pe = %.3f is" \
                % (ion, level, Temp, P), fraction_SB)

    # fig, ax = plt.subplots(1,1, figsize=(12,8))
    #
    # for ion in E_levels_H:
    #     y = []
    #     for T in Temp_range:
    #         y.append(saha_boltz_E(E_ionisation_H, E_levels_H, degeneracy_H, T, ion, 1, p=P))
    #     ax.plot(Temp_range, y, label="Ionisation state I=%.0f" % ion) #Plot solid line
    #     ax.fill_between(Temp_range, 0, y, alpha=0.1) #Plot shaded region beneath curve
    #
    # ax.set_title("Relative strengths of ionisation states of Ca in \
    #                 the ground energy level as a function of temperature", size=16)
    # ax.set_xlabel("Temperature $T$ ($K$)")
    # ax.set_ylabel("Fraction of total H in given ionisation state $\\frac{N_I}{N_{TOT}}$")
    # ax.legend()
    # ax.grid(lw=0.5)
    # plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(12,8))

    for I, col in zip(E_levels_H,\
                ["xkcd:red", "xkcd:blue", "xkcd:green", "xkcd:orange", "xkcd:purple"]):
        for i in range(len(E_levels_H[I])):

            y = []

            for T in Temp_range:
                y.append(saha_boltz_E(E_ionisation_H, E_levels_H, degeneracy_H, T, I, i+1, n=n_el))
            if i==0:
                ax.plot(Temp_range, y, color=col, label="Ionisation state I=%.0f, ground state" % I)
                ax.plot(0,0, color=col, ls="--", lw=1, label="Higher energy levels")
                ax.fill_between(Temp_range, 0, y, alpha=0.1, color=col)
            else:
                ax.plot(Temp_range, y, color=col, ls="--", lw=1)
    ax.set_title("Fraction of total H in given ionisation state $I$ and energy level $i$ as a function of temperature", size=16)
    ax.set_xlabel("Temperature $T$ ($K$)")
    ax.set_ylabel("Fraction of total H in given ionisation state $I$ and energy level $i$")
    ax.legend()
    ax.set_yscale("log")
    ax.set_ylim(10e-20, 1e2)
    ax.grid(lw=0.5)
    plt.show()
