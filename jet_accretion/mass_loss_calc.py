import numpy as np
import matplotlib.pylab as plt
from scipy import integrate
import sys
import argparse

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
================================
constants and jet properties
================================
"""

m_p                       = 1.6726e-27               # kg
AU                        = 1.496e+11                # m
M_sol                     = 1.988e30                 # kg
year                      = 60*60*24*365             # s
degr_to_rad               = np.pi/180.


height_0_AU               = 1                        # AU
height_0_m                = AU * height_0_AU  # m

### from Bollen19
# jet_angle                 = 75.6/180*np.pi           # radians
# jet_density_max           = 1e14                     # m^-3
# jet_density_max_kg_per_m3 = jet_density_max # kg/m^3
# jet_velocity_max          = 1.21e6                   # m/s
# jet_velocity_min          = 1.1e4                    # m/s

### random
# jet_angle                 = 67.27 * degr_to_rad # radians
# velocity_centre           = 898         # km/s
# velocity_edge             = 19.77           # km/s
# jet_velocity_max          = 1000 * velocity_centre                      # m/s
# jet_velocity_min          = 1000 * velocity_edge                        # m/s
# jet_density_max           = 2e15                                        # m^-3
# jet_density_max_kg_per_m3 = jet_density_max                             # kg/m^3

###### Read in the object specific and model parameters ########################
parameters = {}
with open('../jet_accretion/input_data/'+str(object_id)+'/'+str(object_id)+'.dat') as f:
    lines  = f.readlines()[2:]

for l in lines:
    split_lines       = l.split()
    title             = split_lines[0]
    value             = split_lines[1]
    parameters[title] = value

jet_angle                 = eval(parameters['jet_angle']) * degr_to_rad # radians
velocity_centre           = eval(parameters['velocity_centre'])         # km/s
velocity_edge             = eval(parameters['velocity_edge'])           # km/s
jet_type                  = parameters['jet_type']
jet_velocity_max          = 1000 * velocity_centre                      # m/s
jet_velocity_min          = 1000 * velocity_edge                        # m/s
jet_density_max           = 3e15                                        # m^-3
jet_density_max_kg_per_m3 = jet_density_max                             # kg/m^3



###### Calculation based on a simple jet with rho propto theta^8

def func_1(x):
    return np.arctan(x)**8 * x

def func_2(x):
    return np.arctan(x)**10 * x

I_int  = integrate.quad(func_1, 0*np.tan(50/180*np.pi) , np.tan(jet_angle))
II_int = integrate.quad(func_2, 0*np.tan(50/180*np.pi) , np.tan(jet_angle))

jet_mass_loss_particle_per_s = 2. * np.pi * jet_density_max_kg_per_m3\
                * height_0_m**2 * jet_angle**-8\
                * ( jet_velocity_max * I_int[0] + \
                (jet_velocity_min - jet_velocity_max)/jet_angle**2 * II_int[0])

jet_mass_loss_kg_per_s = m_p * jet_mass_loss_particle_per_s
jet_mass_loss_Msol_per_yr =  M_sol**-1 * year * jet_mass_loss_kg_per_s

print('The jet mass-loss rate for a density of %.2e m^-3 is: %.3e kg/s' % (jet_density_max, jet_mass_loss_kg_per_s))
print('The jet mass-loss rate for a density of %.2e m^-3 is: %.3e Msol/year' % (jet_density_max,jet_mass_loss_Msol_per_yr))

if jet_type=="simple_stellar_jet":
    number_radii = 10000
    radius = np.linspace(0, height_0_m*np.tan(jet_angle)+1, number_radii)
    radius_max = height_0_m*np.tan(jet_angle)
    density = jet_density_max * (np.arctan(radius/height_0_m)/ np.arctan(radius_max/height_0_m))**8
    velocity = jet_velocity_max + (jet_velocity_min - jet_velocity_max)*(np.arctan(radius/height_0_m)/ np.arctan(radius_max/height_0_m))**.1

    plt.plot(np.arctan(radius/height_0_m)*180./np.pi, velocity/np.max(velocity))
    plt.plot(np.arctan(radius/height_0_m)*180./np.pi, density/np.max(density))
    plt.show()
    masslosses = M_sol**-1 * year * m_p * density * velocity * 2.* np.pi * radius * radius_max / number_radii
    print(np.sum(masslosses))

if jet_type=="simple_stellar_jet":
    number_radii = 10000
    jet_cavity_angle = 0
    exp_velocity = (np.exp(np.exp(3.5)-1))
    radius = np.linspace(0, height_0_m*np.tan(jet_angle)+1, number_radii)
    radius_max = height_0_m*np.tan(jet_angle)
    density = jet_density_max * (np.arctan(radius/height_0_m)/ np.arctan(radius_max/height_0_m))**8

    poloidal_velocity = np.zeros(number_radii)
    exp_angles = ( np.arctan(radius/height_0_m)/np.arctan(radius_max/height_0_m) )**2

    factor = ( exp_velocity**-(exp_angles) - exp_velocity**-1 ) / ( 1 - exp_velocity**-1 )

    velocity = jet_velocity_min + (jet_velocity_max - jet_velocity_min) * factor

    plt.plot(np.arctan(radius/height_0_m)*180./np.pi, velocity)
    plt.show()
    masslosses = M_sol**-1 * year * m_p * density * velocity * 2.* np.pi * radius * radius_max / number_radii
    print(np.sum(masslosses))
