# use: python EW.py -i nnnnnnn -n yyyymmdd -w 6500


import sys
import getopt
import glob
import pyfits
import os
import pylab as pl
import pyfits as pf
import numpy as np
import csv
import matplotlib
font = {'size' : 18}
matplotlib.rc('font',**font)
args = sys.argv

if (len(args) > 1):
    if "-wa" in args:
        wavea = float(args[args.index("-wa") + 1 ])
    if "-wb" in args:
        waveb = float(args[args.index("-wb") + 1 ])
    else:
        print "usage: python lineflux.py -i nnnnnnnn"


#################
'''
Determine the flux/luminosity of a spectral line
'''
#################

datalistc = glob.glob("/STER/mercator/hermes/20111116/reduced/00384951_HRF_OBJ_ext_CosmicsRemoved_log_merged_c.fits")
# data = np.loadtxt("/home/dylanb/astronomy/thesis/BD46442/norm_spectra/specprimrem/00384951_primrem.txt")
data = np.loadtxt("/home/dylanb/astronomy/jet_geometry/BD46442/6562.8/specprimrem/00384951_primrem.txt")

inspec = datalistc
header = pf.getheader(inspec[0])
if header.get('SNR65') > 1 :
    SNR = header.get('SNR65')
else:
    SNR = 00.07*header.get('EXPTIME')
wavelengthsc = data[:,0]
spec = data[:,1]
# pl.plot(wavelengthsc, spec)
# pl.show()
std_spec = data[:,2]


########
'''
Calculating the line flux
'''
########

indices = np.where((wavelengthsc > wavea) & (wavelengthsc < waveb))
wavelengths = wavelengthsc[indices]
flux = (spec[indices] - 1)*(0.7544245-7.22667e-5*wavelengths)*10/2.5

# For Halpha of BD46442, the normalised spectrum has to be multiplied by 0.7544245-7.22667e-5*wavelength for L = 2500 L_odot
# pl.plot(wavelengths,flux)
# pl.show()
# std_flux = std_spec[indices]
# std_diff = std_flux[0:-1]
diff = np.zeros(np.size(flux)-1)
diff = wavelengths[1:]-wavelengths[0:-1]
integration = flux[0:-1]*diff
# std_integration = (1+1/flux[0:-1])**.5*(diff-integration)/SNR
# std_ew = (np.sum(std_integration**2))**.5
# print 'error', std_ew
# std_ewnorm = (np.sum((std_diff*diff)**2))**.5
# print 'error zoals normaal', std_ewnorm
print 'line luminosity: ', np.sum(integration)


########
#Line luminosity (in Lsol)
########

L_line= np.sum(integration)

########
#accretion luminosity (logaritmic in Lsol)
########

L_acc_log = 2.27 + 1.25 * np.log(L_line)
L_acc = 10**L_acc_log
print 'The log accretion luminosity and accretion luminosity are %2f and %2f' %(L_acc_log, L_acc)

########
#Mass accretion rate per year (logaritmic in Msol/year)
########

M_sec = 0.79 #for inclination of 60degr
R_sec = 2.*M_sec**0.75 # M_odot
print R_sec
G = 1.90809e11 # R_odot M_odot^-1 m^-2 s^-2
L_odot = 3.828*10**26 # kg m^2 s^-3
M_odot = 1.9891*10**30 # kg
year = 3600*24*365. # seconds

M_acc = 1.25*L_acc*R_sec/(G*M_sec) * L_odot * year / M_odot
print M_acc
print 'The mass accretion rate per year is %g' % M_acc
