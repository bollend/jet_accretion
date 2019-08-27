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
        print "usage: python EW.py -i nnnnnnnn"


#################
'''
Determine the equivalent width from the spectrum
'''
#################

datalistc = glob.glob("/STER/mercator/hermes/20111116/reduced/00384951_HRF_OBJ_ext_CosmicsRemoved_log_merged_c.fits")
# wave_range_template, spec_template = np.loadtxt("/home/dylanb/astronomy/thesis/BD46442/norm_spectra/specprimrem/00384951_primrem.txt", unpack = True, usecols = [0,1])
data0 = np.loadtxt("/home/dylanb/astronomy/thesis/BD46442/norm_spectra/specprimrem/00384951_primrem.txt")
data = np.loadtxt("/home/dylanb/astronomy/jet_geometry/BD46442/6562.8/specprimrem/00384951_primrem.txt")

inspec = datalistc
header = pf.getheader(inspec[0])
if header.get('SNR65') > 1 :
    SNR = header.get('SNR65')
else:
    SNR = 00.07*header.get('EXPTIME')
wavelengthsc = data[:,0]
spec = data[:,1]
pl.plot(wavelengthsc, spec)
pl.plot(data0[:,0], data0[:,1], 'k')
pl.show()
std_spec = data[:,2]
########
# Calculating the equivalent width
########
indices = np.where((wavelengthsc > wavea) & (wavelengthsc < waveb))
wavelengths = wavelengthsc[indices]
flux = spec[indices]
pl.plot(wavelengths,flux)
pl.show()
std_flux = std_spec[indices]
std_diff = std_flux[0:-1]
diff = np.zeros(np.size(flux)-1)
diff = wavelengths[1:]-wavelengths[0:-1]
integration = (1-flux[0:-1])*diff
std_integration = (1+1/flux[0:-1])**.5*(diff-integration)/SNR
std_ew = (np.sum(std_integration**2))**.5
print 'error', std_ew
std_ewnorm = (np.sum((std_diff*diff)**2))**.5
print 'error zoals normaal', std_ewnorm
print 'ew apart:', np.sum(integration)
ew_int = np.sum(integration)
#ew = np.sum(1-flux)*((wavelengths[-2]-wavelengths[0])/(np.size(wavelengths)-1))
#print 'ew zoals eerst:', ew
print np.size(wavelengths)-1
print ew_int, std_ew
