import numpy as np
import matplotlib.pylab as plt
import pyfits as pf
import sys
import os,glob
import pylab
from scipy import interpolate
import scipy.constants as c

def get_spectrum(filenumber,night,wave,vrange):
    xmin = wave - (vrange/c.speed_of_light)*wave*1000.-1.
    xmax = wave + (vrange/c.speed_of_light)*wave*1000.+1.
    filename = "/lhome/dylanb/astronomy/objects/"+star+"/*"+filenumber+"_HRF_OBJ_ext_CosmicsRemoved_log_merged_c.fits"
    # filename = "/lhome/dylanb/astronomy/objects/BD46442/spectra/spectra/*"+filenumber+"_HRF_OBJ_ext_CosmicsRemoved_log_merged_c.fits"
    # filename = "/STER/mercator/hermes/"+night+"/reduced/*"+filenumber+"_HRF_OBJ_ext_CosmicsRemoved_log_merged_c.fits"
    print(filename)
    datalist = glob.glob(filename)
    if len(datalist) > 0:
        print(datalist[0])
        # Read the data and the header is resp. 'spec' and 'header'
        spec = pf.getdata(datalist[0])
        header = pf.getheader(datalist[0])
        # Make the equidistant wavelengthgrid using the Fits standard info
        # in the header
        crpix           = header.get('CRPIX1')-1
        crval           = header.get('CRVAL1')
        cdelt           = header.get('CDELT1')
        bvcor           = header.get('BVCOR')
        object          = header.get('OBJECT')
        numberpoints    = len(spec)
        wavelengthbegin = (crval - crpix*cdelt)
        wavelengthend   = crval + (numberpoints-1)*cdelt
        wavelengths     = np.linspace(wavelengthbegin,wavelengthend,numberpoints)
        # normalise the given region and plot the object
        wavelengthslin  = np.zeros(len(spec))
        # rv in reference frame of the primary
        wavelengthslin  = np.exp(wavelengths)
        if filenumber=="421031":
            wavelengthslin = np.exp(wavelengths - np.log(1 - bvcor/vc))
        # elif filenumber=="397189" or filenumber=="399553":
        #     wavelengthslin= np.exp(wavelengths - np.log(1 - bvcor/vc))
        spectrum = np.array([])
        wavmin = min(range(len(wavelengthslin)), key = lambda j: abs(wavelengthslin[j]- (wave - 50)))
        wavmax = min(range(len(wavelengthslin)), key = lambda j: abs(wavelengthslin[j]- (wave + 50)))
        lambdas =  wavelengthslin[wavmin:wavmax]
        #### select 2/3 points to fit polynomial to normalise observed spectrum
        point_flux = np.array([np.median(spec[(xmin-100 <= wavelengthslin) & (wavelengthslin <= xmin-50)]),
                                np.median(spec[(xmax <= wavelengthslin) & (wavelengthslin <= xmax+50)]),
                                np.median(spec[(xmax+200 <= wavelengthslin) & (wavelengthslin <= xmax+300)])])
        point_wave = np.array([xmin-50,xmax+25,xmax+250])
        polyn = np.poly1d(np.polyfit(point_wave,point_flux,2))
        spectrumn = spec[wavmin:wavmax]/polyn(wavelengthslin[wavmin:wavmax])
        ####
    else:
        print("Er is geen log file met dat nummer te vinden")
        lambdas = [0]
        spectrumn = [0]

    return lambdas, spectrumn, object



args = sys.argv


if (len(args) > 1):
    if "-i" in args:
        listfile = args[args.index("-i") + 1 ]
    if "-wave" in args:
        wave = float(args[args.index("-wave") + 1])
    if "-object" in args:
        #file with orbital/stellar parameters
        star = args[args.index("-object") + 1 ]
    if "-range" in args:
        vrange = float(args[args.index("-range") + 1])
        vmin = -vrange
        vmax = +vrange
    else:
       print("usage: python Nice2dplots -i <ster,list> -wave <central wavelength> \
       -range <vel. range>  ")


wavec = wave
# vrange_syn = 500.
xmin = wavec - (vrange/c.speed_of_light)*wavec*1000.
xmax = wavec + (vrange/c.speed_of_light)*wavec*1000.
# print xmin,xmax,vrange_syn,wavec

inputfile = open(listfile,'r')
numberspec = 0
data_spectra = {}
data_wavelength = {}
for line in inputfile:
    items = line.split()
    wave_org,flux_org,nameobject = get_spectrum(items[0],items[9],wave,vrange)

    spectrum_id = float(items[0])
    data_spectra[spectrum_id] = flux_org
    data_wavelength[spectrum_id] = wave_org
    # f = open(str(nameobject) + '_' +str(int(spectrum_id)) + ".txt", 'w')
    alldata = np.vstack((wave_org,flux_org))
    print(alldata.T)
    np.savetxt(str(nameobject) + '_' +str(int(spectrum_id)) + ".txt", alldata)
