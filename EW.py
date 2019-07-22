import numpy as np
import matplotlib.pylab as plt
import sys
import os


def equivalent_width(wave, flux, cut=False, wave_min=0, wave_max=0):
    """
    Determine the equivalent width of a line
    """
    if cut==False:
        
        # Use the whole array to compute the equivalent width

        delta_wave  = (wave[2:] - wave[:-2]) / 2.
        integration = (1 - flux[1:-1]) * delta_wave
        EW          = np.sum(integration)


    else:
        # Cut the wave and flux arrays to the region for which you want to
        # calculate the equivalent width

        indices     = np.where((wave_min < wave) & (wave < wave_max))
        wave_cut    = wave[indices]
        flux_cut    = flux[indices]
        delta_wave  = (wave_cut[2:] + wave_cut[:-2]) / 2.
        integration = (1 - flux_cut[1:-1]) * delta_wave
        EW          = np.sum(integration)

    return EW
