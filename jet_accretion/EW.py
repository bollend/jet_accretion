import numpy as np
import matplotlib.pylab as plt
import sys
import os


def equivalent_width(wav, fl, cut=False, wave_min=0, wave_max=0, use_median=False):
    """
    Determine the equivalent width of a line
    """
    wave = np.array(wav)
    flux = np.array(fl)
    if use_median==True:
        if cut==False:

            # Use the whole array to compute the equivalent width
            # The median value is the continuum level
            median_low  = np.median(flux[0:50])
            median_high = np.median(flux[-50:-1])
            median      = ( median_low + median_high ) / 2

            delta_wave  = (wave[2:] - wave[:-2]) / 2.
            integration = (1 - flux[1:-1]/median) * delta_wave
            EW          = np.sum(integration)
        else:
            # Cut the wave and flux arrays to the region for which you want to
            # calculate the equivalent width

            indices     = np.where((wave_min < wave) & (wave < wave_max))
            wave_cut    = wave[indices]
            flux_cut    = flux[indices]

            median_low  = np.median(flux_cut[0:50])
            median_high = np.median(flux_cut[-50:-1])
            median      = ( median_low + median_high ) / 2

            delta_wave  = (wave_cut[2:] - wave_cut[:-2]) / 2.
            integration = (1 - flux_cut[1:-1]/median) * delta_wave
            EW          = np.sum(integration)

    else:
        if cut==False:

            # Use the whole array to compute the equivalent width
            delta_wave  = (wave[2:] - wave[:-2]) / 2.
            integration = (1 - flux[1:-1]) * delta_wave
            EW          = np.sum(integration)

        else:
            # Cut the wave and flux arrays to the region for which you want to
            # calculate the equivalent width

            indices          = np.where((wave_min < wave) & (wave < wave_max))
            wave_cut         = wave[indices]
            flux_cut         = flux[indices]

            delta_wave  = (wave_cut[2:] - wave_cut[:-2]) / 2.
            integration = (1 - flux_cut[1:-1]) * delta_wave
            EW          = np.sum(integration)
    return EW

def equivalent_width_uncertainty(wav, fl, uncertainty_flux, cut=False, wave_min=0, wave_max=0, use_median=False):
    """
    Determine the equivalent width of a line and its uncertainty
    """
    wave = np.array(wav)
    flux = np.array(fl)
    if type(uncertainty_flux)==float:
        uncertainty_flux = uncertainty_flux * np.ones(len(flux))

    if use_median==True:
        if cut==False:

            # Use the whole array to compute the equivalent width
            # The median value is the continuum level
            median_low  = np.median(flux[0:50])
            median_high = np.median(flux[-50:-1])
            median      = ( median_low + median_high ) / 2

            delta_wave  = (wave[2:] - wave[:-2]) / 2.
            integration = (1 - flux[1:-1]/median) * delta_wave
            EW          = np.sum(integration)
        else:
            # Cut the wave and flux arrays to the region for which you want to
            # calculate the equivalent width

            indices     = np.where((wave_min < wave) & (wave < wave_max))
            wave_cut    = wave[indices]
            flux_cut    = flux[indices]

            median_low  = np.median(flux_cut[0:50])
            median_high = np.median(flux_cut[-50:-1])
            median      = ( median_low + median_high ) / 2

            delta_wave  = (wave_cut[2:] - wave_cut[:-2]) / 2.
            integration = (1 - flux_cut[1:-1]/median) * delta_wave
            EW          = np.sum(integration)

    else:
        if cut==False:

            # Use the whole array to compute the equivalent width
            delta_wave              = (wave[2:] - wave[:-2]) / 2.
            integration             = (1 - flux[1:-1]) * delta_wave
            uncertainty_integration = (uncertainty_flux[1:-1]) * delta_wave
            EW                      = np.sum(integration)
            uncertainty_EW          = np.sum(uncertainty_integration)

        else:
            # Cut the wave and flux arrays to the region for which you want to
            # calculate the equivalent width

            indices              = np.where((wave_min < wave) & (wave < wave_max))
            wave_cut             = wave[indices]
            flux_cut             = flux[indices]
            uncertainty_flux_cut = uncertainty_flux[indices]

            delta_wave              = (wave_cut[2:] - wave_cut[:-2]) / 2.
            integration             = (1 - flux_cut[1:-1]) * delta_wave
            uncertainty_integration = (uncertainty_flux_cut[1:-1]) * delta_wave
            EW                      = np.sum(integration)
            uncertainty_EW          = np.sum(uncertainty_integration)

    return EW, uncertainty_EW
