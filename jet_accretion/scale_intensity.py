'''
A function to rescale the flux/intensity level of the spectra by fitting
a straight line to the template.
'''
import numpy


def scale_intensity(wave_0,
                    wave_template,
                    intensity_template,
                    wave_spec,
                    intensity_spec,
                    norm=True,
                    wave_region=100e-10,
                    wave_0_d=100e-10):
    '''
    Returns the rescaled spectrum.
    norm=True if the spectrum is normalised.
    '''
    if norm=True:
        wave_min_a     = wave_0 - (wave_0_d + wave_region)
        wave_min_b     = wave_0 - (wave_0_d)
        wave_max_a     = wave_0 + (wave_0_d)
        wave_max_b     = wave_0 + (wave_0_d + wave_region)
        median_I_low   = np.median(intensity_template[np.where((wave_template > wave_min_a) & (wave_template < wave_min_b))])
        median_I_high  = np.median(intensity_template[np.where((wave_template > wave_max_a) & (wave_template < wave_max_b))])
        gradient       = (median_I_high - median_I_low) / ( 2 * wave_0_d + wave_region)
        intercept      = median_I_high - gradient *  (wave_0 + wave_0_d + 0.5 * wave_region)
        scaling_interp = np.poly1d(np.array([gradient, intercept]))
        intensity_spec = intensity_spec * scaling_interp(wave_spec)

    return intensity_spec
