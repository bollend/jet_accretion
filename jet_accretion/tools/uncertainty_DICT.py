'''
Create the dictionary containing the standard deviation of the spectra
'''
import pickle


def create_uncertainties(object_id, parameters, InputDir, phases_dict):

    standard_deviation = {}
    signal_to_noise    = {}

    f_snr = open(InputDir+object_id+'_signal_to_noise_halpha.txt', 'rb')
    lines = f_snr.readlines()[:]

    for l in lines:

        l = l.decode('utf-8')
        title = l[:7].strip()
        value = eval(l[7:].split()[0])
        signal_to_noise[title] = value

    f_snr.close()

    if parameters['OTHER']['uncertainty_back']==True:
        with open(InputDir+object_id+'_stdev_init_halpha.txt', 'rb') as f:
            uncertainty_background = pickle.load(f)

        for phase in uncertainty_background:

            standard_deviation[phase] = {}

            for spectrum in uncertainty_background[phase]:

                standard_deviation[phase][spectrum] = \
                            2./signal_to_noise[spectrum] + uncertainty_background[phase][spectrum]
                            # Twice the uncertainty from S/N because the input spectrum is a subtraction between two spectra
                            # --> spec_tot = spec_1 - spec_2
                            # ----> delta_tot = delta_1 + delta_2

    else:

        uncertainty_background = None

        for phase in phases_dict.keys():

                standard_deviation[phase] = {}

                for spectrum in phases_dict[phase]:

                    standard_deviation[phase][spectrum] = \
                                2./signal_to_noise[spectrum]

    return standard_deviation
