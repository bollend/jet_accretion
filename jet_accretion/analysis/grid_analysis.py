import numpy as np
import matplotlib.pylab as plt
from scipy.stats import kde
import argparse
import sys
import seaborn as sns
from scipy.interpolate import interp1d
# from common.myimport import *

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    ============================================================================
"   Analyse the results from the grid search (Temperature and density).
"   This will create a 2D plot of the chi-squared and probability with contours.
    ============================================================================
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

## PLOT - histogram for N(HI), NCNM, NWNM  for all 73 soures ##
fts  = 14
lbsz = 14
lgds = 10

plt.rc('font', weight='bold')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=15)
# plt.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']
plt.rcParams['axes.linewidth']      = 2.

args = sys.argv

parser = argparse.ArgumentParser()

parser.add_argument('-i', dest='inputfile',
                    help='Inputfile with the chi-squared values for each\
                    temperature and density')

args       = parser.parse_args()
InputFile  = args.inputfile

with open(InputFile, 'r') as f:

    lines = f.readlines()
    # jet_temperatures = np.zeros(len(lines))
    # jet_density_log  = np.zeros(len(lines))
    # chi_squared      = np.zeros(len(lines))

    jet_temperatures_list = []
    jet_density_log_list  = []
    chi_squared_list      = []

    for (i,line) in enumerate(lines):

        data_line = line.split('\t')

        if float(data_line[0]) not in jet_temperatures_list:

            jet_temperatures_list.append(float(data_line[0]))

        if float(data_line[1]) not in jet_density_log_list:

            jet_density_log_list.append(float(data_line[1]))

        chi_squared_list.append(float(data_line[2].rsplit('\n')[0]))
        # jet_temperatures[i] = float(data_line[0])
        # jet_density_log[i]  = float(data_line[1])
        # chi_squared_list[i]      = float(data_line[2].rsplit('\n')[0])

jet_temperatures = np.array(jet_temperatures_list)
jet_density_log  = np.log10(np.array(jet_density_log_list))#*(2.527e-6/5.01e15))
print(jet_density_log)
T = 0
rho = 0
chi2 = 0
chi_squared = np.zeros((len(jet_temperatures),len(jet_density_log)))

for jet_temperature in jet_temperatures:

    rho = 0

    for jet_density in jet_density_log:

        chi_squared[T,rho] = chi_squared_list[chi2]
        rho               += 1
        chi2              += 1

    T += 1

###### Chi-squared plots

chi2_min = np.min(chi_squared)
indices_chi2_min = np.where(np.min(chi_squared)==chi_squared)
Temp_min = jet_temperatures[indices_chi2_min[0][0]]
Dens_min = jet_density_log[indices_chi2_min[1][0]]
chi2_1sigma = chi2_min + 2.3
chi2_2sigma = chi2_min + 6.17
chi2_3sigma = chi2_min + 11.8
print(Temp_min, Dens_min)

xi, yi = np.mgrid[jet_temperatures.min():jet_temperatures.max():len(jet_temperatures)*1j, jet_density_log.min():jet_density_log.max():len(jet_density_log)*1j]
zi = chi_squared

fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(11, 5))
# plot_color = plt.cm.Reds_r
plot_color = plt.cm.Blues_r
# plot_color = plt.cm.Greens_r

im1   = axes[0].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plot_color)
cont1 = axes[0].contour(xi, yi, zi.reshape(xi.shape), [chi2_min, chi2_1sigma,chi2_2sigma,chi2_3sigma])
axes[0].clabel(cont1, inline=True, fontsize=10)
axes[0].scatter(Temp_min, Dens_min, color='w')
axes[0].legend()

axes[0].set_xlabel(r'Temperature (K)')
axes[0].set_ylabel(r'log(density) ($\log$(m$^{-3}$))')
# axes[0].set_ylabel(r'log of mass acrretion rate ($\log\dot{M_\odot}$)')
fig.colorbar(im1, label='chi-squared', ax=axes[0])
# plt.show()

###### Probability plots

probabilities = np.exp(-0.5*chi_squared**2)
prob_max = np.exp(-0.5*chi2_min**2)
prob_max_1sigma = np.exp(-0.5*chi2_1sigma**2)
prob_max_2sigma = np.exp(-0.5*chi2_2sigma**2)
prob_max_3sigma = np.exp(-0.5*chi2_3sigma**2)
probabilities_sum = np.sum(probabilities)
probabilities_normalised = probabilities / probabilities_sum

prob_max_n = prob_max / probabilities_sum
prob_max_1sigma_n = prob_max_1sigma / probabilities_sum
prob_max_2sigma_n = prob_max_2sigma / probabilities_sum
prob_max_3sigma_n = prob_max_3sigma / probabilities_sum


im2   = axes[1].pcolormesh(xi,yi, probabilities_normalised.reshape(xi.shape), shading='gouraud', cmap=plot_color)
cont2 = axes[1].contour(xi,yi, probabilities_normalised.reshape(xi.shape))
# axes[1].clabel(cont2, inline=True, fontsize=10)
axes[1].set_xlabel(r'Temperature (K)')
axes[1].set_ylabel(r'log(density) ($\log$(m$^{-3}$))')
# axes[1].set_ylabel(r'log of mass acrretion rate ($\log\dot{M_\odot}$)')
fig.colorbar(im2, label='probability', ax=axes[1])


###### Other stuff

# Find region in 2D probability plot of 1, 2, and 3 sigma interval

interval_sigma = np.array([0.9973,0.9545,0.6827])


num_prob             = 100
probability_max      = probabilities_normalised[indices_chi2_min]
probabilities_values = np.linspace(0,probability_max, num_prob)

prob_interval_value     = [0,0,0]
prob_diff_interval      = [10,10,10]

for (p,prob) in enumerate(probabilities_values):

    prob_higher_indices = np.where(probabilities_normalised > prob)
    prob_interval       = np.sum(probabilities_normalised[prob_higher_indices])

    for (i,interval) in enumerate(interval_sigma):

        if np.abs(prob_interval - interval) < prob_diff_interval[i]:

            prob_diff_interval[i]      = np.abs(prob_interval - interval)
            prob_interval_value[i]     = prob


fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(6, 5))
im2   = axes.pcolormesh(xi,yi, probabilities_normalised.reshape(xi.shape), shading='gouraud', cmap=plot_color)
# cont2 = axes.contour(xi,yi, probabilities_normalised.reshape(xi.shape),  prob_interval_value)
# axes.clabel(cont2, inline=True, fontsize=10)
axes.set_xlabel(r'Temperature (K)')
axes.set_ylabel(r'log(density) ($\log$(m$^{-3}$))')
# axes.set_ylabel(r'log of mass acrretion rate ($\log\dot{M_\odot}$)')
fig.colorbar(im2, label='probability', ax=axes)
plt.show()


###### Find region in 1D probability plot of 1, 2, and 3 sigma interval in temperature

probabilities_temperature = np.sum(probabilities_normalised, axis=1)

# interpolation (cubic)
n_inter = 200

interp_temp = interp1d(jet_temperatures, probabilities_temperature)

jet_temperatures = np.linspace(np.min(jet_temperatures)+0.01, np.max(jet_temperatures)-0.01, n_inter)
probabilities_temperature = interp_temp(jet_temperatures)
probabilities_temperature /= np.sum(probabilities_temperature)

# Find interval

T_max_index = np.where(probabilities_temperature==np.max(probabilities_temperature))
T_max = jet_temperatures[T_max_index]

probability_max      = probabilities_temperature[T_max_index]
probabilities_values = np.linspace(0,probability_max, num_prob)

prob_interval_value_T     = [0,0,0]
prob_diff_interval_T      = [10,10,10]
prob_interval_indices_T   = [[],[],[]]

for (p,prob) in enumerate(probabilities_values):

    prob_higher_indices = np.where(probabilities_temperature > prob)
    prob_interval       = np.sum(probabilities_temperature[prob_higher_indices])

    for (i,interval) in enumerate(interval_sigma):

        if np.abs(prob_interval - interval) < prob_diff_interval_T[i]:

            prob_interval_value_T[i]     = prob
            prob_diff_interval_T[i]      = np.abs(prob_interval - interval)
            prob_interval_indices_T[i]   = prob_higher_indices

# Find region in 1D probability plot of 1, 2, and 3 sigma interval in density

probabilities_density = np.sum(probabilities_normalised, axis=0)

# interpolation (cubic)

interp_dens = interp1d(jet_density_log, probabilities_density)

jet_density_log = np.linspace(np.min(jet_density_log)+0.01, np.max(jet_density_log), n_inter)
probabilities_density = interp_dens(jet_density_log)
probabilities_density /= np.sum(probabilities_density)

# Find interval

d_max_index = np.where(probabilities_density==np.max(probabilities_density))
d_max       = jet_density_log[d_max_index]

probability_max      = probabilities_density[d_max_index]
probabilities_values = np.linspace(0,probability_max, num_prob)

prob_interval_value_d     = [0,0,0]
prob_diff_interval_d      = [10,10,10]
prob_interval_indices_d   = [[],[],[]]

for (p,prob) in enumerate(probabilities_values):

    prob_higher_indices = np.where(probabilities_density > prob)
    prob_interval       = np.sum(probabilities_density[prob_higher_indices])

    for (i,interval) in enumerate(interval_sigma):

        if np.abs(prob_interval - interval) < prob_diff_interval_d[i]:

            prob_interval_value_d[i]     = prob
            prob_diff_interval_d[i]      = np.abs(prob_interval - interval)
            prob_interval_indices_d[i]   = prob_higher_indices

# print(T_max, d_max)
# print(prob_interval_value_d)
# print(prob_diff_interval_d)
# print(prob_interval_indices_d)

fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(11, 5))


## X and Y Labels

fill_colors = ['darkblue', 'blue', 'lightblue']

axes[0].set_xlabel(r'Temperature (K)', fontsize=fts, fontweight='normal')
axes[1].set_xlabel(r'density ($\log$(m$^{-3}$))', fontsize=fts, fontweight='normal')

# axes[1].set_xlabel(r'log of mass acrretion rate ($\log\dot{M_\odot}$)')
axes[0].set_ylabel(r'$\mathrm{Probability\ Temperature}$', fontsize=fts, fontweight='normal')
axes[1].set_ylabel(r'$\mathrm{Probability\ density}$', fontsize=fts, fontweight='normal')

axes[0].plot(jet_temperatures, probabilities_temperature)
axes[0].fill_between(jet_temperatures[prob_interval_indices_T[0]], probabilities_temperature[prob_interval_indices_T[0]], color=fill_colors[0])
axes[0].fill_between(jet_temperatures[prob_interval_indices_T[1]], probabilities_temperature[prob_interval_indices_T[1]], color=fill_colors[1])
axes[0].fill_between(jet_temperatures[prob_interval_indices_T[2]], probabilities_temperature[prob_interval_indices_T[2]], color=fill_colors[2])

axes[1].semilogx(10**jet_density_log, probabilities_density)
axes[1].fill_between(10**jet_density_log[prob_interval_indices_d[0]], probabilities_density[prob_interval_indices_d[0]], color=fill_colors[0])
axes[1].fill_between(10**jet_density_log[prob_interval_indices_d[1]], probabilities_density[prob_interval_indices_d[1]], color=fill_colors[1])
axes[1].fill_between(10**jet_density_log[prob_interval_indices_d[2]], probabilities_density[prob_interval_indices_d[2]], color=fill_colors[2])

# axes[0].set_xlabel(r'Temperature (K)')
# axes[0].set_ylabel('Normalised probability')
plt.show()

std_temp_min = np.min(jet_temperatures[prob_interval_indices_T[2]]) - T_max
std_temp_max = np.max(jet_temperatures[prob_interval_indices_T[2]]) - T_max
print('mean temperature is ', T_max, 'standard deviation is ', std_temp_min, ' and ', std_temp_max)

# probabilities_density = np.sum(probabilities_normalised, axis=0)
mean_temperature = np.sum(jet_temperatures*probabilities_temperature)
mean_density = np.sum(jet_density_log*probabilities_density)

std_temperature = np.sum((mean_temperature - jet_temperatures)**2*probabilities_temperature)**.5
std_density = np.sum((mean_density - jet_density_log)**2*probabilities_density)**.5
print('mean and std temp is', mean_temperature, std_temperature)
print('mean and std density is ', mean_density, std_density)
fig, axes = plt.subplots(1,2)
axes[0].plot(jet_temperatures, probabilities_temperature)
axes[1].plot(jet_density_log, probabilities_density)
plt.show()
