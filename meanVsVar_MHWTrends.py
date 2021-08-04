'''

  Software which uses the MHW definition
  of Hobday et al. (2015) applied to
  AR1 statistical simulated SST

'''

# Load required modules

import numpy as np
import scipy as sp
from scipy import io
from scipy import signal
from scipy import linalg
from scipy import stats
from datetime import date
from netCDF4 import Dataset

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
import mpl_toolkits.basemap as bm

import deseason as ds
import ecoliver as ecj

import marineHeatWaves as mhw
import trendSimAR1

header = '/home/ecoliver/Desktop/data/'
header = '/home/oliver/data/'

#
# Load in range of AR1 parameters
#

#outfile = header + 'MHWs/Trends/mhw_census.2016.excessTrends.lores.combine'
outfile = header + 'MHWs/MeanVar/trendsMeanVar_AR1Fit'
data = np.load(outfile+'.npz')
map_tau = data['ar1_tau']
map_sig_eps = data['ar1_sig_eps']
lon_map = data['lon_map']
lat_map = data['lat_map']

# Make data mask based on land and ice
matobj = io.loadmat(header + 'MHWs/Trends/NOAAOISST_iceMean.mat')
ice_mean = matobj['ice_mean']
ice_longestRun = matobj['ice_longestRun']
#dl = 4
datamask = np.ones(map_tau.shape)
datamask[np.isnan(map_tau)] = np.nan
#datamask[ice_longestRun[::dl,::dl]>=6.] = np.nan # mask where ice had runs of 6 days or longer
datamask[ice_longestRun>=6.] = np.nan # mask where ice had runs of 6 days or longer
datamask[lat_map<=-65.,:] = np.nan
map_tau *= datamask
map_sig_eps *= datamask

# Plot distribution of tau-sig_eps

TAU, SIG_EPS = np.mgrid[0:120:0.1, 0:1:0.01]
positions = np.vstack([TAU.ravel(), SIG_EPS.ravel()])
kernel = stats.gaussian_kde(np.array([ecj.nonans(map_tau.flatten()), ecj.nonans(map_sig_eps.flatten())]))
pdf = np.reshape(kernel(positions), TAU.shape)

plt.figure()
plt.clf()
plt.plot(map_tau.flatten(), map_sig_eps.flatten(), 'ko', alpha=0.5, zorder=10, markeredgewidth=0, markersize=4)
plt.contourf(TAU, SIG_EPS, pdf, levels=[0.01,0.02,0.05,0.10,0.15,0.2,0.3,0.5,0.75,0.95], zorder=20, cmap=plt.cm.hot)
H = plt.colorbar()
plt.ylim(0.05, 0.65)
plt.xlim(0, 80)
plt.clim(-0.025, 0.9)
plt.xlabel(r'Autoregressive time scale, $\tau$ (days)')
plt.ylabel(r'Error standard deviation, $\sigma_\epsilon$ ($^\circ$C)')
H.set_label('Probability density')
plt.legend(['Individual observations'], loc='upper right')
# plt.savefig('../../documents/22_Mean_vs_Variance/figures/pdf_AR1_orig.png', bbox_inches='tight', pad_inches=0.05, dpi=300)

#
# Distribution of trends (mean, variance)
#

outfile = header + 'MHWs/Trends/sst_meanVarSkew_tr'
data = np.load(outfile + '.npz')
SST_mean_tr = data['SST_mean_tr'][::dl,::dl]
SST_var_tr = data['SST_var_tr'][::dl,::dl]
#SST_mean_tr_TS = data['SST_mean_tr_TS']
#SST_var_tr_TS = data['SST_var_tr_TS']
SST_mean_tr *= datamask
SST_var_tr *= datamask

# Plot distribution of trends

MEAN, VAR = np.mgrid[-1:1:0.01, -0.6:3:0.01]
positions = np.vstack([MEAN.ravel(), VAR.ravel()])
kernel = stats.gaussian_kde(np.array([ecj.nonans(SST_mean_tr.flatten()), ecj.nonans(SST_var_tr.flatten())]))
pdf = np.reshape(kernel(positions), MEAN.shape)

plt.figure()
plt.clf()
plt.plot(SST_mean_tr.flatten()*10, SST_var_tr.flatten()*10, 'ko', alpha=0.5, zorder=10, markeredgewidth=0, markersize=4)
plt.contourf(MEAN*10, VAR*10, pdf, zorder=20, cmap=plt.cm.hot, levels=[2, 10, 100, 500, 1000, 1500, 2000, 4000])
H = plt.colorbar()
plt.ylim(-0.4, 2)
plt.xlim(-0.4, 1)
plt.xlabel(r'Linear trend in mean SST ($^\circ$C decade$^{-1}$)')
plt.ylabel(r'Linear trend in SST variance ($^\circ$C$^2$ decade$^{-1}$)')
H.set_label('Probability density')
plt.legend(['Individual observations'], loc='upper right')
# plt.savefig('../../documents/22_Mean_vs_Variance/figures/pdf_Trends_orig.png', bbox_inches='tight', pad_inches=0.05, dpi=300)

#
# Run some simulations with specified trends for mean/variance
#

t, dates, T, year, month, day, doy = ecj.timevector([1982,1,1], [2017,12,31])

#tau, sig_eps, trends, means = trendSimAR1.simulate(t, sst_obs=None, seas_obs=None, sst_trend_obs=0.0, sst_trend_var=1.0, N_ens=10, params=[a, sig_eps])
#print trends['count']

#sst = tsa.arima_process.arma_generate_sample([1,-a], [1], T, sigma=sig_eps, burnin=100)
#trendMean_dec = 0.3
#trendVar_dec = 0.1
#trendVar_dec_applied = trendVar_dec / (2*np.var(sst))
# Effect of neglecting nonlinearity
# plt.clf()
# plt.plot(t, 1 + 2*trendVar_dec_applied*(t-t[0])/10./365.25)
# plt.plot(t, 1 + 2*trendVar_dec_applied*(t-t[0])/10./365.25 + (trendVar_dec_applied*(t-t[0])/10./365.25)**2)

keys = ['count', 'intensity_max_max', 'duration', 'total_days']
units = {'count': 'events', 'intensity_max_max': '$^\circ$C', 'duration': 'days', 'total_days': 'days'}
N_keys = len(keys)

N_ens = 100 #1000 # Number of ensembles, per trend value
N_tr = 10 # Number of trend values to consider
trend_max_mean = 1.0 # Maximum trend (1 degree per decade)
trend_max_var = 1.5 # Maximum trend (1 degree per decade)
trend_range_mean = np.linspace(0., 1., N_tr)*trend_max_mean
trend_range_var = np.linspace(0., 1., N_tr)*trend_max_var

    tau = 13.
    a = np.exp(-1/tau)
    sig_eps = 0.3
    #
    trends_mean = {}
    trends_var = {}
    for key in keys:
        trends_mean[key] = np.zeros((N_ens,N_tr))
        trends_var[key] = np.zeros((N_ens,N_tr))
    
    for i_tr in range(N_tr):
        print k+1, len(tau_sample), i_tr+1, 'of', N_tr
        # Mean trend
        tau, sig_eps, trends0, means = trendSimAR1.simulate(t, sst_obs=None, seas_obs=None, sst_trend_obs=trend_range_mean[i_tr], sst_trend_var=0., N_ens=N_ens, params=[a, sig_eps])
        for key in keys:
            trends_mean[key][:,i_tr] = trends0[key]
        # Variance trend
        tau, sig_eps, trends0, means = trendSimAR1.simulate(t, sst_obs=None, seas_obs=None, sst_trend_obs=0., sst_trend_var=trend_range_var[i_tr], N_ens=N_ens, params=[a, sig_eps])
        for key in keys:
            trends_var[key][:,i_tr] = trends0[key]

# Analyse results

plt.figure(figsize=(18,5))
plt.clf()
i = 0
for key in keys:
    # Mean
    plt.subplot(2, N_keys, i+1)
    plt.plot(trend_range_mean, 10*trends_mean[key].T, '-', color=(0.5,0.5,0.5), alpha=0.5)
    plt.plot(trend_range_mean, 10*np.mean(trends_mean[key], axis=0), 'k-o', linewidth=2)
    plt.plot(trend_range_mean, 10*np.percentile(trends_mean[key], 2.5, axis=0), 'b-o', linewidth=2)
    plt.plot(trend_range_mean, 10*np.percentile(trends_mean[key], 97.5, axis=0), 'r-o', linewidth=2)
    plt.plot(trend_range_mean, 0*trend_range_mean, 'k:')
    plt.xlabel(r'SST trend [$^\circ$C decade$^{-1}$]')
    plt.ylabel(r'MHW ' + key + ' trend [' + units[key] + ' decade$^{-1}$]')
    plt.title(r'$\tau$ = ' + str(tau) + ' days, $\sigma_\epsilon$ = ' + str(sig_eps) + ' $^\circ$C')
    #if key == 'duration':
    #    plt.legend(['Simulated trends', 'Ensemble mean', 'Ensemble 2.5th pctle.', 'Ensemble 97.5th pctle.'], loc='upper left')
    # Variance
    plt.subplot(2, N_keys, i+1 + N_keys)
    plt.plot(trend_range_var, 10*trends_var[key].T, '-', color=(0.5,0.5,0.5), alpha=0.5)
    plt.plot(trend_range_var, 10*np.mean(trends_var[key], axis=0), 'k-o', linewidth=2)
    plt.plot(trend_range_var, 10*np.percentile(trends_var[key], 2.5, axis=0), 'b-o', linewidth=2)
    plt.plot(trend_range_var, 10*np.percentile(trends_var[key], 97.5, axis=0), 'r-o', linewidth=2)
    plt.plot(trend_range_var, 0*trend_range_var, 'k:')
    plt.xlabel(r'SST variance trend [$^\circ$C$^2$ decade$^{-1}$]')
    plt.ylabel(r'MHW ' + key + ' trend [' + units[key] + ' decade$^{-1}$]')
    #
    i += 1

# plt.savefig('../../documents/22_Mean_vs_Variance/figures/simulated_trends_orig.png', bbox_inches='tight', pad_inches=0.05, dpi=300)

plt.figure(figsize=(18,5))
plt.clf()
i = 0
for key in ['total_days', 'intensity_max_max']:
    # Mean
    plt.subplot(2, 2, i+1)
    plt.plot(trend_range_mean, 10*trends_mean[key].T, '-', color=(0.5,0.5,0.5), alpha=0.25)
    plt.plot(trend_range_mean, 10*np.mean(trends_mean[key], axis=0), 'k-o', linewidth=2)
    plt.plot(trend_range_mean, 10*np.percentile(trends_mean[key], 2.5, axis=0), 'b-o', linewidth=2)
    plt.plot(trend_range_mean, 10*np.percentile(trends_mean[key], 97.5, axis=0), 'r-o', linewidth=2)
    plt.plot(trend_range_mean, 0*trend_range_mean, 'k:')
    plt.xlabel(r'SST trend [$^\circ$C decade$^{-1}$]')
    plt.ylabel(r'MHW ' + key + ' trend [' + units[key] + ' decade$^{-1}$]')
    plt.title(r'$\tau$ = ' + str(tau) + ' days, $\sigma_\epsilon$ = ' + str(sig_eps) + ' $^\circ$C')
    #if key == 'duration':
    #    plt.legend(['Simulated trends', 'Ensemble mean', 'Ensemble 2.5th pctle.', 'Ensemble 97.5th pctle.'], loc='upper left')
    # Variance
    plt.subplot(2, 2, i+1 + 2)
    plt.plot(trend_range_var, 10*trends_var[key].T, '-', color=(0.5,0.5,0.5), alpha=0.25)
    plt.plot(trend_range_var, 10*np.mean(trends_var[key], axis=0), 'k-o', linewidth=2)
    plt.plot(trend_range_var, 10*np.percentile(trends_var[key], 2.5, axis=0), 'b-o', linewidth=2)
    plt.plot(trend_range_var, 10*np.percentile(trends_var[key], 97.5, axis=0), 'r-o', linewidth=2)
    plt.plot(trend_range_var, 0*trend_range_var, 'k:')
    plt.xlabel(r'SST variance trend [$^\circ$C$^2$ decade$^{-1}$]')
    plt.ylabel(r'MHW ' + key + ' trend [' + units[key] + ' decade$^{-1}$]')
    #
    i += 1

# plt.savefig('../../documents/22_Mean_vs_Variance/figures/simulated_trends_totDaysIMax_orig.png', bbox_inches='tight', pad_inches=0.05, dpi=300)


