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

outfile = header + 'MHWs/Trends/mhw_census.2016.excessTrends.lores.combine'
data = np.load(outfile+'.npz')
map_tau = data['ar1_tau']
map_sig_eps = data['ar1_sig_eps']
lon_map = data['lon_map']
lat_map = data['lat_map']

# Make data mask based on land and ice
matobj = io.loadmat(header + 'MHWs/Trends/NOAAOISST_iceMean.mat')
ice_mean = matobj['ice_mean']
ice_longestRun = matobj['ice_longestRun']
dl = 4
datamask = np.ones(map_tau.shape)
datamask[np.isnan(map_tau)] = np.nan
datamask[ice_longestRun[::dl,::dl]>=6.] = np.nan # mask where ice had runs of 6 days or longer
datamask[lat_map<=-65.,:] = np.nan
map_tau *= datamask
map_sig_eps *= datamask

# Plot distribution of tau-sig_eps

TAU, SIG_EPS = np.mgrid[0:120:0.1, 0:1:0.01]
positions = np.vstack([TAU.ravel(), SIG_EPS.ravel()])
kernel = stats.gaussian_kde(np.array([ecj.nonans(map_tau.flatten()), ecj.nonans(map_sig_eps.flatten())]))
pdf = np.reshape(kernel(positions), TAU.shape)

# Find the (tau, sigmas) over which to sample
tau_sample, sig_sample = np.mgrid[2:60+1:2, 0.1:0.6+0.02:0.02]
tau_sample = tau_sample.flatten()
sig_sample = sig_sample.flatten()
keep = np.array([])
for k in range(len(tau_sample)):
    tau0 = tau_sample[k]
    sig0 = sig_sample[k]
    i, j = ecj.findxy(TAU, SIG_EPS, (tau0, sig0))
    if pdf[j,i] >= 0.001:
        keep = np.append(keep, k)
        #print tau0

tau_sample = tau_sample[keep.astype(int)]
sig_sample = sig_sample[keep.astype(int)]

#
# Run some simulations with specified trends for mean/variance
#

t, dates, T, year, month, day, doy = ecj.timevector([1982,1,1], [2017,12,31])

keys = ['count', 'intensity_max_max', 'duration', 'total_days']
units = {'count': 'events', 'intensity_max_max': '$^\circ$C', 'duration': 'days', 'total_days': 'days'}
N_keys = len(keys)

N_ens = 500 # Number of ensembles, per trend value
#N_tr = 10 # Number of trend values to consider
#trend_max_mean = 1.0 # Maximum trend (per decade)
#trend_max_var = 1.5 # Maximum trend (per decade)
#trend_min_mean = -0.4 # Minimum trend (per decade)
#trend_min_var =  # Minimum trend (per decade)
#trend_range_mean = np.linspace(trend_min_mean, trend_max_mean, N_tr)
#trend_range_var = np.linspace(trend_min_var, trend_max_var, N_tr)
trend_range_mean = np.array([-0.4, -0.2, -0.1, 0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
trend_range_var = np.array([-0.2, -0.1, 0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5])
N_tr = np.min([len(trend_range_mean), len(trend_range_var)])

for k in range(len(tau_sample)):
    #tau = 13.
    tau = tau_sample[k]
    a = np.exp(-1/tau)
    #sig_eps = 0.3
    sig_eps = sig_sample[k]
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
    # Save data
    outfile = header + 'MHWs/MeanVar/simTrends/simTrends_k' + str(k).zfill(3)
    np.savez(outfile, trend_range_mean=trend_range_mean, trend_range_var=trend_range_var, trends_mean=trends_mean, trends_var=trends_var, keys=keys, N_keys=N_keys, units=units)

