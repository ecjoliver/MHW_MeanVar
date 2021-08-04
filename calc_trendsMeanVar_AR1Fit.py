'''

  Software which uses the MHW definition
  of Hobday et al. (2015) applied to 
  select SST time series around the globe

'''

# Load required modules

import numpy as np
from scipy import io
from scipy import linalg
from scipy import stats
from scipy import signal
from datetime import date
from netCDF4 import Dataset

from matplotlib import pyplot as plt
import mpl_toolkits.basemap as bm

import ecoliver as ecj

import marineHeatWaves as mhw
import trendSimAR1

# Load some select time series

#
# observations
#

#pathroot = '/mnt/erebor/'
#pathroot = '/home/ecoliver/Desktop/'
pathroot = '/data/home/oliver/'
header = pathroot+'data/sst/noaa_oi_v2/avhrr/'
file0 = header + '1982/avhrr-only-v2.19820101.nc'
t = np.arange(date(1982,1,1).toordinal(),date(2017,12,31).toordinal()+1)
T = len(t)
year = np.zeros((T))
month = np.zeros((T))
day = np.zeros((T))
for i in range(T):
    year[i] = date.fromordinal(t[i]).year
    month[i] = date.fromordinal(t[i]).month
    day[i] = date.fromordinal(t[i]).day

#
# lat and lons of obs
#

fileobj = Dataset(file0, 'r')
lon = fileobj.variables['lon'][:].astype(float)
lat = fileobj.variables['lat'][:].astype(float)
fill_value = fileobj.variables['sst']._FillValue.astype(float)
scale = fileobj.variables['sst'].scale_factor.astype(float)
offset = fileobj.variables['sst'].add_offset.astype(float)
fileobj.close()

#
# Size of mhwBlock variable
#

matobj = io.loadmat(header + 'timeseries/avhrr-only-v2.ts.' + str(300).zfill(4) + '.mat')
mhws, clim = mhw.detect(t, matobj['sst_ts'][300,:])
mhwBlock = mhw.blockAverage(t, mhws)
years = mhwBlock['years_centre']
NB = len(years)

#
# initialize some variables
#

X = len(lon)
Y = len(lat)
i_which = range(0,X) #,10)
j_which = range(0,Y) #,10)
#i_which = range(0,X,4)
#j_which = range(0,Y,4)
DIM = (len(j_which), len(i_which))
N_ts = np.zeros((len(j_which), len(i_which), NB))
SST_mean_ts = np.zeros((len(j_which), len(i_which), NB))
SST_var_ts = np.zeros((len(j_which), len(i_which), NB))
SST_skew_ts = np.zeros((len(j_which), len(i_which), NB))
lon_map =  np.NaN*np.zeros(len(i_which))
lat_map =  np.NaN*np.zeros(len(j_which))
ar1_tau = np.NaN*np.zeros(DIM)
ar1_sig_eps = np.NaN*np.zeros(DIM)

#
# loop through locations
#

icnt = 0
for i in i_which:
    print i, 'of', len(lon)-1
#   load SST
    matobj = io.loadmat(header + 'timeseries/avhrr-only-v2.ts.' + str(i+1).zfill(4) + '.mat')
    sst_ts = matobj['sst_ts']
    lon_map[icnt] = lon[i]
#   loop over j
    jcnt = 0
    for j in j_which:
        lat_map[jcnt] = lat[j]
        sst = sst_ts[j,:].copy()
        if np.logical_not(np.isfinite(sst.sum())) + ((sst<-1).sum()>0): # check for land, ice
            jcnt += 1
            continue
#   Count number of MHWs of each length
        mhws, clim = mhw.detect(t, sst)
        # Fit AR1 model
        a, tmp, sig_eps = trendSimAR1.ar1fit(signal.detrend(sst - clim['seas']))
        ar1_tau[jcnt,icnt] = -1/np.log(a)
        ar1_sig_eps[jcnt,icnt] = sig_eps
        # SST mean, var, skew
        SST_mean_ij = np.zeros(years.shape)
        SST_var_ij = np.zeros(years.shape)
        SST_skew_ij = np.zeros(years.shape)
        for yr in range(len(years)):
            SST_mean_ij[yr] = np.mean(sst[year==years[yr]])
            SST_var_ij[yr] = np.var((sst - clim['seas'])[year==years[yr]])
            SST_skew_ij[yr] = stats.skew((sst - clim['seas'])[year==years[yr]])
        #
        SST_mean_ts[jcnt,icnt,:] = SST_mean_ij
        SST_var_ts[jcnt,icnt,:] = SST_var_ij
        SST_skew_ts[jcnt,icnt,:] = SST_skew_ij
        # Up counts
        jcnt += 1
    icnt += 1
    # Save data so far
    outfile = pathroot + 'data/MHWs/MeanVar/trendsMeanVar_AR1Fit'
    np.savez(outfile, lon_map=lon_map, lat_map=lat_map, SST_mean_ts=SST_mean_ts, SST_var_ts=SST_var_ts, SST_skew_ts=SST_skew_ts, years=year, ar1_tau=ar1_tau, ar1_sig_eps=ar1_sig_eps)

#
# Calculate trends
#

infile = pathroot + 'data/MHWs/MeanVar/trendsMeanVar_AR1Fit'

data = np.load(infile+'.npz')
lon_map = data['lon_map']
lat_map = data['lat_map']
years = np.unique(data['years'])
SST_mean_ts = data['SST_mean_ts']
SST_var_ts = data['SST_var_ts']
SST_skew_ts = data['SST_skew_ts']

SST_mean_ts[SST_mean_ts==0] = np.nan
SST_var_ts[SST_var_ts==0] = np.nan
SST_skew_ts[SST_skew_ts==0] = np.nan

# Calculate trends
SST_mean_tr = np.nan*np.zeros(SST_mean_ts[:,:,0].shape)
SST_var_tr = np.nan*np.zeros(SST_mean_ts[:,:,0].shape)
SST_skew_tr = np.nan*np.zeros(SST_mean_ts[:,:,0].shape)
SST_mean_dtr = np.nan*np.zeros(SST_mean_ts[:,:,0].shape)
SST_var_dtr = np.nan*np.zeros(SST_mean_ts[:,:,0].shape)
SST_skew_dtr = np.nan*np.zeros(SST_mean_ts[:,:,0].shape)
for i in range(len(lon_map)):
    print i+1, 'of', len(lon_map)
    for j in range(len(lat_map)):
        if ~np.isnan(SST_mean_ts[j,i,:].sum()):
            SST_mean_tr[j,i], SST_mean_dtr[j,i] = ecj.trend(years, SST_mean_ts[j,i,:])[1:]
        if ~np.isnan(SST_var_ts[j,i,:].sum()):
            SST_var_tr[j,i], SST_var_dtr[j,i] = ecj.trend(years, SST_var_ts[j,i,:])[1:]
        if ~np.isnan(SST_skew_ts[j,i,:].sum()):
            SST_skew_tr[j,i], SST_skew_dtr[j,i] = ecj.trend(years, SST_skew_ts[j,i,:])[1:]

# Save data
outfile = pathroot + 'data/MHWs/MeanVar/trendsMeanVar_AR1Fit_tr'
# np.savez(outfile, SST_mean_tr=SST_mean_tr, SST_var_tr=SST_var_tr, SST_skew_tr=SST_skew_tr, SST_mean_dtr=SST_mean_dtr, SST_var_dtr=SST_var_dtr, SST_skew_dtr=SST_skew_dtr)






