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
datamask = np.ones(map_tau.shape)
datamask[np.isnan(map_tau)] = np.nan
datamask[ice_longestRun>=6.] = np.nan # mask where ice had runs of 6 days or longer
datamask[lat_map<=-65.,:] = np.nan
map_tau *= datamask
map_sig_eps *= datamask

# Plot distribution of tau-sig_eps

TAU, SIG_EPS = np.mgrid[0:120:0.1, 0:1:0.01]
positions = np.vstack([TAU.ravel(), SIG_EPS.ravel()])
kernel = stats.gaussian_kde(np.array([ecj.nonans(map_tau.flatten()), ecj.nonans(map_sig_eps.flatten())]))
pdf_AR1 = np.reshape(kernel(positions), TAU.shape)

# TAU.flatten()[np.argmax(pdf_AR1)]
# SIG_EPS.flatten()[np.argmax(pdf_AR1)]
# np.nanpercentile(map_tau.flatten(), [0.5, 99.5])
# np.nanpercentile(map_sig_eps.flatten(), [0.5, 99.5])

# Find the (tau, sigmas) over which to sample
tau_sample, sig_sample = np.mgrid[2:60+1:2, 0.1:0.6+0.02:0.02]
tau_sample = tau_sample.flatten()
sig_sample = sig_sample.flatten()
keep = np.array([])
for k in range(len(tau_sample)):
    tau0 = tau_sample[k]
    sig0 = sig_sample[k]
    i, j = ecj.findxy(TAU, SIG_EPS, (tau0, sig0))
    if pdf_AR1[j,i] >= 0.001:
        keep = np.append(keep, k)

tau_sample = tau_sample[keep.astype(int)]
sig_sample = sig_sample[keep.astype(int)]

# Plot parameters
domain = [-65, 0, 70, 360]
domain_draw = [-60, 60, 60, 360]
dlat = 30
dlon = 60
proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
llon_map, llat_map = np.meshgrid(lon_map, lat_map)
lonproj, latproj = proj(llon_map, llat_map)
bg_col = '0.6'
cont_col = '0.0'

cmap = cmap1 = plt.get_cmap('viridis')
cmap = mpl.colors.ListedColormap(cmap(np.floor(np.linspace(0, 255, 7)).astype(int)))

plt.figure(figsize=(12,5))
plt.clf()
plt.subplot(2,2,1, axisbg=bg_col)
proj.fillcontinents(color=(0,0,0), lake_color=None, ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False], dashes=[3,900])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False], dashes=[3,900])
plt.pcolor(lonproj, latproj, np.ma.masked_invalid(map_tau), cmap=cmap)
plt.clim(0, 70)
plt.title(r'(A) Autoregressive time scale ($\tau$)')
H = plt.colorbar()
H.set_label(r'days')
#
plt.subplot(2,2,3, axisbg=bg_col)
proj.fillcontinents(color=(0,0,0), lake_color=None, ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False], dashes=[3,900])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True], dashes=[3,900])
plt.pcolor(lonproj, latproj, np.ma.masked_invalid(map_sig_eps), cmap=cmap)
plt.clim(0.0, 0.7)
plt.title(r'(B) Error standard deviation ($\sigma_\epsilon$)')
H = plt.colorbar()
H.set_label(r'$^\circ$C')
#
plt.subplot(1,2,2)
plt.plot(map_tau.flatten(), map_sig_eps.flatten(), 'ko', alpha=0.5, zorder=10, markeredgewidth=0, markersize=4)
plt.contourf(TAU, SIG_EPS, pdf_AR1, levels=[0.01,0.02,0.05,0.10,0.15,0.2,0.3,0.5,0.75,0.95], zorder=20, cmap=plt.cm.hot)
H = plt.colorbar(orientation='horizontal')
plt.ylim(0.05, 0.65)
plt.xlim(0, 80)
plt.clim(-0.025, 0.9)
plt.xlabel(r'Autoregressive time scale, $\tau$ (days)')
plt.ylabel(r'Error standard deviation, $\sigma_\epsilon$ ($^\circ$C)')
H.set_label('Probability density')
plt.legend(['Individual observations'], loc='upper right')
plt.title('(C) Probability distribution')
# plt.savefig('../../documents/22_Mean_vs_Variance/figures/pdf_AR1_orig.png', bbox_inches='tight', pad_inches=0.05, dpi=300)


#
# Distribution of trends (mean, variance)
#

outfile = header + 'MHWs/MeanVar/trendsMeanVar_AR1Fit_tr'
data = np.load(outfile + '.npz')
SST_mean_tr = data['SST_mean_tr']
SST_var_tr = data['SST_var_tr']
SST_mean_tr *= datamask
SST_var_tr *= datamask

# Plot distribution of trends

MEAN, VAR = np.mgrid[-1:1:0.01, -0.6:3:0.01]
positions = np.vstack([MEAN.ravel(), VAR.ravel()])
kernel = stats.gaussian_kde(np.array([ecj.nonans(SST_mean_tr.flatten()), ecj.nonans(SST_var_tr.flatten())]))
pdf_tr = np.reshape(kernel(positions), MEAN.shape)

cmap = cmap1 = plt.get_cmap('RdBu_r')
cmap = mpl.colors.ListedColormap(cmap(np.floor(np.linspace(0, 255, 8)).astype(int)))

plt.figure(figsize=(12,5))
plt.clf()
plt.subplot(2,2,1, axisbg=bg_col)
proj.fillcontinents(color=(0,0,0), lake_color=None, ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False], dashes=[3,900])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False], dashes=[3,900])
plt.pcolor(lonproj, latproj, np.ma.masked_invalid(SST_mean_tr*10), cmap=cmap)
plt.clim(-0.8, 0.8)
plt.title(r'(A) Linear trend in mean SST')
H = plt.colorbar()
H.set_label(r'$^\circ$C decade$^{-1}$')
#
plt.subplot(2,2,3, axisbg=bg_col)
proj.fillcontinents(color=(0,0,0), lake_color=None, ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False], dashes=[3,900])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True], dashes=[3,900])
plt.pcolor(lonproj, latproj, np.ma.masked_invalid(SST_var_tr*10), cmap=cmap)
plt.clim(-0.4, 0.4)
plt.title(r'(B) Linear trend in SST variance')
H = plt.colorbar()
H.set_label(r'$^\circ$C$^2$ decade$^{-1}$')
#
plt.subplot(1,2,2)
plt.plot(SST_mean_tr.flatten()*10, SST_var_tr.flatten()*10, 'ko', alpha=0.5, zorder=10, markeredgewidth=0, markersize=4)
plt.contourf(MEAN*10, VAR*10, pdf_tr, zorder=20, cmap=plt.cm.hot, levels=[2, 10, 100, 500, 1000, 1500, 2000, 4000])
#levels=[0.01,0.02,0.05,0.10,0.15,0.2,0.3,0.5,0.75,0.95], zorder=20, cmap=plt.cm.hot)
H = plt.colorbar(orientation='horizontal')
plt.ylim(-0.5, 2.5)
plt.xlim(-0.6, 1.1)
plt.xlabel(r'Linear trend in mean SST ($^\circ$C decade$^{-1}$)')
plt.ylabel(r'Linear trend in SST var. ($^\circ$C$^2$ decade$^{-1}$)')
H.set_label('Probability density')
plt.legend(['Individual observations'], loc='upper right')
plt.title('(C) Probability distribution')
# plt.savefig('../../documents/22_Mean_vs_Variance/figures/pdf_Trends_orig.png', bbox_inches='tight', pad_inches=0.05, dpi=300)

#
# Calculate relative dominance of mean/variance trend by spatial grid cell
#

metrics = ['total_days', 'intensity_max_max']
trendsSim_indep = {}
trendsSim_mean = {}
trendsSim_var = {}
for metric in metrics:
    trendsSim_indep[metric] = np.nan*np.zeros(map_tau.shape)
    trendsSim_mean[metric] = {'mean': np.nan*np.zeros(map_tau.shape), 'lower': np.nan*np.zeros(map_tau.shape), 'upper': np.nan*np.zeros(map_tau.shape)}
    trendsSim_var[metric] = {'mean': np.nan*np.zeros(map_tau.shape), 'lower': np.nan*np.zeros(map_tau.shape), 'upper': np.nan*np.zeros(map_tau.shape)}

# Location of interest
#i = 330
#j = 125
for i in range(len(lon_map)):
    print i+1, len(lon_map)
    for j in range(len(lat_map)):
        if np.isnan(map_tau[j,i]) + np.isnan(map_sig_eps[j,i]):
            continue
        #print lat_map[j], lon_map[i]
        # AR1 parameters at that location
        tau = map_tau[j,i]
        sig_eps = map_sig_eps[j,i]
        # Nearest AR1 parameter values from subsampled distribution (sample index of)
        #tau = ecj.find_nearest(np.unique(tau_sample), tau)[0]
        #sig_eps = ecj.find_nearest(np.unique(sig_sample), sig_eps)[0]
        # Sample index
        #k = np.where((tau_sample == tau) * (sig_sample == sig_eps))[0][0]
        k = np.argmin((tau_sample - tau)**2 + (sig_sample - sig_eps)**2)
        # Load in data for this index
        infile = header + 'MHWs/MeanVar/simTrends/simTrends_k' + str(k).zfill(3) + '.npz'
        try:
            data = np.load(infile)
        except:
            continue
        trend_range_mean = data['trend_range_mean']
        trend_range_var = data['trend_range_var']
        trends_mean = data['trends_mean'].item()
        trends_var = data['trends_var'].item()
        keys = data['keys']
        N_keys = data['N_keys']
        units = data['units'].item()
        # Find observed mean and variance trends by location
        #if (SST_mean_tr[j,i] < 0) + (SST_var_tr[j,i] < 0):
        #    continue
        #sign_mean = np.sign(SST_mean_tr[j,i])
        #sign_var = np.sign(SST_var_tr[j,i])
        #im = ecj.find_nearest(trend_range_mean, np.abs(SST_mean_tr[j,i]*10))[1]
        #iv = ecj.find_nearest(trend_range_var, np.abs(SST_var_tr[j,i]*10))[1]
        SST_mean_tr0 = SST_mean_tr[j,i]*10
        SST_var_tr0 = SST_var_tr[j,i]*10
        #im = ecj.find_nearest(trend_range_mean, SST_mean_tr[j,i]*10)[1]
        #iv = ecj.find_nearest(trend_range_var, SST_var_tr[j,i]*10)[1]
        for metric in metrics:
            #KS, p = stats.ks_2samp(np.sign(trends_mean[metric][:,im]), np.sign(trends_var[metric][:,iv])) # if p small, distributions different
            # Test for independence of mean/variance roles
            #if p < 0.05:
            #    trendsSim_indep[metric][j,i] = 1.
            #else:
            #    trendsSim_indep[metric][j,i] = 0.
            # Mean simulated trend values
            #trendsSim_mean[metric]['mean'][j,i] = np.mean(trends_mean[metric][:,im])
            #trendsSim_var[metric]['mean'][j,i] = np.mean(trends_var[metric][:,iv])
            #trendsSim_mean[metric]['lower'][j,i] = np.percentile(trends_mean[metric][:,im], 2.5)
            #trendsSim_var[metric]['lower'][j,i] = np.percentile(trends_var[metric][:,iv], 2.5)
            #trendsSim_mean[metric]['upper'][j,i] = np.percentile(trends_mean[metric][:,im], 97.5)
            #trendsSim_var[metric]['upper'][j,i] = np.percentile(trends_var[metric][:,iv], 97.5)
            #
            trendsSim_mean[metric]['mean'][j,i] = np.interp(SST_mean_tr0, trend_range_mean, np.mean(trends_mean[metric], axis=0))
            trendsSim_var[metric]['mean'][j,i] = np.interp(SST_var_tr0, trend_range_var, np.mean(trends_var[metric], axis=0))
            trendsSim_mean[metric]['lower'][j,i] = np.interp(SST_mean_tr0, trend_range_mean, np.percentile(trends_mean[metric], 2.5, axis=0))
            trendsSim_var[metric]['lower'][j,i] = np.interp(SST_var_tr0, trend_range_var, np.percentile(trends_var[metric], 2.5, axis=0))
            trendsSim_mean[metric]['upper'][j,i] = np.interp(SST_mean_tr0, trend_range_mean, np.percentile(trends_mean[metric], 97.5, axis=0))
            trendsSim_var[metric]['upper'][j,i] = np.interp(SST_var_tr0, trend_range_var, np.percentile(trends_var[metric], 97.5, axis=0))

trendSim_mean_nonZero = {}
trendSim_var_nonZero = {}
for metric in metrics:
    trendSim_mean_nonZero[metric] = ~(trendsSim_mean[metric]['lower'] < 0.) * (trendsSim_mean[metric]['upper'] > 0.)
    trendSim_var_nonZero[metric] = ~(trendsSim_var[metric]['lower'] < 0.) * (trendsSim_var[metric]['upper'] > 0.)

# Define some trend types
trendsSim_type = {}
for metric in metrics:
    trendsSim_type[metric] = np.nan*trendsSim_mean[metric]['mean']
    # Type 1: Both are not different from zero
    trendsSim_type[metric][(~trendSim_mean_nonZero[metric]) * (~trendSim_var_nonZero[metric])] = 1.
    # Type 4: Both are different from zero
    trendsSim_type[metric][trendSim_mean_nonZero[metric] * trendSim_var_nonZero[metric]] = 4.
    # Type 3: Mean trend is different from zero, variance trend not
    trendsSim_type[metric][trendSim_mean_nonZero[metric] * (~trendSim_var_nonZero[metric])] = 3.
    # Type 2: Variance trend is different from zero, mean trend not
    trendsSim_type[metric][(~trendSim_mean_nonZero[metric]) * trendSim_var_nonZero[metric]] = 2.
    # Now mask out the crap again
    trendsSim_type[metric][np.isnan(trendsSim_mean[metric]['mean'])] = np.nan

# Proportions
prop = {}
for metric in metrics:
    prop[metric] = {}
    for type in range(1,4+1):
        prop[metric][type] = 100*(trendsSim_type[metric]==1.*type).sum()*1. / np.sum(~np.isnan(trendsSim_type[metric]))

# Plot it up!!!

typeColours = [[1., 1., 1.], [1., 200./255, 102./255], [1., 105./255, 0.], [158./255, 0., 0.]]
typeColours = [[1., 1., 1.], [30./255, 60./255, 180./255], [1., 105./255, 0.], [158./255, 0., 0.]] #[255./255, 215./255, 0./255]]
cmap1 = mpl.colors.ListedColormap(np.array(typeColours))
cmap1.set_bad(color = 'k', alpha = 0.)

plt.figure(figsize=(7,6))
plt.clf()
plt.subplot(2,1,1, axisbg=bg_col)
proj.fillcontinents(color=(0,0,0), lake_color=None, ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False], dashes=[3,900])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,False], dashes=[3,900])
plt.pcolor(lonproj, latproj, np.ma.masked_invalid(trendsSim_type['total_days']), cmap=cmap1)
plt.clim(0.5, 4.5)
plt.title('(A) Exposure (Annnual MHW days)')
H = plt.colorbar()
H.set_ticks([1, 2, 3, 4])
#H.set_ticklabels(['Neither\n(' + '{:.1f}'.format(prop['total_days'][1]) + '%)', 'Var.\n(' + '{:.1f}'.format(prop['total_days'][2]) + '%)', 'Mean\n(' + '{:.1f}'.format(prop['total_days'][3]) + '%)', 'Both\n(' + '{:.1f}'.format(prop['total_days'][4]) + '%)'])
H.set_ticklabels(['1. Neither\n(' + '{:.1f}'.format(prop['total_days'][1]) + '%)', '2. Var.\n(' + '{:.1f}'.format(prop['total_days'][2]) + '%)', '3. Mean\n(' + '{:.1f}'.format(prop['total_days'][3]) + '%)', '4. Both\n(' + '{:.1f}'.format(prop['total_days'][4]) + '%)'])
plt.subplot(2,1,2, axisbg=bg_col)
proj.fillcontinents(color=(0,0,0), lake_color=None, ax=None, zorder=None, alpha=None)
proj.drawparallels(range(domain_draw[0],domain_draw[2]+1,dlat), labels=[True,False,False,False], dashes=[3,900])
proj.drawmeridians(range(domain_draw[1],domain_draw[3]+1,dlon), labels=[False,False,False,True], dashes=[3,900])
plt.pcolor(lonproj, latproj, np.ma.masked_invalid(trendsSim_type['intensity_max_max']), cmap=cmap1)
plt.clim(0.5, 4.5)
plt.title('(B) Intensity (Maximum MHW intensity)')
H = plt.colorbar()
H.set_ticks([1, 2, 3, 4])
H.set_ticklabels(['1. Neither\n(' + '{:.1f}'.format(prop['intensity_max_max'][1]) + '%)', '2. Var.\n(' + '{:.1f}'.format(prop['intensity_max_max'][2]) + '%)', '3. Mean\n(' + '{:.1f}'.format(prop['intensity_max_max'][3]) + '%)', '4. Both\n(' + '{:.1f}'.format(prop['intensity_max_max'][4]) + '%)'])

# plt.savefig('../../documents/22_Mean_vs_Variance/figures/types_orig.png', bbox_inches='tight', pad_inches=0.05, dpi=300)

tau = TAU.flatten()[np.argmax(pdf_AR1)]
sig_eps = SIG_EPS.flatten()[np.argmax(pdf_AR1)]
k = np.argmin((tau_sample - tau)**2 + (sig_sample - sig_eps)**2)
infile = header + 'MHWs/MeanVar/simTrends/simTrends_k' + str(k).zfill(3) + '.npz'
data = np.load(infile)
trend_range_mean = data['trend_range_mean']
trend_range_var = data['trend_range_var']
trends_mean = data['trends_mean'].item()
trends_var = data['trends_var'].item()
keys = data['keys']
N_keys = data['N_keys']
units = data['units'].item()
yrange = {'total_days': [-30, 75], 'intensity_max_max': [-0.7, 1.3]}
titles = ['(A)', '(B)', '(C)', '(D)']
metricName = {'total_days': 'MHW Exposure', 'intensity_max_max': 'MHW Intensity'}

plt.figure(figsize=(10,8))
plt.clf()
i = 0
for key in ['total_days', 'intensity_max_max']:
    # Mean
    AX = plt.subplot(2, 2, 2*i + 1)
    plt.plot(trend_range_mean, 10*np.mean(trends_mean[key], axis=0), 'k-o', linewidth=2, zorder=50)
    plt.plot(trend_range_mean, 10*np.percentile(trends_mean[key], 2.5, axis=0), 'b-o', linewidth=2, zorder=50)
    plt.plot(trend_range_mean, 10*np.percentile(trends_mean[key], 97.5, axis=0), 'r-o', linewidth=2, zorder=50)
    plt.plot(trend_range_mean, 10*trends_mean[key].T, '-', color='0.65', alpha=0.25, zorder=20)
    plt.plot(trend_range_mean, 0*trend_range_mean, 'k:', zorder=30)
    plt.ylim(yrange[key][0], yrange[key][1])
    if i == 1:
        plt.xlabel(r'SST mean trend [$^\circ$C decade$^{-1}$]')
    else:
        AX.set_xticklabels([])
    plt.ylabel(r'' + metricName[key] + ' trend [' + units[key] + ' decade$^{-1}$]')
    #plt.title(r'$\tau$ = ' + str(tau) + ' days, $\sigma_\epsilon$ = ' + str(sig_eps) + ' $^\circ$C')
    plt.title(titles[2*i])
    if key == 'total_days':
        plt.legend(['Ensemble mean', 'Ensemble 2.5th pctle.', 'Ensemble 97.5th pctle.', 'Simulated trends'], loc='upper left')
    # Variance
    AX = plt.subplot(2, 2, 2*i + 2)
    plt.plot(trend_range_var, 10*np.mean(trends_var[key], axis=0), 'k-o', linewidth=2, zorder=50)
    plt.plot(trend_range_var, 10*np.percentile(trends_var[key], 2.5, axis=0), 'b-o', linewidth=2, zorder=50)
    plt.plot(trend_range_var, 10*np.percentile(trends_var[key], 97.5, axis=0), 'r-o', linewidth=2, zorder=50)
    plt.plot(trend_range_var, 10*trends_var[key].T, '-', color='0.65', alpha=0.25, zorder=20)
    plt.plot(trend_range_var, 0*trend_range_var, 'k:', zorder=30)
    plt.ylim(yrange[key][0], yrange[key][1])
    if i == 1:
        plt.xlabel(r'SST variance trend [$^\circ$C$^2$ decade$^{-1}$]')
    else:
        AX.set_xticklabels([])
    AX.set_yticklabels([])
    plt.title(titles[2*i+1])
    #
    i += 1

# plt.savefig('../../documents/22_Mean_vs_Variance/figures/simulated_trends_totDaysIMax_orig.png', bbox_inches='tight', pad_inches=0.05, dpi=300)


