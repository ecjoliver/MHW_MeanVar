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

# Plot parameters
domain = [-65, 0, 70, 360]
domain_draw = [-60, 60, 60, 360]
dlat = 30
dlon = 60
#proj = bm.Basemap(projection='merc', llcrnrlat=domain[0], llcrnrlon=domain[1], urcrnrlat=domain[2], urcrnrlon=domain[3], resolution='i')
lon0 = 160
proj = bm.Basemap(projection='eck4', lon_0=lon0-360, resolution='l')
bg_col = '0.6'

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

cmap = cmap1 = plt.get_cmap('RdBu_r')
cmap = mpl.colors.ListedColormap(cmap(np.floor(np.linspace(0, 255, 8)).astype(int)))

trend = np.ones(SST_mean_tr.shape)
trend[SST_mean_tr*10. >= 0.1] = 2.
#trend[(SST_mean_tr*10. >= 0.) * (SST_var_tr*10 >= 0.15)] = 4.
#trend[(SST_mean_tr*10. < 0.) * (SST_var_tr*10 >= 0.15)] = 2.
trend[SST_var_tr*10 >= 0.15] = 3.
trend *= datamask

typeColours = [[1., 1., 1.], [250./255, 128./255, 114./255], [158./255, 0., 0.]]
cmap1 = mpl.colors.ListedColormap(np.array(typeColours))
cmap1.set_bad(color = 'k', alpha = 0.)

i_20E = np.where(lon_map>lon0+180)[0][0]
lon_map = np.append(lon_map[i_20E:], lon_map[:i_20E]+360)
trend = np.append(trend[:,i_20E:], trend[:,:i_20E], axis=1)

smoothed = np.ma.masked_invalid(trend) - ecj.spatial_filter(np.ma.masked_invalid(trend), 0.25, 2., 2.)

llon_map, llat_map = np.meshgrid(lon_map - 360, lat_map)
lonproj, latproj = proj(llon_map, llat_map)

#plt.figure(figsize=(12,5))
plt.clf()
plt.subplot(1,1,1, facecolor=bg_col)
proj.fillcontinents(color='0.25', lake_color='w', ax=None, zorder=None, alpha=None)
proj.drawcoastlines(color='k', linewidth=0.25)
#plt.pcolor(lonproj, latproj, np.ma.masked_invalid(trend), cmap=cmap1)
#plt.pcolor(lonproj, latproj, smoothed, cmap=cmap1)
plt.contourf(lonproj, latproj, smoothed, cmap=cmap1)
plt.clim(0.5, 3.5)

# plt.savefig('warming.pdf', bbox_inches='tight')
# plt.savefig('warming.png', bbox_inches='tight', dpi=600)



