# MHW_MeanVar
Code in support of Oliver, E. C. J. (2019), Mean warming not variability drives marine heatwave trends, Climate Dynamics, 53(3), pp. 1653-1659, doi: 10.1007/s00382-019-04707-2.

Files:

 - ecoliver.py, deseason.py are scripts with basic support functions
 - trendSimAR1.py includes functions to perform fitting and simulating of an AR1 process
 - meanVsVar_MHWTrends_systematicLoop_analyse.py includes most of the final analysis to produce the publication figures (Figs. 2, 4, 5, 6)
 - meanVsVar_MHWTrends_systematicLoop.py generates the various simTrends_k***.npz files that are analysed in the previous script
 - calc_trendsMeanVar_AR1Fit.py generates the needed trendsMeanVar_AR1Fit_tr.npz file used in the above analysis script
