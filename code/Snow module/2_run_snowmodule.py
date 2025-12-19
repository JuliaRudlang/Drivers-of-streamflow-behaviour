#%%
# This script is adapted from F.van Oorschot: https://github.com/fvanoorschot/global_sr_module
#%%
# import packages
#import sklearn
#from sklearn.linear_model import LinearRegression
from scipy.stats import gaussian_kde
#import statsmodels.api as sm
#from statsmodels.regression.linear_model import OLS
#from statsmodels.tools import add_constant
#import matplotlib as mpl
import random
from scipy.optimize import minimize
import pandas as pd
import numpy as np

import glob
import os

#!pip install pathos # might need to install this package!!
from pathos.threading import ThreadPool as Pool

#%% Info
# Data needed is a csv for each catchment containinf time series of P, Ep and T
# As well as csv. files for each cathment containing mean elevation and elevation bands (calculated with GEE)

#%%
# Introducing the functions used in this script
# The calculation ( I ran it here rather than calling in the function)
# here the snow calculation functions are provided
def snow_calculation(catch_id,work_dir):
    # load p-ep-tas timeseries
    file = glob.glob(f'/scratch/juliarudlang/Data/P_Ep_T/{catch_id}.csv')[0]
    ts = pd.read_csv(f'{file}',index_col=0)
    p = ts.p.values
    tas = ts.tas.values

    # load mean elevation
    file = glob.glob(f'{work_dir}/stats/ele_{catch_id}.csv')[0]
    elm = pd.read_csv(f'{file}',index_col=0)

    # load elevation zones
    file = glob.glob(f'{work_dir}/el_zones/{catch_id}*.csv')[0]
    elz = pd.read_csv(f'{file}',index_col=0)
    
    # snow parameters
    TT = -1
    MF = 2.5
    
    # make empty matrices with timseries rows and elevation zones columns
    Pm_el = np.zeros((len(p),len(elz))) # melt water
    Pl_el = np.zeros((len(p),len(elz))) # liquid precipitation
    Ps_el = np.zeros((len(p),len(elz))) # solid precipitation

    for j in range(len(elz)):
        Ss = np.zeros(len(p)) # snow storage
        Pm = np.zeros(len(p)) # melt water
        Pl = np.zeros(len(p)) # liquid precipitation
        Ps = np.zeros(len(p)) # solid precipitation
        T = np.zeros(len(p)) # temperature

        # temperature difference for elevation zone
        el_dif = elm.mean_ele - elz.loc[j]['mean_el']
        dt = (el_dif/1000.)*6.4

        # loop over timesteps and compute ps, pm and pl
        for i in range(0,len(p)):               
            Tmean = tas[i]
            T[i] = Tmean+dt

            if T[i]>TT:
                Pl[i] = p[i]
                Ps[i] = 0

                Pm[i] = min(Ss[i],MF*(T[i]-TT))
                Ss[i] = max(0, Ss[i]-Pm[i])
            else:
                Ps[i] = p[i]
                Pl[i] = 0
                Ss[i] = Ss[i]+Ps[i]
            if i<len(p)-1:
                Ss[i+1] = Ss[i]

        # scale to coverage of elevation zone and add to matrix
        Ps = Ps * elz.loc[j]['frac']
        Pm = Pm * elz.loc[j]['frac']
        Pl = Pl * elz.loc[j]['frac']
        Pm_el[:,j] = Pm
        Pl_el[:,j] = Pl
        Ps_el[:,j] = Ps

    # sum fractions of pm, pl and ps
    pm = Pm_el.sum(axis=1)
    pl = Pl_el.sum(axis=1)
    ps = Ps_el.sum(axis=1)
	

    # add to forcing dataframe
    ts['pm'] = pm
    ts['pl'] = pl
    ts['ps'] = ps

	
    ts.to_csv(f'{work_dir}/output/timeseries/{catch_id}.csv')
    
    
    
def run_function_parallel_snow(
    catch_list=list,
    work_dir_list=list,
    # threads=None
    threads=100
    ):
    """
    Runs function snow_calculation  in parallel.

    catch_list:  str, list, list of catchmet ids
    work_dir_list:     str, list, list of work dir
    threads:         int,       number of threads (cores), when set to None use all available threads

    Returns: None
    """
    # Set number of threads (cores) used for parallel run and map threads
    if threads is None:
        pool = Pool()
    else:
        pool = Pool(nodes=threads)
    # Run parallel models
    results = pool.map(
        snow_calculation,
        catch_list,
        work_dir_list,
    )
	
	
#%%	

work_dir='/scratch/juliarudlang/Snow_Module' # Change to your workdir

# here open and print the list of ids of the snow catchments
catch_id_list = np.genfromtxt(f'{work_dir}/catch_id_list_snow_t_and_p.txt',dtype='str')

#%%
# check which catchments are missing
catch_list = catch_id_list
el_id_list=[]
for filepath in glob.iglob(f'{work_dir}/output/timeseries/*.csv'):
    f = os.path.split(filepath)[1] # remove full path
    f = f[:-4] # remove .year extension
    el_id_list.append(f)
dif = list(set(catch_id_list) - set(el_id_list))
len(dif)

# Run function in parallel again for the catchments that where not run the first time
work_dir_list = [work_dir] * len(dif)
run_function_parallel_snow(dif, work_dir_list)