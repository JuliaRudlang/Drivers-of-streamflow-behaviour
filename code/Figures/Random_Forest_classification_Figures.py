# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 11:47:01 2025

@author: juliarudlang
"""

#%% Load in packages
import geopandas as gpd                                      # Pandas for geospatial analysis
from shapely.geometry import Point, Polygon                  # Module used for geospatial analysis     
import pymannkendall as mk                                   # Module used for trend-computation
from plotly.offline import plot
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings                                              
import datetime                                              # Datetime module pretty useful for time-series analysis
import tqdm as tqdm                                          # Useful module to access the progress bar of a loop
import os
import glob
warnings.simplefilter(action='ignore', category=Warning)     # Module useful for taking out some unecessary warnings
from scipy.interpolate import interp1d
import seaborn as sns
import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib
from mpl_toolkits.axes_grid1 import inset_locator
from sklearn.ensemble import RandomForestClassifier  
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score  
import seaborn as sns  
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import itertools
from sklearn.metrics import r2_score
import random
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.patches as patches
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import string
from matplotlib.colors import to_rgba

#%% Hydrological signatures (mainly need the cluster label column)
hydro_sign = pd.read_csv(r'U:\Hydrological Data\EStreams\Data 071025\hydro_sign_clustered_101025.csv', encoding='utf-8')
hydro_sign.set_index("basin_id", inplace=True)
hydro_sign 
#%% # Load the climate and landscape data  
file_path = 'U:\Hydrological Data\EStreams\Data 071025\climate_landscape_071025.csv'

# Read the file
climate_landscape = pd.read_csv(file_path)
climate_landscape.set_index("basin_id", inplace=True)
climate_landscape

#%% Make a copy 
clusters_df = hydro_sign.copy()
#%% Create a colour map 
color_mapping2 = ['#1f77b4',  # Blue")
                '#e377c2',  # Magenta ")
                 '#9467bd', # Purple
                  '#2ca02c',  # Green ")
                 '#17becf',  # Cyan ")
                 '#c51b8a',  # Pink")
                 '#aec7e8',  # Light blue")
                 '#ff7f0e',  # Orange ")
                  '#005a32',  # Dark green")
                  '#d62728'  # Red ")
                 ]

#%%
# The following section renames and divides the different signatures 
#%% Only climate signatures corresponds to 2-C
# taking out ndvi and lai variance and parde range 
climate_clean_df = climate_landscape.drop(columns={'mean_Artifical_Surfaces',
       'mean_Agricultural_areas', 'mean_Wetlands', 'LAI_median',
        'NDVI_median',  'Irr_median',
        'mean_Forest', 'mean_Scrub',
       'mean_Open_natural_areas', 'frac_sand_gravel', 'frac_silt_clay',
       'frac_organic_content_median', 'soil_twac_median',
       'soil_bulk_density_median', 'soil_root_depth_median',
       'fraction_treecover', 'Sandstones', 'Mudstones', 'Carbonates',
       'Other/lowperm', 'cont_high_perm', 'cont_medium_perm', 'cont_low_perm',
       'cont_verylow_perm', 'bedrock_depth', 'ele_mt_mean', 'slp_dg_mean',
       'flat_area_fra', 'steep_area_fra', 'elon_ratio', 'strm_dens',
       'reservoir_storage', 'lakes_area_relative_to_catchA', 'LAI_parde_range', 'NDVI_parde_range','LAI_variance','NDVI_variance'})
#%% Vegetation and landcvoer, experimemnt 3-VLC
vegetation_landcover_df = climate_landscape.drop(columns={'pet_mean', 'aridity', 'p_seasonality', 'T_range', 'SI_PET',
       'phi_P_PET', 'annual_intensity_PM_median',
       'annual_median_fraction_snow', 'interannual_var_fraction_snow',
       'longest_duration_annual_snow_cover_median',
       'Median_mean_Annual_Duration_Snow_cover', 'annual_intensity_PL_median',
       'Ea_var', 'Ea_mean', 'pet_var', 'PL_var', 'PM_var', 'PS_var', 'PL_sin',
       'PL_cos', 'median_PL_annual_days', 'var_PL_annual_days', 'hpl_freq',
       'hpl_dur', 'lpl_freq', 'lpl_dur', 'T_mean', 'PL_mean',
       'circular_var_timing_PL', 'Longterm_Median_Duration_wd',
       'annual_longest_wd_periods_median', 'median_annual_max_wd',
       'var_annual_max_wd', 'Median_annual_days_wd', 'var_annual_days_wd',
       'median_declining_slope_max_wd', 'median_uprising_slope_max_wd',
       'circ_var_first_negative_WD', 'circ_var_last_negative_WD', 'wd_max_sin',
       'wd_max_cos', 'circ_var_timing_max_wd','PM_mean',
       'First_wd_sin', 'First_wd_cos', 'Last_wd_sin', 'Last_wd_cos','mean_Artifical_Surfaces',
       'mean_Agricultural_areas','Irr_median', 'frac_sand_gravel', 'frac_silt_clay',
       'frac_organic_content_median', 'soil_twac_median',
       'soil_bulk_density_median', 'soil_root_depth_median',
       'Sandstones', 'Mudstones', 'Carbonates',
       'Other/lowperm', 'cont_high_perm', 'cont_medium_perm', 'cont_low_perm',
       'cont_verylow_perm', 'bedrock_depth', 'ele_mt_mean', 'slp_dg_mean',
       'flat_area_fra', 'steep_area_fra', 'elon_ratio', 'strm_dens',
       'reservoir_storage', 'lakes_area_relative_to_catchA'})

#%% Vegetation and lancover, experiment 4-VLS
vegetation_landcover_static_df = climate_landscape.drop(columns={'pet_mean', 'aridity', 'p_seasonality', 'T_range', 'SI_PET',
       'phi_P_PET', 'annual_intensity_PM_median',
       'annual_median_fraction_snow', 'interannual_var_fraction_snow',
       'longest_duration_annual_snow_cover_median',
       'Median_mean_Annual_Duration_Snow_cover', 'annual_intensity_PL_median',
       'Ea_var', 'Ea_mean', 'pet_var', 'PL_var', 'PM_var', 'PS_var', 'PL_sin',
       'PL_cos', 'median_PL_annual_days', 'var_PL_annual_days', 'hpl_freq',
       'hpl_dur', 'lpl_freq', 'lpl_dur', 'T_mean', 'PL_mean',
       'circular_var_timing_PL', 'Longterm_Median_Duration_wd',
       'annual_longest_wd_periods_median', 'median_annual_max_wd',
       'var_annual_max_wd', 'Median_annual_days_wd', 'var_annual_days_wd',
       'median_declining_slope_max_wd', 'median_uprising_slope_max_wd',
       'circ_var_first_negative_WD', 'circ_var_last_negative_WD', 'wd_max_sin',
       'wd_max_cos', 'circ_var_timing_max_wd','PM_mean',
       'First_wd_sin', 'First_wd_cos', 'Last_wd_sin', 'Last_wd_cos','mean_Artifical_Surfaces',
       'mean_Agricultural_areas','Irr_median', 'frac_sand_gravel', 'frac_silt_clay',
       'frac_organic_content_median', 'soil_twac_median',
       'soil_bulk_density_median', 'soil_root_depth_median',
       'Sandstones', 'Mudstones', 'Carbonates',
       'Other/lowperm', 'cont_high_perm', 'cont_medium_perm', 'cont_low_perm',
       'cont_verylow_perm', 'bedrock_depth', 'ele_mt_mean', 'slp_dg_mean',
       'flat_area_fra', 'steep_area_fra', 'elon_ratio', 'strm_dens',
       'reservoir_storage', 'lakes_area_relative_to_catchA', 'LAI_parde_range', 'NDVI_parde_range','LAI_variance','NDVI_variance'})

#%% Soil, geology and topogrpahy signatures, experiemnt 5-SGT
# keep the signatures associated with soil and geology and topography
soil_geology_topography_df = climate_landscape.drop(columns={'pet_mean', 'aridity', 'p_seasonality', 'T_range', 'SI_PET',
       'phi_P_PET', 'annual_intensity_PM_median',
       'annual_median_fraction_snow', 'interannual_var_fraction_snow',
       'longest_duration_annual_snow_cover_median',
       'Median_mean_Annual_Duration_Snow_cover', 'annual_intensity_PL_median',
       'Ea_var', 'Ea_mean', 'pet_var', 'PL_var', 'PM_var', 'PS_var', 'PL_sin',
       'PL_cos', 'median_PL_annual_days', 'var_PL_annual_days', 'hpl_freq',
       'hpl_dur', 'lpl_freq', 'lpl_dur', 'T_mean', 'PL_mean',
       'circular_var_timing_PL', 'Longterm_Median_Duration_wd',
       'annual_longest_wd_periods_median', 'median_annual_max_wd',
       'var_annual_max_wd', 'Median_annual_days_wd', 'var_annual_days_wd',
       'median_declining_slope_max_wd', 'median_uprising_slope_max_wd',
       'circ_var_first_negative_WD', 'circ_var_last_negative_WD', 'wd_max_sin',
       'wd_max_cos', 'circ_var_timing_max_wd','PM_mean',
       'First_wd_sin', 'First_wd_cos', 'Last_wd_sin', 'Last_wd_cos', 'LAI_parde_range', 'NDVI_parde_range','LAI_variance','NDVI_variance','mean_Artifical_Surfaces',
       'mean_Agricultural_areas', 'mean_Wetlands', 'LAI_median',
        'NDVI_median',  'Irr_median',
        'mean_Forest', 'mean_Scrub',
       'mean_Open_natural_areas', 
       'fraction_treecover',
       'reservoir_storage', 'lakes_area_relative_to_catchA'})

#%% Only anthropogenic attributes, experiment 6-A
#This is actually Human Influence
lumped_df = climate_landscape.drop(columns={'pet_mean', 'aridity', 'p_seasonality', 'T_range', 'SI_PET',
       'phi_P_PET', 'annual_intensity_PM_median',
       'annual_median_fraction_snow', 'interannual_var_fraction_snow',
       'longest_duration_annual_snow_cover_median',
       'Median_mean_Annual_Duration_Snow_cover', 'annual_intensity_PL_median',
       'Ea_var', 'Ea_mean', 'pet_var', 'PL_var', 'PM_var', 'PS_var', 'PL_sin',
       'PL_cos', 'median_PL_annual_days', 'var_PL_annual_days', 'hpl_freq',
       'hpl_dur', 'lpl_freq', 'lpl_dur', 'T_mean', 'PL_mean',
       'circular_var_timing_PL', 'Longterm_Median_Duration_wd',
       'annual_longest_wd_periods_median', 'median_annual_max_wd',
       'var_annual_max_wd', 'Median_annual_days_wd', 'var_annual_days_wd',
       'median_declining_slope_max_wd', 'median_uprising_slope_max_wd',
       'circ_var_first_negative_WD', 'circ_var_last_negative_WD', 'wd_max_sin',
       'wd_max_cos', 'circ_var_timing_max_wd','PM_mean',
       'First_wd_sin', 'First_wd_cos', 'Last_wd_sin', 'Last_wd_cos', 'LAI_parde_range', 'NDVI_parde_range','LAI_variance','NDVI_variance',
        'mean_Wetlands', 'LAI_median',
        'NDVI_median',
        'mean_Forest', 'mean_Scrub',
       'mean_Open_natural_areas', 'frac_sand_gravel', 'frac_silt_clay',
       'frac_organic_content_median', 'soil_twac_median',
       'soil_bulk_density_median', 'soil_root_depth_median',
       'fraction_treecover', 'Sandstones', 'Mudstones', 'Carbonates',
       'Other/lowperm', 'cont_high_perm', 'cont_medium_perm', 'cont_low_perm',
       'cont_verylow_perm', 'bedrock_depth', 'ele_mt_mean', 'slp_dg_mean',
       'flat_area_fra', 'steep_area_fra', 'elon_ratio', 'strm_dens'})

#%% # Labels into categories: More detailed
human_influence = ['Irr_median', 'reservoir_storage', 'lakes_area_relative_to_catchA', 'mean_Artifical_Surfaces', 'mean_Agricultural_areas']

landcover = ['fraction_treecover', 'mean_Forest', 'mean_Scrub', 'mean_Open_natural_areas', 'mean_Wetlands']

vegetation_static = ['LAI_median', 'NDVI_median']
vegetation_dynamic = ['LAI_parde_range', 'NDVI_parde_range','LAI_variance','NDVI_variance']

soil_geology = ['frac_sand_gravel',  'frac_silt_clay',  'frac_organic_content_median',  'soil_twac_median',  'soil_bulk_density_median', 'soil_root_depth_median',  'Sandstones',  'Mudstones',  'Carbonates',  'Other/lowperm',  'cont_high_perm',  'cont_medium_perm',  'cont_low_perm',  'cont_verylow_perm',  'bedrock_depth']

climate = ['pet_mean', 'aridity', 'p_seasonality', 'T_range', 'SI_PET',
       'phi_P_PET', 'Ea_var', 'Ea_mean', 'pet_var', 'T_mean' ]

pliquid = ['PL_mean','PL_var', 'PL_sin','PL_cos', 'median_PL_annual_days', 'var_PL_annual_days', 'hpl_freq','annual_intensity_PL_median','hpl_dur', 'lpl_freq', 'lpl_dur','circular_var_timing_PL']

water_deficit = ['Longterm_Median_Duration_wd',
       'annual_longest_wd_periods_median', 'median_annual_max_wd',
       'var_annual_max_wd', 'Median_annual_days_wd', 'var_annual_days_wd',
       'median_declining_slope_max_wd', 'median_uprising_slope_max_wd',
       'circ_var_first_negative_WD', 'circ_var_last_negative_WD', 'wd_max_sin',
       'wd_max_cos', 'circ_var_timing_max_wd', 'First_wd_sin', 'First_wd_cos', 'Last_wd_sin', 'Last_wd_cos']

snow = ['PM_var', 'PS_var', 'annual_median_fraction_snow', 'interannual_var_fraction_snow',
       'longest_duration_annual_snow_cover_median',
       'Median_mean_Annual_Duration_Snow_cover', 'annual_intensity_PM_median','PM_mean']

topography = [ 'ele_mt_mean', 'slp_dg_mean',
       'flat_area_fra', 'steep_area_fra', 'elon_ratio', 'strm_dens']

#%% # Map attribute names to the variables for nicer name visualisation
feature_name_mapping = {
    'pet_mean' : r"$C_{\mathrm{Ep}}$", 
    'aridity' : r"$C_{\mathrm{AI}}$",
    'p_seasonality': r"$C_{\phi(P)}$", 
    'T_range': r"$C_{\mathrm{R(T)}}$", 
    'SI_PET': r"$C_{\phi(\mathrm{Ep})}$",
    'phi_P_PET': r"$C_{\Delta \phi(P,\mathrm{Ep})}$", 
    'annual_intensity_PM_median': r"$C_{\mathrm{I}(P_M)}$",
    'annual_median_fraction_snow' : r"$C_{\mathrm{F(Snow)}}$", 
    'interannual_var_fraction_snow' : r"$C_{\mathrm{V,F(Snow)}}$",
    'longest_duration_annual_snow_cover_median' : r"$C_{\mathrm{D(SnowC_{Max})}}$",
    'Median_mean_Annual_Duration_Snow_cover' : r"$C_{\mathrm{D(SnowC)}}$", 
    'annual_intensity_PL_median' : r"$C_{\mathrm{I}(P_L)}$",
    'Ea_var' : r"$C_{\mathrm{V,Ea}}$", 
    'Ea_mean' : r"$C_{\mathrm{Ea}}$", 
    'pet_var' : r"$C_{\mathrm{V,Ep}}$", 
    'PL_var' : r"$C_{\mathrm{V,P_L}}$", 
    'PM_var' : r"$C_{\mathrm{V,(P_M)}}$", 
    'PS_var' : r"$C_{\mathrm{V,P_S}}$", 

    'PL_sin' : r"$C_{\mathrm{t}(P_{L, max})_{\mathrm{WS}}}$",
    'PL_cos' : r"$C_{\mathrm{t}(P_{L,max})_{\mathrm{SA}}}$", 

    'median_PL_annual_days' : r"$C_{\mathrm{N}(P_L)}$", 
    'var_PL_annual_days' : r"$C_{\mathrm{V,N}(P_L)}$", 

    'hpl_freq' : r"$C_{\mathrm{f}(P_{LH})}$",
    'hpl_dur' : r"$C_{\mathrm{D}(P_{LH})}$", 
    'lpl_freq' : r"$C_{\mathrm{f}(P_{LL})}$", 
    'lpl_dur' : r"$C_{\mathrm{D}(P_{LL})}$", 

    'T_mean' : r"$C_T$", 
    'PL_mean' : r"$C_{P_L}$",

    'circular_var_timing_PL' : r"$C_{\mathrm{V,t}(P_L)}$", 

    'Longterm_Median_Duration_wd' : r"$C_{\mathrm{D(WD)}}$",
    'annual_longest_wd_periods_median' : r"$C_{\mathrm{D(WD_{Max})}}$", 
    'median_annual_max_wd' : r"$C_{\mathrm{WD}_{Max}}$",
    'var_annual_max_wd' : r"$C_{\mathrm{V,WD}_{Max}}$", 
    'Median_annual_days_wd' : r"$C_{\mathrm{N(WD_{Max})}}$", 
    'var_annual_days_wd' : r"$C_{\mathrm{V,N(WD_{Max})}}$",

    'median_declining_slope_max_wd' : r"$C_{\mathrm{S_D}(WD_{Max})}$", 
    'median_uprising_slope_max_wd' : r"$C_{\mathrm{S_R}(WD_{Max})}$",

    'circ_var_first_negative_WD' : r"$C_{\mathrm{V,t}(WD_{First})}$", 
    'circ_var_last_negative_WD': r"$C_{\mathrm{V,t}(WD_{Last})}$", 

    'wd_max_sin' : r"$C_{\mathrm{t}(WD_{Max})_{\mathrm{WS}}}$",
    'wd_max_cos' : r"$C_{\mathrm{t}(WD_{Max})_{\mathrm{SA}}}$", 
    'circ_var_timing_max_wd' : r"$C_{\mathrm{V,t}(WD_{Max})}}$",

    'PM_mean' : r"$C_{P_M}$",

    'First_wd_sin' : r"$C_{\mathrm{t}(WD_{First})_{\mathrm{WS}}}$", 
    'First_wd_cos' : r"$C_{\mathrm{t}(WD_{First})_{\mathrm{SA}}}$", 
    'Last_wd_sin' : r"$C_{\mathrm{t}(WD_{Last})_{\mathrm{WS}}}$", 
    'Last_wd_cos' : r"$C_{\mathrm{t}(WD_{Last})_{\mathrm{SA}}}$", 

    'LAI_parde_range' : r"$V_{\mathrm{R(LAI_{Parde})}}$", 
    'NDVI_parde_range' : r"$V_{\mathrm{R(NDVI_{Parde})}}$",

    'LAI_variance' : r"$V_{\mathrm{V,LAI}}$",
    'NDVI_variance' :  r"$V_{\mathrm{V,NDVI}}$",

    'LAI_median' : r"$V_{\mathrm{LAI}}$", 
    'NDVI_median' : r"$V_{\mathrm{NDVI}}$",

    'fraction_treecover': r"$V_{\mathrm{F(TC)}}$", 
    'mean_Forest' : r"$V_{\mathrm{F(Forest)}}$", 
    'mean_Scrub' : r"$V_{\mathrm{F(Shrub)}}$", 
    'mean_Open_natural_areas' : r"$V_{\mathrm{F(Open)}}$", 
    'mean_Wetlands' : r"$V_{\mathrm{F(Wetland)}}$", 

    'Irr_median' : r"$A_{\mathrm{Irr}}$", 
    'reservoir_storage' : r"$A_{\mathrm{ResStorage}}$", 
    'lakes_area_relative_to_catchA' : r"$A_{\mathrm{Lake_A}}$", 
    'mean_Artifical_Surfaces' : r"$A_{\mathrm{F(AS)}}$", 
    'mean_Agricultural_areas' : r"$A_{\mathrm{F(Agri)}}$",

    'frac_sand_gravel' : r"$S_{\mathrm{F(SG)}}$",  
    'frac_silt_clay' : r"$S_{\mathrm{F(SC)}}$",  
    'frac_organic_content_median' : r"$S_{\mathrm{F(OC)}}$",  
    'soil_twac_median' : r"$S_{\mathrm{TWAC}}$",  
    'soil_bulk_density_median' : r"$S_{\mathrm{SBD}}$", 
    'soil_root_depth_median' : r"$S_{\mathrm{RD}}$", 

    'Sandstones' : r"$G_{\mathrm{F(SS)}}$",  
    'Mudstones' : r"$G_{\mathrm{F(MS)}}$",  
    'Carbonates' : r"$G_{\mathrm{F(C)}}$",  
    'Other/lowperm' : r"$G_{\mathrm{F(LP)}}$",  
    'cont_high_perm' : r"$G_{\mathrm{F(Perm_H)}}$",  
    'cont_medium_perm' : r"$G_{\mathrm{F(Perm_M)}}$",  
    'cont_low_perm' : r"$G_{\mathrm{F(Perm_L)}}$",  
    'cont_verylow_perm' : r"$G_{\mathrm{F(Perm_{VL})}}$",  

    'bedrock_depth' : r"$G_{\mathrm{BD}}$",

    'ele_mt_mean' : r"$T_{\mathrm{Elev}}$", 
    'slp_dg_mean' : r"$T_{\mathrm{Slp}}$",

    'flat_area_fra' : r"$T_{\mathrm{F(A_F)}}$", 
    'steep_area_fra' : r"$T_{\mathrm{F(A_S)}}$", 
    'elon_ratio' : r"$T_{\mathrm{ER}}$", 
    'strm_dens' : r"$T_{\mathrm{DD}}$"
}

#%% Overview of all simulations 6,7,8,9 and 10 clusters,monte carlo hyper parameter tuning.
#Load data
all_results_df_6 = pd.read_csv(r'U:\Hydrological Data\EStreams\Data 071025\temp_rf_run_6C_5000.csv')
all_results_df_7 = pd.read_csv(r'U:\Hydrological Data\EStreams\Data 071025\temp_rf_run_7C_5000.csv')
all_results_df_8 = pd.read_csv(r'U:\Hydrological Data\EStreams\Data 071025\temp_rf_run_8C_5000.csv')
all_results_df_9 = pd.read_csv(r'U:\Hydrological Data\EStreams\Data 071025\temp_rf_run_9C_5000.csv')
all_results_df_10 = pd.read_csv(r'U:\Hydrological Data\EStreams\Data 071025\temp_rf_run_10C_5000.csv')

#%% Plot results
# Define colors and cluster sets
colors = ['#159895', '#33A9A6', '#52BBB7', '#70CCC8', '#8EDDD9'] # change to CL experiment colours
dfs = [all_results_df_10, all_results_df_9, all_results_df_8, all_results_df_7, all_results_df_6]
labels = ['10 HRT', '9 HRT', '8 HRT', '7 HRT', '6 HRT']
subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)']

# Create 5 subplots (1 row, 5 columns)
fig, axes = plt.subplots(1, 5, figsize=(18, 6), sharex=True, sharey=True)

for i, ax in enumerate(axes):
    ax.scatter(dfs[i]['train_acc'], dfs[i]['test_acc'], color=colors[i], zorder=3)
    ax.plot([0.2, 1], [0.2, 1], 'k--', zorder=4)
    # Include subplot letter in the title
    ax.set_title(f"{subplot_labels[i]} {labels[i]}", fontsize=18, loc='left')
    ax.set_xlim(0.4, 1)
    ax.set_ylim(0.4, 1)
    ax.tick_params(axis='both', which='major', labelsize=18, pad=7)
    ax.grid(True, linestyle='--', alpha=0.6, zorder=2)

    if i == 0:
        ax.set_ylabel("Validation accuracy (-)", fontsize=18)
        ax.set_xlabel("Training accuracy (-)", fontsize=18)
        ax.scatter(0.69, 0.60, color='red', marker='X', s=90, edgecolor='black', zorder=5) #The used RF model validation and testing accuracy 

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(r'U:\Hydrological Data\EStreams\Figures\1st paper\Version 2\5000Runs_perfomance_Indvidual_678910C_V2.pdf', bbox_inches='tight', dpi=300)
#%% 1-CL
# Climate and Landscape
X = climate_landscape
y = clusters_df['C_k10_AllS_NOClimate_remapped'] 


# Initialize the KFold class  # use stratified fold instead!!!!!
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

fold_importances_0 = []
# Lists to store results
train_scores_0 = []
test_scores_0 = []
all_y_test_0 = []
all_y_pred_0 = []
all_y_score_0 = []
y_pred_full_0 = pd.Series(index=X.index, dtype=int)

# Initialize lists
precisions_0 = []
recalls_0 = []
f1s_0 = []
cumulative_cm_0 = np.zeros((10, 10), dtype=int) # For the cumulative confusion matrix
fold_cms_0 = []
#for the ROC curves
n_classes = 10  # clusters

# Store FPR, TPR, and AUCs for each class across folds
fpr_dict_0 = {i: [] for i in range(n_classes)}
tpr_dict_0 = {i: [] for i in range(n_classes)}
auc_dict_0 = {i: [] for i in range(n_classes)}
tpr_interp_dict_0 = {i: [] for i in range(n_classes)}
all_folds_roc_0 = []  
# Store the mean FPR for interpolation  -- why is this needed?
mean_fpr = np.linspace(0, 1, 100)

for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Initialize and train the model 
        model = RandomForestClassifier(n_estimators=494, max_depth=10, min_samples_split=3, min_samples_leaf=13, 
                                   min_weight_fraction_leaf=0.0, class_weight='balanced', n_jobs=-1, random_state=42)
        model.fit(X_train, y_train)
    
    # Predict and evaluate on training set
        y_train_pred = model.predict(X_train)
    
    # Predict and evaluate on test set
        y_test_pred = model.predict(X_test)
    
      # Get predicted probabilities for ROC
        y_score = model.predict_proba(X_test)

        all_y_test_0.append(y_test)
        all_y_pred_0.append(y_test_pred)
        all_y_score_0.append(y_score)
        y_pred_full_0.loc[X_test.index] = y_test_pred

        train_scores_0.append(model.score(X_train, y_train))
        test_scores_0.append(model.score(X_test, y_test))
        
        cm = confusion_matrix(y_test, y_test_pred, labels=np.arange(10))
        fold_cms_0.append(cm)
        cumulative_cm_0 += cm

        #For ROC curve
        # Binarize the output for multiclass ROC
        y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))
        fold_roc_data = []
        # For each class
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            fold_roc_data.append((fpr, tpr, roc_auc))

            # Store raw values for fold-wise plotting
            fpr_dict_0[i].append(fpr)
            tpr_dict_0[i].append(tpr)
            auc_dict_0[i].append(roc_auc)

            # For averaging
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tpr_interp_dict_0[i].append(interp_tpr)
            
        # Save per fold
        all_folds_roc_0.append(fold_roc_data)
        # Compute permutation importances (incMSE)
        result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
        # Collect fold importances
        fold_importances_0.append(result.importances_mean) 

        # Append scores for this fold
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_test_pred, average='macro') # macro = unweighted mean across clases, weighted = mean weigteth by support
        precisions_0.append(precision)
        recalls_0.append(recall)
        f1s_0.append(f1)

        print(f"Training accuracy: {train_scores_0[-1]:.4f}")
        print(f"Testing accuracy: {test_scores_0[-1]:.4f}")

# Mean metrics across folds  - more correct to do mean or median?
print(f"Average Precision: {np.mean(precisions_0):.3f} ± {np.std(precisions_0):.3f}")
print(f"Average Recall:    {np.mean(recalls_0):.3f} ± {np.std(recalls_0):.3f}")
print(f"Average F1 Score:  {np.mean(f1s_0):.3f} ± {np.std(f1s_0):.3f}")
print(f"Average test accurcay:  {np.mean(test_scores_0):.3f} ± {np.std(test_scores_0):.3f}")
print(f"Average training accurcay:  {np.mean(train_scores_0):.3f} ± {np.std(train_scores_0):.3f}")

# Possible to get the average confusion matrix
# Normalize by rows (true classes)
cm_avg_0 = cumulative_cm_0.astype('float') / cumulative_cm_0.sum(axis=1, keepdims=True) * 100  # % format

#next add in feature importance
# Create a DataFrame for importances and correlation (all folds saved)
importances_fold_0_df = pd.DataFrame({
    'Feature': X.columns,
    "fold_1": fold_importances_0[0],
    "fold_2": fold_importances_0[1],
    "fold_3": fold_importances_0[2],
    "fold_4": fold_importances_0[3],
    "fold_5": fold_importances_0[4],
    "fold_6": fold_importances_0[5],
    "fold_7": fold_importances_0[6],
    "fold_8": fold_importances_0[7],
    "fold_9": fold_importances_0[8],
    "fold_10": fold_importances_0[9]    
})

# Average importances over folds
avg_importances_0 = np.median(fold_importances_0, axis=0)
# --- Feature Importance ---
importances_array_0 = np.array(fold_importances_0)  # shape: (n_folds, n_features)

importance_df_0 = pd.DataFrame({
    'Feature': X.columns,
    'Mean Importance': importances_array_0.mean(axis=0),
    'Median Importance': np.median(importances_array_0, axis=0),
    'Std Importance': importances_array_0.std(axis=0)
})

# Save to output
#Save true labels (y_test)
y_test_all = pd.concat(all_y_test_0)
y_test_all = y_test_all.rename_axis('basin_id').reset_index()
#y_test_all.to_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\y_test_0.csv', index=False)

# Save predicted labels
y_pred_all = pd.Series(np.concatenate(all_y_pred_0))
#y_pred_all.to_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\y_pred_0.csv', index=False)

#Save predicted probabilities
# Flatten list of arrays
y_score_all = np.vstack(all_y_score_0)
y_score_df = pd.DataFrame(y_score_all)#, columns=[f'class_{i}' for i in range(n_classes)])
#y_score_df.to_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\y_score_0.csv', index=False)

# Save average confusion matrix
cm_avg_df = pd.DataFrame(cm_avg_0)#, columns=[f'pred_{i}' for i in range(n_classes)],
                         #index=[f'true_{i}' for i in range(n_classes)])
#cm_avg_df.to_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\cm_0.csv')

#Save feature importance
#importance_df_0.to_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\FeatureImp_0.csv')

#Saving the data per fold:

n_folds = len(all_y_test_0)

for fold_idx in range(n_folds):
    # ----- True labels -----
    y_test_fold = all_y_test_0[fold_idx]
    y_test_fold = y_test_fold.rename_axis('basin_id').reset_index()  # optional if you have basin_id
   # y_test_fold.to_csv(f'U:/Hydrological Data/EStreams/ML_results_041125/y_test_0_fold{fold_idx+1}.csv', index=False)

    # ----- Predicted labels -----
    y_pred_fold = pd.Series(all_y_pred_0[fold_idx])
    #y_pred_fold.to_csv(f'U:/Hydrological Data/EStreams/ML_results_041125/y_pred_0_fold{fold_idx+1}.csv', index=False)

    # ----- Predicted probabilities -----
    y_score_fold = pd.DataFrame(all_y_score_0[fold_idx])
    #y_score_fold.to_csv(f'U:/Hydrological Data/EStreams/ML_results_041125/y_score_0_fold{fold_idx+1}.csv', index=False)


all_interp_tprs = []

for class_idx, fold_list in tpr_interp_dict_0.items():
    for fold_idx, interp_tpr in enumerate(fold_list):
        col_name = f'class{class_idx}_fold{fold_idx+1}'
        all_interp_tprs.append(pd.Series(interp_tpr, name=col_name))

# Concatenate all series as columns
df_all = pd.concat(all_interp_tprs, axis=1)
#df_all.to_csv('U:/Hydrological Data/EStreams/ML_results_041125/tpr_interp_0_all.csv', index=False)


#and how to use later is: df = pd.read_csv('tpr_interp_0_all.csv')
# Mean TPR for class 0:   , mean_tpr_class0 = df[[c for c in df.columns if 'class0' in c]].mean(axis=1), std_tpr_class0  = df[[c for c in df.columns if 'class0' in c]].std(axis=1)
#%%
# Version 2: Only climate (meterological)
X = climate_clean_df
y = clusters_df['C_k10_AllS_NOClimate_remapped'] 

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Print samples/features
#print(f"Number of training samples: {X_train.shape[0]}")
##print(f"Number of features: {X_train.shape[1]}")
#print(f"Ratio of samples to features: {X_train.shape[0]/X_train.shape[1]:.2f}")

# Initialize the KFold class  # use stratified fold instead!!!!!
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

fold_importances_2 = []
# Lists to store results
train_scores_2 = []
test_scores_2 = []
all_y_test_2 = []
all_y_pred_2 = []
all_y_score_2 = []
y_pred_full_2 = pd.Series(index=X.index, dtype=int)

# Initialize lists
precisions_2 = []
recalls_2 = []
f1s_2 = []
cumulative_cm_2 = np.zeros((10, 10), dtype=int) # For the cumulative confusion matrix
fold_cms_2 = []
#for the ROC curves
n_classes = 10  # clusters

# Store FPR, TPR, and AUCs for each class across folds
fpr_dict_2 = {i: [] for i in range(n_classes)}
tpr_dict_2 = {i: [] for i in range(n_classes)}
auc_dict_2 = {i: [] for i in range(n_classes)}
tpr_interp_dict_2 = {i: [] for i in range(n_classes)}
all_folds_roc_2 = []  
# Store the mean FPR for interpolation  -- why is this needed?
mean_fpr = np.linspace(0, 1, 100)

for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Initialize and train the model 
        model = RandomForestClassifier(n_estimators=494, max_depth=10, min_samples_split=3, min_samples_leaf=13, 
                                   min_weight_fraction_leaf=0.0, class_weight='balanced', n_jobs=-1, random_state=42)
        model.fit(X_train, y_train)
    
    # Predict and evaluate on training set
        y_train_pred = model.predict(X_train)
    
    # Predict and evaluate on test set
        y_test_pred = model.predict(X_test)
    
      # Get predicted probabilities for ROC
        y_score = model.predict_proba(X_test)

        all_y_test_2.append(y_test)
        all_y_pred_2.append(y_test_pred)
        all_y_score_2.append(y_score)
        y_pred_full_2.loc[X_test.index] = y_test_pred

        train_scores_2.append(model.score(X_train, y_train))
        test_scores_2.append(model.score(X_test, y_test))
        
        cm = confusion_matrix(y_test, y_test_pred, labels=np.arange(10))
        fold_cms_2.append(cm)
        cumulative_cm_2 += cm

        #For ROC curve
        # Binarize the output for multiclass ROC
        y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))
        fold_roc_data = []
        # For each class
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            fold_roc_data.append((fpr, tpr, roc_auc))

            # Store raw values for fold-wise plotting
            fpr_dict_2[i].append(fpr)
            tpr_dict_2[i].append(tpr)
            auc_dict_2[i].append(roc_auc)

            # For averaging
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tpr_interp_dict_2[i].append(interp_tpr)
            
        # Save per fold
        all_folds_roc_2.append(fold_roc_data)
        # Compute permutation importances (incMSE)
        result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
        #incMSE_importances = result.importances_mean
        # Collect fold importances
        fold_importances_2.append(result.importances_mean)

        # Append scores for this fold
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_test_pred, average='macro') # macro = unweighted mean across clases, weighted = mean weigteth by support
        precisions_2.append(precision)
        recalls_2.append(recall)
        f1s_2.append(f1)

        #print("\
        #Very constrained Random Forest:")
        print(f"Training accuracy: {train_scores_2[-1]:.4f}")
        print(f"Testing accuracy: {test_scores_2[-1]:.4f}")

# Mean metrics across folds  - more correct to do mean or median?
print(f"Average Precision: {np.mean(precisions_2):.3f} ± {np.std(precisions_2):.3f}")
print(f"Average Recall:    {np.mean(recalls_2):.3f} ± {np.std(recalls_2):.3f}")
print(f"Average F1 Score:  {np.mean(f1s_2):.3f} ± {np.std(f1s_2):.3f}")
print(f"Average test accurcay:  {np.mean(test_scores_2):.3f} ± {np.std(test_scores_2):.3f}")
print(f"Average training accurcay:  {np.mean(train_scores_2):.3f} ± {np.std(train_scores_2):.3f}")

# Possible to get the average confusion matrix
# Normalize by rows (true classes)
cm_avg_2 = cumulative_cm_2.astype('float') / cumulative_cm_2.sum(axis=1, keepdims=True) * 100  # % format

#next add in feature importance
# Create a DataFrame for importances and correlation (all folds saved)
importances_fold_2_df = pd.DataFrame({
    'Feature': X.columns,
    "fold_1": fold_importances_2[0],
    "fold_2": fold_importances_2[1],
    "fold_3": fold_importances_2[2],
    "fold_4": fold_importances_2[3],
    "fold_5": fold_importances_2[4],
    "fold_6": fold_importances_2[5],
    "fold_7": fold_importances_2[6],
    "fold_8": fold_importances_2[7],
    "fold_9": fold_importances_2[8],
    "fold_10": fold_importances_2[9]    
})

# Average importances over folds
avg_importances_2 = np.median(fold_importances_2, axis=0)
# --- Feature Importance ---
importances_array_2 = np.array(fold_importances_2)  # shape: (n_folds, n_features)

importance_df_2 = pd.DataFrame({
    'Feature': X.columns,
    'Mean Importance': importances_array_2.mean(axis=0),
    'Median Importance': np.median(importances_array_2, axis=0),
    'Std Importance': importances_array_2.std(axis=0)
})

# Save to output
#Save true labels (y_test)
y_test_all = pd.concat(all_y_test_2)
y_test_all = y_test_all.rename_axis('basin_id').reset_index()
y_test_all.to_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\y_test_2.csv', index=False)

# Save predicted labels
y_pred_all = pd.Series(np.concatenate(all_y_pred_2))
y_pred_all.to_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\y_pred_2.csv', index=False)

#Save predicted probabilities
# Flatten list of arrays
y_score_all = np.vstack(all_y_score_2)
y_score_df = pd.DataFrame(y_score_all)#, columns=[f'class_{i}' for i in range(n_classes)])
y_score_df.to_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\y_score_2.csv', index=False)

# Save average confusion matrix
cm_avg_df = pd.DataFrame(cm_avg_2)#, columns=[f'pred_{i}' for i in range(n_classes)],
                         #index=[f'true_{i}' for i in range(n_classes)])
cm_avg_df.to_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\cm_2.csv')

#Save feature importance
importance_df_2.to_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\FeatureImp_2.csv')


#Saving the data per fold:

n_folds = len(all_y_test_2)

for fold_idx in range(n_folds):
    # ----- True labels -----
    y_test_fold = all_y_test_2[fold_idx]
    y_test_fold = y_test_fold.rename_axis('basin_id').reset_index()  # optional if you have basin_id
    y_test_fold.to_csv(f'U:/Hydrological Data/EStreams/ML_results_041125/y_test_2_fold{fold_idx+1}.csv', index=False)

    # ----- Predicted labels -----
    y_pred_fold = pd.Series(all_y_pred_2[fold_idx])
    y_pred_fold.to_csv(f'U:/Hydrological Data/EStreams/ML_results_041125/y_pred_2_fold{fold_idx+1}.csv', index=False)

    # ----- Predicted probabilities -----
    y_score_fold = pd.DataFrame(all_y_score_2[fold_idx])
    y_score_fold.to_csv(f'U:/Hydrological Data/EStreams/ML_results_041125/y_score_2_fold{fold_idx+1}.csv', index=False)


all_interp_tprs = []

for class_idx, fold_list in tpr_interp_dict_2.items():
    for fold_idx, interp_tpr in enumerate(fold_list):
        col_name = f'class{class_idx}_fold{fold_idx+1}'
        all_interp_tprs.append(pd.Series(interp_tpr, name=col_name))

# Concatenate all series as columns
df_all = pd.concat(all_interp_tprs, axis=1)
df_all.to_csv('U:/Hydrological Data/EStreams/ML_results_041125/tpr_interp_2_all.csv', index=False)
#%% 3-VLC
# Version 3: vegetation and landcover dynamic + static
X = vegetation_landcover_df
y = clusters_df['C_k10_AllS_NOClimate_remapped'] 

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Print samples/features
#print(f"Number of training samples: {X_train.shape[0]}")
#print(f"Number of features: {X_train.shape[1]}")
#print(f"Ratio of samples to features: {X_train.shape[0]/X_train.shape[1]:.2f}")

# Initialize the KFold class  # use stratified fold instead!!!!!
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

fold_importances_3 = []
train_scores_3 = []
test_scores_3 = []
all_y_test_3 = []
all_y_pred_3 = []
all_y_score_3 = []
y_pred_full_3 = pd.Series(index=X.index, dtype=int)

# Initialize lists
precisions_3 = []
recalls_3 = []
f1s_3 = []
cumulative_cm_3 = np.zeros((10, 10), dtype=int)
fold_cms_3 = []
#for the ROC curves
n_classes = 10  # clusters

# Store FPR, TPR, and AUCs for each class across folds
fpr_dict_3 = {i: [] for i in range(n_classes)}
tpr_dict_3 = {i: [] for i in range(n_classes)}
auc_dict_3 = {i: [] for i in range(n_classes)}
tpr_interp_dict_3 = {i: [] for i in range(n_classes)}
all_folds_roc_3 = []  
mean_fpr = np.linspace(0, 1, 100)

for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Initialize and train the model 
        model = RandomForestClassifier(n_estimators=494, max_depth=10, min_samples_split=3, min_samples_leaf=13, 
                                   min_weight_fraction_leaf=0.0, class_weight='balanced', n_jobs=-1, random_state=42)
        model.fit(X_train, y_train)
    
    # Predict and evaluate on training set
        y_train_pred = model.predict(X_train)
    
    # Predict and evaluate on test set
        y_test_pred = model.predict(X_test)
    
      # Get predicted probabilities for ROC
        y_score = model.predict_proba(X_test)

        all_y_test_3.append(y_test)
        all_y_pred_3.append(y_test_pred)
        all_y_score_3.append(y_score)
        y_pred_full_3.loc[X_test.index] = y_test_pred
        
        train_scores_3.append(model.score(X_train, y_train))
        test_scores_3.append(model.score(X_test, y_test))
        
        cm = confusion_matrix(y_test, y_test_pred, labels=np.arange(10))
        fold_cms_3.append(cm)
        cumulative_cm_3 += cm

        #For ROC curve
        # Binarize the output for multiclass ROC
        y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))
        fold_roc_data = []
        # For each class
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            fold_roc_data.append((fpr, tpr, roc_auc))

            # Store raw values for fold-wise plotting
            fpr_dict_3[i].append(fpr)
            tpr_dict_3[i].append(tpr)
            auc_dict_3[i].append(roc_auc)
            # For averaging
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tpr_interp_dict_3[i].append(interp_tpr)
            
        # Save per fold
        all_folds_roc_3.append(fold_roc_data)
        # Compute permutation importances (incMSE)
        result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
        incMSE_importances = result.importances_mean

        # Collect fold importances
        fold_importances_3.append(result.importances_mean)

        # Append scores for this fold
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_test_pred, average='macro') # macro = unweighted mean across clases, weighted = mean weigteth by support
 
        precisions_3.append(precision)
        recalls_3.append(recall)
        f1s_3.append(f1)

        print(f"Training accuracy: {train_scores_3[-1]:.4f}")
        print(f"Testing accuracy: {test_scores_3[-1]:.4f}")

print(f"Average Precision: {np.mean(precisions_3):.3f} ± {np.std(precisions_3):.3f}")
print(f"Average Recall:    {np.mean(recalls_3):.3f} ± {np.std(recalls_3):.3f}")
print(f"Average F1 Score:  {np.mean(f1s_3):.3f} ± {np.std(f1s_3):.3f}")
print(f"Average test accuracy:  {np.mean(test_scores_3):.3f} ± {np.std(test_scores_3):.3f}")
print(f"Average training accuracy:  {np.mean(train_scores_3):.3f} ± {np.std(train_scores_3):.3f}")

# Possible to get the average confusion matrix
# Normalize by rows (true classes)
cm_avg_3 = cumulative_cm_3.astype('float') / cumulative_cm_3.sum(axis=1, keepdims=True) * 100


#next add in feature importance
# Create a DataFrame for importances and correlation (all folds saved)
importances_fold_3_df = pd.DataFrame({
    'Feature': X.columns,
    "fold_1": fold_importances_3[0],
    "fold_2": fold_importances_3[1],
    "fold_3": fold_importances_3[2],
    "fold_4": fold_importances_3[3],
    "fold_5": fold_importances_3[4],
    "fold_6": fold_importances_3[5],
    "fold_7": fold_importances_3[6],
    "fold_8": fold_importances_3[7],
    "fold_9": fold_importances_3[8],
    "fold_10": fold_importances_3[9]    
})

# Average importances over folds
avg_importances_3 = np.median(fold_importances_3, axis=0)
# --- Feature Importance ---
# Feature importance
importances_array_3 = np.array(fold_importances_3)  # shape: (n_folds, n_features)

importance_df_3 = pd.DataFrame({
    'Feature': X.columns,
    'Mean Importance': importances_array_3.mean(axis=0),
    'Median Importance': np.median(importances_array_3, axis=0),
    'Std Importance': importances_array_3.std(axis=0)
})


# Save to output
#Save true labels (y_test)
y_test_all = pd.concat(all_y_test_3)
y_test_all = y_test_all.rename_axis('basin_id').reset_index()
y_test_all.to_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\y_test_3.csv', index=False)

# Save predicted labels
y_pred_all = pd.Series(np.concatenate(all_y_pred_3))
y_pred_all.to_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\y_pred_3.csv', index=False)

#Save predicted probabilities
# Flatten list of arrays
y_score_all = np.vstack(all_y_score_3)
y_score_df = pd.DataFrame(y_score_all)#, columns=[f'class_{i}' for i in range(n_classes)])
y_score_df.to_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\y_score_3.csv', index=False)

# Save average confusion matrix
cm_avg_df = pd.DataFrame(cm_avg_3)#, columns=[f'pred_{i}' for i in range(n_classes)],index=[f'true_{i}' for i in range(n_classes)])
cm_avg_df.to_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\cm_3.csv')

#Save feature importance
importance_df_3.to_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\FeatureImp_3.csv')

#Saving the data per fold:

n_folds = len(all_y_test_3)

for fold_idx in range(n_folds):
    # ----- True labels -----
    y_test_fold = all_y_test_3[fold_idx]
    y_test_fold = y_test_fold.rename_axis('basin_id').reset_index()  # optional if you have basin_id
    y_test_fold.to_csv(f'U:/Hydrological Data/EStreams/ML_results_041125/y_test_3_fold{fold_idx+1}.csv', index=False)

    # ----- Predicted labels -----
    y_pred_fold = pd.Series(all_y_pred_3[fold_idx])
    y_pred_fold.to_csv(f'U:/Hydrological Data/EStreams/ML_results_041125/y_pred_3_fold{fold_idx+1}.csv', index=False)

    # ----- Predicted probabilities -----
    y_score_fold = pd.DataFrame(all_y_score_3[fold_idx])
    y_score_fold.to_csv(f'U:/Hydrological Data/EStreams/ML_results_041125/y_score_3_fold{fold_idx+1}.csv', index=False)


all_interp_tprs = []

for class_idx, fold_list in tpr_interp_dict_3.items():
    for fold_idx, interp_tpr in enumerate(fold_list):
        col_name = f'class{class_idx}_fold{fold_idx+1}'
        all_interp_tprs.append(pd.Series(interp_tpr, name=col_name))

# Concatenate all series as columns
df_all = pd.concat(all_interp_tprs, axis=1)
df_all.to_csv('U:/Hydrological Data/EStreams/ML_results_041125/tpr_interp_3_all.csv', index=False)

#%% 4-VLS
# Vegetation static
X = vegetation_landcover_static_df
y = clusters_df['C_k10_AllS_NOClimate_remapped'] 


# Initialize the KFold class  # use stratified fold instead!!!!!
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

fold_importances_4 = []
# Lists to store results
train_scores_4  = []
test_scores_4  = []
all_y_test_4  = []
all_y_pred_4  = []
all_y_score_4  = []
y_pred_full_4  = pd.Series(index=X.index, dtype=int)

# Initialize lists
precisions_4  = []
recalls_4  = []
f1s_4  = []
cumulative_cm_4  = np.zeros((10, 10), dtype=int) # For the cumulative confusion matrix
fold_cms_4 = []
#for the ROC curves
n_classes = 10  # clusters

# Store FPR, TPR, and AUCs for each class across folds
fpr_dict_4  = {i: [] for i in range(n_classes)}
tpr_dict_4  = {i: [] for i in range(n_classes)}
auc_dict_4  = {i: [] for i in range(n_classes)}
tpr_interp_dict_4  = {i: [] for i in range(n_classes)}
all_folds_roc_4  = []  
# Store the mean FPR for interpolation  -- why is this needed?
mean_fpr = np.linspace(0, 1, 100)

for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Initialize and train the model 
        model = RandomForestClassifier(n_estimators=494, max_depth=10, min_samples_split=3, min_samples_leaf=13, 
                                   min_weight_fraction_leaf=0.0, class_weight='balanced', n_jobs=-1, random_state=42)
        model.fit(X_train, y_train)
    
    # Predict and evaluate on training set
        y_train_pred = model.predict(X_train)
    
    # Predict and evaluate on test set
        y_test_pred = model.predict(X_test)
    
      # Get predicted probabilities for ROC
        y_score = model.predict_proba(X_test)

        all_y_test_4 .append(y_test)
        all_y_pred_4 .append(y_test_pred)
        all_y_score_4 .append(y_score)
        y_pred_full_4 .loc[X_test.index] = y_test_pred

        train_scores_4.append(model.score(X_train, y_train))
        test_scores_4.append(model.score(X_test, y_test))
        cm = confusion_matrix(y_test, y_test_pred, labels=np.arange(10))
        fold_cms_4.append(cm)
        cumulative_cm_4 += cm

        #For ROC curve
        # Binarize the output for multiclass ROC
        y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))
        fold_roc_data = []
        # For each class
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            fold_roc_data.append((fpr, tpr, roc_auc))

            # Store raw values for fold-wise plotting
            fpr_dict_4[i].append(fpr)
            tpr_dict_4[i].append(tpr)
            auc_dict_4[i].append(roc_auc)

            # For averaging
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tpr_interp_dict_4[i].append(interp_tpr)
            
        # Save per fold
        all_folds_roc_4.append(fold_roc_data)
        # Compute permutation importances (incMSE)
        result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)

        # Collect fold importances
        fold_importances_4.append(result.importances_mean) 

        # Append scores for this fold
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_test_pred, average='macro') # macro = unweighted mean across clases, weighted = mean weigteth by support
        precisions_4.append(precision)
        recalls_4.append(recall)
        f1s_4.append(f1)

        print(f"Training accuracy: {train_scores_4[-1]:.4f}")
        print(f"Testing accuracy: {test_scores_4[-1]:.4f}")

# Mean metrics across folds  - more correct to do mean or median?
print(f"Average Precision: {np.mean(precisions_4):.3f} ± {np.std(precisions_4):.3f}")
print(f"Average Recall:    {np.mean(recalls_4):.3f} ± {np.std(recalls_4):.3f}")
print(f"Average F1 Score:  {np.mean(f1s_4):.3f} ± {np.std(f1s_4):.3f}")
print(f"Average test accurcay:  {np.mean(test_scores_4):.3f} ± {np.std(test_scores_4):.3f}")
print(f"Average training accurcay:  {np.mean(train_scores_4):.3f} ± {np.std(train_scores_4):.3f}")

# Possible to get the average confusion matrix
# Normalize by rows (true classes)
cm_avg_4 = cumulative_cm_4.astype('float') / cumulative_cm_4.sum(axis=1, keepdims=True) * 100  # % format

#next add in feature importance
# Create a DataFrame for importances and correlation (all folds saved)
importances_fold_4_df = pd.DataFrame({
    'Feature': X.columns,
    "fold_1": fold_importances_4[0],
    "fold_2": fold_importances_4[1],
    "fold_3": fold_importances_4[2],
    "fold_4": fold_importances_4[3],
    "fold_5": fold_importances_4[4],
    "fold_6": fold_importances_4[5],
    "fold_7": fold_importances_4[6],
    "fold_8": fold_importances_4[7],
    "fold_9": fold_importances_4[8],
    "fold_10": fold_importances_4[9]    
})

# Average importances over folds
avg_importances_4 = np.median(fold_importances_4, axis=0)
# --- Feature Importance ---
importances_array_4 = np.array(fold_importances_4)  # shape: (n_folds, n_features)

importance_df_4 = pd.DataFrame({
    'Feature': X.columns,
    'Mean Importance': importances_array_4.mean(axis=0),
    'Median Importance': np.median(importances_array_4, axis=0),
    'Std Importance': importances_array_4.std(axis=0)
})



# Save to output
#Save true labels (y_test)
y_test_all = pd.concat(all_y_test_4)
y_test_all = y_test_all.rename_axis('basin_id').reset_index()
y_test_all.to_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\y_test_4.csv', index=False)

# Save predicted labels
y_pred_all = pd.Series(np.concatenate(all_y_pred_4))
y_pred_all.to_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\y_pred_4.csv', index=False)

#Save predicted probabilities
# Flatten list of arrays
y_score_all = np.vstack(all_y_score_4)
y_score_df = pd.DataFrame(y_score_all)#, columns=[f'class_{i}' for i in range(n_classes)])
y_score_df.to_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\y_score_4.csv', index=False)

# Save average confusion matrix
cm_avg_df = pd.DataFrame(cm_avg_4)
cm_avg_df.to_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\cm_4.csv')

#Save feature importance
importance_df_4.to_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\FeatureImp_4.csv')

#Saving the data per fold:

n_folds = len(all_y_test_4)

for fold_idx in range(n_folds):
    # ----- True labels -----
    y_test_fold = all_y_test_4[fold_idx]
    y_test_fold = y_test_fold.rename_axis('basin_id').reset_index()  # optional if you have basin_id
    y_test_fold.to_csv(f'U:/Hydrological Data/EStreams/ML_results_041125/y_test_4_fold{fold_idx+1}.csv', index=False)

    # ----- Predicted labels -----
    y_pred_fold = pd.Series(all_y_pred_4[fold_idx])
    y_pred_fold.to_csv(f'U:/Hydrological Data/EStreams/ML_results_041125/y_pred_4_fold{fold_idx+1}.csv', index=False)

    # ----- Predicted probabilities -----
    y_score_fold = pd.DataFrame(all_y_score_4[fold_idx])
    y_score_fold.to_csv(f'U:/Hydrological Data/EStreams/ML_results_041125/y_score_4_fold{fold_idx+1}.csv', index=False)


all_interp_tprs = []

for class_idx, fold_list in tpr_interp_dict_4.items():
    for fold_idx, interp_tpr in enumerate(fold_list):
        col_name = f'class{class_idx}_fold{fold_idx+1}'
        all_interp_tprs.append(pd.Series(interp_tpr, name=col_name))

# Concatenate all series as columns
df_all = pd.concat(all_interp_tprs, axis=1)
df_all.to_csv('U:/Hydrological Data/EStreams/ML_results_041125/tpr_interp_4_all.csv', index=False)
#%% 5-SGT
# Version 5: Soil, geology, topography
X = soil_geology_topography_df
y = clusters_df['C_k10_AllS_NOClimate_remapped'] 

#Initialize the KFold class  # use stratified fold instead!!!!!
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

fold_importances_5 = []
# Lists to store results
train_scores_5 = []
test_scores_5 = []
all_y_test_5 = []
all_y_pred_5 = []
all_y_score_5 = []
y_pred_full_5 = pd.Series(index=X.index, dtype=int)

# Initialize lists
precisions_5 = []
recalls_5 = []
f1s_5 = []
cumulative_cm_5 = np.zeros((10, 10), dtype=int) # For the cumulative confusion matrix
fold_cms_5 = []
#for the ROC curves
n_classes = 10  # clusters

# Store FPR, TPR, and AUCs for each class across folds
fpr_dict_5 = {i: [] for i in range(n_classes)}
tpr_dict_5 = {i: [] for i in range(n_classes)}
auc_dict_5 = {i: [] for i in range(n_classes)}
tpr_interp_dict_5 = {i: [] for i in range(n_classes)}
all_folds_roc_5 = []  
# Store the mean FPR for interpolation  -- why is this needed?
mean_fpr = np.linspace(0, 1, 100)

for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Initialize and train the model 
        model = RandomForestClassifier(n_estimators=494, max_depth=10, min_samples_split=3, min_samples_leaf=13, 
                                   min_weight_fraction_leaf=0.0, class_weight='balanced', n_jobs=-1, random_state=42)
        model.fit(X_train, y_train)
    
    # Predict and evaluate on training set
        y_train_pred = model.predict(X_train)
    
    # Predict and evaluate on test set
        y_test_pred = model.predict(X_test)
    
      # Get predicted probabilities for ROC
        y_score = model.predict_proba(X_test)

        all_y_test_5.append(y_test)
        all_y_pred_5.append(y_test_pred)
        all_y_score_5.append(y_score)
        y_pred_full_5.loc[X_test.index] = y_test_pred

        train_scores_5.append(model.score(X_train, y_train))
        test_scores_5.append(model.score(X_test, y_test))
        cm = confusion_matrix(y_test, y_test_pred, labels=np.arange(10))
        fold_cms_5.append(cm)
        cumulative_cm_5 += cm

        #For ROC curve
        # Binarize the output for multiclass ROC
        y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))
        fold_roc_data = []
        # For each class
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            fold_roc_data.append((fpr, tpr, roc_auc))

            # Store raw values for fold-wise plotting
            fpr_dict_5[i].append(fpr)
            tpr_dict_5[i].append(tpr)
            auc_dict_5[i].append(roc_auc)

            # For averaging
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tpr_interp_dict_5[i].append(interp_tpr)
            
        # Save per fold
        all_folds_roc_5.append(fold_roc_data)
        # Compute permutation importances (incMSE)
        result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
        # Collect fold importances
        fold_importances_5.append(result.importances_mean) 

        # Append scores for this fold
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_test_pred, average='macro') # macro = unweighted mean across clases, weighted = mean weigteth by support
        precisions_5.append(precision)
        recalls_5.append(recall)
        f1s_5.append(f1)

        print(f"Training accuracy: {train_scores_5[-1]:.4f}")
        print(f"Testing accuracy: {test_scores_5[-1]:.4f}")

# Mean metrics across folds  - more correct to do mean or median?
print(f"Average Precision: {np.mean(precisions_5):.3f} ± {np.std(precisions_5):.3f}")
print(f"Average Recall:    {np.mean(recalls_5):.3f} ± {np.std(recalls_5):.3f}")
print(f"Average F1 Score:  {np.mean(f1s_5):.3f} ± {np.std(f1s_5):.3f}")
print(f"Average test accurcay:  {np.mean(test_scores_5):.3f} ± {np.std(test_scores_5):.3f}")
print(f"Average training accurcay:  {np.mean(train_scores_5):.3f} ± {np.std(train_scores_5):.3f}")

# Possible to get the average confusion matrix
# Normalize by rows (true classes)
cm_avg_5 = cumulative_cm_5.astype('float') / cumulative_cm_5.sum(axis=1, keepdims=True) * 100  # % format

#next add in feature importance
# Create a DataFrame for importances and correlation (all folds saved)
importances_fold_5_df = pd.DataFrame({
    'Feature': X.columns,
    "fold_1": fold_importances_5[0],
    "fold_2": fold_importances_5[1],
    "fold_3": fold_importances_5[2],
    "fold_4": fold_importances_5[3],
    "fold_5": fold_importances_5[4],
    "fold_6": fold_importances_5[5],
    "fold_7": fold_importances_5[6],
    "fold_8": fold_importances_5[7],
    "fold_9": fold_importances_5[8],
    "fold_10": fold_importances_5[9]    
})

# Average importances over folds
avg_importances_5 = np.median(fold_importances_5, axis=0)
# --- Feature Importance ---
importances_array_5 = np.array(fold_importances_5)  # shape: (n_folds, n_features)

importance_df_5 = pd.DataFrame({
    'Feature': X.columns,
    'Mean Importance': importances_array_5.mean(axis=0),
    'Median Importance': np.median(importances_array_5, axis=0),
    'Std Importance': importances_array_5.std(axis=0)
})




# Save to output
#Save true labels (y_test)
y_test_all = pd.concat(all_y_test_5)
y_test_all = y_test_all.rename_axis('basin_id').reset_index()
y_test_all.to_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\y_test_5.csv', index=False)

# Save predicted labels
y_pred_all = pd.Series(np.concatenate(all_y_pred_5))
y_pred_all.to_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\y_pred_5.csv', index=False)

#Save predicted probabilities
# Flatten list of arrays
y_score_all = np.vstack(all_y_score_5)
y_score_df = pd.DataFrame(y_score_all)
y_score_df.to_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\y_score_5.csv', index=False)

# Save average confusion matrix
cm_avg_df = pd.DataFrame(cm_avg_5)
cm_avg_df.to_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\cm_5.csv')

#Save feature importance
importance_df_5.to_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\FeatureImp_5.csv')


#Saving the data per fold:

n_folds = len(all_y_test_5)

for fold_idx in range(n_folds):
    # ----- True labels -----
    y_test_fold = all_y_test_5[fold_idx]
    y_test_fold = y_test_fold.rename_axis('basin_id').reset_index()  # optional if you have basin_id
    y_test_fold.to_csv(f'U:/Hydrological Data/EStreams/ML_results_041125/y_test_5_fold{fold_idx+1}.csv', index=False)

    # ----- Predicted labels -----
    y_pred_fold = pd.Series(all_y_pred_5[fold_idx])
    y_pred_fold.to_csv(f'U:/Hydrological Data/EStreams/ML_results_041125/y_pred_5_fold{fold_idx+1}.csv', index=False)

    # ----- Predicted probabilities -----
    y_score_fold = pd.DataFrame(all_y_score_5[fold_idx])
    y_score_fold.to_csv(f'U:/Hydrological Data/EStreams/ML_results_041125/y_score_5_fold{fold_idx+1}.csv', index=False)


all_interp_tprs = []

for class_idx, fold_list in tpr_interp_dict_5.items():
    for fold_idx, interp_tpr in enumerate(fold_list):
        col_name = f'class{class_idx}_fold{fold_idx+1}'
        all_interp_tprs.append(pd.Series(interp_tpr, name=col_name))

# Concatenate all series as columns
df_all = pd.concat(all_interp_tprs, axis=1)
df_all.to_csv('U:/Hydrological Data/EStreams/ML_results_041125/tpr_interp_5_all.csv', index=False)

#%% 6-A
# Anthropogenic attributes
X = lumped_df
y = clusters_df['C_k10_AllS_NOClimate_remapped'] 

# Initialize the KFold class  # use stratified fold instead!!!!!
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

fold_importances_6 = []
# Lists to store results
train_scores_6  = []
test_scores_6  = []
all_y_test_6  = []
all_y_pred_6  = []
all_y_score_6  = []
y_pred_full_6  = pd.Series(index=X.index, dtype=int)

# Initialize lists
precisions_6  = []
recalls_6  = []
f1s_6  = []
cumulative_cm_6  = np.zeros((10, 10), dtype=int) # For the cumulative confusion matrix
fold_cms_6 = []

#for the ROC curves
n_classes = 10  # clusters

# Store FPR, TPR, and AUCs for each class across folds
fpr_dict_6  = {i: [] for i in range(n_classes)}
tpr_dict_6  = {i: [] for i in range(n_classes)}
auc_dict_6  = {i: [] for i in range(n_classes)}
tpr_interp_dict_6  = {i: [] for i in range(n_classes)}
all_folds_roc_6  = []  
# Store the mean FPR for interpolation  -- why is this needed?
mean_fpr = np.linspace(0, 1, 100)

for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Initialize and train the model 
        model = RandomForestClassifier(n_estimators=494, max_depth=10, min_samples_split=3, min_samples_leaf=13, 
                                   min_weight_fraction_leaf=0.0, class_weight='balanced', n_jobs=-1, random_state=42)
        model.fit(X_train, y_train)
    
    # Predict and evaluate on training set
        y_train_pred = model.predict(X_train)
    
    # Predict and evaluate on test set
        y_test_pred = model.predict(X_test)
    
      # Get predicted probabilities for ROC
        y_score = model.predict_proba(X_test)

        all_y_test_6 .append(y_test)
        all_y_pred_6 .append(y_test_pred)
        all_y_score_6 .append(y_score)
        y_pred_full_6 .loc[X_test.index] = y_test_pred

        train_scores_6.append(model.score(X_train, y_train))
        test_scores_6.append(model.score(X_test, y_test))
        
        cm = confusion_matrix(y_test, y_test_pred, labels=np.arange(10))
        fold_cms_6.append(cm)
        cumulative_cm_6 += cm

        #For ROC curve
        # Binarize the output for multiclass ROC
        y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))
        fold_roc_data = []
        # For each class
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            fold_roc_data.append((fpr, tpr, roc_auc))

            # Store raw values for fold-wise plotting
            fpr_dict_6[i].append(fpr)
            tpr_dict_6[i].append(tpr)
            auc_dict_6[i].append(roc_auc)

            # For averaging
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tpr_interp_dict_6[i].append(interp_tpr)
            
        # Save per fold
        all_folds_roc_6.append(fold_roc_data)
        # Compute permutation importances (incMSE)
        result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)

        # Collect fold importances
        fold_importances_6.append(result.importances_mean) 

        # Append scores for this fold
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_test_pred, average='macro') # macro = unweighted mean across clases, weighted = mean weigteth by support
        precisions_6.append(precision)
        recalls_6.append(recall)
        f1s_6.append(f1)


        print(f"Training accuracy: {train_scores_6[-1]:.4f}")
        print(f"Testing accuracy: {test_scores_6[-1]:.4f}")

# Mean metrics across folds  - more correct to do mean or median?
print(f"Average Precision: {np.mean(precisions_6):.3f} ± {np.std(precisions_6):.3f}")
print(f"Average Recall:    {np.mean(recalls_6):.3f} ± {np.std(recalls_6):.3f}")
print(f"Average F1 Score:  {np.mean(f1s_6):.3f} ± {np.std(f1s_6):.3f}")
print(f"Average test accurcay:  {np.mean(test_scores_6):.3f} ± {np.std(test_scores_6):.3f}")
print(f"Average training accurcay:  {np.mean(train_scores_6):.3f} ± {np.std(train_scores_6):.3f}")

# Possible to get the average confusion matrix
# Normalize by rows (true classes)
cm_avg_6 = cumulative_cm_6.astype('float') / cumulative_cm_6.sum(axis=1, keepdims=True) * 100  # % format

#next add in feature importance
# Create a DataFrame for importances and correlation (all folds saved)
importances_fold_6_df = pd.DataFrame({
    'Feature': X.columns,
    "fold_1": fold_importances_6[0],
    "fold_2": fold_importances_6[1],
    "fold_3": fold_importances_6[2],
    "fold_4": fold_importances_6[3],
    "fold_5": fold_importances_6[4],
    "fold_6": fold_importances_6[5],
    "fold_7": fold_importances_6[6],
    "fold_8": fold_importances_6[7],
    "fold_9": fold_importances_6[8],
    "fold_10": fold_importances_6[9]    
})

# Average importances over folds
avg_importances_6 = np.median(fold_importances_6, axis=0)
# --- Feature Importance ---
importances_array_6 = np.array(fold_importances_6)  # shape: (n_folds, n_features)

importance_df_6 = pd.DataFrame({
    'Feature': X.columns,
    'Mean Importance': importances_array_6.mean(axis=0),
    'Median Importance': np.median(importances_array_6, axis=0),
    'Std Importance': importances_array_6.std(axis=0)
})


# Save to output
#Save true labels (y_test)
y_test_all = pd.concat(all_y_test_6)
y_test_all = y_test_all.rename_axis('basin_id').reset_index()
y_test_all.to_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\y_test_6.csv', index=False)

# Save predicted labels
y_pred_all = pd.Series(np.concatenate(all_y_pred_6))
y_pred_all.to_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\y_pred_6.csv', index=False)

#Save predicted probabilities
# Flatten list of arrays
y_score_all = np.vstack(all_y_score_6)
y_score_df = pd.DataFrame(y_score_all)
y_score_df.to_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\y_score_6.csv', index=False)

# Save average confusion matrix
cm_avg_df = pd.DataFrame(cm_avg_6)
cm_avg_df.to_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\cm_6.csv')

#Save feature importance
importance_df_6.to_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\FeatureImp_6.csv')


#Saving the data per fold:

n_folds = len(all_y_test_6)

for fold_idx in range(n_folds):
    # ----- True labels -----
    y_test_fold = all_y_test_6[fold_idx]
    y_test_fold = y_test_fold.rename_axis('basin_id').reset_index()  # optional if you have basin_id
    y_test_fold.to_csv(f'U:/Hydrological Data/EStreams/ML_results_041125/y_test_6_fold{fold_idx+1}.csv', index=False)

    # ----- Predicted labels -----
    y_pred_fold = pd.Series(all_y_pred_6[fold_idx])
    y_pred_fold.to_csv(f'U:/Hydrological Data/EStreams/ML_results_041125/y_pred_6_fold{fold_idx+1}.csv', index=False)

    # ----- Predicted probabilities -----
    y_score_fold = pd.DataFrame(all_y_score_6[fold_idx])
    y_score_fold.to_csv(f'U:/Hydrological Data/EStreams/ML_results_041125/y_score_6_fold{fold_idx+1}.csv', index=False)


all_interp_tprs = []

for class_idx, fold_list in tpr_interp_dict_6.items():
    for fold_idx, interp_tpr in enumerate(fold_list):
        col_name = f'class{class_idx}_fold{fold_idx+1}'
        all_interp_tprs.append(pd.Series(interp_tpr, name=col_name))

# Concatenate all series as columns
df_all = pd.concat(all_interp_tprs, axis=1)
df_all.to_csv('U:/Hydrological Data/EStreams/ML_results_041125/tpr_interp_6_all.csv', index=False)
#%%
#ALSO NEED TO SAVE THIS SO RUN AGAIN:
    #tpr_interp_dict_list = [tpr_interp_dict_0, tpr_interp_dict_2, tpr_interp_dict_3, tpr_interp_dict_4, tpr_interp_dict_5, tpr_interp_dict_6]
    #auc_dict_list = [auc_dict_0, auc_dict_2, auc_dict_3, auc_dict_4, auc_dict_5, auc_dict_6]

#%% LOAD ALL DATA


all_y_test_0 = pd.read_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\y_test_0.csv', index_col='basin_id')
all_y_test_2 = pd.read_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\y_test_2.csv', index_col='basin_id')
all_y_test_3 = pd.read_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\y_test_3.csv', index_col='basin_id')
all_y_test_4 = pd.read_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\y_test_4.csv', index_col='basin_id')
all_y_test_5 = pd.read_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\y_test_5.csv', index_col='basin_id')
all_y_test_6 = pd.read_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\y_test_6.csv', index_col='basin_id')


all_y_pred_0 = pd.read_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\y_pred_0.csv')
all_y_pred_2 = pd.read_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\y_pred_2.csv')
all_y_pred_3 = pd.read_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\y_pred_3.csv')
all_y_pred_4 = pd.read_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\y_pred_4.csv')
all_y_pred_5 = pd.read_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\y_pred_5.csv')
all_y_pred_6 = pd.read_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\y_pred_6.csv')


all_y_score_0 = pd.read_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\y_score_0.csv')
all_y_score_2 = pd.read_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\y_score_2.csv')
all_y_score_3 = pd.read_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\y_score_3.csv')
all_y_score_4 = pd.read_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\y_score_4.csv')
all_y_score_5 = pd.read_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\y_score_5.csv')
all_y_score_6 = pd.read_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\y_score_6.csv')

cm_avg_0 = pd.read_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\cm_0.csv', index_col=0)
cm_avg_2 = pd.read_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\cm_2.csv', index_col=0)
cm_avg_3 = pd.read_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\cm_3.csv', index_col=0)
cm_avg_4 = pd.read_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\cm_4.csv', index_col=0)
cm_avg_5 = pd.read_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\cm_5.csv', index_col=0)
cm_avg_6 = pd.read_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\cm_6.csv', index_col=0)


importance_df_0 = pd.read_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\FeatureImp_0.csv')
importance_df_2 = pd.read_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\FeatureImp_2.csv')
importance_df_3 = pd.read_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\FeatureImp_3.csv')
importance_df_4 = pd.read_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\FeatureImp_4.csv')
importance_df_5 = pd.read_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\FeatureImp_5.csv')
importance_df_6 = pd.read_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\FeatureImp_6.csv')

#%% Load results from all folds
import glob
#experiment_id = [0,2,3,4,5,6]
out_dir = r'U:\Hydrological Data\EStreams\ML_results_041125'

def load_folds(experiment_id, out_dir):
    # Load all folds for a given experiment
    # True labels
    y_test_files = sorted(glob.glob(os.path.join(out_dir, f'y_test_{experiment_id}_fold*.csv')))
    all_y_test = [pd.read_csv(f)['C_k10_AllS_NOClimate_remapped'] for f in y_test_files]  # replace 'your_label_column' with actual column

    # Predicted labels
    y_pred_files = sorted(glob.glob(os.path.join(out_dir, f'y_pred_{experiment_id}_fold*.csv')))
    all_y_pred = [pd.read_csv(f).squeeze() for f in y_pred_files]  # squeeze converts 1-column DF to Series

    # Predicted probabilities
    y_score_files = sorted(glob.glob(os.path.join(out_dir, f'y_score_{experiment_id}_fold*.csv')))
    all_y_score = [pd.read_csv(f).values for f in y_score_files]

    return all_y_test, all_y_pred, all_y_score

#%% Load the folds
all_y_test_0_fold, all_y_pred_0_fold, all_y_score_0_fold = load_folds(0, out_dir)
all_y_test_2_fold, all_y_pred_2_fold, all_y_score_2_fold = load_folds(2, out_dir)
all_y_test_3_fold, all_y_pred_3_fold, all_y_score_3_fold = load_folds(3, out_dir)
all_y_test_4_fold, all_y_pred_4_fold, all_y_score_4_fold = load_folds(4, out_dir)
all_y_test_5_fold, all_y_pred_5_fold, all_y_score_5_fold = load_folds(5, out_dir)
all_y_test_6_fold, all_y_pred_6_fold, all_y_score_6_fold = load_folds(6, out_dir)

#%%
tpr_interp_0_all_df = pd.read_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\tpr_interp_0_all.csv')
tpr_interp_2_all_df = pd.read_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\tpr_interp_2_all.csv')
tpr_interp_3_all_df = pd.read_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\tpr_interp_3_all.csv')
tpr_interp_4_all_df = pd.read_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\tpr_interp_4_all.csv')
tpr_interp_5_all_df = pd.read_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\tpr_interp_5_all.csv')
tpr_interp_6_all_df = pd.read_csv(r'U:\Hydrological Data\EStreams\ML_results_041125\tpr_interp_6_all.csv')

#%% Feature importance plot
import matplotlib.patches as mpatches
bar_height = 0.7  # constant thickness
max_bars = 20     # max slots to keep spacing consistent
# Create subplots: 3 rows × 2 columns
fig, axes = plt.subplots(3, 2, figsize=(14, 20))
axes = axes.flatten()

# Put your six DataFrames in a list
importance_dfs = [
    importance_df_0,
    importance_df_2,
    importance_df_3,
    importance_df_4,
    importance_df_5,
    importance_df_6
]

legend_patches = [
    mpatches.Patch(color='tab:blue', label='Climate'),
    mpatches.Patch(color='#2E8B57', label='Vegetation & Landcover'),
    mpatches.Patch(color='#A6D854', label='Vegetation & Landcover Static'),
    mpatches.Patch(color='saddlebrown', label='Soil, Geology & Topography'),
    mpatches.Patch(color='#9EC1CF', label='Anthropogenic')
    
]

# Prepare legend patches (you can adjust which ones to include)
# Category mapping for colors
def get_colors(features):
    return [
        '#A6D854' if f in vegetation_static 
        else 'saddlebrown' if f in soil_geology 
        else '#9EC1CF' if f in human_influence 
        else 'saddlebrown' if f in topography 
        else '#A6D854' if f in landcover 
        else 'tab:blue' if f in pliquid 
        else 'tab:blue' if f in water_deficit 
        else 'tab:blue' if f in snow 
        else '#2E8B57' if f in vegetation_dynamic 
        else 'tab:blue'
        for f in features
    ]

# titles for each subplot
titles = [
    "1-CL",
    "2-C",
    "3-VLC",
    "4-VLS",
    "5-SGT",
    "6-A"
]
# Letters for subplot labels
subplot_letters = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

for i, (df, ax) in enumerate(zip(importance_dfs, axes)):
    # Sort and select up to max_bars
    df_sorted = df.sort_values(by='Median Importance', ascending=False).head(max_bars)
    df_display = df_sorted.copy()
    df_display['PrettyName'] = df_display['Feature'].map(feature_name_mapping).fillna(df_display['Feature'])

    colors = get_colors(df_display['Feature'])

    # Number of actual bars to plot
    n_bars = len(df_display)

    padding = 0.5  # space at top and bottom in "slots"
    y_positions = [padding + (max_bars - 2*padding) * (j + 0.5)/n_bars for j in range(n_bars)]

    # Plot bars
    ax.barh(y_positions, df_display['Median Importance'], color=colors, height=bar_height, align='center')
    ax.errorbar(
    df_display['Median Importance'], 
    y_positions, 
    xerr=[np.minimum(df_display['Std Importance'], df_display['Median Importance']), 
          df_display['Std Importance']], fmt='.', color='black')
    # Fix y-axis so all subplots have same bar spacing
    ax.set_ylim(0, max_bars)

    # Set y-ticks and labels
    ax.set_yticks(y_positions)
    ax.set_xlabel("Decrease in accuracy (-)", fontsize=16)
    ax.set_yticklabels(df_display['PrettyName'])
    ax.tick_params(axis='both', labelsize=18)

    # Invert so top is first feature
    ax.invert_yaxis()

    # X limits and titles with letters
    #ax.set_xlim(-0.01, 0.185)
    ax.set_xlim(left=0)
    ax.set_title(f"{subplot_letters[i]} {titles[i]}", fontsize=16, loc='left')

# Add one legend for the whole figure
fig.legend(
    handles=legend_patches,
    loc='lower center',
    ncol=2,
    bbox_to_anchor=(0.42, -0.07),
    prop={'size': 17}  # <-- increase the font size here
)

plt.tight_layout()
plt.savefig(r'U:\Hydrological Data\EStreams\Figures\1st paper\Version 2\FeatureImportance_allversions_10C_V2.pdf',bbox_inches='tight',dpi=300)
plt.savefig(r'U:\Hydrological Data\EStreams\Figures\1st paper\Version 2\FeatureImportance_allversions_10C_V2.png',bbox_inches='tight',dpi=300)

#%% Confusion Matrices
# Experiment colors

exp_colors = [
    "#159895",  # teal (lightseagreen adjusted; slightly bluer)
    "#1f77b4",  # tab:blue (unchanged)
    "#2E8B57",  # sea green (for darker vegetation; less red)
    "#A6D854",  # lime-ish green (lighter vegetation; shifted toward yellow)
    "#8B4513",  # saddlebrown (unchanged, soil)
    "#9EC1CF"   # slightly darker lightsteelblue for better contrast
]
# List of confusion matrices
cms = [cm_avg_0, cm_avg_2, cm_avg_3, cm_avg_4, cm_avg_5, cm_avg_6]

# experiment titles
titles = [
    "1-CL",
    "2-C",
    "3-VLC",
    "4-VLS",
    "5-SGT",
    "6-A"
]


letters = list(string.ascii_lowercase)

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 17))
#plt.subplots_adjust(wspace=0.01, hspace=0.3) 
for i, (ax, cm, title) in enumerate(zip(axes.ravel(), cms, titles)):
    # Normalize values to [0,1] for alpha scaling
    #cm_norm = cm.astype(float) / cm.values.max() if cm.values.max() > 0 else cm 
    cm_max = np.max(cm.values) if isinstance(cm, pd.DataFrame) else np.max(cm)
    cm_norm = cm.astype(float) / cm_max if cm_max > 0 else cm
    
    cmap = LinearSegmentedColormap.from_list("my_cmap", ["white", exp_colors[i]])
    ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1)
    
    for (row, col), val in np.ndenumerate(cm):
        color = exp_colors[i]
        alpha = cm_norm.iloc[row, col]
        rect = patches.Rectangle((col - 0.5, row - 0.5), 1, 1,
                                 linewidth=1, edgecolor="none",
                                 facecolor=color, alpha=alpha)
        ax.add_patch(rect)
    
        # automatic text color
        text_color = "white" if cm_norm.iloc[row, col] > 0.5 else "black"
        ax.text(col, row, f"{val:.0f}", ha="center", va="center",
                fontsize=13, color=text_color)
        

        
        # Add red outlines to diagonal cells
        for j in range(cm.shape[0]):
            rect = patches.Rectangle(
                (j - 0.5, j - 0.5), 1, 1,
                linewidth=2, edgecolor='red', facecolor='none'
                )
            ax.add_patch(rect)
    
    # ticks
    tick_labels = [str(k) for k in range(1, cm.shape[0]+1)]
    ax.set_xticks(range(cm.shape[0]))
    ax.set_yticks(range(cm.shape[0]))
    ax.set_xticklabels(tick_labels, size=15)
    ax.set_yticklabels(tick_labels, size=15)
    ax.set_xlabel('Catchment % predicted HRT', fontsize=14)
    ax.set_ylabel('Catchment % actual HRT', fontsize=14)
    
    ax.set_title(f"({letters[i]}) {title}", fontsize=16, loc='left')

plt.subplots_adjust(wspace=-0.2, hspace=0.275)
#plt.tight_layout() 
plt.savefig(r'U:\Hydrological Data\EStreams\Figures\1st paper\Version 2\ConfusionMatrix_allversions_10C_V2.pdf',bbox_inches='tight',dpi=300)
plt.savefig(r'U:\Hydrological Data\EStreams\Figures\1st paper\Version 2\ConfusionMatrix_allversions_10C_V2.png',bbox_inches='tight',dpi=300)


#%% Confusion matrix main driver plots (pre-calculation)

# Store confusion matrices in a dict
cms = {
    'exp_2': cm_avg_2,
    'exp_3': cm_avg_3,
    'exp_4': cm_avg_4,
    'exp_5': cm_avg_5,
    'exp_6': cm_avg_6
}

n_clusters = cm_avg_2.shape[0]

# Initialize DataFrame
results = pd.DataFrame(index=range(n_clusters), columns=['best_exp', 'best_acc', 'second_best_acc', 'diff'])

for i in range(n_clusters):
    # Extract diagonal values for cluster i
    diag_values = {exp: cm.iloc[i, i] for exp, cm in cms.items()}
    
    # Sort by accuracy descending
    sorted_acc = sorted(diag_values.items(), key=lambda x: x[1], reverse=True)
    
    # Record best, second best and difference
    best_exp, best_acc = sorted_acc[0]
    second_best_acc = sorted_acc[1][1]
    diff = best_acc - second_best_acc
    
    results.loc[i] = [best_exp, best_acc, second_best_acc, diff]

# Optional: convert numeric columns to float
results[['best_acc', 'second_best_acc', 'diff']] = results[['best_acc', 'second_best_acc', 'diff']].astype(float)

print(results)

#%% DRIVERS OF CONFUSION MATRIX plot
from matplotlib.colors import to_rgb, to_hex
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


n_levels = 4  # number of shades per experiment
light_factor = 0.6  # lightest shade = 50% of base color

exp_colors = [
    "#1f77b4",  # tab:blue
    "#2E8B57",  # sea green
    "#A6D854",  # lime-ish green
    "#8B4513",  # saddlebrown
    "#9EC1CF"   # lightsteelblue
]
exp_map = {'exp_2': 0, 'exp_3': 1, 'exp_4': 2, 'exp_5': 3, 'exp_6':4}
legend_labels = ["2-C", "3-VLC", "4-VLS", "5-SGT", "6-A"]
exp_plot_order = ['exp_2', 'exp_5', 'exp_3', 'exp_4', 'exp_6']  # order to plot


# Create shades function
def make_shades(base_color):
    base_rgb = np.array(to_rgb(base_color))

    #two darkest shades 
    darkest1 = base_rgb * 0.58       # very dark
    darkest2 = base_rgb * 0.87       # dark make bigger to make them less dark

    # ---- medium shade ----
    lighter = base_rgb * 0.75 + np.array([1,1,1]) * 0.25

    # ---- lightest shade ----
    lightest = base_rgb * 0.5 + np.array([1,1,1]) * 0.51 # colour as x and y, decreas x and increase y slightly to make lighter

    # return LIGHT → DARK
    return [
        to_hex(lightest),
        to_hex(lighter),
        to_hex(darkest2),
        to_hex(darkest1)
    ]

# Quantile-based scaling
# Map catchments to their best experiment
catchment_clusters = hydro_sign['C_k10_AllS_NOClimate_remapped']  # cluster 0-9
catchment_best_exp = catchment_clusters.map(lambda x: results.loc[x, 'best_exp'])
catchment_diff = catchment_clusters.map(lambda x: results.loc[x, 'diff'])

bins = [0, 3, 6, 9, 13]  # bin edges
labels = [0, 1, 2, 3]    # 0=lightest, 3=darkest

diff_scaled = pd.cut(catchment_diff, bins=bins, labels=labels, include_lowest=True)

# Assign colors
catchment_color = []
for catchment in catchment_clusters.index:
    cluster = catchment_clusters.loc[catchment]
    best_exp = results.loc[cluster, 'best_exp']
    level = diff_scaled.loc[catchment]

    if pd.isna(level):
        # Optional: assign a default colour if diff is NaN
        catchment_color.append("#cccccc")  # light grey
    else:
        shades = make_shades(exp_colors[exp_map[best_exp]])
        catchment_color.append(shades[int(level)])  # cast to int

# Add colors and experiment order to dataframe
hydro_sign['best_exp'] = catchment_best_exp
hydro_sign['plot_color'] = catchment_color
hydro_sign['plot_order'] = hydro_sign['best_exp'].apply(lambda x: exp_plot_order.index(x))
hydro_sign_sorted = hydro_sign.sort_values('plot_order')


# Plot map
fig = plt.figure(figsize=(8, 9))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cartopy.feature.OCEAN, zorder=100, facecolor='powderblue', alpha=0.2, edgecolor='k', linewidth=0.08)
ax.set_global()
ax.set_xlim(-28, 35)
ax.set_ylim(34, 72)

sc = ax.scatter(
    hydro_sign_sorted.lon_snap,
    hydro_sign_sorted.lat_snap,
    c=hydro_sign_sorted.plot_color,
    marker='o',
    s=3.5,
    edgecolors='face',
    linewidths=0.1,
    zorder=2
)

# Compact info box with manual white box + labels on the left
present_exps = [exp for exp in exp_plot_order if exp in hydro_sign['best_exp'].unique()]


# Create inset axes
info_box = inset_axes(
    ax, width="26%", height="16%", loc='lower left',
    bbox_to_anchor=(0.01, 0.08, 1, 1),
    bbox_transform=ax.transAxes, borderpad=0
)

# Make the axes background white with black border
info_box.patch.set_facecolor('white')
info_box.patch.set_edgecolor('black')
info_box.patch.set_linewidth(0.5)


# Parameters for bars
n_shades = 4
bar_height = 0.18
bar_width = 0.18
label_x = 0.02
bars_x_start = 0.24
y_spacing = 0.17
top_padding = 0.1
bottom_padding = 0.02

n_bars = len(present_exps)


# Compute positions for bars
y_start = 1 - top_padding - bar_height           # topmost bar inside axes
bottom_bar_y = y_start - (n_bars-1) * y_spacing


present_exps = ['exp_2', 'exp_3', 'exp_5']

# Draw bars + labels
for i, exp in enumerate(present_exps):  # first = top
    base_color = exp_colors[exp_map[exp]]
    shades = make_shades(base_color)
    y = y_start - i * y_spacing

    # Label
    info_box.text(label_x, y + bar_height/2, legend_labels[exp_map[exp]],
                  va='center', ha='left', fontsize=7, zorder=10)

    # Horizontal bars
    for j, shade in enumerate(shades):
        info_box.add_patch(Rectangle(
            (bars_x_start + j*bar_width, y),
            width=bar_width, height=bar_height,
            facecolor=shade, edgecolor='none', zorder=3
        ))


# x axis ticks
max_diff = 12
tick_vals = np.arange(0, max_diff+1, 3)
bar_width_total = n_shades * bar_width
tick_positions = bars_x_start + (tick_vals / max_diff) * bar_width_total

for x in tick_positions:
    info_box.vlines(x, bottom_bar_y - 0.015, bottom_bar_y - 0.005,
                    color='k', lw=0.4, zorder=5)

# Tick labels
for x, val in zip(tick_positions, tick_vals):
    info_box.text(x, bottom_bar_y - 0.035, str(val),
                  fontsize=7, ha='center', va='top', zorder=10)

# X-axis label
info_box.text(bars_x_start + bar_width_total/2,
              bottom_bar_y - 0.19,
              "Prediction diff. (% points)",
              fontsize=6, ha='center', va='top', zorder=10)


# Final adjustments
info_box.set_xlim(0, 1)
info_box.set_ylim(0, 1)
info_box.set_xticks([])
info_box.set_yticks([])
for spine in info_box.spines.values():
    spine.set_visible(False)
    
plt.savefig(r'U:\Hydrological Data\EStreams\Figures\1st paper\Version 2\ConfusionMatrix_DRIVERS_temp_allversions_10C.pdf',bbox_inches='tight',dpi=300)
plt.savefig(r'U:\Hydrological Data\EStreams\Figures\1st paper\Version 2\ConfusionMatrix_DRIVERS_temp_allversions_10C.png',bbox_inches='tight',dpi=300)

#%%

from sklearn.metrics import auc, confusion_matrix
from sklearn.preprocessing import label_binarize
import numpy as np

# Your titles
titles = [
    "1-CL",
    "2-C",
    "3-VLC",
    "4-VLS",
    "5-SGT",
    "6-A"
]

# List of your RF run data for convenience
all_y_test_list = [all_y_test_0, all_y_test_2, all_y_test_3, all_y_test_4, all_y_test_5, all_y_test_6]
all_y_pred_list = [all_y_pred_0, all_y_pred_2, all_y_pred_3, all_y_pred_4, all_y_pred_5, all_y_pred_6]
all_y_score_list = [all_y_score_0, all_y_score_2, all_y_score_3, all_y_score_4, all_y_score_5, all_y_score_6]
tpr_interp_dict_list = [tpr_interp_dict_0, tpr_interp_dict_2, tpr_interp_dict_3, tpr_interp_dict_4, tpr_interp_dict_5, tpr_interp_dict_6]
auc_dict_list = [auc_dict_0, auc_dict_2, auc_dict_3, auc_dict_4, auc_dict_5, auc_dict_6]



fig, axes = plt.subplots(3, 2, figsize=(14, 19))  # 3 rows x 2 columns
axes = axes.ravel()  # flatten axes array for easy indexing

for idx in range(6):
    ax = axes[idx]

    y_test_all = all_y_test_list[idx].values.ravel()
    y_pred_all = all_y_pred_list[idx].values.ravel()
    y_score_all = all_y_score_list[idx].values

    n_classes = len(np.unique(y_test_all))
    y_test_bin = label_binarize(y_test_all, classes=np.arange(n_classes))
    y_pred_bin = label_binarize(y_pred_all, classes=np.arange(n_classes))

    dot_points = []  # store (fpr_point, tpr_point, color) tuples

    # --- plot ROC curves first ---
    for i in range(n_classes):
        mean_tpr = np.mean(tpr_interp_dict_list[idx][i], axis=0)
        std_tpr = np.std(tpr_interp_dict_list[idx][i], axis=0)
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(auc_dict_list[idx][i])

        ax.plot(mean_fpr, mean_tpr, lw=2, color=color_mapping2[i],
                label=f'HRT {i+1} (AUC = {mean_auc:.2f} ± {std_auc:.2f})')
        ax.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                        color=color_mapping2[i], alpha=0.2)

        # compute the confusion matrix point but don't plot yet
        y_pred_all_argmax = np.argmax(y_score_all, axis=1)
        y_true_bin = (y_test_all == i).astype(int)
        y_pred_bin_i = (y_pred_all_argmax == i).astype(int)
        cm = confusion_matrix(y_true_bin, y_pred_bin_i)
        tn, fp, fn, tp = cm.ravel()
        fpr_point = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tpr_point = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        dot_points.append((fpr_point, tpr_point, color_mapping2[i]))

    # --- now plot all dots on top ---
    for fpr_point, tpr_point, color in dot_points:
        ax.plot(fpr_point, tpr_point, 'o', color=color,
                markersize=7, markeredgecolor='k', zorder=10)

    # rest of styling
    ax.plot([0, 1], [0, 1], 'k--', lw=1, zorder=1)
    ax.set_xlabel('False Positive Rate (-)', fontsize=14)
    ax.set_ylabel('True Positive Rate (-)', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_title(f"({chr(97+idx)}) {titles[idx]}", fontsize=16, loc='left')
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True)

plt.tight_layout()
plt.savefig(r'U:\Hydrological Data\EStreams\Figures\1st paper\Version 2\ROC_PER_CLUSTER_allversions_10C_V2.pdf',bbox_inches='tight',dpi=300)
plt.savefig(r'U:\Hydrological Data\EStreams\Figures\1st paper\Version 2\ROC_PER_CLUSTER_allversions_10C_V2.png',bbox_inches='tight',dpi=300)

#%%
#%% ROC curve
#%% ROC curve with shaded area
all_runs = {
    0: (all_y_test_0_fold, all_y_score_0_fold, all_y_pred_0_fold),
    2: (all_y_test_2_fold, all_y_score_2_fold, all_y_pred_2_fold),
    3: (all_y_test_3_fold, all_y_score_3_fold, all_y_pred_3_fold),
    4: (all_y_test_4_fold, all_y_score_4_fold, all_y_pred_4_fold),
    5: (all_y_test_5_fold, all_y_score_5_fold, all_y_pred_5_fold),
    6: (all_y_test_6_fold, all_y_score_6_fold, all_y_pred_6_fold)
}

run_labels = {
    0: "1-CL",
    2: "2-C",
    3: "3-VLC",
    4: "4-VLS",
    5: "5-SGT",
    6: "6-A"
}

# your experiment colors in the correct order
colors = [
    "#159895",  # teal (lightseagreen adjusted; slightly bluer)
    "#1f77b4",  # tab:blue (unchanged)
    "#2E8B57",  # sea green (for darker vegetation; less red)
    "#A6D854",  # lime-ish green (lighter vegetation; shifted toward yellow)
    "#8B4513",  # saddlebrown (unchanged, soil)
    "#9EC1CF"   # slightly darker lightsteelblue for better contrast
]

run_order = [0, 2, 3, 4, 5, 6]  # explicit order to match colors / titles

# List of your RF run data for convenience
all_y_test_list = [all_y_test_0_fold, all_y_test_2_fold, all_y_test_3_fold, all_y_test_4_fold, all_y_test_5_fold, all_y_test_6_fold]
all_y_pred_list = [all_y_pred_0_fold, all_y_pred_2_fold, all_y_pred_3_fold, all_y_pred_4_fold, all_y_pred_5_fold, all_y_pred_6_fold]
all_y_score_list = [all_y_score_0_fold, all_y_score_2_fold, all_y_score_3_fold, all_y_score_4_fold, all_y_score_5_fold, all_y_score_6_fold]
tpr_interp_dict_list = [tpr_interp_0_all_df, tpr_interp_2_all_df, tpr_interp_3_all_df, tpr_interp_4_all_df, tpr_interp_5_all_df, tpr_interp_6_all_df]
#auc_dict_list = [auc_dict_0, auc_dict_2, auc_dict_3, auc_dict_4, auc_dict_5, auc_dict_6]



for idx in range(6):
    y_test_all = np.concatenate(all_y_test_list[idx])
    y_pred_all = np.concatenate(all_y_pred_list[idx])
    y_score_all = np.vstack(all_y_score_list[idx])

mean_fpr = np.linspace(0, 1, 100)

# helper functions for combining fold lists/series/arrays robustly
def concat_labels(obj):
    # obj can be a list of arrays/series, a single array/series, or a pandas concat-able list
    if isinstance(obj, (list, tuple)):
        try:
            return np.concatenate([np.asarray(x).ravel() for x in obj])
        except Exception:
            # fallback to pandas concat (works if elements are Series)
            return pd.concat(obj).values.ravel()
    elif isinstance(obj, (np.ndarray, pd.Series)):
        return np.asarray(obj).ravel()
    else:
        raise ValueError("Unsupported labels object type")

def vstack_scores(obj):
    # obj expected to be a list of (n_samples_fold x n_classes) arrays or a single 2D array
    if isinstance(obj, (list, tuple)):
        return np.vstack([np.asarray(x) for x in obj])
    else:
        return np.asarray(obj)
# define a common FPR grid for interpolation
#mean_fpr = np.linspace(0, 1, 100)

plt.figure(figsize=(10, 8))

for color, run in zip(colors, run_order):
    entry = all_runs[run]
    if len(entry) == 3:
        all_y_test, all_y_score, all_y_pred = entry
    elif len(entry) == 2:
        all_y_test, all_y_score = entry
        all_y_pred = None
    else:
        raise ValueError("each all_runs entry must have 2 or 3 items (y_test, y_score[, y_pred])")

    # combine folds robustly
    y_test_all = concat_labels(all_y_test)
    y_score_all = vstack_scores(all_y_score)

    # number of classes
    n_classes = y_score_all.shape[1] if y_score_all.ndim > 1 else len(np.unique(y_test_all))
    classes = np.arange(n_classes)

    # binarize labels
    y_test_bin_all = label_binarize(y_test_all, classes=classes)

    # --- collect fold-level ROC curves ---
    # if all_y_score is a list of folds
    tprs = []
    aucs = []
    for fold_y_test, fold_y_score in zip(all_y_test, all_y_score):
        fold_y_test_bin = label_binarize(fold_y_test, classes=classes)
        fpr, tpr, _ = roc_curve(fold_y_test_bin.ravel(), fold_y_score.ravel())
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)
        aucs.append(auc(fpr, tpr))

    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    label = f'{run_labels[run]} (AUC = {mean_auc:.2f} ± {std_auc:.2f})'
    plt.plot(mean_fpr, mean_tpr, color=color, lw=2, label=label)
    plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color=color, alpha=0.2)

    # --- final decision point ---
    if all_y_pred is None:
        y_pred_all = np.argmax(y_score_all, axis=1)
    else:
        y_pred_all = concat_labels(all_y_pred)

    y_pred_bin_all = label_binarize(y_pred_all, classes=classes)
    y_true_flat = y_test_bin_all.ravel()
    y_pred_flat = y_pred_bin_all.ravel()

    TP = np.sum((y_true_flat == 1) & (y_pred_flat == 1))
    FP = np.sum((y_true_flat == 0) & (y_pred_flat == 1))
    FN = np.sum((y_true_flat == 1) & (y_pred_flat == 0))
    TN = np.sum((y_true_flat == 0) & (y_pred_flat == 0))

    fpr_point = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    tpr_point = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    plt.scatter(fpr_point, tpr_point, s=90, color=color, edgecolors='k', zorder=6)

# chance line
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('False Positive Rate (-)', fontsize=15)
plt.ylabel('True Positive Rate (-)', fontsize=15)
plt.xticks(size=13)
plt.yticks(size=13)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend(loc='lower right', fontsize=13)
plt.grid(True)
plt.tight_layout()
plt.savefig(r'U:\Hydrological Data\EStreams\Figures\1st paper\Version 2\ROC_curve_allversions_10C.pdf',bbox_inches='tight', dpi=300)
plt.savefig(r'U:\Hydrological Data\EStreams\Figures\1st paper\Version 2\ROC_curve_allversions_10C.png',bbox_inches='tight', dpi=300)

#%% Test and set up for roc
# Micro avregaed, so all of the test values for each fold are taken out and taken together
# thus there is no shaded area, because we are treating everything 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
#%% 

all_runs = {
    0: (all_y_test_0_fold, all_y_score_0_fold, all_y_pred_0_fold),
    2: (all_y_test_2_fold, all_y_score_2_fold, all_y_pred_2_fold),
    3: (all_y_test_3_fold, all_y_score_3_fold, all_y_pred_3_fold),
    4: (all_y_test_4_fold, all_y_score_4_fold, all_y_pred_4_fold),
    5: (all_y_test_5_fold, all_y_score_5_fold, all_y_pred_5_fold),
    6: (all_y_test_6_fold, all_y_score_6_fold, all_y_pred_6_fold)
}

run_labels = {
    0: "1-CL",
    2: "2-C",
    3: "3-VLC",
    4: "4-VLS",
    5: "5-SGT",
    6: "6-A"
}

#  experiment colors in the correct order
colors = [
    "#159895",  # teal (lightseagreen adjusted; slightly bluer)
    "#1f77b4",  # tab:blue (unchanged)
    "#2E8B57",  # sea green (for darker vegetation; less red)
    "#A6D854",  # lime-ish green (lighter vegetation; shifted toward yellow)
    "#8B4513",  # saddlebrown (unchanged, soil)
    "#9EC1CF"   # slightly darker lightsteelblue for better contrast
]

run_order = [0, 2, 3, 4, 5, 6]  # explicit order to match colors / titles

# List of your RF run data for convenience
all_y_test_list = [all_y_test_0_fold, all_y_test_2_fold, all_y_test_3_fold, all_y_test_4_fold, all_y_test_5_fold, all_y_test_6_fold]
all_y_pred_list = [all_y_pred_0_fold, all_y_pred_2_fold, all_y_pred_3_fold, all_y_pred_4_fold, all_y_pred_5_fold, all_y_pred_6_fold]
all_y_score_list = [all_y_score_0_fold, all_y_score_2_fold, all_y_score_3_fold, all_y_score_4_fold, all_y_score_5_fold, all_y_score_6_fold]
tpr_interp_dict_list = [tpr_interp_0_all_df, tpr_interp_2_all_df, tpr_interp_3_all_df, tpr_interp_4_all_df, tpr_interp_5_all_df, tpr_interp_6_all_df]
#%%

def concat_labels(label_list):
    """Safely concatenate a list of 1D label arrays."""
    return np.concatenate(label_list, axis=0)

def vstack_scores(score_list):
    """Safely stack a list of 2D score arrays."""
    return np.vstack(score_list)

#%% New example #2
def micro_roc_with_std(all_y_test_folds, all_y_score_folds, n_classes, color, label):
    # define common FPR grid
    mean_fpr = np.linspace(0, 1, 101)

    tprs_interp = []
    aucs = []

    classes = np.arange(n_classes)

    for y_test_fold, y_score_fold in zip(all_y_test_folds, all_y_score_folds):
        # binarize labels for this fold
        y_test_bin = label_binarize(y_test_fold, classes=classes)

        # flatten for micro ROC
        y_true_flat = y_test_bin.ravel()
        y_score_flat = y_score_fold.ravel()

        fpr, tpr, _ = roc_curve(y_true_flat, y_score_flat)

        # interpolate TPR onto the common mean_fpr grid
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs_interp.append(tpr_interp)

        fold_auc = auc(fpr, tpr)
        aucs.append(fold_auc)

    tprs_interp = np.array(tprs_interp)
    mean_tpr = tprs_interp.mean(axis=0)
    std_tpr = tprs_interp.std(axis=0)

    mean_tpr[-1] = 1.0

    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    # plot mean curve
    plt.plot(mean_fpr, mean_tpr,
             color=color,
             lw=2,
             label=label + " micro (AUC = " + str(round(mean_auc, 3)) + " ± " + str(round(std_auc, 3)) + ")")

    # plot shaded std band
    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tpr_lower, tpr_upper,
                     color=color, alpha=0.2)
    

#%% Test new example #3
def micro_roc_with_std(all_y_test_folds, all_y_score_folds, n_classes, color, label):
    # define common FPR grid
    mean_fpr = np.linspace(0, 1, 101)

    tprs_interp = []
    aucs = []

    classes = np.arange(n_classes)

    for y_test_fold, y_score_fold in zip(all_y_test_folds, all_y_score_folds):
        # binarize labels for this fold
        y_test_bin = label_binarize(y_test_fold, classes=classes)

        # flatten for micro ROC
        y_true_flat = y_test_bin.ravel()
        y_score_flat = y_score_fold.ravel()

        fpr, tpr, _ = roc_curve(y_true_flat, y_score_flat)
        print(fpr)

        # interpolate TPR onto the common mean_fpr grid
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs_interp.append(tpr_interp)

        fold_auc = auc(fpr, tpr)
        aucs.append(fold_auc)

    tprs_interp = np.array(tprs_interp)
    mean_tpr = tprs_interp.mean(axis=0)
    std_tpr = tprs_interp.std(axis=0)

    mean_tpr[-1] = 1.0

    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    # plot mean curve
    plt.plot(
        mean_fpr,
        mean_tpr,
        color=color,
        lw=2,
        label=label + " (AUC = " + str(round(mean_auc, 2)) +
              " ± " + str(round(std_auc, 2)) + ")"
    )

    # plot shaded std band
    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color=color, alpha=0.2)
    
def macro_operating_point_from_cm(cm):
    # cm: (n_classes, n_classes)
    n_classes = cm.shape[0]
    tpr_list = []
    fpr_list = []

    for k in range(n_classes):
        tp_k = cm[k, k]
        fn_k = cm[k, :].sum() - tp_k       # row sum minus diagonal
        fp_k = cm[:, k].sum() - tp_k       # column sum minus diagonal
        tn_k = cm.sum() - (tp_k + fn_k + fp_k)

        # avoid division by zero
        if tp_k + fn_k > 0:
            tpr_k = tp_k / float(tp_k + fn_k)
        else:
            tpr_k = 0.0

        if fp_k + tn_k > 0:
            fpr_k = fp_k / float(fp_k + tn_k)
        else:
            fpr_k = 0.0

        tpr_list.append(tpr_k)
        fpr_list.append(fpr_k)

    tpr_arr = np.array(tpr_list)
    fpr_arr = np.array(fpr_list)

    # macro-average over classes, ignoring NaNs if any
    macro_tpr = np.nanmean(tpr_arr)
    macro_fpr = np.nanmean(fpr_arr)

    return macro_fpr, macro_tpr

#%% Plotting

fold_cms_exp = [fold_cms_0, fold_cms_2, fold_cms_3, fold_cms_4, fold_cms_5, fold_cms_6]
cumulative_cm_exp = [cumulative_cm_0, cumulative_cm_2, cumulative_cm_3, cumulative_cm_4, cumulative_cm_5, cumulative_cm_6]
plt.figure(figsize=(7, 7))

for color, exp, fold_cms_list, cum_cm in zip(colors, run_order, fold_cms_exp, cumulative_cm_exp):

    # unpack your stored data per experiment
    all_y_test_exp, all_y_score_exp, _ = all_runs[exp]  # assuming this structure

    # infer number of classes
    n_classes = all_y_score_exp[0].shape[1]

    # 1) plot micro-ROC with std band for this experiment
    micro_roc_with_std(
        all_y_test_folds=all_y_test_exp,
        all_y_score_folds=all_y_score_exp,
        n_classes=n_classes,
        color=color,
        label=run_labels[exp]
    )

# 2) plot per-fold operating points (small markers)
    #    fold_cms_list is the list of confusion matrices for this experiment
    for cm_fold in fold_cms_list:
        fpr_fold, tpr_fold = macro_operating_point_from_cm(cm_fold)
        #print(f"frp:{exp} : {fpr_fold}")
        #print(f"trp:{exp} : {tpr_fold}")
        plt.scatter(
            fpr_fold,
            tpr_fold,
            color=color,
            s=30,
            alpha=0.7,
            edgecolor="none",
            zorder=3
        )
        
    if exp == 3:
        fpr_global, tpr_global = macro_operating_point_from_cm(cum_cm)
        plt.scatter(
            fpr_global,
            tpr_global,
            color=color,
            alpha = 0.8,
            s=80,
            marker="o",
            edgecolor="k",
            linewidth=0.9,
            #label=run_labels[exp] + " operating point",
            zorder=5
        )
    elif exp == 5:
         fpr_global, tpr_global = macro_operating_point_from_cm(cum_cm)
         plt.scatter(
             fpr_global,
             tpr_global,
             color=color,
             alpha = 0.9,
             s=100,
             marker="o",
             edgecolor="k",
             linewidth=0.8,
             #label=run_labels[exp] + " operating point",
             zorder=4
         )
        
    else:

        # 3) plot global operating point from cumulative confusion matrix (big marker)
        fpr_global, tpr_global = macro_operating_point_from_cm(cum_cm)
        plt.scatter(
            fpr_global,
            tpr_global,
            color=color,
            alpha = 0.8,
            s=80,
            marker="o",
            edgecolor="k",
            linewidth=0.9,
            #label=run_labels[exp] + " operating point",
            zorder=4
            )

# diagonal line
plt.plot([0, 1], [0, 1], "k--", lw=1)

plt.xlabel("False Positive Rate (-)", fontsize=15)
plt.ylabel("True Positive Rate (-)", fontsize=15)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True)

# you may want to manage legend size: micro curves + operating points
plt.legend(loc="lower right", fontsize=10)
plt.tight_layout()
plt.savefig(r'U:\Hydrological Data\EStreams\Figures\1st paper\Version 2\Roc_test.pdf',bbox_inches='tight', dpi=300)
plt.savefig(r'U:\Hydrological Data\EStreams\Figures\1st paper\Version 2\Roc_test.png',bbox_inches='tight', dpi=300)

#%% COMBO NEW FIGURE V2 use this one
# --- Figure with 2 subpanels ---


# -----------------------
# --- Panel (a) data ---
all_runs_panel_a = {
    0: (test_scores_0, train_scores_0),
    2: (test_scores_2, train_scores_2),
    3: (test_scores_3, train_scores_3),
    4: (test_scores_4, train_scores_4),
    5: (test_scores_5, train_scores_5),
    6: (test_scores_6, train_scores_6)
}



run_labels = {
    0: "1-CL",
    2: "2-C",
    3: "3-VLC",
    4: "4-VLS",
    5: "5-SGT",
    6: "6-A"
}

colors = [
    "#159895",  # teal
    "#1f77b4",  # blue
    "#2E8B57",  # sea green
    "#A6D854",  # lime-ish green
    "#8B4513",  # brown
    "#9EC1CF"   # lightsteelblue
]

run_order = [0, 2, 3, 4, 5, 6]


# -----------------------
# --- Create figure and subplots ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# -----------------------
# --- Panel (a): training vs testing scatter + boxplots ---
box_width = 0.04  # horizontal width of boxplots

for i, run_id in enumerate(all_runs_panel_a.keys()):
    test_scores, train_scores = all_runs_panel_a[run_id]
    test_scores = np.array(test_scores)
    train_scores = np.array(train_scores)
    
    # Compute quartiles
    q1, q2, q3 = np.percentile(test_scores, [25, 50, 75])
    iqr = q3 - q1
    lower_whisker = max(min(test_scores), q1 - 1.5 * iqr)
    upper_whisker = min(max(test_scores), q3 + 1.5 * iqr)
    
    # Center x-position
    x_center = np.mean(train_scores)
    
    # Draw box
    rect = plt.Rectangle(
        (x_center - box_width/2, q1),
        box_width,
        q3 - q1,
        facecolor=colors[i],
        edgecolor="black",
        alpha=0.6,
        zorder=1
    )
    ax1.add_patch(rect)
    
    # Draw whiskers
    ax1.vlines(x_center, lower_whisker, q1, color='black', alpha=0.3, lw=1.2)
    ax1.vlines(x_center, q3, upper_whisker, color='black', alpha=0.3, lw=1.2)

# Scatter overlay
for i, run_id in enumerate(all_runs_panel_a.keys()):
    test_scores, train_scores = all_runs_panel_a[run_id]
    ax1.scatter(
        train_scores,
        test_scores,
        color=colors[i],
        label=run_labels[run_id],
        alpha=0.8,
        s=80,
        edgecolor='k',
        zorder=2
    )

# Diagonal reference line
ax1.plot([0.25, 0.75], [0.25, 0.75], 'k--', lw=1.5, zorder=3)

# Styling
ax1.set_xlabel("Training accuracy (-)", fontsize=15)
ax1.set_ylabel("Testing accuracy (-)", fontsize=15)
ax1.set_xlim(0.25, 0.75)
ax1.set_ylim(0.25, 0.75)
ax1.tick_params(axis='both', which='major', labelsize=13)
ax1.legend(loc='upper left', fontsize=12)
ax1.grid(True, alpha=0.3)




for color, exp, fold_cms_list, cum_cm in zip(colors, run_order, fold_cms_exp, cumulative_cm_exp):

    # unpack your stored data per experiment
    all_y_test_exp, all_y_score_exp, _ = all_runs[exp]  # assuming this structure

    # infer number of classes
    n_classes = all_y_score_exp[0].shape[1]

    # 1) plot micro-ROC with std band for this experiment
    micro_roc_with_std(
        all_y_test_folds=all_y_test_exp,
        all_y_score_folds=all_y_score_exp,
        n_classes=n_classes,
        color=color,
        label=run_labels[exp]
    )

# 2) plot per-fold operating points (small markers)
    #    fold_cms_list is the list of confusion matrices for this experiment
    for cm_fold in fold_cms_list:
        fpr_fold, tpr_fold = macro_operating_point_from_cm(cm_fold)
        ax2.scatter(
            fpr_fold,
            tpr_fold,
            color=color,
            s=30,
            alpha=0.7,
            edgecolor="none",
            zorder=3
        )
        
    if exp == 3:
        fpr_global, tpr_global = macro_operating_point_from_cm(cum_cm)
        ax2.scatter(
            fpr_global,
            tpr_global,
            color=color,
            alpha = 0.8,
            s=78,
            marker="o",
            edgecolor="k",
            linewidth=0.9,
            #label=run_labels[exp] + " operating point",
            zorder=5
        )
    elif exp == 5:
         fpr_global, tpr_global = macro_operating_point_from_cm(cum_cm)
         ax2.scatter(
             fpr_global,
             tpr_global,
             color=color,
             alpha = 0.9,
             s=105,
             marker="o",
             edgecolor="k",
             linewidth=0.8,
             #label=run_labels[exp] + " operating point",
             zorder=4
         )
        
    else:

        # 3) plot global operating point from cumulative confusion matrix (big marker)
        fpr_global, tpr_global = macro_operating_point_from_cm(cum_cm)
        ax2.scatter(
            fpr_global,
            tpr_global,
            color=color,
            alpha = 0.8,
            s=80,
            marker="o",
            edgecolor="k",
            linewidth=0.9,
            #label=run_labels[exp] + " operating point",
            zorder=4
            )

# diagonal line
ax2.plot([0, 1], [0, 1], "k--", lw=1)

ax2.set_xlabel("False Positive Rate (-)", fontsize=15)
ax2.set_ylabel("True Positive Rate (-)", fontsize=15)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='lower right', fontsize=12)
ax2.tick_params(axis='both', which='major', labelsize=13)


ax1.set_title("(a)", loc='left', fontsize=16,  pad=10)
ax2.set_title("(b)", loc='left', fontsize=16, pad=10)

plt.subplots_adjust(wspace=0.2) 
plt.savefig(r'U:\Hydrological Data\EStreams\Figures\1st paper\Version 2\RF_TestingvsTraining_ROCCRUVE_10C_V2.pdf',bbox_inches='tight', dpi=300)
plt.savefig(r'U:\Hydrological Data\EStreams\Figures\1st paper\Version 2\RF_TestingvsTraining_ROCCRUVE_10C_V2.png',bbox_inches='tight', dpi=300)



#%%

all_scores = [test_scores_0, test_scores_2, test_scores_3, test_scores_4, test_scores_5, test_scores_6]

titles2 = [
    "1-CL",
    "2-C",
    "3-VLC",
    "4-VLS",
    "5-SGT",
    "6-A"
]

colors = [
    "#159895",  # teal
    "#1f77b4",  # blue
    "#2E8B57",  # sea green
    "#A6D854",  # lime-ish green
    "#8B4513",  # brown
    "#9EC1CF"   # lightsteelblue
]

plt.figure(figsize=(10, 6))
ax = plt.gca()

# Plot fold scores per experiment
for i, scores in enumerate(all_scores):
    folds = np.arange(1, len(scores)+1)
    scores_array = np.array(scores)
    
    mean_scores = scores_array  # fold scores themselves
    std_scores = np.std(scores_array)  # standard deviation across folds
    
    ax.plot(folds, mean_scores, marker='o', color=colors[i], label=titles2[i], zorder=2)
    ax.fill_between(folds, mean_scores - std_scores, mean_scores + std_scores,
                    color=colors[i], alpha=0.2, zorder=3)

ax.set_xlabel("Cross-validation Fold", fontsize=14)
ax.set_ylabel("Testing Accuracy (-)", fontsize=14)
ax.set_xticks(folds)
ax.set_ylim(0.23, 0.7)
ax.set_xlim(1, 10)
ax.grid(alpha=0.3)
#ax.legend(fontsize=12, markerscale=0.8, frameon=True, edgecolor='black')
ax.tick_params(axis='both', which='major', labelsize=14)
ax.legend(
    fontsize=14, markerscale=0.8, frameon=False,
    ncol=len(all_scores), loc='upper center', bbox_to_anchor=(0.5, -0.15))

plt.tight_layout()
plt.savefig(r'U:\Hydrological Data\EStreams\Figures\1st paper\Version 2\RF_TestingACC_perfold_10C.pdf',bbox_inches='tight', dpi=300)
plt.savefig(r'U:\Hydrological Data\EStreams\Figures\1st paper\Version 2\RF_TestingACC_perfold_10C.png',bbox_inches='tight', dpi=300)


#%% Correct Incorrect 
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
import string
from matplotlib.ticker import MaxNLocator
# Data
y_true = y.to_numpy().ravel()
y_pred = y_pred_full_6  # change depedning on the cluster you want to plot 0, 2,3,4,5,6 (only change here)
y_pred = y_pred_full_6.astype(int) # change depedning on the cluster you want to plot 0, 2,3,4,5,6 (only change here)
correct_predictions = y_true == y_pred
incorrect_predictions = ~correct_predictions

longitudes = hydro_sign.lon_snap
latitudes = hydro_sign.lat_snap

# Plot settings
cmap_sr = ListedColormap(color_mapping2[:10])
norm = BoundaryNorm(np.linspace(0, 10, 11), cmap_sr.N)
x = np.arange(10)
bar_width = 0.8
total_catchments = len(y_true)

import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(20, 25))
axes = []
rows = 5
cols = 4
width = 0.26
height = 0.12
h_gap = 0.02
v_start = 0.85

for i in range(rows):
    row_axes = []
    bottom = v_start - i*(height + h_gap)
    for j in range(cols):

        # Pair index (0 or 1)
        pair = j // 2
        # Position inside the pair (0=correct, 1=incorrect)
        pos = j % 2

        # Base left offset for each pair
        if pair == 0:
            base_left = 0.05
        else:
            base_left = 0.05 + 2*(width + 0.005) + 0.05  # <-- big space before pair 2

        # Inside each pair: tiny gap between correct/incorrect
        left = base_left + pos*(width + 0.005)  # small 0.005 gap inside pair

        ax = fig.add_axes([left, bottom, width, height], projection=ccrs.PlateCarree())
        row_axes.append(ax)
    axes.append(row_axes)

# Flatten axes list into a single list
all_axes = [ax for row in axes for ax in row]

panel_labels = [f"({c})" for c in string.ascii_lowercase[:10]]

label_idx = 0


clusters = np.unique(y_true)  # 10 clusters

for idx, cluster_id in enumerate(clusters):
    row = idx // 2  # two clusters per row
    col_base = (idx % 2) * 2  # base column for this cluster

    # Mask for this cluster
    cluster_mask = y_true == cluster_id
    correct_mask = cluster_mask & correct_predictions
    incorrect_mask = cluster_mask & incorrect_predictions

    # LEFT column = Correct predictions
    ax_correct = axes[row][col_base]
    ax_correct.coastlines()
    ax_correct.add_feature(cartopy.feature.OCEAN, zorder=100,
                           facecolor='powderblue', alpha=0.2, edgecolor='k', linewidth=0.1)
    ax_correct.set_extent([-28, 35, 34, 72], crs=ccrs.PlateCarree())
    
    #plot incorrect in lgith grey
    ax_correct.scatter(longitudes[incorrect_mask], latitudes[incorrect_mask],
                       c='lightgrey', marker='o', s=10, edgecolors='none', zorder=2)
    #plot correct in cluster colour
    ax_correct.scatter(longitudes[correct_mask], latitudes[correct_mask],
                       c=color_mapping2[cluster_id], marker='o', s=10, edgecolors='none', zorder=3)
    
    
    ax_correct.set_title(f'{panel_labels[label_idx]}  HRT {cluster_id+1} – Correct',fontsize=14, loc='left')
    label_idx += 1
    
    # Inset histogram for correct
    total_per_cluster = [np.sum(y_true == c) for c in clusters]
    correct_per_cluster = [np.sum((y_true == c) & correct_predictions) for c in clusters]

    colors_bg = [color_mapping2[c] if c == cluster_id else 'grey' for c in clusters]
    alphas_bg = [0.3 if c == cluster_id else 0.3 for c in clusters]  # faded bars

    colors_fg = [color_mapping2[c] if c == cluster_id else 'none' for c in clusters]

    ax_inset = inset_axes(ax_correct, width="16%", height="35%", loc='lower left', borderpad=3)
    # Background bars (total catchments)
    for i, (tot, c, a) in enumerate(zip(total_per_cluster, colors_bg, alphas_bg)):
        ax_inset.bar(i, tot, width=bar_width, color=c, alpha=a, edgecolor='none')
        # Foreground bars (correct predictions)
        ax_inset.bar(cluster_id, correct_per_cluster[cluster_id], width=bar_width, color=color_mapping2[cluster_id], edgecolor='none')

    ax_inset.set_ylim(0, max(total_per_cluster) * 1.1)
    ticks = clusters
    tick_labels = [str(c+1) for c in clusters]
    tick_labels[-1] = "  10"   # or "   10" if you want more space
    ax_inset.set_xticks(ticks)
    ax_inset.set_xticklabels(tick_labels, fontsize=7)
    ax_inset.tick_params(axis='y', labelsize=8)

    # Optional: secondary y-axis for percent
    def counts_to_percent(count): return (count / total_catchments) * 100
    def percent_to_counts(percent): return (percent / 100) * total_catchments
    secax = ax_inset.secondary_yaxis('right', functions=(counts_to_percent, percent_to_counts))
    secax.tick_params(axis='y', labelsize=8)
    percent_ticks = np.arange(5, 105, 5)
    secax.set_yticks(percent_ticks)
    secax.set_yticklabels([f'{int(p)}%' for p in percent_ticks])

    # RIGHT column = Incorrect predictions (colored by predicted cluster)
    ax_incorrect = axes[row][col_base + 1]
    ax_incorrect.coastlines()
    ax_incorrect.add_feature(cartopy.feature.OCEAN, zorder=100,
                             facecolor='powderblue', alpha=0.2, edgecolor='k', linewidth=0.1)
    ax_incorrect.set_extent([-28, 35, 34, 72], crs=ccrs.PlateCarree())

    ax_incorrect.scatter(longitudes[incorrect_mask], latitudes[incorrect_mask],
                         c=[color_mapping2[p] for p in y_pred[incorrect_mask]],
                         marker='o', s=10, edgecolors='none')
    ax_incorrect.set_title(f'HRT {cluster_id +1} – Incorrect', fontsize=14, loc='left')

    # Histogram for incorrect: distribution of predicted clusters
    predicted_incorrect = y_pred[incorrect_mask]
    counts_predicted = [np.sum(predicted_incorrect == c) for c in clusters]
    colors_predicted = [color_mapping2[c] for c in clusters]

    ax_inset = inset_axes(ax_incorrect, width="17%", height="35%", loc='lower left', borderpad=3)
    ax_inset.bar(clusters, counts_predicted, color=colors_predicted, edgecolor='none')
    ax_inset.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ticks = clusters
    tick_labels = [str(c+1) for c in clusters]
    tick_labels[-1] = "  10"   # or "   10" if you want more space
    ax_inset.set_xticks(ticks)
    ax_inset.set_xticklabels(tick_labels, fontsize=7)
    ax_inset.tick_params(axis='y', labelsize=8)

# Shared colorbar
cbar_ax = fig.add_axes([0.25, 0.26, 0.72, 0.012]) # left, bottom, widht, height
cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap_sr),
                    cax=cbar_ax, orientation='horizontal')
cbar.set_ticks(np.arange(0.5, 10.5, 1))
cbar.set_ticklabels(np.arange(1, 11, 1))
cbar.set_label('HRT', fontsize=16)
cbar.ax.tick_params(labelsize=16)

plt.savefig(r'U:\Hydrological Data\EStreams\Figures\1st paper\Version 2\map_CL_correct_incorrect_2C_V2.pdf',bbox_inches='tight', dpi=300)
plt.savefig(r'U:\Hydrological Data\EStreams\Figures\1st paper\Version 2\map_CL_correct_incorrect_6A_V2.png',bbox_inches='tight', dpi=300)


#%% Functions for weighted correlation of the ranking of the attributes in the feature importance

# Weighted correlation functions 
def weighted_stats(x, y, w):
    """Return weighted mean, covariance, variances, and correlation."""
    w = np.asarray(w, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    sw = w.sum()
    if sw == 0:
        raise ValueError("Sum of weights is zero.")

    # weighted means
    mx = np.sum(w * x) / sw
    my = np.sum(w * y) / sw

    # deviations
    dx = x - mx
    dy = y - my

    # weighted covariance and variances
    cov = np.sum(w * dx * dy) / sw
    var_x = np.sum(w * dx**2) / sw
    var_y = np.sum(w * dy**2) / sw

    denom = np.sqrt(var_x * var_y)
    corr = cov / denom if denom > 0 else np.nan

    return mx, my, cov, var_x, var_y, corr


def weighted_pearson(x, y, w):
    """Return weighted Pearson correlation."""
    return weighted_stats(x, y, w)[5]


def weighted_spearman(rank_x, rank_y, w):
    """Compute weighted Spearman by applying weighted Pearson to ranks."""
    return weighted_pearson(rank_x, rank_y, w)
#%%

# Sort importance DataFrames 
importance_df_0_sorted = importance_df_0.sort_values('Median Importance', ascending=False)
importance_df_2_sorted = importance_df_2.sort_values('Median Importance', ascending=False)

#  Merge and compute mean importance 
merged_importance2 = pd.merge(
    importance_df_0_sorted[['Feature', 'Median Importance']],
    importance_df_2_sorted[['Feature', 'Median Importance']],
    on='Feature', how='inner', suffixes=('_0', '_2')
)

merged_importance2['Mean_importance'] = (
    merged_importance2['Median Importance_0'] + merged_importance2['Median Importance_2']
) / 2

#  Shift to make all non-negative 
min_val = merged_importance2['Mean_importance'].min()
merged_importance2['Mean_importance_shifted'] = merged_importance2['Mean_importance'] - min_val

#  Normalize to sum=1 
merged_importance2['Weight_norm'] = merged_importance2['Mean_importance_shifted'] / merged_importance2['Mean_importance_shifted'].sum()

#  Rank mappings 
features2 = importance_df_2_sorted['Feature'].tolist()
features0 = importance_df_0_sorted['Feature'].tolist()

features0_ordered_subset = [f for f in features0 if f in set(features2)]

rank2_map = {feature: rank for rank, feature in enumerate(features2, start=1)}
rank0_subset_map = {feature: rank for rank, feature in enumerate(features0_ordered_subset, start=1)}

# Create ranking DataFrame 
ranks_df2 = pd.DataFrame({'Feature': features2})
ranks_df2['Rank_2'] = ranks_df2['Feature'].map(rank2_map)
ranks_df2['Rank_0_subset'] = ranks_df2['Feature'].map(rank0_subset_map)
ranks_df2 = ranks_df2.dropna(subset=['Rank_0_subset']).copy()
ranks_df2['Rank_0_subset'] = ranks_df2['Rank_0_subset'].astype(int)

#  Merge with normalized weights 
ranks_df2 = ranks_df2.merge(
    merged_importance2[['Feature', 'Weight_norm']],
    on='Feature', how='left'
)

#  Add nice feature names 
ranks_df2['Feature_nice'] = ranks_df2['Feature'].map(feature_name_mapping)

#  Weighted Pearson correlation function 
def weighted_stats2(x, y, w):
    w = np.asarray(w, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sw = w.sum()
    if sw == 0:
        raise ValueError("Sum of weights is zero.")
    mx = (w * x).sum() / sw
    my = (w * y).sum() / sw
    dx = x - mx
    dy = y - my
    cov = (w * dx * dy).sum() / sw
    var_x = (w * dx**2).sum() / sw
    var_y = (w * dy**2).sum() / sw
    denom = np.sqrt(var_x * var_y)
    corr = cov / denom if denom > 0 else np.nan
    return mx, my, cov, var_x, var_y, corr

def weighted_pearson2(x, y, w):
    return weighted_stats(x, y, w)[5]

#  Compute correlations 
r_weighted2 = weighted_pearson2(ranks_df2['Rank_2'], ranks_df2['Rank_0_subset'], ranks_df2['Weight_norm'])
r2_weighted2 = r_weighted2**2
r_unweighted2 = ranks_df2['Rank_2'].corr(ranks_df2['Rank_0_subset'], method='pearson')

print(f"Weighted Pearson r = {r_weighted2:.3f}, R² = {r2_weighted2:.3f}")
print(f"Unweighted Pearson r = {r_unweighted2:.3f}")

#  Plot 
fig, ax = plt.subplots(figsize=(6,6))

# Scatter with size proportional to normalized weights
sizes = ranks_df2['Weight_norm'] * 4000
sc = ax.scatter(
    ranks_df2['Rank_2'], ranks_df2['Rank_0_subset'],
    s=sizes, c='#1f77b4', alpha=0.7, edgecolor='black', linewidth=0.6
)

# 1-1 line
ax.plot([-0.5, 48], [-0.5, 48], 'k--', alpha=0.7, linewidth=1)

# Axes limits and labels
ax.set_xlim(-0.5, 48)
ax.set_ylim(-0.5, 48)
ax.set_xticks(range(1, 48, 2))
ax.set_yticks(range(1, 48, 2))
ax.set_xlabel('Rank in 2-C')
ax.set_ylabel('Rank in 1-CL')
ax.grid(alpha=0.4, linestyle='--')

# Label top 3 features by weight
top_n = 3
for _, row in ranks_df2.nlargest(top_n, 'Weight_norm').iterrows():
    ax.text(row['Rank_2'] + 0.15, row['Rank_0_subset'], row['Feature_nice'],
            fontsize=9, va='center')

# Size legend
size_values_full  = np.linspace(ranks_df2['Weight_norm'].min(), ranks_df2['Weight_norm'].max(), 4)
size_values = size_values_full[1:]  # skip minimum
size_values = np.round(size_values, 3)
marker_sizes = size_values * 4000

for val, size in zip(size_values, marker_sizes):
    ax.scatter([], [], s=size, color='#1f77b4', alpha=0.7, edgecolor='k', label=f'{val:.3f}')

# Place horizontal legend below
fig.legend(
    title="Normalized weight of mean Feature Importance",
    loc='upper center',
    bbox_to_anchor=(0.5, 0.05),
    ncol=len(size_values),
    labelspacing=1.2,
    frameon=False
)

#%% Ranking for 4-VLS
# Sort both importance DataFrames by descending importance
importance_df_0_sorted = importance_df_0.sort_values('Median Importance', ascending=False)
importance_df_4_sorted = importance_df_4.sort_values('Median Importance', ascending=False)

# Merge and compute mean importance and normalized weights
merged_importance4 = pd.merge(
    importance_df_0_sorted[['Feature', 'Median Importance']],
    importance_df_4_sorted[['Feature', 'Median Importance']],
    on='Feature', how='inner', suffixes=('_0', '_4'))

merged_importance4['Mean_importance'] = (merged_importance4['Median Importance_0'] + merged_importance4['Median Importance_4']) / 2
merged_importance4['Weight_norm'] = (merged_importance4['Mean_importance'] / merged_importance4['Mean_importance'].sum())

# Rank order mapping
features4 = importance_df_4_sorted['Feature'].tolist()
features0 = importance_df_0_sorted['Feature'].tolist()

# keep only overlapping features, preserve df0 order
features0_ordered_subset = [f for f in features0 if f in set(features4)]

# mapping: feature → rank
rank4_map = {feature: rank for rank, feature in enumerate(features4, start=1)}
rank0_subset_map = {feature: rank for rank, feature in enumerate(features0_ordered_subset, start=1)}

# Combine into ranking dataframe
ranks_df4 = pd.DataFrame({'Feature': features4})
ranks_df4['Rank_4'] = ranks_df4['Feature'].map(rank4_map)
ranks_df4['Rank_0_subset'] = ranks_df4['Feature'].map(rank0_subset_map)
ranks_df4 = ranks_df4.dropna(subset=['Rank_0_subset']).copy()
ranks_df4['Rank_0_subset'] = ranks_df4['Rank_0_subset'].astype(int)

# Merge with weights
ranks_df4 = ranks_df4.merge(
    merged_importance4[['Feature', 'Weight_norm']],
    on='Feature', how='left'
)

# Compute weighted and unweighted correlations
r_weighted4 = weighted_pearson(ranks_df4['Rank_4'], ranks_df4['Rank_0_subset'], ranks_df4['Weight_norm'])
r2_weighted4 = r_weighted4**2
r_unweighted4 = ranks_df4['Rank_4'].corr(ranks_df4['Rank_0_subset'], method='pearson')

print(f"Weighted Pearson r = {r_weighted4:.3f}, R² = {r2_weighted4:.3f}")
print(f"Unweighted Pearson r = {r_unweighted4:.3f}")

# ---------- Plot ----------
# Assume you already have a list of nice names matching the feature order
 #df_display['PrettyName'] = df_display['Feature'].map(feature_name_mapping).fillna(df_display['Feature'])

#nice_name_map = dict(zip(importance_df_0_sorted['Feature'], nice_names))
ranks_df4['Feature_nice'] = ranks_df4['Feature'].map(feature_name_mapping)

fig, ax = plt.subplots(figsize=(6,6))

sizes = ranks_df4['Weight_norm'] * 4000
sc = plt.scatter(
    ranks_df4['Rank_4'], ranks_df4['Rank_0_subset'],
    s=sizes, c='#A6D854', alpha=0.7, edgecolor='black', linewidth=0.6
)

plt.plot([-0.5, 8], [-0.5, 8], 'k--', alpha=0.7, linewidth=1)
plt.xlim(-0.5, 8)
plt.ylim(-0.5, 8)
plt.yticks(range(1, 8, 1))
plt.xticks(range(1, 8, 1))
plt.xlabel('Rank in 4-VLC')
plt.ylabel('Rank in 1-CL')
plt.grid(alpha=0.4, linestyle='--')

# Label top features by weight
top_n = 3
for _, row in ranks_df4.nlargest(top_n, 'Weight_norm').iterrows():
    plt.text(row['Rank_4'] + 0.15, row['Rank_0_subset'], row['Feature_nice'],
             fontsize=9, va='center')

## Size legend
# --- Create a horizontal size legend ---
import matplotlib.lines as mlines

# Define representative sizes
size_values = np.linspace(ranks_df4['Weight_norm'].min(), ranks_df4['Weight_norm'].max(), 4)
size_values = np.round(size_values, 3)
marker_sizes = size_values * 2500

# Add markers below plot
for i, (val, size) in enumerate(zip(size_values, marker_sizes)):
    plt.scatter([], [], s=size, color='#A6D854', alpha=0.7, edgecolor='k',
                label=f'{val:.3f}')

# Place legend below figure (centered horizontally)
fig.legend(
    title="Normalized weight of mean Feature Importance",
    loc='upper center',
    bbox_to_anchor=(0.5, -0.0009),
    ncol=len(size_values),
    labelspacing=1.2,
    frameon=False
)

plt.tight_layout()
colors = [
    "#159895",  # teal (lightseagreen adjusted; slightly bluer)
    "#1f77b4",  # tab:blue (unchanged)
    "#2E8B57",  # sea green (for darker vegetation; less red)
    "#A6D854",  # lime-ish green (lighter vegetation; shifted toward yellow)
    "#8B4513",  # saddlebrown (unchanged, soil)
    "#9EC1CF"   # slightly darker lightsteelblue for better contrast
]
#%% Ranking 5-SGT
# Sort both importance DataFrames by descending importance
importance_df_0_sorted = importance_df_0.sort_values('Median Importance', ascending=False)
importance_df_5_sorted = importance_df_5.sort_values('Median Importance', ascending=False)

# Merge and compute mean importance and normalized weights
merged_importance5 = pd.merge(
    importance_df_0_sorted[['Feature', 'Median Importance']],
    importance_df_5_sorted[['Feature', 'Median Importance']],
    on='Feature', how='inner', suffixes=('_0', '_5'))

merged_importance5['Mean_importance'] = (
    merged_importance5['Median Importance_0'] + merged_importance5['Median Importance_5']) / 2
merged_importance5 ['Weight_norm'] = (
    merged_importance5['Mean_importance'] / merged_importance5['Mean_importance'].sum())

# Rank order mapping
features5 = importance_df_5_sorted['Feature'].tolist()
features0 = importance_df_0_sorted['Feature'].tolist()

# keep only overlapping features, preserve df0 order
features0_ordered_subset = [f for f in features0 if f in set(features5)]

# mapping: feature → rank
rank5_map = {feature: rank for rank, feature in enumerate(features5, start=1)}
rank0_subset_map = {feature: rank for rank, feature in enumerate(features0_ordered_subset, start=1)}

# Combine into ranking dataframe
ranks_df5 = pd.DataFrame({'Feature': features5})
ranks_df5['Rank_5'] = ranks_df5['Feature'].map(rank5_map)
ranks_df5['Rank_0_subset'] = ranks_df5['Feature'].map(rank0_subset_map)
ranks_df5 = ranks_df5.dropna(subset=['Rank_0_subset']).copy()
ranks_df5['Rank_0_subset'] = ranks_df5['Rank_0_subset'].astype(int)

# Merge with weights
ranks_df5 = ranks_df5.merge(
    merged_importance5[['Feature', 'Weight_norm']],
    on='Feature', how='left'
)

# Compute weighted and unweighted correlations
r_weighted5 = weighted_pearson(ranks_df5['Rank_5'], ranks_df5['Rank_0_subset'], ranks_df5['Weight_norm'])
r2_weighted5 = r_weighted5**2
r_unweighted5 = ranks_df5['Rank_5'].corr(ranks_df5['Rank_0_subset'], method='pearson')

print(f"Weighted Pearson r = {r_weighted5:.3f}, R² = {r2_weighted5:.3f}")
print(f"Unweighted Pearson r = {r_unweighted5:.3f}")

# ---------- Plot ----------
# Assume you already have a list of nice names matching the feature order
 #df_display['PrettyName'] = df_display['Feature'].map(feature_name_mapping).fillna(df_display['Feature'])

#nice_name_map = dict(zip(importance_df_0_sorted['Feature'], nice_names))
ranks_df5['Feature_nice'] = ranks_df5['Feature'].map(feature_name_mapping)

fig, ax = plt.subplots(figsize=(6,6))

sizes = ranks_df5['Weight_norm'] * 4000
sc = plt.scatter(
    ranks_df5['Rank_5'], ranks_df5['Rank_0_subset'],
    s=sizes, c='#8B4513', alpha=0.7, edgecolor='black', linewidth=0.6
)

plt.plot([-0.5, 22], [-0.5, 22], 'k--', alpha=0.7, linewidth=1)
plt.xlim(-0.5, 22)
plt.ylim(-0.5, 22)
plt.yticks(range(1, 22, 1))
plt.xticks(range(1, 22, 1))
plt.xlabel('Rank in 5-SGT')
plt.ylabel('Rank in 1-CL')
plt.grid(alpha=0.4, linestyle='--')

# Label top features by weight
top_n = 3
for _, row in ranks_df5.nlargest(top_n, 'Weight_norm').iterrows():
    plt.text(row['Rank_5'] + 0.15, row['Rank_0_subset'], row['Feature_nice'],
             fontsize=9, va='center')

## Size legend
# --- Create a horizontal size legend ---
import matplotlib.lines as mlines

# Define representative sizes
size_values = np.linspace(ranks_df5['Weight_norm'].min(), ranks_df5['Weight_norm'].max(), 4)
size_values = np.round(size_values, 3)
marker_sizes = size_values * 2500

# Add markers below plot
for i, (val, size) in enumerate(zip(size_values, marker_sizes)):
    plt.scatter([], [], s=size, color='#8B4513', alpha=0.7, edgecolor='k',
                label=f'{val:.3f}')

# Place legend below figure (centered horizontally)
fig.legend(
    title="Normalized weight of mean Feature Importance",
    loc='upper center',
    bbox_to_anchor=(0.5, -0.0009),
    ncol=len(size_values),
    labelspacing=1.2,
    frameon=False
)

plt.tight_layout()
colors = [
    "#159895",  # teal (lightseagreen adjusted; slightly bluer)
    "#1f77b4",  # tab:blue (unchanged)
    "#2E8B57",  # sea green (for darker vegetation; less red)
    "#A6D854",  # lime-ish green (lighter vegetation; shifted toward yellow)
    "#8B4513",  # saddlebrown (unchanged, soil)
    "#9EC1CF"   # slightly darker lightsteelblue for better contrast
]
#%% Ranking 6-A
#  Main analysis 
# Sort both importance DataFrames by descending importance
importance_df_0_sorted = importance_df_0.sort_values('Median Importance', ascending=False)
importance_df_6_sorted = importance_df_6.sort_values('Median Importance', ascending=False)

# Merge and compute mean importance and normalized weights
merged_importance6 = pd.merge(
    importance_df_0_sorted[['Feature', 'Median Importance']],
    importance_df_6_sorted[['Feature', 'Median Importance']],
    on='Feature', how='inner', suffixes=('_0', '_6'))

merged_importance6['Mean_importance'] = (
    merged_importance6['Median Importance_0'] + merged_importance6['Median Importance_6']) / 2
merged_importance6['Weight_norm'] = (
    merged_importance6['Mean_importance'] / merged_importance6['Mean_importance'].sum())

# Rank order mapping
features6 = importance_df_6_sorted['Feature'].tolist()
features0 = importance_df_0_sorted['Feature'].tolist()

# keep only overlapping features, preserve df0 order
features0_ordered_subset = [f for f in features0 if f in set(features6)]

# mapping: feature → rank
rank6_map = {feature: rank for rank, feature in enumerate(features6, start=1)}
rank0_subset_map = {feature: rank for rank, feature in enumerate(features0_ordered_subset, start=1)}

# Combine into ranking dataframe
ranks_df6 = pd.DataFrame({'Feature': features6})
ranks_df6['Rank_6'] = ranks_df6['Feature'].map(rank6_map)
ranks_df6['Rank_0_subset'] = ranks_df6['Feature'].map(rank0_subset_map)
ranks_df6 = ranks_df6.dropna(subset=['Rank_0_subset']).copy()
ranks_df6['Rank_0_subset'] = ranks_df6['Rank_0_subset'].astype(int)

# Merge with weights
ranks_df6 = ranks_df6.merge(
    merged_importance6[['Feature', 'Weight_norm']],
    on='Feature', how='left'
)

# Compute weighted and unweighted correlations
r_weighted6 = weighted_pearson(ranks_df6['Rank_6'], ranks_df6['Rank_0_subset'], ranks_df6['Weight_norm'])
r2_weighted6 = r_weighted6**2
r_unweighted6 = ranks_df6['Rank_6'].corr(ranks_df6['Rank_0_subset'], method='pearson')

print(f"Weighted Pearson r = {r_weighted6:.3f}, R² = {r2_weighted6:.3f}")
print(f"Unweighted Pearson r = {r_unweighted6:.3f}")

# ---------- Plot ----------
# Assume you already have a list of nice names matching the feature order
 #df_display['PrettyName'] = df_display['Feature'].map(feature_name_mapping).fillna(df_display['Feature'])

#nice_name_map = dict(zip(importance_df_0_sorted['Feature'], nice_names))
ranks_df6['Feature_nice'] = ranks_df6['Feature'].map(feature_name_mapping)

plt.figure(figsize=(6,6))

sizes = ranks_df6['Weight_norm'] * 4000
sc = plt.scatter(
    ranks_df6['Rank_6'], ranks_df6['Rank_0_subset'],
    s=sizes, c='#9EC1CF', alpha=0.7, edgecolor='black', linewidth=0.6
)

plt.plot([-0.5, 6], [-0.5, 6], 'k--', alpha=0.7, linewidth=1)
plt.xlim(-0.5, 6)
plt.ylim(-0.5, 6)
plt.yticks(range(1, 6, 1))
plt.xticks(range(1, 6, 1))
plt.xlabel('Rank in 6-A')
plt.ylabel('Rank in 1-CL')
plt.grid(alpha=0.4, linestyle='--')

# Label top features by weight
top_n = 3
for _, row in ranks_df6.nlargest(top_n, 'Weight_norm').iterrows():
    plt.text(row['Rank_6'] + 0.15, row['Rank_0_subset'], row['Feature_nice'],
             fontsize=9, va='center')

## Size legend
# --- Create a horizontal size legend ---
import matplotlib.lines as mlines

# Define representative sizes
size_values = np.linspace(ranks_df6['Weight_norm'].min(), ranks_df6['Weight_norm'].max(), 4)
size_values = np.round(size_values, 3)
marker_sizes = size_values * 2000

# Add markers below plot
for i, (val, size) in enumerate(zip(size_values, marker_sizes)):
    plt.scatter([], [], s=size, color='#9EC1CF', alpha=0.7, edgecolor='k',
                label=f'{val:.3f}')

# Place legend below figure (centered horizontally)
plt.legend(
    title="Normalised weight of mean Feature Importance",
    loc='upper center',
    bbox_to_anchor=(0.5, -0.12),
    ncol=len(size_values),
    labelspacing=1.2,
    frameon=False
)

plt.tight_layout()
colors = [
    "#159895",  # teal (lightseagreen adjusted; slightly bluer)
    "#1f77b4",  # tab:blue (unchanged)
    "#2E8B57",  # sea green (for darker vegetation; less red)
    "#A6D854",  # lime-ish green (lighter vegetation; shifted toward yellow)
    "#8B4513",  # saddlebrown (unchanged, soil)
    "#9EC1CF"   # slightly darker lightsteelblue for better contrast
]

#%%Ranking 3-VLC
# Sort both importance DataFrames by descending importance
importance_df_0_sorted = importance_df_0.sort_values('Median Importance', ascending=False)
importance_df_3_sorted = importance_df_3.sort_values('Median Importance', ascending=False)

# Merge and compute mean importance and normalized weights
merged_importance3 = pd.merge(
    importance_df_0_sorted[['Feature', 'Median Importance']],
    importance_df_3_sorted[['Feature', 'Median Importance']],
    on='Feature', how='inner', suffixes=('_0', '_3')
)
merged_importance3['Mean_importance'] = (
    merged_importance3['Median Importance_0'] + merged_importance3['Median Importance_3']
) / 2
merged_importance3['Weight_norm'] = (
    merged_importance3['Mean_importance'] / merged_importance3['Mean_importance'].sum()
)

# Rank order mapping
features3 = importance_df_3_sorted['Feature'].tolist()
features0 = importance_df_0_sorted['Feature'].tolist()

# keep only overlapping features, preserve df0 order
features0_ordered_subset = [f for f in features0 if f in set(features3)]

# mapping: feature → rank
rank3_map = {feature: rank for rank, feature in enumerate(features3, start=1)}
rank0_subset_map = {feature: rank for rank, feature in enumerate(features0_ordered_subset, start=1)}

# Combine into ranking dataframe
ranks_df3 = pd.DataFrame({'Feature': features3})
ranks_df3['Rank_3'] = ranks_df3['Feature'].map(rank3_map)
ranks_df3['Rank_0_subset'] = ranks_df3['Feature'].map(rank0_subset_map)
ranks_df3 = ranks_df3.dropna(subset=['Rank_0_subset']).copy()
ranks_df3['Rank_0_subset'] = ranks_df3['Rank_0_subset'].astype(int)

# Merge with weights
ranks_df3 = ranks_df3.merge(
    merged_importance3[['Feature', 'Weight_norm']],
    on='Feature', how='left'
)

# Compute weighted and unweighted correlations
r_weighted3 = weighted_pearson(ranks_df3['Rank_3'], ranks_df3['Rank_0_subset'], ranks_df3['Weight_norm'])
r2_weighted3 = r_weighted3**2
r_unweighted3 = ranks_df3['Rank_3'].corr(ranks_df3['Rank_0_subset'], method='pearson')

print(f"Weighted Pearson r = {r_weighted3:.3f}, R² = {r2_weighted3:.3f}")
print(f"Unweighted Pearson r = {r_unweighted3:.3f}")

# ---------- Plot ----------
# Assume you already have a list of nice names matching the feature order
 #df_display['PrettyName'] = df_display['Feature'].map(feature_name_mapping).fillna(df_display['Feature'])

#nice_name_map = dict(zip(importance_df_0_sorted['Feature'], nice_names))
ranks_df3['Feature_nice'] = ranks_df3['Feature'].map(feature_name_mapping)

plt.figure(figsize=(6,6))

sizes = ranks_df3['Weight_norm'] * 4000
sc = plt.scatter(
    ranks_df3['Rank_3'], ranks_df3['Rank_0_subset'],
    s=sizes, c='#2E8B57', alpha=0.7, edgecolor='black', linewidth=0.6
)

plt.plot([0.5, 12], [0.5, 12], 'k--', alpha=0.7, linewidth=1)
plt.xlim(0.5, 12)
plt.ylim(0.5, 12)
plt.yticks(range(1, 12, 1))
plt.xticks(range(1, 12, 1))
plt.xlabel('Rank in 3-VLC')
plt.ylabel('Rank in 1-CL')
plt.grid(alpha=0.4, linestyle='--')

# Label top features by weight
top_n = 3
for _, row in ranks_df3.nlargest(top_n, 'Weight_norm').iterrows():
    plt.text(row['Rank_3'] + 0.15, row['Rank_0_subset'], row['Feature_nice'],
             fontsize=9, va='center')

## Size legend
# --- Create a horizontal size legend ---
import matplotlib.lines as mlines

# Define representative sizes
size_values = np.linspace(ranks_df3['Weight_norm'].min(), ranks_df3['Weight_norm'].max(), 4)
size_values = np.round(size_values, 3)
marker_sizes = size_values * 4000

# Add markers below plot
for i, (val, size) in enumerate(zip(size_values, marker_sizes)):
    plt.scatter([], [], s=size, color='#2E8B57', alpha=0.7, edgecolor='k',
                label=f'{val:.3f}')

# Place legend below figure (centered horizontally)
plt.legend(
    title="Normalised weight of mean Feature Importance",
    loc='upper center',
    bbox_to_anchor=(0.5, -0.12),
    ncol=len(size_values),
    labelspacing=1.2,
    frameon=False
)

plt.title(f"Weighted R² = {r2_weighted3:.3f}")
plt.tight_layout()
#%% Plotted all together
dfs = [ranks_df2, ranks_df3, ranks_df4, ranks_df5, ranks_df6]
colors = ['#1f77b4', '#2E8B57', '#A6D854', '#8B4513', '#9EC1CF']
xlabels = ['Rank in 2-C', 'Rank in 3-VLC', 'Rank in 4-VLS', 'Rank in 5-SGT', 'Rank in 6-A']
xlims = [(-0.5,48), (0.5,12), (-0.5,8), (-0.5,22), (-0.5,6)]
xticks_list = [range(1,48,2), range(1,12,1), range(1,8,1), range(1,22,1), range(1,6,1)]
letters = ['a)', 'b)', 'c)', 'd)', 'e)']
r2_list = [r2_weighted2, r2_weighted3, r2_weighted4, r2_weighted5, r2_weighted6]


fig, axes = plt.subplots(1,5, figsize=(25,5), sharey=False)

for i, (df, color, xlabel, xlim_vals, xticks, letter) in enumerate(zip(dfs, colors, xlabels, xlims, xticks_list, letters)):
    ax = axes[i]
    
    # Scatter points
    sizes = df['Weight_norm'] * 2000
    ax.scatter(df[f'Rank_{i+2}'], df['Rank_0_subset'],
               s=sizes, c=color, alpha=0.7, edgecolor='black', linewidth=0.6)

        # 1-1 line
    ax.plot([xlim_vals[0], xlim_vals[1]], [xlim_vals[0], xlim_vals[1]], 'k--', alpha=0.7, linewidth=1)

    # Axes limits and labels
    xrange = xlim_vals[1] - xlim_vals[0]
    ax.set_xlim(xlim_vals[0], xlim_vals[1])  # add padding for legend dots
    ax.set_ylim(xlim_vals[0], xlim_vals[1])
    ax.set_xticks(xticks)
    ax.set_yticks(xticks)
    ax.set_xlabel(xlabel)
    if i == 0:
        ax.set_ylabel('Rank in 1-CL')
        ax.grid(alpha=0.4, linestyle='--')

    # Subplot title with letter + weighted R²
    ax.set_title(f"{letter}  Weighted R² = {r2_list[i]:.3f}", loc='left', fontweight='bold')
    
    # Top features
    top_n = 3
    for _, row in df.nlargest(top_n,'Weight_norm').iterrows():
        ax.text(row[f'Rank_{i+2}']+0.15, row['Rank_0_subset'], row['Feature_nice'],
                fontsize=9, va='center')
    
    # Subplot letter
    
    ax.set_title(f"{letter}  Weighted R² = {r2_list[i]:.3f}", loc='left', fontweight='bold')

    # --- Horizontal size legend per subplot ---
    size_values_full = np.linspace(df['Weight_norm'].min(), df['Weight_norm'].max(), 4)
    size_values = size_values_full[1:]  # skip min
    marker_sizes = size_values * 2000
    
    for val, size in zip(size_values, marker_sizes):
        ax.scatter([], [], s=size, color=color, alpha=0.7, edgecolor='k', label=f'{val:.3f}')
    
    # Position legend just below the subplot
    ax.legend(title="Norm. weight", loc='upper center',
              bbox_to_anchor=(0.5, -0.15), ncol=len(size_values),
              frameon=False, labelspacing=1.2)
    
plt.tight_layout()
fig.subplots_adjust(bottom=0.25)  # leave space for all legends
plt.show()
#%% Plotted together


# Example list of rank DataFrames and colors for the 5 experiments
rank_dfs = [ranks_df2, ranks_df3, ranks_df4, ranks_df5, ranks_df6]
colors = ['#1f77b4', 'darkgreen', '#A6D854', '#8B4513', '#9EC1CF']
xlims = [(-0.5,48), (0.5,12), (-0.5,8), (-0.5,22), (-0.5,6)]
xticks_list = [range(1,48,3), range(1,12,1), range(1,8,1), range(1,22,2), range(1,6,1)]
letters = ['(a)', '(b)', '(c)', '(d)', '(e)']
titles = ['2-C', '3-VLC', '4-VLS', '5-SGT', '6-A']
r_squared_list = [r_weighted2, r_weighted3, r_weighted4, r_weighted5, r_weighted6]

fig = plt.figure(figsize=(14, 16))
gs = fig.add_gridspec(3, 2, height_ratios=[1,1,1], hspace=0.6, wspace=0.3)

for i, (df, color, letter, title, r2, xlim, xticks) in enumerate(
    zip(rank_dfs, colors, letters, titles, r_squared_list, xlims, xticks_list)):

    # Select axes
    if i < 4:  # first 4 plots: 2 per row
        ax = fig.add_subplot(gs[i//2, i%2])
    else:  # last plot: first column of third row
        ax = fig.add_subplot(gs[2, 0])

    # Scatter
    sizes = df['Weight_norm'] * 2000
    sc = ax.scatter(df.iloc[:,1], df.iloc[:,2], s=sizes, c=color,
                    alpha=0.7, edgecolor='black', linewidth=0.6)

    # 1-1 line
    max_rank = max(df.iloc[:,1].max(), df.iloc[:,2].max()) + 1
    ax.plot([xlim[0], xlim[1]], [xlim[0], xlim[1]], 'k--', alpha=0.7, linewidth=1)

    # Axes limits and ticks
    ax.set_xlim(xlim)
    ax.set_ylim(xlim)
    ax.set_xticks(xticks)
    ax.set_yticks(xticks)
    ax.set_xlabel(f'Rank in {title}', fontsize=12)
    ax.set_ylabel('Rank in 1-CL', fontsize=12)
    ax.grid(alpha=0.4, linestyle='--')
    ax.tick_params(axis='both', which='major', labelsize=12)
    # Label top 3 features by weight
    #top_n = 3
    #for _, row in df.nlargest(top_n, 'Weight_norm').iterrows():
     #   ax.text(row.iloc[1] + 0.15, row.iloc[2], row['Feature_nice'], fontsize=11, va='center')

    # Horizontal legend for sizes (skip minimum)
    size_values_full = np.linspace(df['Weight_norm'].min(), df['Weight_norm'].max(), 4)
    size_values = size_values_full[1:]
    marker_sizes = size_values * 2000
    for val, msize in zip(size_values, marker_sizes):
        ax.scatter([], [], s=msize, color=color, alpha=0.7, edgecolor='k', label=f'{val:.3f}')

    # Place horizontal legend below each subplot
    ax.legend(title="Normalised weight", fontsize=12, loc='upper center',
              bbox_to_anchor=(0.5, -0.18), ncol=len(size_values),
              frameon=False, labelspacing=1.2)

    # Title with letter and weighted R²
    ax.set_title(f"{letter} Weighted R = {r2:.3f}", loc='left', fontsize=13)

plt.tight_layout()
plt.savefig(r'U:\Hydrological Data\EStreams\Figures\1st paper\Version 2\FI_rankingscatter_v2_notext_10C_V2.pdf',bbox_inches='tight', dpi=300)
plt.savefig(r'U:\Hydrological Data\EStreams\Figures\1st paper\Version 2\FI_rankingscatter_v2_notext_10C_V2.png',bbox_inches='tight', dpi=300)


