# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 11:32:53 2025

@author: juliarudlang
"""

import geopandas as gpd                                      # Pandas for geospatial analysis
from shapely.geometry import Point, Polygon                  # Module used for geospatial analysis     
import pymannkendall as mk                                   # Module used for trend-computation
from plotly.offline import plot
#import contextily as cx
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
import seaborn as sns
import matplotlib as mpl
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib
from mpl_toolkits.axes_grid1 import inset_locator
import geopandas as gpd                                      # Pandas for geospatial analysis
from shapely.geometry import Point, Polygon                  # Module used for geospatial analysis     
import pymannkendall as mk                                   # Module used for trend-computation
from plotly.offline import plot
#import contextily as cx
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings                                              
import datetime                                              # Datetime module pretty useful for time-series analysis
from utils.utils import check_data
from utils.utils import calculate_hydro_year
from utils.utils import calculate_season
from scipy import stats
from tqdm import tqdm                                     # Pandas for geospatial analysis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import copy
from pathlib import Path
import plotly.express as px
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from kneed import KneeLocator
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import matplotlib.colors as mcolors
from scipy.interpolate import interp1d

#%%
# Load in timeseries subset in mm/day
timeseries_EU = pd.read_csv(r'U:\Hydrological Data\EStreams\Data 071025\timeseries_EU_mmd_071025.csv', index_col=0) 
timeseries_EU.index = pd.to_datetime(timeseries_EU.index)
timeseries_EU.index.name = ""

#%% Load climate and landscape attributes dataframe
file_path = 'U:\Hydrological Data\EStreams\Data 071025\climate_landscape_071025.csv'

# Read the file
climate_landscape = pd.read_csv(file_path)
climate_landscape.set_index("basin_id", inplace=True)
climate_landscape
#%% Load hydro signatures dataframe
hydro_sign = pd.read_csv(r'U:\Hydrological Data\EStreams\Data 071025\hydro_sign_clustered_101025.csv', encoding='utf-8')
hydro_sign.set_index("basin_id", inplace=True)
hydro_sign

#%% Define visually nicer names for the figures
signature_nice_names = {
    "q_mean": r"$H_{{\mathrm{Q}}}$",
    "Q_var": r"$H_{\mathrm{V, \,Q}}$",
    "slope_sawicz": r"$H_{\mathrm{FDC}}$",
    "var_slope_sawicz": r"$H_{\mathrm{V,\,FDC}}$",
    "baseflow_index": r"$H_{\mathrm{BFI}}$",
    "hfd_mean": r"$H_{{\mathrm{t(HFD)}}}$",
    "hfd_var": r"$H_{\mathrm{V,\, t(HFD)}}$",
    'hfd_sin': r'$H_{\mathrm{t(HFD)_{WS}}}$',
    'hfd_cos': r'$H_{\mathrm{t(HFD)_{SA}}}$',
    "q_5": r"$H_{\mathrm{Q_5}}$",
    "q_95": r"$H_{\mathrm{Q_{95}}}$",
    "q5_var": r"$H_{\mathrm{V,\,{Q_{5}}}}$",
    "q95_var": r"$H_{\mathrm{V,\, {Q_{95}}}}$",
    "hq_freq": r"$H_{\mathrm{f(Q_{H})}}$",
    "hq_dur": r"$H_{\mathrm{D(Q_{H})}}$",
    "lq_freq": r"$H_{\mathrm{f(Q_{L})}}$",
    "lq_dur": r"$H_{\mathrm{D(Q_{L})}}$",
    "hq_freq_var": r"$H_{\mathrm{V,\, f(Q_{H})}}$",
    "hq_dur_var": r"$H_{\mathrm{V,\, D(Q_{H})}}$",
    "lq_freq_var": r"$H_{\mathrm{V,\,f(Q_{L})}}$",
    "lq_dur_var": r"$H_{\mathrm{V,\, D(Q_{L})}}$",
    "hq_day": r"$H_{\mathrm{t(Q_{H})}}$",
    "lq_day": r"$H_{\mathrm{t(Q_{L})}}$",
    'hq_sin': r'$H_{\mathrm{t({Q}_{H})_{WS}}}$',
    'hq_cos': r'$H_{\mathrm{t({Q}_{H})_{SA}}}$',
    'lq_sin': r'$H_{\mathrm{t({Q}_{L})_{WS}}}$',
    'lq_cos': r'$H_{\mathrm{t({Q}_{L})_{SA}}}$',
    "Hq_doy_circular_var": r"$H_{\mathrm{V, \,t(Q_{H})}}$",
    "Lq_doy_circular_var": r"$H_{\mathrm{V,\, t(Q_{L})}}$",
    "days_between_hq_lq": r"$H_{\mathrm{N(Q_{H}–Q_{L})}}$",
    "circ_var_days_between_hq_lq": r"$H_{\mathrm{V, \,N(Q_{H}–\text{Q}_{L})}}$",
    "zero_q_freq": r"$H_{\mathrm{f(Q_{Z})}}$",
    "qzero_var": r"$H_{\mathrm{V,\, f(Q_{Z})}}$",
    "rld": r"$H_{\mathrm{RLD}}$",
    "fld": r"$H_{\mathrm{FLD}}$",
    "Flashiness_index": r"$H_{\mathrm{RBI}}$",
    "gini_mean": r"$H_{{\mathrm{GC}}}$",
    "gini_var": r"$H_{\mathrm{V,\, GC}}$",
    "cv": r"$H_{\mathrm{CV}}$",
    "Parde_range": r"$H_{\mathrm{R(Parde)}}$",
    "Parde_range_var": r"$H_{\mathrm{V,\, R(Parde)}}$",
    "autocorr_lag1": r"$H_{\mathrm{AC_{1}}}$",
    "autocorr_lag30": r"$H_{\mathrm{AC_{30}}}$",
    "std_slope_sawicz": r"$H^{\sigma}_{\mathrm{FDC}}$",
    "Q_std": r"$H^{\sigma}_{\mathrm{Q}}$",
    "Hq_doy_circular_std": r"$H^{\sigma}_{\mathrm{t(Q_{H})}}$",
    "Lq_doy_circular_std": r"$H^{\sigma}_{\mathrm{t(Q_{L})}}$",
    "circ_std_days_between_hq_lq": r"$H^{\sigma}_{\mathrm{N(Q_{H}–Q_{L})}}$",
    "Parde_range_std": r"$H^{\sigma}_{\mathrm{R(Parde)}}$",
    "hq_dur_std": r"$H^{\sigma}_{\mathrm{D(Q_{H})}}$",
    "gini_std": r"$H^{\sigma}_{\mathrm{GC}}$",
    "hq_freq_std": r"$H^{\sigma}_{\mathrm{f(Q_{H})}}$",
    "lq_dur_std": r"$H^{\sigma}_{\mathrm{D(Q_{L})}}$",
    "lq_freq_std": r"$H^{\sigma}_{\mathrm{f(Q_{L})}}$",
    "q95_std": r"$H^{\sigma}_{\mathrm{Q_{95}}}$",
    "q5_std": r"$H^{\sigma}_{\mathrm{Q_{5}}}$",
    "qzero_std": r"$H^{\sigma}_{\mathrm{f(Q_{Z})}}$",
    "hfd_std": r"$H^{\sigma}_{\mathrm{HFD}}$"
}

#%% Create a colour scheme for the 10 HRTs for all figures
color_mapping2 = ['#1f77b4',  # Blue")
                '#e377c2',  # Magenta")
                 '#9467bd', # purple")
                  '#2ca02c',  # Green")
                 '#17becf',  # Cyan")
                 '#c51b8a',  # Pink")
                 '#aec7e8',  # Light blue")
                 '#ff7f0e',  # Orange ")
                  '#005a32',  # Dark green")
                  '#d62728'  # Red")
                 ]
                  
                  
                  
# Create a dictionary to map cluster labels (0-9) to colors in color_mapping2
cluster_color_map = {cluster: color for cluster, color in enumerate(color_mapping2)}

# Print the cluster-color mapping to know which cluster gets which color
for cluster, color in cluster_color_map.items():
    print(f'Cluster {cluster} -> Color {color}')
    
    
#%% Main map pot of all HRTs
import matplotlib.ticker as mticker
from matplotlib.cm import ScalarMappable
longitudes = hydro_sign.lon_snap
latitudes = hydro_sign.lat_snap
cluster = hydro_sign['C_k10_AllS_NOClimate_remapped']
# Total number of catchments
total_catchments = 7175

                 
# Create a colormap directly from your fixed colors
cmap_sr = ListedColormap(color_mapping2[:10])  # Adjust the slice if fewer colors are needed
bounds = np.linspace(0, 10, 11)
norm = mpl.colors.BoundaryNorm(bounds, cmap_sr.N)
cluster_order = [7, 5, 0, 4, 8, 3, 9,6, 2, 1]

fig = plt.figure(figsize=(7, 7), dpi=600)  # width ~12 cm
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cartopy.feature.OCEAN, zorder=100, facecolor='powderblue', alpha=0.2, edgecolor='k', linewidth=0.0008)
ax.set_global()
ax.set_xlim(-28, 35)
ax.set_ylim(34, 72)

for c in cluster_order:
    mask = cluster == c
    ax.scatter(longitudes[mask], latitudes[mask], 
               c=[color_mapping2[c]], 
               s=1.3, edgecolors='face', linewidths=0.05)
    
# Inset colorbar below the map
sc = plt.cm.ScalarMappable(cmap=cmap_sr, norm=norm)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
cax = inset_axes(ax,
                 width="143%",  # width relative to axes (adjust as needed)
                 height="4%",  # thickness
                 loc='lower center',
                 bbox_to_anchor=(0.15, -0.075, 0.7, 1),  # left, bottom, width, height relative to ax
                 bbox_transform=ax.transAxes)

cbar = plt.colorbar(sc, cax=cax, orientation='horizontal')
cbar.set_ticks(np.arange(0.5, 10.5, 1))
cbar.set_ticklabels(np.arange(1, 11))
cbar.ax.tick_params(labelsize=8)
cbar.set_label('HRT')
for spine in cbar.ax.spines.values():
    spine.set_linewidth(0.5)


# ADD THIS SECTION FOR THE HISTOGRAM INSET:
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

inset_ax = inset_axes(ax, width="12%", height="35%", loc='lower left',
                      bbox_to_anchor=(0.09, 0.05, 1, 1), bbox_transform=ax.transAxes)

sns.countplot(x='C_k10_AllS_NOClimate_remapped', data=hydro_sign,
              palette=color_mapping2[:10], ax=inset_ax)

for spine in inset_ax.spines.values():
    spine.set_linewidth(0.5)   # thinner box lines

inset_ax.set_xlabel('', fontsize=6)
inset_ax.set_ylabel('$n$ catchments', fontsize=7)#$n$
inset_ax.set_yticks(np.arange(200, 1600, 400))
inset_ax.tick_params(axis='both', labelsize=8)

inset_ax.set_xticks([])
inset_ax.set_xticklabels([])

# Secondary y-axis for percentage
def counts_to_percent(count):
    return (count / total_catchments) * 100

def percent_to_counts(percent):
    return percent * total_catchments / 100
# Secondary y-axis
secax = inset_ax.secondary_yaxis('right', functions=(counts_to_percent, percent_to_counts))
secax.tick_params(axis='y', labelsize=8)
secax.set_yticks(np.arange(5, 105, 5))
secax.set_yticklabels([f'{int(p)}%' for p in np.arange(5, 105, 5)])
# Thinner spines for the secondary y-axis
for spine in secax.spines.values():
    spine.set_linewidth(0.5)


plt.savefig(r'U:\Hydrological Data\EStreams\Figures\1st paper\Version 2\map_10cluster.pdf',bbox_inches='tight',dpi=300)
plt.savefig(r'U:\Hydrological Data\EStreams\Figures\1st paper\Version 2\map_10cluster.png',bbox_inches='tight',dpi=300)   
#%% Plotting and saving for individual HRTs
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
import numpy as np
import matplotlib as m


longitudes = hydro_sign.lon_snap
latitudes = hydro_sign.lat_snap
cluster = hydro_sign['C_k10_AllS_NOClimate_remapped']

# Total number of catchments
total_catchments = 7175

# Colormap for the 10 clusters
cmap_sr = ListedColormap(color_mapping2[:10])
bounds = np.linspace(0, 10, 11)
norm = BoundaryNorm(bounds, cmap_sr.N)

# Cluster order (largest first)
cluster_order = [7, 5, 0, 4, 8, 3, 9,6, 2, 1]

for c_highlight in cluster_order:
    fig = plt.figure(figsize=(4, 4), dpi=300)  # smaller figure
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cartopy.feature.OCEAN, zorder=100, facecolor='powderblue', alpha=0.2, edgecolor='k', linewidth=0.05)
    ax.set_global()
    ax.set_xlim(-28, 35)
    ax.set_ylim(34, 72)

    # Only plot the highlighted cluster
    mask = cluster == c_highlight
    ax.scatter(longitudes[mask], latitudes[mask],
               c=[color_mapping2[c_highlight]], s=2.4, edgecolors='face', linewidths=0.05)

    # Histogram inset
    inset_ax = inset_axes(ax, width="15%", height="33%", loc='lower left',
                          bbox_to_anchor=(0.095, 0.06, 1, 1), bbox_transform=ax.transAxes)

    # Highlighted cluster in color, others in grey
    colors_hist = ['lightgrey'] * 10
    colors_hist[c_highlight] = color_mapping2[c_highlight]

    sns.countplot(x='C_k10_AllS_NOClimate_remapped', data=hydro_sign,
                  palette=colors_hist, ax=inset_ax)

    inset_ax.set_xlabel('', fontsize=4.5)  # smaller font
    inset_ax.set_ylabel('', fontsize=6)
    inset_ax.tick_params(axis='both', labelsize=7.5)
    inset_ax.set_yticks(np.arange(200, 1600, 400))
    inset_ax.set_xticks([])
    inset_ax.set_xticklabels([])
    for spine in inset_ax.spines.values():
        spine.set_linewidth(0.5)   # thinner box lines

    ax.set_title(f"HRT {c_highlight + 1}", loc='left', fontsize=8)  # smaller title
    plt.tight_layout()

    # Save figure
    plt.savefig(f"U:/Hydrological Data/EStreams/Figures/1st paper/Version 2/clustermap_{c_highlight}.png", dpi=300, bbox_inches='tight')

#%% make subset for k-means clustering plot
features = ['q_mean', 'slope_sawicz', 'var_slope_sawicz', 'baseflow_index', 'hfd_sin', 'hfd_cos','q_5', 'q_95', 'hq_freq', 'hq_dur', 'lq_freq', 'lq_dur', 'zero_q_freq', 'cv', 'gini_mean', 'Parde_range', 'autocorr_lag1', 'autocorr_lag30', 'rld', 'hq_sin', 'hq_cos', 'lq_sin', 'lq_cos', 'days_between_hq_lq', 'fld', 'Flashiness_index', 'Q_var', 'Hq_doy_circular_var', 'Lq_doy_circular_var', 'circ_var_days_between_hq_lq', 'Parde_range_var', 'gini_var','hq_dur_var', 'hq_freq_var', 'lq_dur_var', 'lq_freq_var', 'q95_var', 'q5_var', 'qzero_var', 'hfd_var']
subsss = hydro_sign[features]

# make  copy of subset
df_subset = subsss.copy()

# fill subset
df_filled = df_subset.fillna(0)
 

# Scale the data
scaler = MinMaxScaler()
X_std = scaler.fit_transform(df_filled)

#%% Elbow method and silhouette score
# Assuming X_std is already defined
inertia = []
silhouette_scores = []
K = range(2, 21)

for k in K:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X_std)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_std, kmeans.labels_))

# Set figure size: width = 12 cm (≈ 4.72 in), height ~ 3 in for 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(6, 3), dpi=300)

# Common plot style
plot_kwargs = {'marker': 'x', 'linestyle': '-', 'color': 'tab:blue', 'linewidth': 1.5, 'markersize': 5}

# Elbow plot
axes[0].plot(K, inertia, **plot_kwargs)
axes[0].set_xlabel('Number of HRTs', fontsize=11)
axes[0].set_xlim(2, 20)
axes[0].set_xticks(range(2, 21, 2))
axes[0].set_ylabel('Inertia (-)', fontsize=11)
axes[0].set_title('a)', fontsize=12, loc='left')
axes[0].tick_params(axis='both', which='major', labelsize=10, width=1.0, length=6)

# Silhouette score plot
axes[1].plot(K, silhouette_scores, **plot_kwargs)
axes[1].set_xlabel('Number of HRTs', fontsize=11)
axes[1].set_xlim(2, 20)
axes[1].set_xticks(range(2, 21, 2))
axes[1].set_ylabel('Silhouette Score (-)', fontsize=11)
axes[1].set_title('b)', fontsize=12, loc='left')
axes[1].tick_params(axis='both', which='major', labelsize=10, width=1.0, length=6)

plt.tight_layout()
plt.savefig(r"U:\Hydrological Data\EStreams\Figures\1st paper\Version 2\kmeans_elbow_silhouette.pdf", bbox_inches='tight', dpi=300)


#%% Define subset of the full dataframe
signatures = ['q_mean', 'slope_sawicz', 'var_slope_sawicz', 'baseflow_index', 'hfd_mean','q_5', 'q_95', 'hq_freq', 'hq_dur', 'lq_freq', 'lq_dur', 'zero_q_freq', 'cv', 'gini_mean', 'Parde_range', 'autocorr_lag1', 'autocorr_lag30', 'rld', 'hq_day', 'lq_day', 'days_between_hq_lq', 'fld', 'Flashiness_index', 'Q_var', 'Hq_doy_circular_var', 'Lq_doy_circular_var', 'circ_var_days_between_hq_lq', 'Parde_range_var', 'gini_var','hq_dur_var', 'hq_freq_var', 'lq_dur_var', 'lq_freq_var', 'q95_var', 'q5_var', 'qzero_var', 'hfd_var']
signatures2 = ['q_mean', 'slope_sawicz',  'baseflow_index', 'hfd_mean','q_5', 'q_95', 'hq_freq', 'hq_dur', 'lq_freq', 'lq_dur', 'hq_day', 'lq_day', 'days_between_hq_lq','zero_q_freq',  'cv', 'gini_mean', 'Parde_range', 'autocorr_lag1', 'autocorr_lag30', 'rld', 'fld', 'Flashiness_index', 'Q_std','std_slope_sawicz','hfd_std', 'Hq_doy_circular_std', 'Lq_doy_circular_std', 'circ_std_days_between_hq_lq', 'hq_freq_std','hq_dur_std', 'lq_dur_std', 'lq_freq_std', 'q95_std', 'q5_std', 'qzero_std','Parde_range_std', 'gini_std']


#%% The subsets we acutally use
subset = hydro_sign[signatures]
hydro_sing_clustered = subset.copy()
#%% Map the signaure units to the signature
signature_units = {
    'q_mean' : '(mm d$^{-1}$)',
    'slope_sawicz' : '(–)',
    'baseflow_index': '(–)',
    'hfd_mean': '(doy)',

    'q_5' : '(mm d$^{-1}$)',
    'q_95' : '(mm d$^{-1}$)',

    'hq_freq': '(d yr$^{-1}$)',
    'hq_dur' : '(d)',
    'lq_freq' : '(d yr$^{-1}$)',
    'lq_dur' : '(d)',

    'hq_day' : '(doy)',
    'lq_day' : '(doy)',
    'days_between_hq_lq' : '(d)',

    'zero_q_freq' : '(d yr$^{-1}$)',

    'cv' : '(–)',
    'gini_mean' : '(–)',
    'Parde_range' : '(–)',

    'autocorr_lag1' : '(–)',
    'autocorr_lag30' : '(–)',

    'rld' : '(d$^{-1}$)',
    'fld' : '(d$^{-1}$)',
    'Flashiness_index' : '(–)',

    'Q_std': '(mm d$^{-1}$)',
    'std_slope_sawicz' : '(–)',

    'hfd_std' : '(doy)',
    'Hq_doy_circular_std' : '(doy)',
    'Lq_doy_circular_std' : '(doy)',
    'circ_std_days_between_hq_lq' : '(d)',

    'hq_freq_std' : '(d yr$^{-1}$)',
    'hq_dur_std' : '(d)',
    'lq_dur_std' : '(d)',
    'lq_freq_std' : '(d yr$^{-1}$)',

    'q95_std' : '(mm d$^{-1}$)',
    'q5_std' : '(mm d$^{-1}$)',
    'qzero_std' : '(d yr$^{-1}$)',

    'Parde_range_std': '(–)',
    'gini_std': '(–)'
}
#%% Boxplot for all hydrological signatures. 
import math
import matplotlib.pyplot as plt
import seaborn as sns

n_vars = 37
n_cols = 4
n_rows = math.ceil(n_vars / n_cols)

fig, axes = plt.subplots(
    n_rows, n_cols, figsize=(28, 33),
    constrained_layout=False
)

axes = axes.flatten()

for i, signature in enumerate(signatures2[:n_vars]):
    nice_name = signature_nice_names.get(signature, signature)
    unit = signature_units.get(signature, '')

    # --- Boxplot ---
    sns.boxplot(
        x='Cluster_1_to_10', y=signature, data=hydro_sign,
        palette=color_mapping2, ax=axes[i],
        width=0.8, showfliers=False, showcaps=False, linewidth=0.6
    )

    # x axis ticks
    if i == 0:
        axes[i].set_xticklabels(range(1, 11))
        axes[i].tick_params(axis='x', labelsize=20)
    else:
        axes[i].set_xticklabels([])

    # REMOVE default x- and y-axis labels
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")

   
    #  y axis custom labels

    # Position of the y-label block (closer to y-axis)
    NAME_X = -0.32   # main label
    UNIT_X = -0.22     # unit label (further left)

    # Add the signature name
    axes[i].text(
        NAME_X, 0.5, nice_name,
        transform=axes[i].transAxes,
        rotation=90, va='center', ha='center',
        fontsize=24
    )

    # Add unit, with more vertical space between labels
    axes[i].text(
        UNIT_X, 0.5, f"{unit}",
        transform=axes[i].transAxes,
        rotation=90, va='center', ha='center',
        fontsize=16
    )

    # Style ticks
    axes[i].tick_params(axis='both', labelsize=20, length=6, width=1)

# Remove unused axes
for j in range(n_vars, len(axes)):
    fig.delaxes(axes[j])

# More horizontal space for y-labels
plt.subplots_adjust(hspace=0.35, wspace=0.47)

#Save figure
plt.savefig(r'U:\Hydrological Data\EStreams\Figures\1st paper\Version 2\boxplot_10C.pdf', bbox_inches='tight', dpi=300)

#%% Make  copy of the timeseries and hydrological signatures dataframe
#hydro_sign.columns # optional check for column names

hydro_sign_C = hydro_sign.copy()

timeseries_EU_copy = timeseries_EU.copy()

#%% Precompute Mean monthly Q
# use group by index month mean or median
monthly_mean = timeseries_EU_copy.groupby(timeseries_EU_copy.index.month).mean()
#monthly_median = timeseries_EU_copy.groupby(timeseries_EU_copy.index.month).median()
#%% Clusters ( for correct labeling from 1-10 and nor 0-9)
CK10 = hydro_sign['Cluster_1_to_10']
clusters = np.sort(CK10.unique())

#%% Pre-calculating the flow duration curve
n_exceedance_points = 100
exceedance_common = np.linspace(0, 1, n_exceedance_points)

cluster_fdcs = {}  # dictionary to store mean FDCs and optionally individual interpolated flows

for i, cluster in enumerate(clusters):
    cluster_stations = CK10[CK10 == cluster].index
    all_interpolated_flows = []
    
    for station in cluster_stations:
        data = timeseries_EU_copy[station].dropna()
        sorted_flows = np.sort(data)[::-1]
        exceedence = np.arange(1., len(sorted_flows) + 1) / len(sorted_flows)
        interp_func = interp1d(exceedence, sorted_flows, bounds_error=False, fill_value="extrapolate")
        interpolated_flows = interp_func(exceedance_common)
        all_interpolated_flows.append(interpolated_flows)
    
    cluster_fdcs[cluster] = {
        'mean': np.mean(all_interpolated_flows, axis=0),
        'all': all_interpolated_flows  # optional: only if you want to plot individual curves
    }

#%% Precomputing the high and low flows
bin_edges = np.linspace(0, 366, 31)
cluster_hq_lq_hist = {}

for cluster in clusters:
    cluster_stations = CK10[CK10 == cluster].index
    hq_days = hydro_sign_C.loc[cluster_stations, 'hq_day'].dropna()
    lq_days = hydro_sign_C.loc[cluster_stations, 'lq_day'].dropna()
    
    hq_hist, _ = np.histogram(hq_days, bins=bin_edges)
    lq_hist, _ = np.histogram(lq_days, bins=bin_edges)
    
    cluster_hq_lq_hist[cluster] = {
        'hq': hq_hist,
        'lq': lq_hist
    }


#%% Signatures and boxplots
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
from matplotlib.ticker import LogLocator
from matplotlib.ticker import MaxNLocator

# Boxplot features and nicer labels
boxplot_features = [
    'baseflow_index', 'gini_mean', 'autocorr_lag30', 
    'rld', 'Flashiness_index', 'days_between_hq_lq'
]

signature_nice_names2 = {
    'baseflow_index': r"$H_{\mathrm{BFI}}$",
    'gini_mean': r"$H_{{\mathrm{GC}}}$",
    'days_between_hq_lq': r"$H_{\mathrm{N(Q_{H}–Q_{L})}}$",
    'autocorr_lag30': r"$H_{\mathrm{AC}_{30}}$",
    'rld': r"$H_{\mathrm{RLD}}$",
    'Flashiness_index': r"$H_{\mathrm{RBI}}$",
}

signature_units = {
    'baseflow_index':  "(-)",
    'gini_mean': "(-)",
    'days_between_hq_lq': "(d)",
    'autocorr_lag30': "(-)",
    'rld': "(d$^{-1}$)",
    'Flashiness_index': "(-)"
}

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['mathtext.default'] = 'regular'

# Global parameters
global_max_qmean = 10
global_min_fdc = 1e-2
global_max_fdc = 1e2

# Ticks
bin_edges = np.linspace(0, 366, 31)
day_ticks = [0, 152, 335]
day_labels = ['Jan', 'Jul', 'Dec']
month_ticks = [1, 6, 12]
month_labels = ['Jan', 'Jul', 'Dec']

#Font settings
label_fontsize = 18
tick_fontsize = 16
title_fontsize = 26
title_fontsize1 = 22

# Figure layout
fig = plt.figure(figsize=(28, 26))
gs = gridspec.GridSpec(
    7, 1, figure=fig,
    height_ratios=[1, 1, 1, 1, 1, 0.08, 1.2]  # Bottom row taller for boxplots
)

# Top grid (4 rows × n_clusters columns)
gs_top = gridspec.GridSpecFromSubplotSpec(4, len(clusters), subplot_spec=gs[:5, 0])

# Plot hydro signatures
for i, cluster in enumerate(clusters):
    cluster_stations = CK10[CK10 == cluster].index
    color = color_mapping2[i]

    # Row 1: Monthly mean Q 
    ax = fig.add_subplot(gs_top[0, i])
    for station in cluster_stations:
        ax.plot(monthly_mean[station], c=color, alpha=0.2)
    cluster_mean = monthly_mean[cluster_stations].mean(axis=1)
    ax.plot(cluster_mean, c='black', linewidth=1)
    ax.set_ylim(0, global_max_qmean)
    ax.set_title(f'HRT {cluster}', fontsize=title_fontsize1)
    ax.set_xticks(month_ticks)
    ax.set_xticklabels(month_labels)
    ax.tick_params(axis='x', labelsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)

    if i == 0:
        ax.set_ylabel('$H_{Q}$', fontsize=title_fontsize, labelpad=35)
        # Smaller unit label below
        ax.text(-0.35, 0.5, "(mm d$^{-1}$)", transform=ax.transAxes, rotation=90, va='center', ha='center',fontsize=18)
        ax.set_xlabel('Month', fontsize=label_fontsize)


    # Row 2: Flow Duration Curve (precomputed)
    ax = fig.add_subplot(gs_top[1, i])
    for interpolated_flows in cluster_fdcs[cluster]['all']:
        ax.plot(exceedance_common, interpolated_flows, c=color, alpha=0.2, linewidth=1)
    ax.plot(exceedance_common, cluster_fdcs[cluster]['mean'], c='black', linewidth=1)
    ax.semilogy()
    ax.set_ylim(global_min_fdc, global_max_fdc)
    ax.yaxis.set_major_locator(LogLocator(numticks=3))
    ax.tick_params(length=7, width=1)
    ax.tick_params(axis='x', labelsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["0", "0.5", "1"])

    if i == 0:
        ax.set_ylabel('$H_{{FDC}}$', fontsize=title_fontsize, labelpad=20)
        ax.text(-0.38, 0.5, "(-)", transform=ax.transAxes, rotation=90, va='center', ha='center',fontsize=18)
        ax.set_xlabel('Exceedance P (-)', fontsize=label_fontsize)


    # Row 3: Timing of High Flows 
    ax = fig.add_subplot(gs_top[2, i])
    ax.bar(
        bin_edges[:-1],
        cluster_hq_lq_hist[cluster]['hq'],
        width=np.diff(bin_edges),
        color=color, edgecolor='black', alpha=0.7
    )
    ax.set_xlim(0, 366)
    ax.set_xticks(day_ticks)
    ax.set_xticklabels(day_labels)
    ax.tick_params(axis='x', labelsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5)) 

    if i == 0:
        ax.set_ylabel('$H_{t({Q}_{H})}$', fontsize=title_fontsize, labelpad=25)
        ax.text(-0.35, 0.5, "(n catchments)", transform=ax.transAxes, rotation=90, va='center', ha='center',fontsize=18)
        ax.set_xlabel('doy', fontsize=label_fontsize)


    #  Row 4: Timing of Low Flows 
    ax = fig.add_subplot(gs_top[3, i])
    ax.bar(
        bin_edges[:-1],
        cluster_hq_lq_hist[cluster]['lq'],
        width=np.diff(bin_edges),
        color=color, edgecolor='black', alpha=0.7
    )
    ax.set_xlim(0, 366)
    ax.set_xticks(day_ticks)
    ax.set_xticklabels(day_labels)
    ax.tick_params(axis='x', labelsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5)) 

    if i == 0:
        ax.set_ylabel('$H_{t({Q}_{L})}$', fontsize=title_fontsize, labelpad=25)
        ax.text(-0.35, 0.5, "(n catchments)", transform=ax.transAxes, rotation=90, va='center', ha='center',fontsize=18)
        ax.set_xlabel('doy', fontsize=label_fontsize)


#  Bottom row: 6 boxplots side by side
gs_bottom = gridspec.GridSpecFromSubplotSpec(
    1, len(boxplot_features),
    subplot_spec=gs[6, 0], wspace=0.5
)

for j, feature in enumerate(boxplot_features):

    ax = fig.add_subplot(gs_bottom[0, j])
    ax.set_clip_on(False)

    sns.boxplot(
        x='Cluster_1_to_10',
        y=feature,
        data=hydro_sign,
        palette=color_mapping2,
        ax=ax,
        showfliers=False,
        showcaps=False,
        width=0.6
    )

    # Get latex name and unit
    label_latex = signature_nice_names2.get(feature, feature)
    unit = signature_units.get(feature, "")

    # Remove default titles + y labels
    ax.set_title("")
    ax.set_ylabel("")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5)) 

    #  PLACE TITLE (LaTeX) ON Y AXIS 
    MAIN_X = -0.35      # closer to axis
    UNIT_X = -0.24      # unit slightly further left

    # Main signature name
    ax.text(
        MAIN_X, 0.5, label_latex,
        transform=ax.transAxes,
        rotation=90,
        va='center', ha='center',
        fontsize=title_fontsize
    )

    # Unit below it
    ax.text(
        UNIT_X, 0.5, unit,
        transform=ax.transAxes,
        rotation=90,
        va='center', ha='center',
        fontsize=label_fontsize
    )

    # Tick styling
    ax.tick_params(axis='x', labelsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)

    # No per-subplot x labels
    ax.set_xlabel("")


fig.text(0.5, 0.03, "HRT", ha='center', va='center', fontsize=22)

# Panel labels for the 4 rows (a–d)
fig.text(0.01, 0.965, "(a)", fontsize=26,  va='center')
fig.text(0.01, 0.78, "(b)", fontsize=26,  va='center')
fig.text(0.01, 0.59, "(c)", fontsize=26,  va='center')
fig.text(0.01, 0.4, "(d)", fontsize=26,  va='center')



# y-position for all boxplot labels (same height as the y-axis title)
y_box = 0.2     # adjust as needed to align with boxplot row

# x-positions for each boxplot panel label 
x_positions = [0.01, 0.18, 0.336, 0.5, 0.67, 0.833]   # one per boxplot

boxplot_labels = ["(e)", "(f)", "(g)", "(h)", "(i)", "(j)"]

for x, label in zip(x_positions, boxplot_labels):
    fig.text(x, y_box, label,
             fontsize=26,  ha='left', va='center')
    
# Final layout adjustments
plt.tight_layout(pad=1.2)
plt.subplots_adjust(
    left=0.05, right=0.98, top=0.97, bottom=0.06,
    hspace=0.2, wspace=0.38
)


plt.savefig(r'U:\Hydrological Data\EStreams\Figures\1st paper\Version 2\hydrologicalsignatures_andboxplot_qmean_10C_181125_V2.pdf', bbox_inches='tight', dpi=300)
#%%  Confuson matrix distribution of correlation with H and CL together in one plot

# hydrological signatures
df_numeric_H = hydro_sign.select_dtypes(include=[np.number])

#Compute correlation matrix
corr_matrix_H= df_numeric_H.corr()

# Extract all r values excluding the diagonal
r_values_H = corr_matrix_H.values[np.triu_indices_from(corr_matrix_H, k=1)]  # upper triangle, k=1 excludes diagonal

# Take absolute values
r_values_abs_H = np.abs(r_values_H)

#Climate and landscape attributes
# Ensure it's numerical by selecting only numeric columns
df_numeric_CL = climate_landscape.select_dtypes(include=[np.number])

# Compute correlation matrix
corr_matrix_CL= df_numeric_CL.corr()

#Extract all r values excluding the diagonal
r_values_CL = corr_matrix_CL.values[np.triu_indices_from(corr_matrix_CL, k=1)]  # upper triangle, k=1 excludes diagonal

# Take absolute values
r_values_abs_CL = np.abs(r_values_CL)


# Convert cm to inches for figure size (optional)
cm_to_inch = 1 / 2.54
fig_width = 17 * cm_to_inch  # ≈ 6.7 inches
fig_height = 8 * cm_to_inch  # ≈ 3.1 inches

#Plot
fig, axes = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)

# Panel (a): hydrological signatures
sns.boxplot(
    x=r_values_abs_H,
    color="powderblue",
    showfliers=False,
    showcaps=False,
    boxprops=dict(facecolor='powderblue', edgecolor='black', linewidth=1),
    whiskerprops=dict(color='black', linewidth=1.2),
    medianprops=dict(color='black', linewidth=1.5),
    ax=axes[0]
)
    
axes[0].set_title("(a) Hydrological signatures", loc='left', fontsize=12)
axes[0].set_xlabel("Correlation coefficient (|r|)")
axes[0].set_ylabel("")  # optional for cleaner layout


# Panel (b): Climate and landscape attributes
sns.boxplot(
    x=r_values_abs_CL,
    color="#159895",
    showfliers=False,
    showcaps=False,
    whiskerprops=dict(color='black', linewidth=1.2),
    medianprops=dict(color='black', linewidth=1.5),
    ax=axes[1]
)

axes[1].set_title("(b) Climate &Landscape attributes", loc='left', fontsize=12)
axes[1].set_xlabel("Correlation coefficient (|r|)")
axes[1].set_ylabel("")

plt.savefig(r'U:\Hydrological Data\EStreams\Figures\1st paper\Version 2\Correlation_H_CL_boxplot_V2.pdf',bbox_inches='tight', dpi=300)


#%% normalised mean and std of all signatures
signatures = ['q_mean', 'q_5', 'q_95','hq_freq', 'lq_freq','zero_q_freq','hq_dur',  'lq_dur', 'hfd_mean', 'hq_day', 'lq_day','days_between_hq_lq', 'slope_sawicz', 'rld',  'fld', 'Flashiness_index','gini_mean', 'Parde_range',  'autocorr_lag1', 'autocorr_lag30', 'baseflow_index',  'cv',    'Q_var', 'q95_var', 'q5_var','hq_freq_var','lq_freq_var', 'qzero_var', 'hq_dur_var',  'lq_dur_var', 'hfd_var','Hq_doy_circular_var', 'Lq_doy_circular_var', 'circ_var_days_between_hq_lq','var_slope_sawicz',   'gini_var',  'Parde_range_var']

# font sizes 
#axis_labelsize = 22
#tick_labelsize = 20
#text_labelsize = 19

# Compute cluster means
cluster_means = hydro_sign.groupby('C_k10_AllS_NOClimate_remapped')[signatures].mean()

# Standardize
cluster_means_std = (cluster_means - cluster_means.mean()) / cluster_means.std()

# Replace column names with nice names
cluster_means_std.columns = [signature_nice_names.get(col, col) for col in cluster_means_std.columns]

# Change cluster index from 0–9 to 1–10
cluster_means_std.index = cluster_means_std.index + 1

# Plot heatmap
plt.figure(figsize=(12, 4))
sns.heatmap(
    cluster_means_std, 
    cmap='RdBu_r', 
    center=0,
    annot=False, 
    fmt='.2f',
    cbar_kws={'label': r"$\sigma(\text{normalised mean})$", 'shrink': 0.7, 'pad': 0.02}
)

# Axis labels and ticks
#plt.ylabel('Hydrological Response Type', fontsize=13, labelpad=10)
plt.ylabel('HRT', fontsize=13, labelpad=10)
plt.xlabel('', fontsize=14)  # optional: remove or customize
plt.xticks(rotation=90, fontsize=12)
plt.yticks(fontsize=12)

# Colorbar label size
cbar = plt.gca().collections[0].colorbar
cbar.ax.tick_params(labelsize=12)
cbar.set_ticks(np.arange(-2, 2.1, 1))
cbar.set_label(r"$\sigma \,(\text{normalised mean})$", fontsize=13)

plt.tight_layout()
plt.savefig(r'U:\Hydrological Data\EStreams\Figures\1st paper\Version 2\normalised_deviationmean_10C.pdf', bbox_inches='tight', dpi=300)

#%% PCA
features = ['q_mean', 'slope_sawicz', 'var_slope_sawicz', 'baseflow_index', 'hfd_sin', 'hfd_cos','q_5', 'q_95', 'hq_freq', 'hq_dur', 'lq_freq', 'lq_dur', 'zero_q_freq', 'cv', 'gini_mean', 'Parde_range', 'autocorr_lag1', 'autocorr_lag30', 'rld', 'hq_sin', 'hq_cos', 'lq_sin', 'lq_cos', 'days_between_hq_lq', 'fld', 'Flashiness_index', 'Q_var', 'Hq_doy_circular_var', 'Lq_doy_circular_var', 'circ_var_days_between_hq_lq', 'Parde_range_var', 'gini_var','hq_dur_var', 'hq_freq_var', 'lq_dur_var', 'lq_freq_var', 'q95_var', 'q5_var', 'qzero_var', 'hfd_var']

#%% PCA
lf = len(features)
print('length of features:', lf)

X = hydro_sign[features]
X = X.fillna(0)

X = StandardScaler().fit_transform(X) #standardize all values
X = pd.DataFrame(X,columns=features)

pca = PCA(n_components=lf)
components = pca.fit_transform(X)

loadings = pca.components_ #eigenvector, length represents variance var explains
 
# Create dataframe
pca_df = pd.DataFrame(data=components[:,0:2],columns=['PC1', 'PC2'])

#Scale
pca_df_scaled = pca_df.copy()
 
scaler_df = pca_df[['PC1', 'PC2']]
scaler = 1 / (scaler_df.max() - scaler_df.min()) 
 
for index in scaler.index:
    pca_df_scaled[index] *= scaler[index]

per_var  = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
per_var

#%% Loadings calculation pc1 and pc2
ld_df = pd.DataFrame(index=features, columns=['xs','ys'])
ld_df['xs'] = loadings[0]
ld_df['ys'] = loadings[1]
ld_df['len'] = np.sqrt(loadings[0]**2+loadings[1]**2)
ld_df

#%%Explained variance
# Get explained variance (%)
explained_var = pca.explained_variance_ratio_ * 100

# Create a DataFrame of loadings
loadings_df = pd.DataFrame(
    pca.components_.T,  # transpose so rows = signatures, cols = PCs
    index=features,
    columns=[f"PC{i+1}" for i in range(lf)]
)

# Add variance explained row at the bottom
loadings_df.loc["Explained Variance (%)"] = explained_var

# Save to CSV or Excel
#loadings_df.to_csv("PCA_loadings_streamflow_signatures.csv")
# or:
#loadings_df.to_excel(r"U:\Hydrological Data\EStreams\PCA_loadings_streamflow_signatures.xlsx")





#%% PCA

#Merge cluster info
pca_df_scaled['cluster'] = hydro_sign['C_k10_AllS_NOClimate_remapped'].values

# --- Compute mean and std for each cluster ---
cluster_stats = pca_df_scaled.groupby('cluster')[['PC1', 'PC2']].agg(['mean', 'std'])

# Flatten multiindex columns for convenience
cluster_stats.columns = ['PC1_mean', 'PC1_std', 'PC2_mean', 'PC2_std']

#%%
# Loadings (already have)
loadings_df = pd.DataFrame(
    pca.components_.T,
    index=features,
    columns=[f"PC{i+1}" for i in range(lf)]
)

# Explained variance as fraction
explained_var_frac = explained_var / 100  # e.g., 45% -> 0.45

# Weight loadings by explained variance
weighted_loadings = (loadings_df ** 2) * explained_var_frac  # broadcasting works

# Absolute cumulative contribution across PCs
cum_loadings = weighted_loadings.abs().cumsum(axis=1)


#%% For everything up to and including pc40

# Last column = total cumulative contribution
cum_total = cum_loadings.iloc[:, -1]

# Rank features (highest cumulative contribution = rank 1)
ranked_features = cum_total.sort_values(ascending=False)

# Fraction of cumulative contribution
cum_frac = ranked_features / ranked_features.sum()

# Optional: create a dataframe to store rank, cumulative fraction
rank_df = pd.DataFrame({
    'feature': ranked_features.index,
    'cum_contribution': ranked_features.values,
    'cum_fraction': cum_frac.values
})

rank_df['rank'] = np.arange(1, len(rank_df) + 1)

#%% Now only for PC1-PC13 since that explains above 80% of the variance
# Select only PC1–PC13
selected_pcs = [f"PC{i+1}" for i in range(13)]

# Subset loadings
loadings_subset = loadings_df[selected_pcs]

# Subset explained variance as fraction
explained_var_frac_subset = explained_var_frac[:13]  # first 13 PCs

weighted_loadings_subset = (loadings_subset ** 2 ) * explained_var_frac_subset

cum_loadings_subset = weighted_loadings_subset.cumsum(axis=1)

# Last column = total cumulative contribution for the 13 PCs
cum_total_subset = cum_loadings_subset.iloc[:, -1]

# Rank features
ranked_features_subset = cum_total_subset.sort_values(ascending=False)

# Fraction of total
cum_frac_subset = ranked_features_subset / ranked_features_subset.sum()

# Build a dataframe
rank_df_subset = pd.DataFrame({
    'feature': ranked_features_subset.index,
    'cum_contribution': ranked_features_subset.values,
    'cum_fraction': cum_frac_subset.values
})

rank_df_subset['rank'] = np.arange(1, len(rank_df_subset) + 1)

#%% PC1 and PC2 space with HRT avergaes

#  Common font sizes
axis_labelsize = 18
tick_labelsize = 16
text_labelsize = 17
arrow_head = 0.02  #head width for arrows

dpi_val = 300

# Panel a: Scatter + cluster means

fig, ax = plt.subplots(figsize=(12, 12)) # size 12, 10
ax.scatter(pca_df_scaled['PC1'], pca_df_scaled['PC2'],
           s=5, alpha=0.6, color='lightgrey')
for cluster_id, row in cluster_stats.iterrows():
    ax.errorbar(
        x=row['PC1_mean'], y=row['PC2_mean'],
        xerr=row['PC1_std'], yerr=row['PC2_std'],
        fmt='o', markersize=13,
        color=color_mapping2[cluster_id],
        ecolor=color_mapping2[cluster_id],
        elinewidth=3, capsize=5, alpha=0.9
    )
ax.set_xlabel(f"PC1 ({per_var[0]}%)", fontsize=axis_labelsize)
ax.set_ylabel(f"PC2 ({per_var[1]}%)", fontsize=axis_labelsize)
ax.tick_params(axis="both", which="major", labelsize=tick_labelsize)
ax.set_ylim(-0.2, 0.5)
ax.set_xlim(-0.2, 0.5)
xticks = np.arange(-0.2, 0.51, 0.1)
xticklabels = [""] + [f"{x:.1f}" for x in xticks[1:]]
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)

plt.tight_layout()
plt.savefig(r"U:\Hydrological Data\EStreams\Figures\1st paper\PCA\panel_a.png", dpi=dpi_val, bbox_inches='tight')
#%%
explained_var = pca.explained_variance_ratio_ * 100

# Cumulative sum
cumulative_var = np.cumsum(explained_var)
# Number of PCs
pcs = np.arange(1, len(explained_var) + 1)
pcs2 = np.arange(1, len(explained_var) + 2, 2)
#%%
#font sizes
axis_labelsize = 22
tick_labelsize = 20
text_labelsize = 19
arrow_head = 0.02  # head width for arrows


# Panel b: Explained variance

fig, ax = plt.subplots(figsize=(12, 8)) # og size 12, 6
ax.plot(pcs, cumulative_var, marker='o', color='steelblue', lw=2, label='Cumulative variance')
ax.bar(pcs, explained_var, alpha=0.3, color='lightskyblue', label='PC(n) Variance')
ax.set_xlabel('Principal Component (n)', fontsize=axis_labelsize)
ax.set_ylabel('Explained Variance (%)', fontsize=axis_labelsize)
ax.set_ylim(0, 105)
ax.set_xlim(0.5, 40.5)           # stop at 40
ax.set_xticks(np.arange(1, 41, 2))
#ax.set_xticks(pcs2)
ax.tick_params(axis='both', labelsize=tick_labelsize)
ax.legend(fontsize=16, loc='upper left')
plt.tight_layout()
plt.savefig(r"U:\Hydrological Data\EStreams\Figures\1st paper\PCA\panel_b.png", dpi=dpi_val, bbox_inches='tight')


#%%
#Parameters 
topN = 10
base_color = mcolors.to_rgb("royalblue")  # can change to 'indigo', 'royalblue', etc.

#Merge loadings with global rank info 
ld_ranked = ld_df.merge(
    rank_df_subset[['feature', 'rank']], 
    left_index=True, right_on='feature', how='left'
)

# Select top N based on PC1–PC2 length
top_loadings = ld_ranked.nlargest(topN, "len")

# Max rank for scaling
max_rank = rank_df_subset['rank'].max()

# Functions to scale color and linewidth by global rank 
def blue_shade_rank(rank_value):
    """Map global rank to blue intensity with exponential scaling."""
    if pd.isna(rank_value):
        rank_value = max_rank
    v = (max_rank - rank_value) / (max_rank - 1)  # invert rank: 1=top
    v = (np.exp(5*v) - 1) / (np.exp(5) - 1)       # exponential stretch
    r = 1 - (1 - base_color[0]) * v
    g = 1 - (1 - base_color[1]) * v
    b = 1 - (1 - base_color[2]) * v
    return (r, g, b)

def lw_scale_rank(rank_value):
    """Scale line width based on global rank."""
    if pd.isna(rank_value):
        rank_value = max_rank
    v = (max_rank - rank_value) / (max_rank - 1)
    v = (np.exp(5*v) - 1) / (np.exp(5) - 1) # 3 original, 8 is too much
    return 0.5 + 7 * v  # adjust max thickness if needed
#%%
from adjustText import adjust_text
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# font sizes 
axis_labelsize = 20
tick_labelsize = 18
text_labelsize = 19.5
arrow_head = 0.02  #head width for arrows

# Panel c: PCA loadings with arrows

fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(pca_df_scaled.PC1, pca_df_scaled.PC2, s=5, alpha=0.2, color='lightgrey')
for var in ld_df.index:
    ax.arrow(0, 0, ld_df.loc[var, "xs"], ld_df.loc[var, "ys"],
             color="lightgrey", alpha=0.5, lw=0.8,
             head_width=0.012, length_includes_head=True)

texts = []
for _, row in top_loadings.iterrows():
        rank_val = row['rank']
        ax.arrow(0, 0, row["xs"], row["ys"], color=blue_shade_rank(rank_val),
             alpha=0.9, lw=lw_scale_rank(rank_val), head_width=arrow_head, length_includes_head=True)
        texts.append(ax.text(row["xs"]+0.01, row["ys"]+0.01,
                         signature_nice_names.get(row["feature"], row["feature"]),
                         size=text_labelsize, weight='bold',
                         color='black', bbox=dict(facecolor="white", alpha=0.6, edgecolor="none")))
adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=1))

ax.set_xlabel(f"PC1 ({per_var[0]}%)", fontsize=axis_labelsize)
ax.set_ylabel(f"PC2 ({per_var[1]}%)", fontsize=axis_labelsize)
ax.tick_params(axis="both", which="major", labelsize=tick_labelsize)
ax.set_ylim(-0.3, 0.5)
ax.set_xlim(-0.3, 0.5)
xticks = np.arange(-0.3, 0.51, 0.1)
xticklabels = [""] + [f"{x:.1f}" for x in xticks[1:]]
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
plt.grid(alpha=0.3, zorder=0) 
plt.tight_layout()
plt.savefig(r"U:\Hydrological Data\EStreams\Figures\1st paper\PCA\panel_c.png", dpi=dpi_val, bbox_inches='tight')


#%%
# Panel d: Cumulative contribution 


# font sizes 
axis_labelsize = 22
tick_labelsize = 20
text_labelsize = 19
arrow_head = 0.02  # common head width for arrows

fig, ax = plt.subplots(figsize=(13, 8))
ax.scatter(rank_df_subset['rank'], rank_df_subset['cum_contribution'],
           s=50, color='royalblue', zorder=3)
for i, row in rank_df_subset.iterrows():
    x = row['rank']
    y = row['cum_contribution']
    label = signature_nice_names.get(row['feature'], row['feature'])
    offset = 0.00008
    va = 'top' if row['rank'] <= 35 else 'bottom'
    y_text = y - offset if va == 'top' else y + offset
    text_color = 'purple' if 'var' in row['feature'].lower() else 'black'
    font_weight = 'bold' if 'var' in row['feature'].lower() else 'normal'
    ax.text(x, y_text, label, ha='center', va=va, rotation=90,
            fontsize=text_labelsize, alpha=0.9, color=text_color, weight=font_weight)

ax.set_xlabel('Rank (-)', fontsize=axis_labelsize)
ax.set_ylabel('Explained variance R² (-)', fontsize=axis_labelsize)
ax.tick_params(axis="both", labelsize=tick_labelsize)
ax.set_xlim(0.2, 41)
ax.set_xticks(np.arange(1, 41, 2))
ax.grid(alpha=0.3, zorder=0)
plt.tight_layout()
plt.savefig(r"U:\Hydrological Data\EStreams\Figures\1st paper\PCA\panel_d.png", dpi=dpi_val, bbox_inches='tight')
