
#%% Load in packages
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
hydro_sign = pd.read_csv(r'U:\Hydrological Data\EStreams\Data 071025\hydro_sign_clustered_101025.csv', encoding='utf-8')
hydro_sign.set_index("basin_id", inplace=True)
hydro_sign

#%% Create a new clustering based on this: 
columns_to_cluster = ['q_mean', 'slope_sawicz', 'var_slope_sawicz', 'baseflow_index', 'hfd_sin', 'hfd_cos','q_5', 'q_95', 'hq_freq', 'hq_dur', 'lq_freq', 'lq_dur', 'zero_q_freq', 'cv', 'gini_mean', 'Parde_range', 'autocorr_lag1', 'autocorr_lag30', 'rld', 'hq_sin', 'hq_cos', 'lq_sin', 'lq_cos', 'days_between_hq_lq', 'fld', 'Flashiness_index', 'Q_var', 'Hq_doy_circular_var', 'Lq_doy_circular_var', 'circ_var_days_between_hq_lq', 'Parde_range_var', 'gini_var','hq_dur_var', 'hq_freq_var', 'lq_dur_var', 'lq_freq_var', 'q95_var', 'q5_var', 'qzero_var', 'hfd_var']
subset = hydro_sign[columns_to_cluster]
#%%%
df_subset = subset.copy()
#%% fill subset
df_filled = df_subset.fillna(0)
#%%
# Scale the data
scaler = MinMaxScaler()
X_std = scaler.fit_transform(df_filled)

#%%
# Step 3: Perform K-means clustering
inertia = []
silhouette_scores = []
K = range(2, 20)

for k in K:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X_std)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_std, kmeans.labels_))

# Plot the Elbow method to find the optimal k
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(K, inertia, 'x-',  color='tab:blue')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.xticks(np.arange(1, 20, 1))
plt.title('Elbow Method for Optimal k')

# Plot the silhouette scores
plt.subplot(1, 2, 2)
plt.plot(K, silhouette_scores, 'x-', color='tab:blue')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.xticks(np.arange(1, 20, 1))
plt.title('Silhouette Scores for Optimal k')

plt.tight_layout()
plt.show()


kl = KneeLocator(range(1, 19), inertia, curve="convex", direction="decreasing")
kl.elbow

#%%
hydro_sing_clustered = hydro_sign.copy()
#%%# run k means
# Can do this for 6, 7, 8, 9 and 10 clusters
kmeans = KMeans(n_clusters=10, random_state=42)
hydro_sing_clustered['C_k10_AllS_NOClimate'] = kmeans.fit_predict(X_std)
cluster_centers = kmeans.cluster_centers_
print("Original KMeans cluster centers (order as output by kmeans):")
print(cluster_centers)

#%% Re lbel the clustering according to the ascending slope of the flow duration curves
cluster_stats = hydro_sing_clustered.groupby('C_k10_AllS_NOClimate')['slope_sawicz'].mean().sort_values()
print("\nMedian slope_sawicz by original cluster:")
print(cluster_stats)

# Create mapping: old label -> new label (ordered by slope_sawicz ascending)
new_labels = {old_label: new_label for new_label, old_label in enumerate(cluster_stats.index.tolist(), start=0)}
print("\nMapping from old to new cluster labels:")
print(new_labels)

# Apply new mapping to the dataframe
hydro_sing_clustered['C_k10_AllS_NOClimate_remapped'] = hydro_sing_clustered['C_k10_AllS_NOClimate'].map(new_labels)


#%%
hydro_sing_clustered.to_csv() # save 