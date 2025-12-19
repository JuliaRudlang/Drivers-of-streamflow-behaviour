

# HYDROLOGICAL SIGNATURES
#%% Import packages
import geopandas as gpd                                      # Pandas for geospatial analysis
from shapely.geometry import Point, Polygon                  # Module used for geospatial analysis     
import pymannkendall as mk                                   # Module used for trend-computation
from plotly.offline import plot
import contextily as cx
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
from utils.utils import check_data
from utils.utils import calculate_hydro_year
from utils.utils import calculate_season
from scipy import stats

#%% Load in data
# Q Timeseries
# Load in timeseries subset in mm/day  
timeseries_EU = pd.read_csv(r'U:\Hydrological Data\EStreams\Data 071025\timeseries_EU_mmd_071025.csv', index_col=0) 
timeseries_EU.index = pd.to_datetime(timeseries_EU.index)
timeseries_EU.index.name = ""


# Network
network_EU = pd.read_csv(r'U:\Hydrological Data\EStreams\Data 071025\Network_071025.csv')
network_EU.set_index("basin_id", inplace = True)
network_EU

# Hydrosignatures 
hydro_sign = pd.read_csv(r'U:\Hydrological Data\EStreams\Data 071025\hydro_sign_clustered_101025.csv', encoding='utf-8')
hydro_sign.set_index("basin_id", inplace=True)
hydro_sign


#%%
timeseries_EU_copy = timeseries_EU.copy() # make a copy of the timeseries just i case
#%% Variance of Q
grouped_by_year = timeseries_EU_copy.groupby(timeseries_EU_copy.index.year).mean()

Q_var = grouped_by_year.var()
# Add calcluated signature to dataframe
hydro_sign[ "Q_var"] = Q_var
#%%
# Timing
#%%
# Calculate the mean of each day over all the years in the timeseries
day_mean = timeseries_EU_copy.groupby(timeseries_EU_copy.index.dayofyear).mean() # Need to use the mean or else the plots look a bit weird (cluster around 0-30 days)

#%% # Creating a new df to hold the results

hydro_signatures_timing = pd.DataFrame(index = network_EU.index, 
                                        columns = [ "hq_day", "hq_angle", "hq_sin", "hq_cos", "hq_day_avg",
                                                   "lq_day", "lq_angle", "lq_sin", "lq_cos", "lq_day_avg"
                                                  ])
#%% Timing of high and low flow for each catchment

for i in timeseries_EU.columns:
    lowest_flow_day = day_mean[i].idxmin()
    highest_flow_day = day_mean[i].idxmax()
    
    m = 365.25
    
    angle_hq = highest_flow_day * ((2 * np.pi) / m)
    angle_lq = lowest_flow_day * ((2 * np.pi) / m)
    
    x_hq = np.cos(angle_hq)
    y_hq = np.sin(angle_hq)
    
    x_lq = np.cos(angle_lq)
    y_lq = np.sin(angle_lq)
    
    # Calculate the average date D (Eq. (2))
    if x_hq > 0 and y_hq >= 0:
        D_hq = np.arctan2(y_hq , x_hq) * m / (2 * np.pi)
    elif x_hq <= 0:
        D_hq = (np.arctan2(y_hq, x_hq) * m / (2 * np.pi)) + np.pi
    elif x_hq > 0 and y_hq < 0:
        D_hq = (np.arctan2(y_hq, x_hq) * m / (2 * np.pi)) + (2 * np.pi)
        
        # Ensure the day of the year is within valid range
    if D_hq < 0:
        D_hq += m
    elif D_hq >= m:
        D_hq -= m
        
        
    # Calculate the average date D 
    if x_lq > 0 and y_lq >= 0:
        D_lq = np.arctan2(y_lq, x_lq) * m / (2 * np.pi)
    elif x_lq <= 0:
        D_lq = (np.arctan2(y_lq, x_lq) * m / (2 * np.pi)) + np.pi 
    elif x_lq > 0 and y_lq < 0:
        D_lq = (np.arctan2(y_lq, x_lq) * m / (2 * np.pi)) + (2 * np.pi) 
        
    if D_lq < 0:
        D_lq += m
    elif D_lq >= m:
        D_lq -= m
        
    # Calculate the concentration R 
    R_hq = np.sqrt(x_hq**2 + y_hq**2)
    R_lq = np.sqrt(x_lq**2 + y_lq**2)
    
    hydro_signatures_timing.loc[i, [ "hq_day", "hq_angle", "hq_sin", "hq_cos", "hq_day_avg",
               "lq_day", "lq_angle", "lq_sin", "lq_cos", "lq_day_avg"
              ]] = highest_flow_day, angle_hq, y_hq, x_hq, D_hq, lowest_flow_day, angle_lq, y_lq, x_lq, D_lq


#%% #Save everything to the Hydrosign df
hydro_sign[ "hq_day"] = hydro_signatures_timing[ "hq_day"]
hydro_sign[ "hq_angle"] = hydro_signatures_timing[ "hq_angle"]
hydro_sign["hq_sin"] = hydro_signatures_timing["hq_sin"]
hydro_sign["hq_cos"] = hydro_signatures_timing["hq_cos"]
hydro_sign["hq_day_avg"] = hydro_signatures_timing["hq_day_avg"]
hydro_sign["lq_day"] = hydro_signatures_timing["lq_day"]
hydro_sign["lq_angle"] = hydro_signatures_timing["lq_angle"]
hydro_sign["lq_sin"] = hydro_signatures_timing["lq_sin"] 
hydro_sign["lq_cos"] = hydro_signatures_timing["lq_cos"]
hydro_sign["lq_day_avg"] = hydro_signatures_timing["lq_day_avg"]



#%% DAYS BETWEEN HIGH AND LOW FLOW DAYS
hydro_signatures_timing_daybetween = pd.DataFrame(index = network_EU.index, 
                                        columns = [ "days_between_hq_lq"
                                                  ])
#%% days between hq and lq
# Define the total number of days in a year as 365.25

for i in timeseries_EU.columns:
    lowest_flow_day = day_mean[i].idxmin()
    highest_flow_day = day_mean[i].idxmax()
    
    m = 365.25
    # Convert day of year to radians (angular values)
    peak_flow_angle = (2 * np.pi / m) * highest_flow_day
    lowest_flow_angle = (2 * np.pi / m) * lowest_flow_day

    # Calculate the angular difference
    angular_difference = peak_flow_angle - lowest_flow_angle

    # Ensure angular difference is within the valid range [0, 2π]
    angular_difference = angular_difference % (2 * np.pi)

    # Convert the angular difference back to days
    days_betweenA = (angular_difference * m) / (2 * np.pi)

    # Adjust for circular nature (if any values are negative)
    days_betweenA = np.where(days_betweenA < 0, m + days_betweenA, days_betweenA)
    
    hydro_signatures_timing_daybetween.loc[i, [ "days_between_hq_lq"]] = days_betweenA

#%% Save to dataframe
hydro_sign["days_between_hq_lq"] = hydro_signatures_timing_daybetween["days_between_hq_lq"]

#%% Do the variance of the days between hq and lq
#%% In order to calculate the variance of the timing of high and low flows...
# Group the timeseries data by year
grouped_by_year = timeseries_EU_copy.groupby(timeseries_EU_copy.index.year)

# Find the date of the minimum flow for each year for each catchment (lowest flow day)
lowest_flow_days = grouped_by_year.idxmin()

# Find the date of the maximum flow for each year for each catchment (highest flow day)
highest_flow_days = grouped_by_year.idxmax()

# Extract the day of the year for the lowest flow days
lowest_flow_doy = lowest_flow_days.apply(lambda x: x.dt.dayofyear)

# Extract the day of the year for the highest flow days
highest_flow_doy = highest_flow_days.apply(lambda x: x.dt.dayofyear)

#%%
# Function to calculate circular variance and mean day (doy)

def circular_stats(days):
    # Convert days to angles (in radians), note: day can be considered discrete (1,...,365)
    angles = (days * 2 * np.pi) / 365.25
    sin_sum = np.mean(np.sin(angles))
    cos_sum = np.mean(np.cos(angles))
    mean_direction = np.arctan2(sin_sum, cos_sum)
    # Ensure positive angle
    if mean_direction < 0:
        mean_direction += 2*np.pi
    R = np.sqrt(sin_sum**2 + cos_sum**2)
    circ_variance = 1 - R
    mean_day = (mean_direction * 365.25) / (2 * np.pi)
    return circ_variance, mean_day

#%% For highest flow days
# Calculate circular variance for each catchment
data = highest_flow_doy
resultsH = {}

for catchment in data.columns:
    circ_var, mean_day = circular_stats(data[catchment])
    resultsH[catchment] = {'circular_variance': circ_var, 'mean_day': mean_day}

# Convert results to DataFrame
results_df_H = pd.DataFrame(resultsH).T
print('Circular stats for each catchment:')
print(results_df_H.head())

# Visualization
# We'll visualize the circular distribution for one catchment as an example, you can modify this for others.
example_catchment = data.columns[0]
angles = (data[example_catchment] * 2 * np.pi) / 365

plt.figure(figsize=(8, 8))
ax = plt.subplot(111, projection='polar')
ax.hist(angles, bins=50, color='teal', alpha=0.75)
ax.set_theta_direction(-1)  # clockwise
ax.set_theta_zero_location('N')  # 0 at the top
plt.title('Distribution of Highest Flow Days for ' + example_catchment)
plt.show()

print('done')
#%% Save data
hydro_sign['Hq_doy_circular_var'] = results_df_H['circular_variance']
#%% For lowest flow day
# Calculate circular variance for each catchment
data = lowest_flow_doy
resultsL = {}

for catchment in data.columns:
    circ_var, mean_day = circular_stats(data[catchment])
    resultsL[catchment] = {'circular_variance': circ_var, 'mean_day': mean_day}

# Convert results to DataFrame
results_df_L = pd.DataFrame(resultsL).T
print('Circular stats for each catchment:')
print(results_df_L.head())

# Visualisation example (to check that it looks alright)
example_catchment = data.columns[0]
angles = (data[example_catchment] * 2 * np.pi) / 365

plt.figure(figsize=(8, 8))
ax = plt.subplot(111, projection='polar')
ax.hist(angles, bins=50, color='teal', alpha=0.75)
ax.set_theta_direction(-1)  # clockwise
ax.set_theta_zero_location('N')  # 0 at the top
plt.title('Distribution of Lowest Flow Days for ' + example_catchment)
plt.show()

print('done')
#%% save results
hydro_sign['Lq_doy_circular_var'] = results_df_L['circular_variance']

#%% Variance of days between 
# Define circular difference function using angles

def calculate_circular_difference(day1, day2):
    # Convert to angles in radians
    angle1 = (day1 * 2 * np.pi) / 365.25
    angle2 = (day2 * 2 * np.pi) / 365.25
    diff = abs(angle1 - angle2)
    # minimal difference in radians
    diff = np.minimum(diff, 2*np.pi - diff)
    # convert back to days
    return (diff * 365.25) / (2 * np.pi)


#%%
# Calculate stats for each catchment
results = {}
for catchment in lowest_flow_doy.columns:
    # align high_flow and low_flow
    series_low = lowest_flow_doy[catchment]
    series_high = highest_flow_doy[catchment]
    # drop nan pairs
    df_temp = pd.concat([series_high, series_low], axis=1, keys=['high', 'low']).dropna()
    if df_temp.empty:
        results[catchment] = {'mean_days_between': np.nan, 'variance_days_between': np.nan}
    else:
        differences = calculate_circular_difference(df_temp['high'], df_temp['low'])
        results[catchment] = {
            'mean_days_between': differences.mean(),
            'variance_days_between': differences.var()
        }

results_df = pd.DataFrame(results).T

print("Summary of results (first few catchments):")
print(results_df.head())

#%% Save the data
hydro_sign['mean_days_between_hq_lq_circular'] = results_df['mean_days_between']
hydro_sign['variance_days_between_hq_lq_circular'] = results_df['variance_days_between']
#%%
print("Summary of results (first few catchments):")
print(results_df.head())

print("\Overall statistics:")
print(results_df.describe())

plt.figure(figsize=(10, 6))
plt.hist(results_df['mean_days_between'].dropna(), bins=30, color='blue', alpha=0.7)
plt.xlabel('Mean Days Between High and Low Flow')
plt.ylabel('Number of Catchments')
plt.title('Distribution of Mean Days Between High and Low Flow Across Catchments')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(results_df['mean_days_between'], results_df['variance_days_between'], alpha=0.5)
plt.xlabel('Mean Days Between High and Low Flow')
plt.ylabel('Variance of Days Between')
plt.title('Mean vs Variance of Days Between High and Low Flow')
plt.show()

print('done')
#%% PARDE RANGE SIGNATURE
# First we caculate the parde coefficient for each motnh
Q_mean_month = timeseries_EU_copy.groupby([timeseries_EU_copy.index.month]).mean()
QA = timeseries_EU_copy.mean()

PK = Q_mean_month / QA
#%% Calculate the Parde range, we need the min and max values

PK_max = PK.max()
PK_min = PK.min()

range_PK = PK_max - PK_min

# Calculate which month of the year the max PK occurs
PK_max_month = PK.idxmax()


# Add to the hydro signatures dataframe
hydro_sign['Parde_range'] = range_PK
hydro_sign['Parde_phase'] = PK_max_month

#%% Do the variance of the parde range over the years!

# Compute mean monthly flow for each year
Q_mean_month_yearly = timeseries_EU_copy.groupby([timeseries_EU_copy.index.year, timeseries_EU_copy.index.month]).mean()

# Compute mean annual flow for each year
Q_mean_yearly = timeseries_EU_copy.groupby(timeseries_EU_copy.index.year).mean()

# Normalize to get Parde coefficient for each year
PK_yearly = Q_mean_month_yearly.div(Q_mean_yearly, level=0)

# Compute min, max, and range for each year
PK_max_yearly = PK_yearly.groupby(level=0).max()
PK_min_yearly = PK_yearly.groupby(level=0).min()
range_PK_yearly = PK_max_yearly - PK_min_yearly

# Compute variance of the Parde range over years
variance_range_PK = range_PK_yearly.var()

print(variance_range_PK)

#%% Save data PK to the hydro sign df
hydro_sign['Parde_range_var'] = variance_range_PK
#%% AUTOCORRELATION
#%% Lag 1 Day

# Specify the maximum lag value you want to consider
max_lag = 1  # 1 day

# Initialise a DataFrame to store autocorrelation results
autocorr_df_lag1 = pd.DataFrame(index=range(1, max_lag + 1))  # Index for lags

# Loop over each column (station) in the original DataFrame
for column in timeseries_EU.columns:
    # Initialise lists to store lag values and corresponding autocorrelation values
    lags = []
    autocorrelations = []

    # Calculate autocorrelation for each lag from 1 to max_lag for the current station
    for lag in range(1, max_lag + 1):
        autocorr_value = timeseries_EU[column].autocorr(lag=lag)
        lags.append(lag)
        autocorrelations.append(autocorr_value)

    # Store autocorrelation values in the autocorr_df DataFrame with column name as station name
    autocorr_df_lag1[column] = autocorrelations

# Display the autocorr_df DataFrame containing autocorrelation values for all stations
autocorr_df_lag1


#%% Lag 30 Days

# Specify the maximum lag value you want to consider
max_lag = 30  # 30 days lag-time

# Initialize a DataFrame to store autocorrelation results
autocorr_df_lag30 = pd.DataFrame(index=range(30, max_lag + 1))  # Index for lags

# Loop over each column (station) in the original DataFrame
for column in timeseries_EU.columns:
    # Initialize lists to store lag values and corresponding autocorrelation values
    lags = []
    autocorrelations = []

    # Calculate autocorrelation for each lag from 1 to max_lag for the current station
    for lag in range(30, max_lag + 1):
        autocorr_value = timeseries_EU[column].autocorr(lag=lag)
        lags.append(lag)
        autocorrelations.append(autocorr_value)

    # Store autocorrelation values in the autocorr_df DataFrame with column name as station name
    autocorr_df_lag30[column] = autocorrelations

# Display the autocorr_df DataFrame containing autocorrelation values for all stations
autocorr_df_lag30

#%% The range between lag 1 and 30
#First transpoing the matrix
autocorr_df_lag1_T = autocorr_df_lag1.T

autocorr_df_lag30_T = autocorr_df_lag30.T

#%% Save autocorrelation variables into the hydro sign dataframe
hydro_sign['autocorr_lag1'] = autocorr_df_lag1_T

hydro_sign['autocorr_lag30'] = autocorr_df_lag30_T

hydro_sign['autocorr_range_1_30'] = hydro_sign['autocorr_lag1'] - hydro_sign['autocorr_lag30']

#%% Richard Bakers: FLASHINESS INDEX (RBI)
#%%
# Function to calculate RBI for a single catchment timeseries
def calculate_rbi(flow_series):
    # Calculate absolute differences between consecutive daily streamflows
    abs_diff = np.abs(flow_series.diff().dropna())
    
    # Total sum of absolute differences (numerator)
    numerator = abs_diff.sum()
    
    # Total sum of daily streamflows (denominator)
    denominator = flow_series.sum()
    
    # Calculate RBI (avoid division by zero if denominator is zero)
    return numerator / denominator if denominator != 0 else np.nan

#%%
timeseries_EU_copy = timeseries_EU.copy() # already defined this variable??
# 1. Calculate Annual RBI
# Group data by year
timeseries_EU_copy['Year'] = timeseries_EU_copy.index.year

# Initialise a dictionary to store RBI results per catchment per year
rbi_annual = {}

# Loop through each catchment (column) to calculate RBI per year
for catchment in timeseries_EU_copy.columns[:-1]:  # Exclude 'Year' column
    rbi_annual[catchment] = timeseries_EU_copy.groupby('Year')[catchment].apply(calculate_rbi)

# Convert the dictionary to a dataframe
rbi_annual_df = pd.DataFrame(rbi_annual)

# 2. Calculate RBI for the entire timeseries
# Remove the 'Year' column before applying the calculation to the entire timeseries
df_no_year = timeseries_EU_copy.drop(columns=['Year'])

# Apply RBI calculation to the entire timeseries for each catchment
rbi_entire_timeseries = df_no_year.apply(calculate_rbi, axis=0)

# Print or display the results
print("Annual RBI per catchment:")
print(rbi_annual_df)

print("\nRBI for the entire timeseries per catchment:")
print(rbi_entire_timeseries)

mean = rbi_annual_df.mean()

#%%
var_rbi = rbi_annual_df.var()
#%% Save the results in the hydro signature df
hydro_sign['Flashiness_index'] = rbi_entire_timeseries

#%% RISING AND FALLING LIMB
timeseries_EU_copy = timeseries_EU.copy() # Creating again because of the added "year" column after the rbi calc.
#%%
hydrometeo_signatures_df = pd.DataFrame(index = network_EU.index, 
                                        columns = [
                                                   "rld"
                                                  ])
#%% RLD
def calculate_rld_with_nan_reset(timeseries):
    # Initialize counters
    rising_count = 0
    total_rising_time = 0
    peak_count = 0

    for i in range(1, len(timeseries)):
        if pd.isna(timeseries[i]) or pd.isna(timeseries[i - 1]):
            # If NaN is encountered, reset rising count
            if rising_count > 0:
                peak_count += 1
                total_rising_time += rising_count
                rising_count = 0  # Reset for next period
            continue  # Skip to the next day if there's a NaN

        if timeseries[i] > timeseries[i - 1]:
            # Rising limb logic
            rising_count += 1
        else:
            # When the rise ends, record the peak and reset
            if rising_count > 0:
                peak_count += 1
                total_rising_time += rising_count
                rising_count = 0  # Reset after counting

    # Final check in case the series ends with a rise
    if rising_count > 0:
        peak_count += 1
        total_rising_time += rising_count

    # Calculate RLD
    rld = peak_count / total_rising_time if total_rising_time > 0 else np.nan  # Change to np.nan for zero division

    return rld  # Return just the RLD value


for gauge in tqdm(timeseries_EU_copy.columns):
    rld_value = calculate_rld_with_nan_reset(timeseries_EU_copy[gauge])
    hydrometeo_signatures_df.loc[gauge, "rld"] = rld_value  # Assign the value directly


#%% Save signature to the hydrosign df
hydro_sign['rld'] = hydrometeo_signatures_df['rld']

#%% FLD (falling limb density) empty df to store the results
hydrometeo_signatures_df = pd.DataFrame(index = network_EU.index, 
                                        columns = [
                                                   "fld"
                                                  ])
#%%fld
def calculate_fld_with_nan_reset(timeseries):
    # Initialize counters
    falling_count = 0
    total_falling_time = 0
    trough_count = 0

    for i in range(1, len(timeseries)):
        if pd.isna(timeseries[i]) or pd.isna(timeseries[i-1]):
            # If NaN is encountered, reset falling count
            if falling_count > 0:
                trough_count += 1
                total_falling_time += falling_count
                falling_count = 0  # Reset for next period
            continue  # Skip to the next day if there's a NaN

        if timeseries[i] < timeseries[i-1]:
            # Falling limb logic
            falling_count += 1
        else:
            # When the fall ends, record the trough and reset
            if falling_count > 0:
                trough_count += 1
                total_falling_time += falling_count
                falling_count = 0  # Reset after counting

    # Final check in case the series ends with a fall
    if falling_count > 0:
        trough_count += 1
        total_falling_time += falling_count

    # Calculate FLD
    fld = trough_count / total_falling_time if total_falling_time > 0 else np.nan  # Change to np.nan for zero division

    return fld  # Return just the FLD value


# Loop through each catchment(here: catchment)
for gauge in tqdm(timeseries_EU_copy.columns):
    hq_gauge = calculate_fld_with_nan_reset(timeseries_EU_copy[gauge])
    
    # Directly assign the FLD value to the DataFrame
    hydrometeo_signatures_df.loc[gauge, "fld"] = hq_gauge  # No need for brackets

# Check the resulting DataFrame
print(hydrometeo_signatures_df)


#%% Save signature to the hydro signature dataframe
hydro_sign['fld'] = hydrometeo_signatures_df['fld']
#%% THIS part is from THiagos script: Q mean. slope of fdc, baseflow index, hfd mean, hfd std, Q5, Q95, Hq freq, Hq dur, Lq freg, Lq dur, zero Q frequency, cv Q

# Adding in the indices of gini and cv from script of T.V.M do Nascimento: https://github.com/thiagovmdon/EStreams/blob/main/code/python/C_computation_signatures_and_indices/estreams_hydrometeorological_signatures.ipynb
cv = pd.read_csv(r'U:\Hydrological Data\EStreams\Subset_v2_290324\Subset171024\Indices\results\timeseries\streamflowindices\yearly_streamflow_cv.csv', index_col=0)
gini = pd.read_csv(r'U:\Hydrological Data\EStreams\Subset_v2_290324\Subset171024\Indices\results\timeseries\streamflowindices\yearly_streamflow_gini.csv', index_col=0)

cv_mean = cv.mean()

hydro_sign['cv'] = cv_mean   
gini_mean = gini.mean()
gini_median = gini.median()
gini_var = gini.var()

#add to the dataframe
hydro_sign['gini_mean'] = gini_mean
hydro_sign["gini_median"] = gini_median
hydro_sign["gini_var"] = gini_var


#%%  Interannual variance of hfd, hq, lg etc.
#Loading in the yearly data of the signatures to cerate the frequency
# The yearly data is calculated from the seasonality script and is based script by T.V.M do Nascimento: https://github.com/thiagovmdon/EStreams/blob/main/code/python/C_computation_signatures_and_indices/estreams_hydrometeorological_signatures.ipynb
hfd = pd.read_csv(r'U:\Hydrological Data\EStreams\Subset_v2_290324\Subset171024\hydrometeorological\hfd_mean_std_var.csv', encoding='utf-8')
hfd.set_index("basin_id", inplace = True)
hfd

hq_dur = pd.read_csv(r'U:\Hydrological Data\EStreams\Subset_v2_290324\Subset171024\hydrometeorological\hq_dur_yearly.csv', encoding='utf-8')
hq_dur.set_index("Unnamed: 0", inplace = True)
hq_dur

hq_freq = pd.read_csv(r'U:\Hydrological Data\EStreams\Subset_v2_290324\Subset171024\hydrometeorological\hq_freq_yearly.csv', encoding='utf-8')
hq_freq.set_index("Unnamed: 0", inplace = True)
hq_freq

lq_dur = pd.read_csv(r'U:\Hydrological Data\EStreams\Subset_v2_290324\Subset171024\hydrometeorological\lq_dur_yearly.csv', encoding='utf-8')
lq_dur.set_index("Unnamed: 0", inplace = True)
lq_dur

lq_freq = pd.read_csv(r'U:\Hydrological Data\EStreams\Subset_v2_290324\Subset171024\hydrometeorological\lq_freq_yearly.csv', encoding='utf-8')
lq_freq.set_index("Unnamed: 0", inplace = True)
lq_freq

q95 = pd.read_csv(r'U:\Hydrological Data\EStreams\Subset_v2_290324\Subset171024\hydrometeorological\q95_yearly.csv', encoding='utf-8')
q95.set_index("Unnamed: 0", inplace = True)
q95

q5 = pd.read_csv(r'U:\Hydrological Data\EStreams\Subset_v2_290324\Subset171024\hydrometeorological\q5_yearly.csv', encoding='utf-8')
q5.set_index("Unnamed: 0", inplace = True)
q5

qzero = pd.read_csv(r'U:\Hydrological Data\EStreams\Subset_v2_290324\Subset171024\hydrometeorological\qzero_df_yearly.csv', encoding='utf-8')
qzero.set_index("Unnamed: 0", inplace = True)
qzero

#%% Calculate the variance of the yearly data
hq_dur_var = hq_dur.var()

hq_freq_var = hq_freq.var()
lq_dur_var = lq_dur.var()
lq_freq_var= lq_freq.var()
q95_var = q95.var()
q5_var = q5.var()
qzero_var = qzero.var()

#%% Save to hydro signatures df

hydro_sign['hq_dur_var'] = hq_dur_var
hydro_sign['hq_freq_var'] = hq_freq_var
hydro_sign['lq_dur_var'] = lq_dur_var
hydro_sign['lq_freq_var'] = lq_freq_var
hydro_sign['q95_var'] = q95_var
hydro_sign['q5_var'] = q5_var
hydro_sign['qzero_var'] = qzero_var

#%% Slight modification of df so it fits with defined function
inpyt = hfd[['hfd_mean']]
input_t = inpyt.T

#%% Circular statistic for hfd
data = input_t
results_hfd = {}

for catchment in data.columns:
    circ_var, mean_day = circular_stats(data[catchment])
    results_hfd[catchment] = {'circular_variance': circ_var, 'mean_day': mean_day}

# Convert results to DataFrame
results_df_hfd = pd.DataFrame(results_hfd).T
print('Circular stats for each catchment:')
print(results_df_hfd.head())
#%% Save to hydrosign
hydro_sign['hfd_var'] = results_df_hfd['circular_variance']
#%% Including var of fdc and hfd var (not the interannual var) 
# code adpated from T.V.M. do Nascimento: https://github.com/thiagovmdon/EStreams/blob/main/code/python/C_computation_signatures_and_indices/estreams_hydrometeorological_signatures.ipynb
# FDC yearly

# Define the function (same as before)
def calculate_fdc_slope(streamflow):
    """
    Compute FDC slopes using the method from Sawicz et al. (2011), Yadav et al. (2007),
    McMillan et al. (2017), and Addor et al. (2018).
    """
    # Remove NaN and zero values
    streamflow = streamflow[~np.isnan(streamflow)]
    streamflow = streamflow[streamflow > 0]  # Avoid log(0)

    if len(streamflow) == 0:
        return {'slope_sawicz': np.nan, 'slope_yadav': np.nan, 'slope_mcmillan': np.nan, 'slope_addor': np.nan}

    # Compute FDC
    quantiles = np.arange(0, 1.001, 0.001) * 100
    fdc = -np.sort(-np.percentile(streamflow, quantiles))

    # Get Q33 and Q66
    q33 = fdc[quantiles == 33.0][0]
    q66 = fdc[quantiles == 66.0][0]
    q_median = np.median(streamflow)
    q_mean = np.mean(streamflow)

    # Compute slopes
    if q66 > 0:
        slope_sawicz = (np.log(q33) - np.log(q66)) / (0.66 - 0.33)
        slope_yadav = ((q33 / q_mean) - (q66 / q_mean)) / (0.66 - 0.33)
        slope_mcmillan = (np.log(q33 / q_median) - np.log(q66 / q_median)) / (0.66 - 0.33)
        q33_perc, q66_perc = np.percentile(streamflow, [33, 66])
        slope_addor = (np.log(q66_perc) - np.log(q33_perc)) / (0.66 - 0.33)
    else:
        slope_sawicz = np.nan
        slope_yadav = np.nan
        slope_mcmillan = np.nan
        slope_addor = np.nan

    return {'slope_sawicz': slope_sawicz, 'slope_yadav': slope_yadav, 'slope_mcmillan': slope_mcmillan, 'slope_addor': slope_addor}

#%% Run ofr each year
timeseries_EU = timeseries_EU.loc['1980' : '2022']
#%%
# Convert Date column to datetime (if not already)
timeseries_EU.index = pd.to_datetime(timeseries_EU.index)

# Get the list of years in the dataset
years = np.arange(1980, 2022)  
catchments = timeseries_EU.columns  # Catchment IDs

# Create an empty dictionary to store results
fdc_slope_results = {slope: pd.DataFrame(index=years, columns=catchments) for slope in ['slope_sawicz', 'slope_yadav', 'slope_mcmillan', 'slope_addor']}

# Loop over years and catchments
for year in years:
    print(f"Processing year: {year}")  # Progress tracking
    
    # Extract data for the current year
    yearly_data = timeseries_EU.loc[str(year)]
    
    for catchment in catchments:
        streamflow = yearly_data[catchment].dropna().values  # Remove NaNs
        
        # Calculate slopes
        slopes = calculate_fdc_slope(streamflow)
        
        # Store results
        for slope in slopes:
            fdc_slope_results[slope].loc[year, catchment] = slopes[slope]

# Convert to DataFrame
for slope in fdc_slope_results:
    fdc_slope_results[slope] = fdc_slope_results[slope].astype(float)  # Ensure numerical data

# Now, fdc_slope_results contains a DataFrame for each slope metric
print(fdc_slope_results['slope_sawicz'])  # Example: Check one of them

#%% calculate the variance
var_Saw = fdc_slope_results['slope_sawicz'].var()
var_addor = fdc_slope_results['slope_addor'].var()
var_yadav = fdc_slope_results['slope_yadav'].var()
var_mcmillan = fdc_slope_results['slope_mcmillan'].var()

#%% Save variance to the hydrosign dataframe
hydro_sign['var_slope_sawicz'] = var_Saw
hydro_sign['var_slope_addor'] = var_addor
hydro_sign['var_slope_yadav'] = var_yadav
hydro_sign['var_slope_mcmillan'] = var_mcmillan

#%% Save final df
hydro_sign.to_csv(r'C:\Users\juliarudlang\OneDrive - Delft University of Technology\Temp_climate_snow\hydro_sign_280225.csv')
