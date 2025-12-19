
#%% Load in packages
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

#%% Loading in data --UPDATE
precipitation_timeseries = pd.read_csv(r'U:\Hydrological Data\EStreams\Temp_climate_snow\precipitation_timeseries_droppedS_180225.csv', index_col = 0)
precipitation_timeseries.index = pd.to_datetime(precipitation_timeseries.index)
precipitation_timeseries.index.name = ""

network_EU = pd.read_csv(r'U:\Hydrological Data\EStreams\Data 071025\Network_071025.csv')
network_EU.set_index("basin_id", inplace = True)
network_EU

#%%% previously used ea, why is this one also shorther like the new one?
Ea = pd.read_csv(r'U:\Hydrological Data\EStreams\Subset_v2_290324\Subset171024\Data_with_dropped_stations\timeseries_Ea_update.csv', index_col=0)
Ea.index = pd.to_datetime(Ea.index)
Ea.index.name = ""
#%%
Ea = Ea.drop(columns=['PT000048', 'ES001380', 'CYGR0014', 'DEHE1013', 'FI000110'])
#%% Cropping the timeseries from 1980 - 2022
Ea = Ea.loc['1980' : '2022']
precipitation_timeseries = precipitation_timeseries.loc['1980' : '2022']
#%% Save again
#Ea.to_csv(r'U:\Hydrological Data\EStreams\Temp_climate_snow\Ea_timeseries_1980_2022.csv')
#%% Create df to store results
climate_sign =pd.DataFrame(index=network_EU.index)
#%% Updated version of the water deficit where it resets to 0 at the start of each March
# this means that the wd becomes more of an annual insight rather than a longterm. How reaslistic is this, and what is good about that rather than the real world?
# The idea behind it is as that it did not look exactly right when not including the reset, because there might be other factors that are drawing water from the system that I am not able to account for

catchments = precipitation_timeseries.columns.to_list()
date_index = pd.date_range('01-01-1980', '12-31-2022', freq='D')

# Initialize the storage_deficit DataFrame
storage_deficit_ea = pd.DataFrame(index=date_index, columns=catchments, dtype=float)

# Set the first row of storage_deficit
storage_deficit_ea.iloc[0] = np.minimum(0, precipitation_timeseries.iloc[0] - Ea.iloc[0])

# Iterate over all timesteps
for i in range(1, len(storage_deficit_ea)):
    # Reset storage to zero on January 1st
    if storage_deficit_ea.index[i].day == 1 and storage_deficit_ea.index[i].month == 3: # reset at 1st of march each year
        storage_deficit_ea.iloc[i] = 0
    else:
        # Fill NaN values in the previous storage with 0 before computation. In order to stop nan values propagating into the calculation. When working with lenght and duration as signatures for wd, the nans or 0 wont be counted anyways.
        previous_sd = storage_deficit_ea.iloc[i - 1].fillna(0)
        current_p = precipitation_timeseries.iloc[i]
        current_ea = Ea.iloc[i]

        # Compute storage deficit for the current timestep
        storage_deficit_ea.iloc[i] = np.minimum(0, previous_sd + current_p - current_ea)


#%% Test to see how it looks like
plt.plot(storage_deficit_ea['FR003106'].loc['1980' : '1981'])
#%% Test to see how it looks like
plt.plot(storage_deficit_ea['FR003106'])
#%% Restricting the Ea so it is sure to be correct (excluding 1980, sort like a "spin-up" time)
wd_ea_corr = storage_deficit_ea.loc['1981' : '2022']

#%% SIGNATURE CALCULATION
#%% MEAN AND MEDIAN DURATION OF WD  - longterm, mean and median over the whole timseries
# Initialize a DataFrame to store results
results_df = pd.DataFrame(index=wd_ea_corr.columns, columns=["Mean_Duration_wd_ea", "Median_Duration_wd_ea"])

# Loop through each catchment
for catchment in wd_ea_corr.columns:
    # Get the storage deficit series for the current catchment
    Sd = wd_ea_corr[catchment]
    
    durations = []  # List to store lengths of deficit periods
    current_gap = 0  # Counter for the current deficit period length
    
    # Iterate over values to count deficit periods
    for value in Sd:
        if value < 0:
            current_gap += 1  # Increment gap length
        else:
            if current_gap >= 0:
                durations.append(current_gap)  # Store completed deficit period length
            current_gap = 0  # Reset gap counter

    # Handle the case where the series ends with a deficit
    if current_gap >= 0:
        durations.append(current_gap)
    
    durationsS = pd.Series(durations)
    filtered_durations = durationsS[durationsS > 0]
    # Compute mean and median durations
    if durations:
        results_df.at[catchment, "Mean_Duration_wd_ea"] = np.mean(filtered_durations)
        results_df.at[catchment, "Median_Duration_wd_ea"] = np.median(filtered_durations)
    else:
        results_df.at[catchment, "Mean_Duration_wd_ea"] = np.nan
        results_df.at[catchment, "Median_Duration_wd_ea"] = np.nan
        
#%%Save results
climate_sign["Longterm_Mean_Duration_wd"] = results_df["Mean_Duration_wd_ea"]
climate_sign["Longterm_Median_Duration_wd"] = results_df["Median_Duration_wd_ea"]
#%% THE LONGEST GAP OF WD
#%% 
# Create an empty dataframe to store the results

longest_gap_periods_ea = pd.DataFrame(index=range(wd_ea_corr.index.year.min(), wd_ea_corr.index.year.max() + 1),
                                   columns=wd_ea_corr.columns)


#%% Longest gap = longest duration of water deficit per year 
# Iterate over each column (catchment)
for col in tqdm(wd_ea_corr.columns):
    # Group by year and process each year for the current column
    grouped_by_year = wd_ea_corr[col].groupby(wd_ea_corr.index.year)
    for year, values in grouped_by_year:
        max_gap = 0
        current_gap = 0
        
        # Iterate over values to find consecutive deficit durations
        for value in values:
            if value < 0:  # wd is negative
                current_gap += 1
                max_gap = max(max_gap, current_gap)
            else:
                current_gap = 0
        
        # Store the longest gap for the year
        longest_gap_periods_ea.at[year, col] = max_gap


#%% Finding the median of the longest dry period
longest_gap_periods_median_ea = longest_gap_periods_ea.median()
longest_gap_periods_mean_ea = longest_gap_periods_ea.mean()
longest_gap_periods_var_ea = longest_gap_periods_ea.var()

#%% annual longest wd, in terms of days
climate_sign['annual_longest_wd_periods_median'] = longest_gap_periods_median_ea
climate_sign['annual_longest_wd_periods_mean'] = longest_gap_periods_median_ea
climate_sign['annual_longest_wd_periods_var'] = longest_gap_periods_median_ea

#%% Median Max water deficit
annual_max_wd = wd_ea_corr.groupby(wd_ea_corr.index.year).min()

median_annual_max_wd = annual_max_wd.median()
mean_annual_max_wd = annual_max_wd.mean()
var_annual_max_wd = annual_max_wd.var()
#save to climate sign
climate_sign['median_annual_max_wd'] = median_annual_max_wd
climate_sign['mean_annual_max_wd'] = mean_annual_max_wd
climate_sign['var_annual_max_wd'] = var_annual_max_wd

#%%  Median days of deficit per year
results_df = pd.DataFrame(index=wd_ea_corr.columns, columns=["Mean_annual_days_wd", "Median_annual_days_wd", "CV_annual_days_wd", "var_annual_days_wd"])
# Loop over all catchments
for catchment in tqdm(wd_ea_corr.columns):
    # Select data for the current catchment
    catchment_data = wd_ea_corr[catchment]
    negative_counts = catchment_data[catchment_data < 0]
    # Group by year and count days with negative values
    negative_countss = negative_counts.groupby(negative_counts.index.year).count()
    
    # Calculate mean and median for the yearly counts
    mean_neg_days = negative_countss.mean()
    median_neg_days = negative_countss.median()
    var_neg_days = negative_countss.var()
    
    std_neg_days = negative_countss.std()
    cv_neg_days = (std_neg_days / mean_neg_days) * 100 if mean_neg_days != 0 else 0
    # Store the results
    results_df.at[catchment, "Mean_annual_days_wd"] = mean_neg_days
    results_df.at[catchment, "Median_annual_days_wd"] = median_neg_days
    results_df.at[catchment, "CV_annual_days_wd"] = cv_neg_days
    results_df.at[catchment,"var_annual_days_wd"] = var_neg_days


#%% Save results  
climate_sign['Mean_annual_days_wd'] = results_df["Mean_annual_days_wd"]
climate_sign['Median_annual_days_wd'] = results_df["Median_annual_days_wd"]
climate_sign['CV_annual_days_wd'] = results_df["CV_annual_days_wd"]
climate_sign['var_annual_days_wd'] = results_df["var_annual_days_wd"]

#%% Declining and uprising slope of the max deficit
# Initialize a dictionary to store the results
results = {}

# Loop over each catchment
for catchment in tqdm(wd_ea_corr.columns):
    slopes = []
    
    # Group by year
    grouped = wd_ea_corr[catchment].groupby(wd_ea_corr.index.year)
    
    for year, year_data in grouped:

        # Convert the year data to a NumPy array
        values = year_data.values
        # Skip if the entire year is NaN
        if np.all(np.isnan(values)):
            continue
        
        max_deficit_idx = np.nanargmin(values)  # Index of the most negative value
        max_deficit = values[max_deficit_idx]  # The most negative value
    
        if max_deficit >= 0:
            # Skip years without deficits
            continue
    
        # Count backward for the declining slope
        days_declining = 0
        for i in range(max_deficit_idx - 1, -1, -1):
            if values[i] < 0:
                days_declining += 1
            elif values[i] == 0:
                days_declining += 1
                break
            else:
                break
        declining_slope = max_deficit / days_declining if days_declining > 0 else np.nan

        # Count forward for the uprising slope
        days_uprising = 0
        for i in range(max_deficit_idx + 1, len(values)):
            if values[i] < 0:
                days_uprising += 1
            elif values[i] == 0:
                days_uprising += 1
                break
            else:
                break
        uprising_slope = max_deficit / days_uprising if days_uprising > 0 else np.nan
    
        # Ensure both slopes are NaN if one is NaN
        if np.isnan(declining_slope):
            uprising_slope = np.nan
        elif np.isnan(uprising_slope):
            declining_slope = np.nan 
        
        # Store the slopes for this year
        slopes.append({"year": year, "declining_slope": declining_slope, "uprising_slope": uprising_slope})
    
    # Save the slopes for this catchment
    results[catchment] = slopes

# Convert results to a dataframe for easier analysis
results_df = {catchment: pd.DataFrame(data) for catchment, data in results.items()}

# Example to view results for a specific catchment
#print(results_df["Catchment1"])

#%%
#results_df = {catchment: pd.DataFrame(data) for catchment, data in results.items()}
#%%
# Now, calculate the overall mean and median for each catchment across all years
summary_data = []

for catchment, df in tqdm(results_df.items()):
    # Calculate mean and median for the declining slope
    mean_declining = df['declining_slope'].mean()
    median_declining = df['declining_slope'].median()
    var_declining = df['declining_slope'].var()
    
    # Calculate mean and median for the uprising slope
    mean_uprising = df['uprising_slope'].mean()
    median_uprising = df['uprising_slope'].median()
    var_uprising = df['uprising_slope'].var()
    
    # Store the summary data for this catchment
    summary_data.append({
        'Catchment': catchment,
        'mean_declining': mean_declining,
        'median_declining': median_declining,
        'var_declining': median_declining,
        'mean_uprising': mean_uprising,
        'median_uprising': median_uprising,
        'var_uprising': median_uprising
    })

# Convert the summary data into a DataFrame
summary_df = pd.DataFrame(summary_data)

# Set the catchment names as the index
summary_df.set_index('Catchment', inplace=True)


#%% Save the data
climate_sign['mean_declining_slope_max_wd'] = summary_df["mean_declining"]
climate_sign['median_declining_slope_max_wd'] = summary_df["median_declining"]
climate_sign['var_declining_slope_max_wd'] = summary_df["var_declining"]
climate_sign['mean_uprising_slope_max_wd'] = summary_df["mean_uprising"]
climate_sign['median_uprising_slope_max_wd'] = summary_df["median_uprising"]
climate_sign['var_uprising_slope_max_wd'] = summary_df["var_uprising"]


#%%
#%%# Function to calculate circular variance and mean day. 
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

#%% Timing of the first and last day of wd with a threhold of 1 week duration!!
# Initialize results storage
results = []

# Loop over all catchments (one column per catchment)
for catchment in tqdm(wd_ea_corr.columns):  
    
    grouped = wd_ea_corr[catchment].groupby(wd_ea_corr.index.year)  # Group by year
    
    # To store yearly results
    first_negatives = []
    last_negatives = []

    for year, year_data in grouped:
        values = year_data.values
        dates = year_data.index
        
        # Convert to numpy arrays for easier manipulation
        values = np.array(values)
        dates = pd.to_datetime(dates)  # Ensure dates are datetime objects

        # Identify sequences of negative values
        negative_mask = values < 0
        negative_groups = []
        start_idx = None
        
        for i, is_negative in enumerate(negative_mask):
            if is_negative:
                if start_idx is None:
                    start_idx = i  # Start of a new sequence
            else:
                if start_idx is not None:
                    # End of the current sequence
                    if i - start_idx >= 7:  # Keep only sequences longer than a week
                        negative_groups.append((start_idx, i - 1))
                    start_idx = None
        
        # Handle the case where the last sequence extends to the end
        if start_idx is not None and len(values) - start_idx >= 7:
            negative_groups.append((start_idx, len(values) - 1))

        # Extract the first and last days of valid sequences
        if negative_groups:
            first_negative_idx = negative_groups[0][0]
            last_negative_idx = negative_groups[-1][1]

            # Get DOY
            first_negative_doy = dates[first_negative_idx].day_of_year
            last_negative_doy = dates[last_negative_idx].day_of_year
        else:
            # No valid sequences in this year
            first_negative_doy = np.nan
            last_negative_doy = np.nan

        # Append to yearly results
        first_negatives.append(first_negative_doy)
        last_negatives.append(last_negative_doy)
        
    first_negatives_array = np.array(first_negatives, dtype=np.float64)
    last_negatives_array = np.array(last_negatives, dtype=np.float64)
    # Calculate mean and median across years for this catchment
    mean_first_negative = np.nanmean(first_negatives_array)
    median_first_negative = np.nanmedian(first_negatives_array)
    var_first_negative = np.nanvar(first_negatives_array)
    
    first_negatives_array_forcirc = first_negatives_array[~np.isnan(first_negatives_array)]
    circ_var_first_negative, mean_day_first_negative = circular_stats(first_negatives_array_forcirc)
    
    mean_last_negative = np.nanmean(last_negatives_array)
    median_last_negative = np.nanmedian(last_negatives_array)
    var_last_negative = np.nanvar(last_negatives_array)
    
    last_negatives_array_forcirc = last_negatives_array[~np.isnan(last_negatives_array)]
    circ_var_last_negative, mean_day_last_negative = circular_stats(last_negatives_array_forcirc)

    # Store results for this catchment
    results.append({
        'catchment': catchment,
        'mean_first_negative': mean_first_negative,
        'median_first_negative': median_first_negative,
        'var_first_negative' : var_first_negative,
        'circ_var_first_negative' : circ_var_first_negative,
        'mean_last_negative': mean_last_negative,
        'median_last_negative': median_last_negative,
        'var_last_negative' : var_last_negative,
        'circ_var_last_negative' : circ_var_last_negative
    })

# Convert results to a DataFrame for easier analysis
results_df = pd.DataFrame(results)

#%%
results_df.set_index('catchment', inplace=True)

#%%

climate_sign['mean_first_doy_WD'] = results_df['mean_first_negative']
climate_sign['median_first_doy_WD'] = results_df['median_first_negative']
climate_sign['circ_var_first_negative_WD'] = results_df['circ_var_first_negative']
climate_sign['mean_last_doy_WD'] = results_df['mean_last_negative']
climate_sign['median_last_doy_WD'] = results_df['median_last_negative']
climate_sign['circ_var_last_negative_WD'] = results_df['circ_var_last_negative']

#%%  annual days with wd - ALREADY CALCULATED - VERSION 2

# Add a 'Year' column
wd_ea_corr['Year'] = wd_ea_corr.index.year

# Filter rows where precipitation > 0 (days it rained)
WD_days = wd_ea_corr.iloc[:, :-1] < 0  

# Count the number of rainy days per year
WD_days_per_year = WD_days.groupby(wd_ea_corr['Year']).sum()

# Calculate the median number of rainy days across all years
median_WD_annual_days = WD_days_per_year.median()
mean_WD_annual_days = WD_days_per_year.mean()
var_WD_annual_days = WD_days_per_year.var()

#%% Save data
climate_sign['median_WD_annual_days_V2'] = median_WD_annual_days
climate_sign['mean_WD_annual_days_V2'] = mean_WD_annual_days
climate_sign['var_WD_annual_days_V2'] = var_WD_annual_days

#%% TIMING OF PEAK WD

wd_ea_corr_copy = wd_ea_corr.copy()
day_mean = wd_ea_corr_copy.groupby(wd_ea_corr_copy.index.dayofyear).mean()


#%%
# Creating a new df to hold the results

hydro_signatures_timing = pd.DataFrame(index = climate_sign.index, 
                                        columns = [ 
                                                   "lq_day", "lq_angle", "lq_sin", "lq_cos", "lq_day_avg"
                                                  ])

#%%
# MEAN
for i in wd_ea_corr_copy.columns:
    lowest_flow_day = day_mean[i].idxmin()

    
    m = 365.25
    
    angle_lq = lowest_flow_day * ((2 * np.pi) / m)
    
    
    x_lq = np.cos(angle_lq)
    y_lq = np.sin(angle_lq)

        
        
    # Calculate the average date D (Eq. (2)) 
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
    R_lq = np.sqrt(x_lq**2 + y_lq**2)
    
    hydro_signatures_timing.loc[i, [ 
               "lq_day", "lq_angle", "lq_sin", "lq_cos", "lq_day_avg"
              ]] =  lowest_flow_day, angle_lq, y_lq, x_lq, D_lq
#%%
climate_sign["wd_max_day"] = hydro_signatures_timing["lq_day"]
climate_sign["wd_max_sin"] = hydro_signatures_timing["lq_sin"]
climate_sign["wd_max_cos"] = hydro_signatures_timing["lq_cos"]
climate_sign["wd_max_day_avg"] = hydro_signatures_timing["lq_day_avg"]

#%% in order to calculate the variance of the timing peak WD, group per year and find min and convert to doy 

grouped_by_year_WD = wd_ea_corr_copy.groupby(wd_ea_corr_copy.index.year)
max_WD_day = grouped_by_year_WD.idxmin()
# Extract the day of the year for the highest flow days
max_WD_day_doy = max_WD_day.apply(lambda x: x.dt.dayofyear)

#%% Circlar variance WD peak timing
# Calculate circular variance for each catchment
data = max_WD_day_doy
resultsL = {}

for catchment in data.columns:
    circ_var, mean_day = circular_stats(data[catchment])
    resultsL[catchment] = {'circular_variance': circ_var, 'mean_day': mean_day}

# Convert results to DataFrame
results_df_L = pd.DataFrame(resultsL).T
print('Circular stats for each catchment:')
print(results_df_L.head())

#%% Save to df
climate_sign["circ_var_timing_max_wd"] = results_df_L["circular_variance"]
#%% SAVE NEW CLIMATE SIGN
climate_sign.to_csv(r'U:\Hydrological Data\EStreams\Temp_climate_snow\climate_sing_WD_030325.csv')