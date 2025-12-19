
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
# Network
network_EU = pd.read_csv(r'U:\Hydrological Data\EStreams\Data 071025\Network_071025.csv')
network_EU.set_index("basin_id", inplace = True)
network_EU

#%%
hydro_sign = pd.read_csv(r'U:\Hydrological Data\EStreams\Data 071025\hydro_sign_clustered_101025.csv', encoding='utf-8')
hydro_sign.set_index("basin_id", inplace = True)
hydro_sign

#%% # Load the climate and landscape data  
file_path = 'U:\Hydrological Data\EStreams\Data 071025\climate_landscape_071025.csv'

# Read the file
climate_sign = pd.read_csv(file_path)
climate_sign.set_index("basin_id", inplace=True)
climate_sign 
#%% Climate timeseries  
precipitation_timeseries = pd.read_csv(r'U:\Hydrological Data\EStreams\Temp_climate_snow\precipitation_timeseries_droppedS_180225.csv', index_col = 0)
precipitation_timeseries.index = pd.to_datetime(precipitation_timeseries.index)
precipitation_timeseries.index.name = ""

pet_timeseries = pd.read_csv(r'U:\Hydrological Data\EStreams\Temp_climate_snow\pet_timeseries_droppedS_180225.csv', index_col = 0)
pet_timeseries.index = pd.to_datetime(pet_timeseries.index)
pet_timeseries.index.name = ""

temperature_timeseries = pd.read_csv(r'U:\Hydrological Data\EStreams\Temp_climate_snow\temperature_timeseries_droppedS_180225.csv', index_col = 0)
temperature_timeseries.index = pd.to_datetime(temperature_timeseries.index)
temperature_timeseries.index.name = ""

# Load in timeseries subset in mm/day
timeseries_EU = pd.read_csv(r'C:\Users\juliarudlang\OneDrive - Delft University of Technology\Temp_climate_snow\timeseries_q_interpolated_droppedS.csv', index_col=0) 
timeseries_EU.index = pd.to_datetime(timeseries_EU.index)
timeseries_EU.index.name = ""

#%%Load snowmodule in order to create a df for each sm component
# Check the files in the subdirectory:
filenames = glob.glob("C:/Users/juliarudlang/OneDrive - Delft University of Technology/Temp_climate_snow/SM/timeseries/*.csv")
print("Number of files:", len(filenames))
print("First file:", filenames[0])

#%% Load in all catchments in the folder of SM
catchment_id = os.path.splitext(os.path.basename(filenames[5]))[0]

#%%
# First, we create an empty DataFrame for the data with a datetime index:
Pliquid_df = pd.DataFrame(index=pd.date_range(start='1980-01-01', end='2022-12-31', freq='D'))
Psolid_df = pd.DataFrame(index=pd.date_range(start='1980-01-01', end='2022-12-31', freq='D'))
pm_df = pd.DataFrame(index=pd.date_range(start='1980-01-01', end='2022-12-31', freq='D'))
tas_df = pd.DataFrame(index=pd.date_range(start='1980-01-01', end='2022-12-31', freq='D'))
p_df = pd.DataFrame(index=pd.date_range(start='1980-01-01', end='2022-12-31', freq='D'))
ep_df = pd.DataFrame(index=pd.date_range(start='1980-01-01', end='2022-12-31', freq='D'))

# Loop for reading and concatenating the data:
for file in tqdm.tqdm(range(len(filenames))):
    
    # Read the data from the CSV file:
    P_file = pd.read_csv(filenames[file], index_col=0)
    P_file.index = pd.to_datetime(P_file.index)
    P_file.index.name = ""
    
    #Get the catchment id
    catchment_id = os.path.splitext(os.path.basename(filenames[file]))[0]
    
    #create subdata
    sub_p = P_file['p']
    sub_pl = P_file['pl']
    Sub_ps = P_file['ps']
    sub_tas = P_file['tas']
    sub_ep = P_file['ep']
    sub_pm = P_file['pm']
    
    # Set columns based on the filename
    sub_p.rename(catchment_id, inplace=True)
    sub_pl.rename(catchment_id, inplace=True)
    Sub_ps.rename(catchment_id, inplace=True)
    sub_tas.rename(catchment_id, inplace=True)
    sub_ep.rename(catchment_id, inplace=True)
    sub_pm.rename(catchment_id, inplace=True)
   
    # Concatenate the DataFrames along the columns (axis=1)
    p_df = pd.concat([p_df, sub_p], axis=1) 
    Pliquid_df = pd.concat([Pliquid_df, sub_pl], axis=1)
    Psolid_df = pd.concat([Psolid_df, Sub_ps], axis=1)
    pm_df = pd.concat([pm_df, sub_pm], axis=1)
    tas_df = pd.concat([tas_df, sub_tas], axis=1)
    ep_df = pd.concat([ep_df, sub_ep], axis=1)
#%% Save the data
p_df.to_csv(r'C:\Users\juliarudlang\OneDrive - Delft University of Technology\Temp_climate_snow\SM_P.csv')
Pliquid_df.to_csv(r'C:\Users\juliarudlang\OneDrive - Delft University of Technology\Temp_climate_snow\SM_PL.csv')
Psolid_df.to_csv(r'C:\Users\juliarudlang\OneDrive - Delft University of Technology\Temp_climate_snowSM_PS.csv')
pm_df.to_csv(r'C:\Users\juliarudlang\OneDrive - Delft University of Technology\Temp_climate_snow\SM_PM.csv')
tas_df.to_csv(r'C:\Users\juliarudlang\OneDrive - Delft University of Technology\Temp_climate_snow\SM_TAS.csv')
ep_df.to_csv(r'C:\Users\juliarudlang\OneDrive - Delft University of Technology\Temp_climate_snow\SM_EP.csv')

#%% Create new dataframe with PL, where for catchments that do not have PL Pl = P
# NB: PL from the snow module is actually liquid precipitation = rainfall!!!
timeseries_pr = precipitation_timeseries.copy()
timeseries_pr.update(Pliquid_df)
#%% Save
timeseries_pr.to_csv(r'C:\Users\juliarudlang\OneDrive - Delft University of Technology\Temp_climate_snow\PR_timeseries.csv')

#%% Reconstructing the full PL (Liquid water going to the catchment), both filled and no filled
# PL = Rainfall + snowmelt
PL_timeseries = Pliquid_df + pm_df
PL_timeseries.to_csv(r'C:\Users\juliarudlang\OneDrive - Delft University of Technology\Temp_climate_snow\PL_timeseries.csv')

# For stations that do not have snowmelt, or has a "liquid" part of the rainfall, their original precipitation will always be rainfall. So that is why we can update the timeseries like this
timeseries_pl_full = precipitation_timeseries.copy()
timeseries_pl_full.update(PL_timeseries)
timeseries_pl_full.to_csv(r'C:\Users\juliarudlang\OneDrive - Delft University of Technology\Temp_climate_snow\PL_full_timeseries.csv')


#%% calculate the Prainfall intensity
grouped_Pr_Y = timeseries_pr.groupby(timeseries_pr.index.year)
# Calculate annual total volume and days with rainfall
annual_volume = grouped_Pr_Y.sum()  # Annual total rainfall
annual_days = grouped_Pr_Y.apply(lambda x: (x > 0).sum())  # Days with non-zero precipitation

annual_intensity_Pr = annual_volume / annual_days
annual_intensity_Pr_median = annual_intensity_Pr.median()  # Or do we use the mean?
annual_intensity_Pr_mean = annual_intensity_Pr.mean()  # Or do we use the mean?

climate_sign["annual_intensity_PR_median"] = annual_intensity_Pr_median
climate_sign["annual_intensity_PR_mean"] = annual_intensity_Pr_mean
#%% Long-term intensity Prainfall
total_volume_Pr = timeseries_pr.sum()  # Total volume over all years
total_days_Pr = (timeseries_pr > 0).sum()  # Count of all days with P liquid > 0
long_term_intensity_Pr = total_volume_Pr / total_days_Pr

#save to dataframe
climate_sign["longterm_intensity_PR"] = long_term_intensity_Pr

#%% Rainfall mean
meanPr = timeseries_pr.mean()

climate_sign["PR_mean"] = meanPr

#%%Temperature long-term mean
meanT = temperature_timeseries.mean()
climate_sign["T_mean"] = meanT

#%% calculate the P intensity

grouped_P_Y = precipitation_timeseries.groupby(precipitation_timeseries.index.year)

# Calculate annual total volume and days with liquid precipitation
annual_volume = grouped_P_Y.sum()  # Annual total liquid precipitation
annual_days = grouped_P_Y.apply(lambda x: (x > 0).sum())  # Days with non-zero precipitation

annual_intensity_P = annual_volume / annual_days
annual_intensity_P_median = annual_intensity_P.median()
annual_intensity_P_mean = annual_intensity_P.mean()

climate_sign["annual_intensity_P_median"] = annual_intensity_P_median
climate_sign["annual_intensity_P_mean"] = annual_intensity_P_mean


total_volume_P = precipitation_timeseries.sum()  # Total volume over all years
total_days_P = (precipitation_timeseries > 0).sum()  # Count of all days with P liquid > 0
long_term_intensity_P = total_volume_P / total_days_P

climate_sign["longterm_intensity_P"] = long_term_intensity_P


#%% Use the snowmelt data and calculate the snowmelt intensity

grouped_PM_Y = pm_df.groupby(pm_df.index.year)

# Calculate annual total volume and days with liquid precipitation
annual_volume = grouped_PM_Y.sum()  # Annual total liquid precipitation
annual_days = grouped_PM_Y.apply(lambda x: (x > 0).sum())  # Days with non-zero precipitation

annual_intensity_PM = annual_volume / annual_days
annual_intensity_PM_median = annual_intensity_PM.median()
annual_intensity_PM_mean = annual_intensity_PM.mean()

climate_sign["annual_intensity_PM_median"] = annual_intensity_PM_median
climate_sign["annual_intensity_PM_mean"] = annual_intensity_PM_mean



total_volume_PM = pm_df.sum()  # Total volume over all years
total_days_PM = (pm_df > 0).sum()  # Count of all days with P liquid > 0
long_term_intensity_PM = total_volume_PM / total_days_PM

climate_sign["longterm_intensity_PM"] = long_term_intensity_PM
    
#%%#%% Fraction snow cover which is the fraction of annual precipiatation falling as snow. need the P and the PS solid to alculate this
# Group both dataframes by year
annual_precip = p_df.groupby(p_df.index.year).sum()  # Annual total precipitation
annual_snow = Psolid_df.groupby(Psolid_df.index.year).sum()        # Annual total snow precipitation

# Calculate the fraction of annual precipitation as snow
fraction_snow = annual_snow / annual_precip

annual_median_fraction_snow = fraction_snow.median()   # Median fraction or the mean?
annual_mean_fraction_snow = fraction_snow.mean()   # Median fraction or the mean?

# Optional: Long-term fraction of precipitation as snow
total_precip = p_df.sum()  # Total precipitation over the entire timeseries
total_snow = Psolid_df.sum()      # Total snow precipitation over the entire timeseries
long_term_fraction_snow = total_snow / total_precip

climate_sign["annual_median_fraction_snow"] = annual_median_fraction_snow
climate_sign["annual_mean_fraction_snow"] = annual_mean_fraction_snow
climate_sign["long_term_fraction_snow"] = long_term_fraction_snow

#Save as a csv (Optional)
fraction_snow.to_csv(r'C:\Users\juliarudlang\OneDrive - Delft University of Technology\Temp_climate_snow\annual_fraction_snowcover.csv')

#%% variance
fraction_snow_var = fraction_snow.var()
climate_sign["interannual_var_fraction_snow"] = fraction_snow_var

#%%#%% SNOW STORAGE CALCULATION
#%% FOR ALL CACHMENT Snow storage
#%% Do snowsotrage again but with a reset on the 1st of september
snow_storage_reset = pd.DataFrame(index=pd.date_range('01-01-1980', '12-31-2022', freq='D'), columns=pm_df.columns)

# Loop through each catchment
for catchment in pm_df.columns:
    # Extract P and Ep for the current catchment
    Pm = pm_df[catchment]
    Ps = Psolid_df[catchment]
    
    # Initialize a Series for the current catchment's storage deficit
    Ss = pd.Series(index=snow_storage_reset.index, dtype=float)
    
    # Calculate the storage deficit for the first day
    Ss.iloc[0] = 0
    

    # Iterate over all timesteps
    for i in range(1, len(Ss)): 
        if Ss.index[i].day == 1 and Ss.index[i].month == 9: # adjusted the month so it is in march instead of january
            # Reset storage to zero on January 1stday and month where we set thre reset
            Ss.iloc[i] = 0 
        
        else:
            Ss.iloc[i] = np.maximum(0, Ps.iloc[i] + Ss.iloc[i-1] - Pm.iloc[i])
        
    # Save the result
    snow_storage_reset[catchment] = Ss
    
#%% Save daily timestep timeseries of snowstorage
snow_storage_reset.to_csv(r'C:\Users\juliarudlang\OneDrive - Delft University of Technology\Temp_climate_snow\Snow_Storage_reset.csv')

#%%#%% Duration Snow cover median max annual  - per year!
longest_snow_cover = pd.DataFrame(index=range(snow_storage_reset.index.year.min(), snow_storage_reset.index.year.max() + 1),
                                   columns=snow_storage_reset.columns)

#%%
from tqdm import tqdm
#%% calculate the longest duration of snowcover for each year for each catchment
#so the maximum snowcover duration for each year
# Iterate over each column (catchment)
for col in tqdm(snow_storage_reset.columns):
    # Group by year and process each year for the current column
    grouped_by_year = snow_storage_reset[col].groupby(snow_storage_reset.index.year)
    for year, values in grouped_by_year:
        max_gap = 0
        current_gap = 0
        
        # Iterate over values to find consecutive deficit durations
        for value in values:
            if value > 0:
                current_gap += 1
                max_gap = max(max_gap, current_gap)
            else:
                current_gap = 0
        
        # Store the longest gap for the year
        longest_snow_cover.at[year, col] = max_gap  # is at a another way of saying append?


#%%#%% Finding the median, mean and variance of the longest snowcover period
longest_snow_cover_median = longest_snow_cover.median()
longest_snow_cover_mean = longest_snow_cover.mean()
longest_snow_cover_var = longest_snow_cover.var()
#%% save to df
climate_sign["longest_duration_annual_snow_cover_median"] = longest_snow_cover_median
climate_sign["longest_duration_annual_snow_cover_mean"] = longest_snow_cover_median
climate_sign["longest_duration_annual_snow_cover_var"] = longest_snow_cover_median

#%% long term mean/median of snowcover
# Initialize a DataFrame to store results
results_df = pd.DataFrame(index=snow_storage_reset.columns, columns=["Mean_Duration_snow_cover", "Median_Duration_Snow_cover"])

# Loop through each catchment
for catchment in snow_storage_reset.columns:
    # Get the snow storage series for the current catchment
    Ss = snow_storage_reset[catchment]
    
    durations = []  # List to store lengths of snow cover periods
    current_gap = 0  # Counter for the current snow cover period length
    
    # Iterate over values to count snow cover periods
    for value in Ss:
        if value > 0:  # Snow storage is positive
            current_gap += 1  # Increment snow cover length
        else:  # Snow storage ends
            if current_gap > 0:  # Only store completed periods
                durations.append(current_gap)
                current_gap = 0  # Reset gap counter

    # Handle the case where the series ends with a snow cover period
    if current_gap > 0:
        durations.append(current_gap)
    
    # Convert durations to a Pandas Series for easier manipulation
    durationsS = pd.Series(durations)
    
    # Filter out invalid durations (if any)
    filtered_durations = durationsS[durationsS > 0]
    
    # Compute mean and median durations
    if not filtered_durations.empty:
        results_df.at[catchment, "Mean_Duration_snow_cover"] = np.mean(filtered_durations)
        results_df.at[catchment, "Median_Duration_Snow_cover"] = np.median(filtered_durations)
    else:
        results_df.at[catchment, "Mean_Duration_snow_cover"] = np.nan
        results_df.at[catchment, "Median_Duration_Snow_cover"] = np.nan

#%%
climate_sign['longterm_Mean_Duration_snow_cover'] = results_df["Mean_Duration_snow_cover"]
climate_sign['longterm_Median_Duration_Snow_cover'] = results_df["Median_Duration_Snow_cover"]
#%%#%% Median and mean snow cover yearly and then the longterm of that

# Initialize a DataFrame to store results
results_df = pd.DataFrame(index=snow_storage_reset.columns, columns=["Mean_mean_Annual_Duration_snow_cover", "Median_mean_Annual_Duration_Snow_cover", "Var_mean_Annual_Duration_Snow_cover"])

# Loop through each catchment
for catchment in snow_storage_reset.columns:
    # Get the snow storage series for the current catchment
    Ss = snow_storage_reset[catchment]
    
    # Group data by year
    annual_groups = Ss.groupby(Ss.index.year)  # Group by year using the datetime index
    
    # List to store annual snow cover durations
    annual_durations = []
    
    # Iterate through each year
    for year, group in annual_groups:
        durations = []  # List to store snow cover periods within the year
        current_gap = 0  # Counter for the current snow cover period
        
        # Iterate over values in the year
        for value in group:
            if value > 0:  # Snow storage is positive
                current_gap += 1  # Increment snow cover length
            else:  # Snow storage ends
                if current_gap > 0:  # Only store completed periods
                    durations.append(current_gap)
                    current_gap = 0  # Reset gap counter
        
        # Handle the case where the year ends with a snow cover period
        if current_gap > 0:
            durations.append(current_gap)
        
        # If there are valid durations, calculate the mean for the year
        if durations:
            annual_durations.append(np.mean(durations))  # Use mean or median as needed
    
    # Compute the long-term mean and median across all years
    if annual_durations:
        results_df.at[catchment, "Mean_mean_Annual_Duration_snow_cover"] = np.mean(annual_durations)
        results_df.at[catchment, "Median_mean_Annual_Duration_Snow_cover"] = np.median(annual_durations)
        results_df.at[catchment, "Var_mean_Annual_Duration_Snow_cover"] = np.var(annual_durations)
    else:
        results_df.at[catchment, "Mean_mean_Annual_Duration_snow_cover"] = np.nan
        results_df.at[catchment, "Median_mean_Annual_Duration_Snow_cover"] = np.nan
        results_df.at[catchment, "Var_mean_Annual_Duration_Snow_cover"] = np.nan

# Display results
print(results_df)

#%% Save data
climate_sign["Mean_mean_Annual_Duration_snow_cover"] = results_df["Mean_mean_Annual_Duration_snow_cover"]
climate_sign["Median_mean_Annual_Duration_Snow_cover"] = results_df["Median_mean_Annual_Duration_Snow_cover"]
climate_sign["Var_mean_Annual_Duration_Snow_cover"] = results_df["Median_mean_Annual_Duration_Snow_cover"]

#%% Rainfall timing
timeseries_pr_copy = timeseries_pr.copy()
day_mean_PR = timeseries_pr_copy.groupby(timeseries_pr_copy.index.dayofyear).mean()

#%%#%% Liquid P intensity

grouped_PL_Y = timeseries_pl_full.groupby(timeseries_pl_full.index.year)

# Calculate annual total volume and days with liquid precipitation
annual_volume = grouped_PL_Y.sum()  # Annual total liquid precipitation
annual_days = grouped_PL_Y.apply(lambda x: (x > 0).sum())  # Days with non-zero precipitation

annual_intensity_PL = annual_volume / annual_days
annual_intensity_PL_median = annual_intensity_PL.median()
annual_intensity_PL_mean = annual_intensity_PL.mean()

climate_sign["annual_intensity_PL_median"] = annual_intensity_PL_median
climate_sign["annual_intensity_PL_mean"] = annual_intensity_PL_mean

total_volume_PL = PL_timeseries.sum()  # Total volume over all years
total_days_PL = (PL_timeseries > 0).sum()  # Count of all days with P liquid > 0
long_term_intensity_PL = total_volume_PL / total_days_PL

climate_sign["longterm_intensity_PL"] = long_term_intensity_PL


#%% Here nedd to do in another script but calculate the q elasticity to PL and one more thing? the phase?

#Directions to this other script
#%% Variance
#%% Nedd to load in ea
Ea = pd.read_csv(r'C:\Users\juliarudlang\OneDrive - Delft University of Technology\Temp_climate_snow\Ea_timeseries_droppedS_270225.csv', index_col=0)
Ea.index = pd.to_datetime(Ea.index)
Ea.index.name = ""
#%% Variance of precipitation and EP  
grouped_PL_Y_mean = timeseries_pl_full.groupby(timeseries_pl_full.index.year).mean()
grouped_PR_Y_mean = timeseries_pr.groupby(timeseries_pr.index.year).mean()
grouped_P_Y_mean= precipitation_timeseries.groupby(precipitation_timeseries.index.year).mean()
grouped_EP_Y_mean = pet_timeseries.groupby(pet_timeseries.index.year).mean()
grouped_Ea_Y_mean = Ea.groupby(Ea.index.year).mean()
grouped_PM_Y_mean = pm_df.groupby(pm_df.index.year).mean()
grouped_PS_Y_mean = Psolid_df.groupby(Psolid_df.index.year).mean()
#%%
P_var = grouped_P_Y_mean.var()
Ea_mean = Ea.mean()
Ea_var = grouped_Ea_Y_mean.var()
EP_var = grouped_EP_Y_mean.var()
PL_var = grouped_PL_Y_mean.var()
PR_var = grouped_PR_Y_mean.var()
snowmelt_var = grouped_PM_Y_mean.var()
PS_var = grouped_PS_Y_mean.var()

#%% Save the variance data to climate df
climate_sign[ "P_var"] = P_var
climate_sign[ "Ea_var"] = Ea_var
climate_sign[ "Ea_mean"] = Ea_mean
climate_sign[ "pet_var"] = EP_var
climate_sign[ "PL_var"] = PL_var
climate_sign[ "PR_var"] = PR_var
climate_sign[ "PM_var"] = snowmelt_var
climate_sign[ "PS_var"] = PS_var

#%% Additoin
PL_mean = timeseries_pl_full.mean()
climate_sign[ "PL_mean"] = PL_mean
#%%# Function to calculate circular variance and mean day
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
#%%#%% Variance of timing of snowmelt and rainfall (their maxes) - preparnig the data
grouped_by_year_PR = timeseries_pr.groupby(timeseries_pr.index.year)
max_PR_day = grouped_by_year_PR.idxmax()
# Extract the day of the year for the highest flow days
max_PR_day_doy = max_PR_day.apply(lambda x: x.dt.dayofyear)

grouped_by_year_PM = pm_df.groupby(pm_df.index.year)
max_PM_day = grouped_by_year_PM.idxmax()
# Extract the day of the year for the highest flow days
max_PM_day_doy = max_PM_day.apply(lambda x: x.dt.dayofyear)

#%%#%% Circlar variance Snowmelt (PM)
# Calculate circular variance for each catchment
data = max_PM_day_doy
resultsL = {}

for catchment in data.columns:
    circ_var, mean_day = circular_stats(data[catchment])
    resultsL[catchment] = {'circular_variance': circ_var, 'mean_day': mean_day}

# Convert results to DataFrame
results_df_L = pd.DataFrame(resultsL).T
print('Circular stats for each catchment:')
print(results_df_L.head())

#%%
climate_sign['circular_var_timing_snowmelt'] = results_df_L['circular_variance']        
#%%#%% PL high and low freq and duration and seasonality
# Calculated with adjusted script from EStreams : https://github.com/thiagovmdon/EStreams/blob/main/code/python/C_computation_signatures_and_indices/estreams_hydrometeorological_signatures.ipynb
PL_hpl_lpl  = pd.read_csv(r'C:\Users\juliarudlang\OneDrive - Delft University of Technology\Temp_climate_snow\PL_hpl_lpl_seasonality.csv', encoding='utf-8')
PL_hpl_lpl.set_index("basin_id", inplace = True)
PL_hpl_lpl

#%% Save to df
climate_sign['hpl_freq'] = PL_hpl_lpl['hp_freq']
climate_sign['hpl_dur'] = PL_hpl_lpl['hp_dur']
climate_sign['lpl_freq'] = PL_hpl_lpl['lp_freq']
climate_sign['lpl_dur'] = PL_hpl_lpl['lp_dur']
#%% P liquid timing timing
timeseries_pl_copy = timeseries_pl_full.copy()
day_mean_PL = timeseries_pl_copy.groupby(timeseries_pl_copy.index.dayofyear).mean()
#%% 
# Creating a new df to hold the results
climate_signatures_timing = pd.DataFrame(index = network_EU.index, 
                                        columns = [ "PL_day", "PL_angle", "PL_sin", "PL_cos", "PL_day_avg" ])
#%%  Timing PL
for i in timeseries_pl_copy.columns:
    highest_flow_day = day_mean_PL[i].idxmax()
    
    m = 365.25
    
    angle_hq = highest_flow_day * ((2 * np.pi) / m)

    
    x_hq = np.cos(angle_hq)
    y_hq = np.sin(angle_hq)
    
    
    # Calculate the average date D 
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
        
    # Calculate the concentration R 
    R_hq = np.sqrt(x_hq**2 + y_hq**2)

    
    climate_signatures_timing.loc[i, [ "PL_day", "PL_angle", "PL_sin", "PL_cos", "PL_day_avg"
              ]] = highest_flow_day, angle_hq, y_hq, x_hq, D_hq

#%% Save output to climate sign df
climate_sign['PL_day'] = climate_signatures_timing['PL_day']
climate_sign['PL_angle'] = climate_signatures_timing['PL_angle']
climate_sign['PL_sin'] = climate_signatures_timing['PL_sin']
climate_sign['PL_cos'] = climate_signatures_timing['PL_cos']
climate_sign['PL_day_avg'] = climate_signatures_timing['PL_day_avg']


#%%
grouped_by_year_PL = timeseries_pl_full.groupby(timeseries_pl_full.index.year)
max_PL_day = grouped_by_year_PL.idxmax()
# Extract the day of the year for the highest flow days
max_PL_day_doy = max_PL_day.apply(lambda x: x.dt.dayofyear)
#%% Circular variance PL
# Calculate circular variance for each catchment
data = max_PL_day_doy
resultsL = {}

for catchment in data.columns:
    circ_var, mean_day = circular_stats(data[catchment])
    resultsL[catchment] = {'circular_variance': circ_var, 'mean_day': mean_day}

# Convert results to DataFrame
results_df_L = pd.DataFrame(resultsL).T
print('Circular stats for each catchment:')
print(results_df_L.head())

#%%
climate_sign['circular_var_timing_PL'] = results_df_L['circular_variance']

#%% day annualy with positive values for PL

# Add a 'Year' column
timeseries_pl_full['Year'] = timeseries_pl_full.index.year

# Filter rows where precipitation > 0 (days it rained)
PL_days = timeseries_pl_full.iloc[:, :-1] > 0  

# Count the number of rainy days per year
PL_days_per_year = PL_days.groupby(timeseries_pl_full['Year']).sum()

# Calculate the median number of rainy days across all years
median_PL_annual_days = PL_days_per_year.median()
mean_PL_annual_days = PL_days_per_year.mean()
var_PL_annual_days = PL_days_per_year.var()
#%% Save data
climate_sign['median_PL_annual_days'] = median_PL_annual_days
climate_sign['mean_PL_annual_days'] = mean_PL_annual_days
climate_sign['var_PL_annual_days'] = var_PL_annual_days