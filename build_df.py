import os
import warnings
import time
import regionmask
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import geopandas as gpd
import cartopy.crs as ccrs
from datetime import datetime as dt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from scipy.interpolate import interp1d
from dateutil.relativedelta import relativedelta


# convenience variable for console output formatting
divider = '--------------------------------------------------------------------------------------------'

# Generalization variables, modify variables here to change targets, interpolation timelags, etc.

# COS target, i.e. where our COS observations are obtained from
# changing the value of cos_site will produce a different pickle file,
# note that when you run GA2M in the other script, you must modify the cos_site there as well
cos_site = 'smo'
cos_file = './SourceData/OCS__GCMS_flask.txt'

# path for saving some plots
results_path = './Results/' + cos_site + '_Results'
if not os.path.isdir('./Results'):
    os.mkdir('./Results')

if not os.path.isdir(results_path):
    os.mkdir(results_path)

# Use COS target specific regions? If false, will use generic regions
# using target specific regions is not yet implemented
use_site_regions = False

# target dates (inclusive), i.e. the time between which our data is constrained
year_start = 2000
year_end = 2018

'''
Generalized timelags, we may wish to use different values for individual data sets
timelags are defined as relativedeltas, Note that if these time lags put us outside of the target dates
then some dataframe points will have null values. Rows with null values are dropped immediately prior to saving the
dataframe as a pickle file.
Currently this method produces a lot of features -- number of time lags * number of regions * number of variables
establishing a single timelag for each region, relative to the OCS target site, is desirable, but not yet complete.
'''
time_delta_general = [('-15d', relativedelta(days=-15)), ('-1m', relativedelta(months=-1)), ('-1m15d', relativedelta(months=-1, days=-15)), ('-2m', relativedelta(months=-2))]

# This function builds the regionmask regions to be used.
# data is averaged across these regions to form features for GA2M
def buildRegions():
    regions = None
    if use_site_regions:
        print('Using regions defined for ' + cos_site + ' is not yet implemented')
        print('Please modify the build_df.py file and set use_site_regions = False')
        print('Aborting')
        exit()
    else:
        names = []
        abbrevs = []
        region_list = []

        print(divider)
        print('Building Regionmask')
        print(divider)

        # South Ocean
        south_ocean = np.array([[-180, -60], [180, -60], [180, -40], [-180, -40]])
        names.append('South Ocean')
        abbrevs.append('SO')
        region_list.append(south_ocean)

        # Arctic Ocean
        arctic_ocean = np.array([[-180, 40], [180, 40], [180, 60], [-180, 60]])
        names.append('Arctic Ocean')
        abbrevs.append('AO')
        region_list.append(arctic_ocean)

        # north east pacific
        ne_pacific = np.array([[-180, 20], [-60, 20], [-60, 40], [-180, 40]])
        names.append('North East Pacific Ocean')
        abbrevs.append('NEP')
        region_list.append(ne_pacific)

        # north west pacific
        nw_pacific = np.array([[120, 20], [180, 20], [180, 40], [120, 40]])
        names.append('North West Pacific Ocean')
        abbrevs.append('NWP')
        region_list.append(nw_pacific)

        # equatorial north east pacific
        eqne_pacific = np.array([[-180, 0], [-60, 0], [-60, 20], [-180, 20]])
        names.append('Equatorial North East Pacific Ocean')
        abbrevs.append('EQNEP')
        region_list.append(eqne_pacific)
    
        # equatorial north west pacific
        eqnw_pacific = np.array([[120, 0], [180, 0], [180, 20], [120, 20]])
        names.append('Equatorial North West Pacific Ocean')
        abbrevs.append('EQNWP')
        region_list.append(eqnw_pacific)

        # equatorial south east pacific
        eqse_pacific = np.array([[-180, -20], [-60, -20], [-60, 0], [-180, 0]])
        names.append('Equatorial South East Pacific Ocean')
        abbrevs.append('EQSEP')
        region_list.append(eqse_pacific)

        # equatorial south west pacific
        eqsw_pacific = np.array([[120, -20], [180, -20], [180, 0], [120, 0]])
        names.append('Equatorial South West Pacific Ocean')
        abbrevs.append('EQSWP')
        region_list.append(eqsw_pacific)

        # south east pacific
        se_pacific = np.array([[-180, -40], [-60, -40], [-60, -20], [-180, -20]])
        names.append('South East Pacific Ocean')
        abbrevs.append('SEP')
        region_list.append(se_pacific)

        # south west pacific
        sw_pacific = np.array([[120, -40], [180, -40], [180, -20], [120, -20]])
        names.append('South West Pacific Ocean')
        abbrevs.append('SWP')
        region_list.append(sw_pacific)

        # north atlantic
        north_atlantic = np.array([[-60, 20], [20, 20], [20, 40], [-60, 40]])
        names.append('North Atlantic Ocean')
        abbrevs.append('NA')
        region_list.append(north_atlantic)

        # equatorial north atlantic
        neq_atlantic = np.array([[-60, 0], [20, 0], [20, 20], [-60, 20]])
        names.append('Equatorial North Atlantic Ocean')
        abbrevs.append('EQNA')
        region_list.append(neq_atlantic)

        # equatorial south atlantic
        seq_atlantic = np.array([[-60, -20], [20, -20], [20, 0], [-60, 0]])
        names.append('Equatorial South Atlantic Ocean')
        abbrevs.append('EQSA')
        region_list.append(seq_atlantic)

        # south atlantic
        south_atlantic = np.array([[-60, -40], [20, -40], [20, -20], [-60, -20]])
        names.append('South Atlantic Ocean')
        abbrevs.append('SA')
        region_list.append(south_atlantic)
    
        # North Indian Ocean
        north_indian = np.array([[20, 0], [120, 0], [120, 20], [20, 20]])
        names.append('North Indian Ocean')
        abbrevs.append('NI')
        region_list.append(north_indian)

        # equatorial  indian
        eq_indian = np.array([[20, -20], [120, -20], [120, 0], [20, 0]])
        names.append('Equatorial Indian Ocean')
        abbrevs.append('EQI')
        region_list.append(eq_indian)
    
        # South Indian Ocean
        south_indian = np.array([[20, -40], [120, -40], [120, -20], [20, -20]])
        names.append('South Indian Ocean')
        abbrevs.append('SI')
        region_list.append(south_indian)

        regions = regionmask.Regions(region_list, names=names, abbrevs=abbrevs, name='Ocean Regions')
        print('Completed')
        print(divider)
        #regions.plot(label='abbrev')
        #plt.show()

    return regions

# used to produce features from geospatial data that is averaged across the provided regions
# file_name_start and file_name_end should be strings which form the beginning and end of the filename, with a space in between 
# being defined by the current year being operated on.
# standardizeLon should be true if the source data uses 0 to 360 longitude values, as the regionmask uses -180 to 180
# if standardizeTime is provided, will add yearly data to monthly data, standardizeTime should be a tuple with the name of the
# starting time column followed by the name of the desired time column
def getRegionalizedMapData(file_name_start, file_name_end, variable_name, regions, start, end, standardizeLon=True, standardizeTime=None):
    print(divider)
    print('Begin adding data for ' + variable_name)
    print(divider)

    data_dict = {}
    for region in regions:
        data_dict[region.abbrev] = []
    # +1 to account for 0 starting index, otherwise ends at end - 1
    for year in range(start, end + 1):
        f_name = file_name_start + str(year) + file_name_end
        print('Begin Processing ' + f_name)
        data = None
        # Some of the downward solar radiation data used a date time format that
        # xarray did not like, this attempts to manually convert it, but may need to be
        # adjusted for future issues
        try:
            data = xr.open_dataset(f_name).load()
        except:
            print("Error opening " + f_name)
            print("Attempting alternative datetime conversion")
            try:
                data = xr.open_dataset(f_name, decode_times=False).load()
                time_since = dt(year=1800, month=1, day=1)
                temp_time = data['time'].to_series()
                funct = lambda x : relativedelta(hours=x)
                time_col = temp_time.apply(funct)
                funct2 = lambda x: time_since + x
                data['time'] = time_col.apply(funct2)
                print("Success!")
            except Exception as e:
                Print("Failed")
                print(e)
                exit()
        print('Done loading data')
        # If necessary, convert 0 to 360 to -180 to 180
        if standardizeLon:
            data = standardizeLongitude(data)

        if standardizeTime is not None:
            data = standardizeTime_Month(data, year, standardizeTime[0], standardizeTime[1])

        # get the mean for each region
        data_mask = regions.mask(data)
        for region in regions:
            region_index = regions.map_keys(region.name)
            region_data = data.where(data_mask == region_index)
            data_mean = region_data.mean(dim=('lat', 'lon'))
            data_dict[region.abbrev].append(data_mean)
        print('Region Data added')
        data.close()
        gc.collect()
        print('Done cleaning up after ' + f_name)
        print('Year ' + str(year) + ' added successfully!')
        print(divider)

    # combine each region's yearly mean into single data set
    print('Combining Years')
    for region in data_dict.keys():
        data_dict[region] = xr.concat(data_dict[region], dim='time')

    print('Success!')
    print(divider)
    return data_dict

# modifies date_coord, which should be the time column fromt he OCS observations by delta_t, a set of 
# timelags labeled and defined as above -- look at time_delta_general
def generateOffsetDates(date_coord, delta_t):
    offset_dates = []
    for delta in delta_t:
        funct = lambda x: x + delta[1]
        current_offset = date_coord.apply(funct)
        offset_dates.append((delta[0], current_offset))
    return offset_dates

# This function sets the longitude coordinate of a dataset to -180 to 180
# assumes data has a lon coordinate. Also assumes that the lon coordinate passed to this function covers the range 0 to 360
def standardizeLongitude(data):
    print(divider)
    print('Standardizing Longitude')
    print(divider)
    data = data.assign_coords(lon = (((data.lon + 180) % 360) -180))
    data = data.sortby(data.lon)
    print('Done standardizing longitude')
    print(divider)
    return data

# Used to add yearly values to monthly climatology data
# takes a xarray dataset, called data, a year to add data for
# tName_start is the name of the time column as it initially appears in the dataset
# tName_end is the name of the created time column found in the returned dataset
def standardizeTime_Month(data, year, tName_start, tName_end):
    print(divider)
    print('Standardizing time')
    print(divider)
    dates = []
    for val in data[tName_start].values:
        date = dt(year=year, month=int(val), day=1)
        dates.append(date)
    data = data.rename({tName_start : tName_end})
    data = data.assign_coords(time=dates)
    print('Done standardizing time')
    print(divider)
    return data

# given region_var - an xarray data column, get interpolated data based on interp_dates and return
def interpData(region_var, interp_dates):
    interp = region_var.interp(time=list(map(str, interp_dates)), method='cubic')
    return interp

# used to retrieve OCS observations from a tab delineated text file found at file_name, which is a path relative to the working
# directory. Time_column_name is a string indicating which column the datetime can be found.
# site_name is used to obtain data for a specific site, and start, and end should be years which are used to constrain the obtained
# data between an inclusive range. Note that this currently only works for the NOAA ocs flask data
def loadCOSData(file_name, time_column_name, site_name, start, end):
    print(divider)
    print('Load COS data for' + site_name + ' from ' + str(start) + ' to ' + str(end))
    print(divider)
    
    print ('Reading file: ' + file_name)
    cos_data = pd.read_csv(file_name, delim_whitespace=True, header=1, parse_dates=[time_column_name])
    cos_data = cos_data.loc[cos_data['site'] == site_name]
    
    # use average of same day observations
    duplicates = cos_data.duplicated(keep=False, subset=[time_column_name])
    duplicate_entries = cos_data.where(duplicates)
    duplicate_entries.dropna(inplace=True)
    unique_dates = duplicate_entries[time_column_name].unique()
    same_day_avg = []
    for date in unique_dates:
        entry_subset = duplicate_entries.where(duplicate_entries[time_column_name] == date)
        entry_subset.dropna(inplace=True)
        ocs_col = entry_subset['OCS_']
        
        mean = ocs_col.mean()
        same_day_avg.append((date, mean))

    cos_data = cos_data.drop_duplicates(subset=[time_column_name])

    print(cos_data)
    for avg in same_day_avg:
        cos_data.loc[cos_data[time_column_name] == avg[0], 'OCS_'] = avg[1]
    print("Average replaced:")
    print(cos_data)

    # this section may be commented out later, it was used simply to plot the relationship between wind direction and
    # OCS measurement variation for same day observations
    '''
    duplicates = cos_data.duplicated(keep=False, subset=[time_column_name])
    duplicate_entries = cos_data.where(duplicates)
    duplicate_entries.dropna(inplace=True)
    print(duplicate_entries)
    print(duplicate_entries.count())
    unique_dates = duplicate_entries[time_column_name].unique()
    wind_difference = []
    cos_difference = []
    for date in unique_dates:
        print(divider)
        print(date)
        entry_subset = duplicate_entries.where(duplicate_entries[time_column_name] == date)
        entry_subset.dropna(inplace=True)
        wind_col = entry_subset['wind_dir']
        ocs_col = entry_subset['OCS_']
        wind_col = wind_col.astype(float)
        ocs_col = ocs_col.astype(float)
        wind_col = wind_col.tolist()
        ocs_col = ocs_col.tolist()
        if ((wind_col[0] <= 0) or (wind_col[1] <= 0)):
            continue
        elif ((wind_col[0] >= 360) or (wind_col[1] >= 360)):
            continue

        ocs_delta = abs(ocs_col[0] - ocs_col[1])
        wind_delta = abs(wind_col[0] - wind_col[1])
        if(wind_delta >= 100):
            continue
        print(ocs_delta)
        print(wind_delta)
        wind_difference.append(wind_delta)
        cos_difference.append(ocs_delta)

    print(divider)
    print("Wind difference: ")
    print(wind_difference)
    print(divider)
    print("OCS difference: ")
    print(cos_difference)
    print(divider)
    fig, ax = plt.subplots(figsize=(16,8))
    ax.scatter(wind_difference, cos_difference)
    ax.set_xlabel("Wind Delta")
    ax.set_ylabel("OCS Delta")
    ax.set_title('Same Day Measurements -- Wind Delta vs. OCS Delta')
    #plt.scatter(wind_difference, cos_difference)
    plt.show()
    
    # print(divider)
    # print('Unique Dates')
    # print(divider)
    # print(unique_dates)
    # print(divider)
    '''
    # end wind_delta vs ocs_delta calculations + plot

    print('Constraining between ' + str(start) + ' - ' + str(end))
    cos_data = cos_data[(cos_data[time_column_name] >= dt(year=start, month=1, day=1)) & (cos_data[time_column_name] < dt(year=end+1, month=1, day=1))]
    
    # remove other variables and return only ocs observations and date
    cos_column_name = 'COS_' + site_name
    print('Building Dataframe')
    cos_data = pd.DataFrame({'time':cos_data[time_column_name], cos_column_name : cos_data['OCS_'], 'OCS_stddev' : cos_data['OCS__sd']})
    cos_data = cos_data.reset_index(drop=True)

    # add previous years
    cos_data['cos_mean-1y'] = np.nan
    previous_year_mean = None
    for year in range (start, (end + 1)):
        year_data = cos_data.loc[cos_data['time'].dt.year == year]
        if previous_year_mean is not None:
            cos_data.loc[(cos_data.time.dt.year == year), 'cos_mean-1y'] = previous_year_mean
        previous_year_mean = year_data[cos_column_name].mean()
        print(year)
        print(previous_year_mean)

    print(divider)
    print(cos_data)
    print(divider)

    print(divider)
    print('Success!')
    print(divider)
    fig, ax = plt.subplots(figsize=(16,8))
    ax.plot(cos_data['time'], cos_data[cos_column_name])
    ax.set_title(site_name + ' OCS over time')
    ax.set_xlabel('Time')
    ax.set_ylabel(site_name + '_OCS')

    fig.savefig(results_path + '/OCS_over_time.png')
    #plt.show()
    return cos_data

# given climatology data with a monthly granularity, adds interpolated data points for each OCS observation
# takes file_name, a path relative to the current working directory where the data can be found
# regions is a regionmask object with all desired regions. Takes a data_frame, to which the interpolated data will be added
# variable_name is a string representing the desired column to be taken from the climatology data
# standardize_lon should be set to false if it is not desirable to convert 0 to 360 longitude to -180 to 180 longitude
def addClimatologyData(file_name, regions, data_frame, variable_name ,standardize_lon=True):
    print(divider)
    print('Adding climatology data for: ' + file_name)
    print(divider)
    data = xr.open_dataset(file_name).load()
    if standardize_lon:
        data = standardizeLongitude(data)
    data_dict = {}
    data_mask = regions.mask(data)
    for region in regions:
        region_index = regions.map_keys(region.name)
        region_data = data.where(data_mask == region_index)
        data_mean = region_data.mean(dim=('lat', 'lon'))
        data_dict[region.abbrev] = []
        data_dict[region.abbrev].append(data_mean)
    for region in data_dict.keys():
        data_dict[region] = xr.concat(data_dict[region], dim='month')
        col_name = region + variable_name
        data_frame[col_name] = np.nan
        data_dict[region] = data_dict[region].to_array()
        for i in range (0, 12):
            data_frame.loc[data_frame['time'].dt.month == (i+1), col_name] = float(data_dict[region].variable[0][i])
    
    print('Data added successfully!')
    print(divider)
    return data_frame

# ---------------------------------------------------------------------------------------------------------------------
# build data frame and save as pickle
# ---------------------------------------------------------------------------------------------------------------------

my_data_frame = loadCOSData(cos_file, 'yyyymmdd', cos_site, year_start, year_end)

# establish data regions
regions = buildRegions()

# get timelagged dates
offset_dates = generateOffsetDates(my_data_frame['time'], time_delta_general)
fig, ax = plt.subplots(figsize=(16,8))
regions.plot(label='abbrev')
fig.savefig(results_path + '/regions.png')
#plt.show()

# add v coordinate wind
vwnd_dict = getRegionalizedMapData('./SourceData/VWND/vwnd.10m.gauss.', '.nc', 'vwnd', regions, year_start, year_end)
for region in vwnd_dict.keys():
    interp = interpData(vwnd_dict[region].vwnd, my_data_frame['time'])
    column_name = region + '_vwnd'
    my_data_frame[column_name] = interp

    for delta in offset_dates:
        column_name = region + '_vwnd' + delta[0]
        interp = interpData(vwnd_dict[region].vwnd, delta[1])
        my_data_frame[column_name] = interp

my_data_frame = my_data_frame.copy()

# add u coordinate wind
uwnd_dict = getRegionalizedMapData('./SourceData/UWND/uwnd.10m.gauss.', '.nc', 'uwnd', regions, year_start, year_end)
for region in uwnd_dict.keys():
    interp = interpData(uwnd_dict[region].uwnd, my_data_frame['time'])
    column_name = region + 'uwnd'
    my_data_frame[column_name] = interp

    for delta in offset_dates:
        column_name = region + '_uwnd' + delta[0]
        interp = interpData(uwnd_dict[region].uwnd, delta[1])
        my_data_frame[column_name] = interp

my_data_frame = my_data_frame.copy()

# add downward solar radiation
dswrf_dict = getRegionalizedMapData('./SourceData/DSRF/dswrf.sfc.gauss.', '.nc', 'dswrf', regions, year_start, year_end)
for region in dswrf_dict.keys():
    interp = interpData(dswrf_dict[region].dswrf, my_data_frame['time'])
    column_name = region + '_dswrf'
    my_data_frame[column_name] = interp

    for delta in offset_dates:
        column_name = region + '_dswrf' + delta[0]
        interp = interpData(dswrf_dict[region].dswrf, delta[1])
        my_data_frame[column_name] = interp

my_data_frame = my_data_frame.copy()

# add salinity
my_data_frame = addClimatologyData('./SourceData/sal_T42.nc', regions, my_data_frame, '_sal')

# add mixed layer depth
my_data_frame = addClimatologyData('./SourceData/MLD_T42.nc', regions, my_data_frame, '_mld')

# add Sea Surface Temperature data
sst_dict = getRegionalizedMapData('./SourceData/sst/sst.day.mean.', '.nc', '_sst', regions, year_start, year_end)
for region in sst_dict.keys():
    interp = interpData(sst_dict[region].sst, my_data_frame['time'])
    column_name = region + '_sst'
    my_data_frame[column_name] = interp

    for delta in offset_dates:
        column_name = region + '_sst' + delta[0]
        interp = interpData(sst_dict[region].sst, delta[1])
        my_data_frame[column_name] = interp

del[[sst_dict]]
gc.collect()

my_data_frame = my_data_frame.copy()

# add CDOM data
cdom_dict = getRegionalizedMapData('./SourceData/CDOM/CDOM_a350_', '.nc', '_cdom', regions, year_start, year_end, standardizeTime=('month', 'time'))
for region in cdom_dict.keys():
    interp = interpData(cdom_dict[region].CDOM_a350, my_data_frame['time'])
    column_name = region + '_cdom'
    my_data_frame[column_name] = interp

    for delta in offset_dates:
        column_name = region + '_cdom' + delta[0]
        interp = interpData(cdom_dict[region].CDOM_a350, delta[1])
        my_data_frame[column_name] = interp

del[[cdom_dict]]
gc.collect()

my_data_frame = my_data_frame.copy()

print (my_data_frame)

if my_data_frame.isnull().values.any():
    total_null = my_data_frame.isnull().sum().sum()
    print('Total null values: ' + str(total_null))

my_data_frame.dropna(inplace=True)
print(my_data_frame)
my_data_frame.to_pickle('COS_Seesaw_dataframe_' + cos_site + '.pkl')

