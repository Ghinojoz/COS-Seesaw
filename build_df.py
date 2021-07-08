'''
Disclaimer: There are a lot of improvements that can be made to improve the performance of this script.
it is currently quite slow, but I will be making optimizations as I learn more about xarray and working with geospatial data
'''
import numpy as np
import pandas as pd
from datetime import datetime as dt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from scipy.interpolate import interp1d
from dateutil.relativedelta import relativedelta
import warnings
import matplotlib.pyplot as plt
import xarray as xr
import time
import regionmask
import geopandas as gpd
import warnings
import cartopy.crs as ccrs
import gc

# ----------------------------------------------------------------------------------------------------------------------------------------
# Generalization variables, modify the variables here to change targets, interpolation timelags, etc.
# text divider for console output
# ----------------------------------------------------------------------------------------------------------------------------------------
divider = '--------------------------------------------------------------------------------------------------------'

# COS target
# Assumes that this target can be found in the OCS__GCMS_flask.txt file
cos_target = 'cgo'

# target dates (inclusive)
year_start = 2000
year_end = 2018

# generalized timelags, or SST timelags if other variables use different timelag periods 
# store these as tuples , with the first element being the suffix to appear at then end of lagged data
# and the second element representing the corresponding relativedelta from the dateutil module
time_delta = [("-3m", relativedelta(months=-3)), ("-2m", relativedelta(months=-2)), ("-1m", relativedelta(months=-1))]

# ----------------------------------------------------------------------------------------------------------------------------------------

# can be applied to columns of dataframes, e.g.
# decimal_years = my_data_frame['dd-mm-yyyy'].apply(toYearDecimal)
def toYearDecimal(date):
    year = date.year
    startOfThisYear = dt(year=year, month=1, day=1)
    startOfNextYear = dt(year=year+1, month=1, day=1)
    yearElapsed = time.mktime(date.timetuple()) - time.mktime(startOfThisYear.timetuple())
    yearDuration = time.mktime(startOfNextYear.timetuple()) - time.mktime(startOfThisYear.timetuple())
    decimal = yearElapsed/yearDuration
    decimal = round(decimal,5)
    return year + decimal

# returns all values in the dataframe between the start and endyear as indicated by the dateColumn (inclusive!)
def constrainDataFrameByYears(df, dateColumn, startyear, endyear):
    return df[(df[dateColumn] >= dt(year=startyear, month=1, day=1)) & (df[dateColumn] < dt(year=endyear+1, month=1, day=1))]

# regions must be a regions object which defines the desired regions
def addSSTData(data_frame, regions, start, end):
    # my_data_frame.reset_index(inplace=True, drop=True)
    data_frame.reset_index(inplace=True, drop=True)

    # ------------------------------------------------------------------------------------------------------------------------------------
    # generate offset date arrays for interpolation
    offset_dates = []
    for delta in time_delta:
        funct = lambda x: x + delta[1]
        current_offset = my_data_frame['time'].apply(funct)
        offset_dates.append((delta[0], current_offset))
    # ------------------------------------------------------------------------------------------------------------------------------------

    # sst_dict is used to store sst_mean entries for each region. Regions are the key for the dictionary
    # each entry in the lists associated with a region key represents the sst_mean dataset for 1 year
    sst_dict = {}
    for region in regions:
        sst_dict[region.abbrev] = []
        # add a column to our data frame for each region
        column_name = region.abbrev + '_sst'
        data_frame[column_name] = np.nan
    # +1 to account for 0 starting index, otherwise ends at end - 1
    for year in range(start, end + 1):
        f_name = './SourceData/sst/sst.day.mean.' + str(year) + '.nc'
        print("Now processing " + f_name)
        noaa_data = xr.open_dataset(f_name).load()
        print('Done loading data')
        noaa_data = noaa_data.assign_coords(lon = (((noaa_data.lon + 180) % 360) -180))
        noaa_data = noaa_data.sortby(noaa_data.lon)
        print('Done standardizing coordinates')
        sst_mask = regions.mask(noaa_data)
        for region in regions:
            region_index = regions.map_keys(region.name)
            region_sst = noaa_data.where(sst_mask == region_index)
            sst_mean = region_sst.mean(dim=('lat', 'lon'))
            sst_dict[region.abbrev].append(sst_mean)

        noaa_data.close()
        gc.collect()
        print('Done cleaning up after ' + f_name)
        print('Year '+ str(year) + ' added successfully' )
        print('-------------------------------------------------------------------------------------')
    # now the list of each regions' yearly mean must be combined into a single dataset
    for region in sst_dict.keys():
        sst_dict[region] = xr.concat(sst_dict[region], dim='time')
        interp = sst_dict[region].sst.interp(time=list(map(str, data_frame['time'])), method='cubic')
        interp.to_dataframe()
        column_name = region + '_sst'
        data_frame[column_name] = interp

        # add the date offset interpolation for each offset and each region
        for delta in offset_dates:
            column_name = region + '_sst' + delta[0]
            interp = sst_dict[region].sst.interp(time=list(map(str, delta[1])), method='cubic')
            data_frame[column_name] = interp

    return data_frame
'''
my_data_frame = pd.DataFrame()
# Read jungfraujoch data and constrain to years between 2000-2018, convert dates to decimal time if necessary
print("retrieving jfj_data, and constraining to 2000-2018")
jfj_data = pd.read_csv('./SourceData/jfj_OCS_trop_VMR_DailyMeans_4Elliott.txt', header=3, delim_whitespace=True, skiprows=[4], parse_dates=['dd-mmm-yyyy'])
jfj_data = constrainDataFrameByYears(jfj_data, 'dd-mmm-yyyy', 2000, 2018)
print("success")

# strip down to just cos, cos_stdev and time
jfj_data = pd.DataFrame({'time': jfj_data['dd-mmm-yyyy'], 'COS_JFJ': jfj_data['ppt']})
'''

# to modify the COS target, simply change the cos_target variable at the top of the script
my_data_frame = pd.DataFrame()
print(divider)
print('load COS data and constrain to desired years: ' + str(year_start) + '-' + str(year_end))
cos_data = pd.read_csv('./SourceData/OCS__GCMS_flask.txt', delim_whitespace=True, header=1, parse_dates=['yyyymmdd'])
cos_data = cos_data.loc[cos_data['site'] == cos_target]
cos_data = constrainDataFrameByYears(cos_data, 'yyyymmdd', year_start, year_end)

# strip and add to dataframe
cos_column_name = 'COS_' + cos_target
cos_data = pd.DataFrame({'time': cos_data['yyyymmdd'], cos_column_name: cos_data['OCS_']})
my_data_frame = cos_data
print('Success!')
print('Result:')
print(my_data_frame)

print(divider)
# free up space and collect garbage
print('Free cos_data space and collect garbage')
del[[cos_data]]
gc.collect()

print('Success!')

# generate our ocean regions
north_hemisphere = np.array([[-180, 0], [-180, 90], [180, 90], [180, 0]])
south_hemisphere = np.array([[-180, 0], [-180, -90], [180, -90], [180, 0]])
names = ['Northern hemisphere', 'Southern hemisphere']
abbrevs = ['NH', 'SH']
regions = regionmask.Regions([north_hemisphere, south_hemisphere], names=names, abbrevs=abbrevs, name='Hemispheres')

#regions = regionmask.defined_regions.ar6.ocean
print(divider)
print('Begin adding sst data for year range: ' + str(year_start) + '-' + str(year_end))
print(divider)
print('With the following timelags:')
for lag in time_delta:
    print(lag)
print(divider)


my_data_frame = addSSTData(my_data_frame, regions, year_start, year_end)
print(my_data_frame)

if my_data_frame.isnull().values.any():
    total_null = my_data_frame.isnull().sum().sum()
    print('Total null values: ' + str(total_null))

# remove time from dataframe?
# my_data_frame.drop('time',axis=1)

# drop observations where a variable value is null
my_data_frame.dropna(inplace=True)
print(my_data_frame)
my_data_frame.to_pickle('COS_Seesaw_dataframe.pkl')
