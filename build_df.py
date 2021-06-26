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
import warnings
import matplotlib.pyplot as plt
import xarray as xr
import time
import regionmask
import geopandas as gpd
import warnings
import cartopy.crs as ccrs
import gc

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
def addSSTData(data_frame, regions, year_start, years):
    # my_data_frame.reset_index(inplace=True, drop=True)
    data_frame.reset_index(inplace=True, drop=True)
    # sst_dict is used to store sst_mean entries for each region. Regions are the key for the dictionary
    # each entry in the lists associated with a region key represents the sst_mean dataset for 1 year
    sst_dict = {}
    for region in regions:
        sst_dict[region.abbrev] = []
        column_name = region.abbrev + '_sst'
        data_frame[column_name] = np.nan
    for i in range(years):
        f_name = './SourceData/sst/sst.day.mean.' + str(year_start + i) + '.nc'
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
        print('Year '+ str(2000 + i) + ' added successfully' )
        print('-------------------------------------------------------------------------------------')
    # now the list of each regions' yearly mean must be combined into a single dataset
    for region in sst_dict.keys():
        sst_dict[region] = xr.concat(sst_dict[region], dim='time')
        sst_dict[region].sst.plot(label=region)
        interp = sst_dict[region].sst.interp(time=list(map(str, data_frame['time'])), method='cubic')
        print(interp)
        interp.to_dataframe()
        print(interp)
        #interp.reset_index(inplace=True, drop=True)
        #interp.reset_index(drop=True)
        column_name = region + '_sst'
        label = region + '_interp'
        #interp.rename({'sst': column_name})
        #print(interp)
        #interp.columns = [column_name]
        #interp[column_name].plot.scatter(x='time', y=column_name, label=label)
        interp.plot(label=label, marker = 'o')
        data_frame[column_name] = interp

    plt.legend()
    plt.show()

    return data_frame

my_data_frame = pd.DataFrame()
# Read jungfraujoch data and constrain to years between 2000-2018, convert dates to decimal time if necessary
print("retrieving jfj_data, and constraining to 2000-2018")
jfj_data = pd.read_csv('./SourceData/jfj_OCS_trop_VMR_DailyMeans_4Elliott.txt', header=3, delim_whitespace=True, skiprows=[4], parse_dates=['dd-mmm-yyyy'])
jfj_data = constrainDataFrameByYears(jfj_data, 'dd-mmm-yyyy', 2000, 2018)
print("success")

# strip down to just cos, cos_stdev and time
jfj_data = pd.DataFrame({'time': jfj_data['dd-mmm-yyyy'], 'COS_JFJ': jfj_data['ppt']})

# Read manua load data from OCS_GCMS_flask.txt and constrain to years between 2000-2018, convert dates to decimal times
print('load mlo data and constrain to 2000-2018')
mlo_data = pd.read_csv('./SourceData/OCS__GCMS_flask.txt', delim_whitespace=True, header=1, parse_dates=['yyyymmdd'])
mlo_data = mlo_data.loc[mlo_data['site'] == 'mlo']
mlo_data = constrainDataFrameByYears(mlo_data, 'yyyymmdd', 2000, 2018)
print('success')

my_data_frame = jfj_data
#delete COS sets
del [[jfj_data, mlo_data]]
gc.collect()
print('--------------------------------------------------------------------')

# generate our ocean regions
north_hemisphere = np.array([[-180, 0], [-180, 90], [180, 90], [180, 0]])
south_hemisphere = np.array([[-180, 0], [-180, -90], [180, -90], [180, 0]])
names = ['Northern hemisphere', 'Southern hemisphere']
abbrevs = ['NH', 'SH']
regions = regionmask.Regions([north_hemisphere, south_hemisphere], names=names, abbrevs=abbrevs, name='Hemispheres')
#regions = regionmask.defined_regions.ar6.ocean

my_data_frame = addSSTData(my_data_frame, regions, 2000, 19)
print(my_data_frame)

if my_data_frame.isnull().values.any():
    total_null = my_data_frame.isnull().sum().sum()
    print('Total null values: ' + str(total_null))

# remove time from dataframe?
# my_data_frame.drop('time',axis=1)
my_data_frame.to_pickle('COS_Seesaw_dataframe.pkl')
