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

# generalization variables, modify variables here to change targets, interpolation timelags, etc.

divider = '--------------------------------------------------------------------------------------------'

# COS target, i.e. where our COS observations are obtained from
cos_site = 'cgo'
cos_file = './SourceData/OCS__GCMS_flask.txt'

# target dates (inclusive), i.e. the time between which our data is constrained
year_start = 2000
year_end = 2018

# generalized timelags, may wish to use different values for individual data sets
time_delta_general = [('-15d', relativedelta(days=-15)), ('-1m', relativedelta(months=-1)), ('-1m15d', relativedelta(months=-1, days=-15)), ('-2m', relativedelta(months=-2))]

def buildRegions():
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

    '''
    names = ['South Ocean', 'Arctic Ocean', 'North East Pacific Ocean', 'North West Pacific Ocean', 'Equatorial North East Pacific Ocean', 'Equatorial North West Pacific Ocean', 'Equatorial South East Pacific Ocean', 'Equatorial South West Pacific Ocean', 'South East Pacific Ocean', 'South West Pacific Ocean', 'North Atlantic Ocean', 'Equatorial North Atlantic Ocean', 'Equatorial South Atlantic Ocean', 'South Atlantic Ocean', 'North Indian Ocean', 'Equatorial South Indian Ocean', 'South Indian Ocean']
    abbrevs = ['SO', 'AO', 'NEP', 'NWP', 'EQNEP', 'EQNWP', 'EQSEP', 'EQSWP', 'SEP', 'SWP', 'NA', 'EQNA', 'EQSA', 'SA', 'NI', 'EQI', 'SI']
    region_list = [south_ocean, arctic_ocean, ne_pacific, nw_pacific, eqne_pacific, eqnw_pacific, eqse_pacific, eqsw_pacific, se_pacific, sw_pacific, north_atlantic, neq_atlantic, seq_atlantic, south_atlantic, north_indian, eqs_indian, south_indian]
    '''
    regions = regionmask.Regions(region_list, names=names, abbrevs=abbrevs, name='Ocean Regions')
    print('Completed')
    print(divider)
    regions.plot(label='abbrev')
    plt.show()
    return regions

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
        data = xr.open_dataset(f_name).load()
        print('Done loading data')
        if standardizeLon:
            data = standardizeLongitude(data)

        if standardizeTime is not None:
            data = standardizeTime_Month(data, year, standardizeTime[0], standardizeTime[1])

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

def generateOffsetDates(date_coord, delta_t):
    offset_dates = []
    for delta in delta_t:
        funct = lambda x: x + delta[1]
        current_offset = date_coord.apply(funct)
        offset_dates.append((delta[0], current_offset))
    return offset_dates

def standardizeLongitude(data):
    print(divider)
    print('Standardizing Longitude')
    print(divider)
    data = data.assign_coords(lon = (((data.lon + 180) % 360) -180))
    data = data.sortby(data.lon)
    print('Done standardizing longitude')
    print(divider)
    return data

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


def interpData(region_var, interp_dates):
    interp = region_var.interp(time=list(map(str, interp_dates)), method='cubic')
    return interp

def loadCOSData(file_name, time_column_name, site_name, start, end):
    print(divider)
    print('Load COS data for' + site_name + ' from ' + str(start) + ' to ' + str(end))
    print(divider)
    
    print ('Reading file: ' + file_name)
    cos_data = pd.read_csv(file_name, delim_whitespace=True, header=1, parse_dates=[time_column_name])
    cos_data = cos_data.loc[cos_data['site'] == site_name]
    
    print('Constraining between ' + str(start) + ' - ' + str(end))
    cos_data = cos_data[(cos_data[time_column_name] >= dt(year=start, month=1, day=1)) & (cos_data[time_column_name] < dt(year=end+1, month=1, day=1))]
    # strip to COS and return dataframe
    cos_column_name = 'COS_' + site_name
    
    print('Building Dataframe')
    cos_data = pd.DataFrame({'time':cos_data[time_column_name], cos_column_name : cos_data['OCS_']})

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
    # year_data = cos_data.loc[cos_data['time'].dt.year == 2000]
    # print(year_data)
    # print(divider)
    # print(year_data.mean(numeric_only=True))

    print(divider)
    print('Success!')
    print(divider)
    plt.plot(cos_data['time'], cos_data[cos_column_name])
    plt.show()
    return cos_data

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

def addSSRD(regions, start, end):
    data_dict = {}
    for region in regions:
        data_dict[region.abbrev] = []

    for year in range(start, end + 1):
        f_name = './SourceData/ssrd/ssrd_' + str(year) + '_monthlymeandiurnalT42.nc'
        data = xr.open_dataset(f_name).load()
        dates = []
        for val in data['month'].values:
            date = dt(year=year, month=int(val), day=15)
            dates.append(date)
        data = data.rename({'month' : 'time'})
        data = data.assign_coords(time=dates)

        # print('Begin processing' + f_name)
        # data = xr.open_dataset(f_name).load()
        # print('Done loading data')

        data_mask = regions.mask(data)
        for region in regions:
            region_index = regions.map_keys(region.name)
            region_data = data.where(data_mask == region_index)
            data_mean = region_data.mean(dim=('lat', 'lon'))
            data_dict[region.abbrev].append(data_mean)

    for region in data_dict.keys():
        data_dict[region] = xr.concat(data_dict[region], dim='time')
        print(divider)
        print(region)
        print(divider)
        print(data_dict[region])
        print(divider)

    return data_dict

# ---------------------------------------------------------------------------------------------------------------------
# build data frame and save as pickle
# ---------------------------------------------------------------------------------------------------------------------

my_data_frame = loadCOSData(cos_file, 'yyyymmdd', cos_site, year_start, year_end)

# establish data regions
# regions = regionmask.defined_regions.ar6.ocean
regions = buildRegions()

# get timelagged dates
offset_dates = generateOffsetDates(my_data_frame['time'], time_delta_general)

# regions.plot(label='abbrev')
# plt.show()

f_name = './SourceData/DSRF/dswrf.sfc.gauss.2008.nc'
#dswrf_data = xr.open_dataset(f_name, decode_times=False).load()
dswrf_data = xr.open_dataset(f_name, decode_times=False).load()
print(dswrf_data)
dswrf_data.dswrf[0].plot()
plt.show()
time_since = dt(year=1800, month=1, day=1)
print(time_since)
temp_time = dswrf_data['time'].to_series()
print(type(temp_time))
print(temp_time)
funct = lambda x : relativedelta(hours=x)
time_col = temp_time.apply(funct)
funct2 = lambda x : time_since + x
temp_time = time_col.apply(funct2)
dswrf_data['time'] = temp_time
print(divider)
print("Ending Col")
print(divider)
print(dswrf_data)
print(divider)
dswrf_data.dswrf[0].plot()
plt.show()

dswrf_dict = getRegionalizedMapData('./SourceData/DSRF/dswrf.sfc.gauss.', '.nc', 'dswrf', regions, year_start, year_end)
for region in dswrf_dict.keys():
    interp = interpData(dswrf_dict[region].dswrf, my_data_frame['time'])
    column_name = region + '_dswrf'
    my_data_frame[column_name] = interp

    for delta in offset_dates:
        column_name = region + '_dswrf' + delta[0]
        interp = interpData(dswrf_dict[region].dswrf, delta[1])
        my_data_frame[column_name] = interp

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
    interp.plot()
    plt.show()

    for delta in offset_dates:
        column_name = region + '_sst' + delta[0]
        interp = interpData(sst_dict[region].sst, delta[1])
        my_data_frame[column_name] = interp

del[[sst_dict]]
gc.collect()

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


print (my_data_frame)

if my_data_frame.isnull().values.any():
    total_null = my_data_frame.isnull().sum().sum()
    print('Total null values: ' + str(total_null))

my_data_frame.dropna(inplace=True)
print(my_data_frame)
my_data_frame.to_pickle('COS_Seesaw_dataframe_' + cos_site + '.pkl')

