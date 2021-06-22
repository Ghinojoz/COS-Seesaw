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

def dostuff(data):
    data_mean = data.mean(dim=('lat', 'lon'))
    print('plotting')
    fig = plt.figure()
    data_mean.sst.plot.line(label='mean_sst_CAR')
    plt.legend()
    plt.show()

my_data_frame = pd.DataFrame()

# Read jungfraujoch data and constrain to years between 2000-2018, convert dates to decimal time if necessary
print("retrieving jfj_data, and constraining to 2000-2018")
jfj_data = pd.read_csv('./SourceData/jfj_OCS_trop_VMR_DailyMeans_4Elliott.txt', header=3, delim_whitespace=True, skiprows=[4], parse_dates=['dd-mmm-yyyy'])
jfj_data = constrainDataFrameByYears(jfj_data, 'dd-mmm-yyyy', 2000, 2018)
print("success")

# strip down to just ocs, cos_stdev and time
jfj_data = pd.DataFrame({'time': jfj_data['dd-mmm-yyyy'], 'COS_JFJ' : jfj_data['ppt'], 'COS_STDEV_JFJ': jfj_data['stdev']})
jfj_data.to_pickle('COS-seesaw.pkl')

# Read manua load data from OCS_GCMS_flask.txt and constrain to years between 2000-2018, convert dates to decimal times
print('load mlo data and constrain to 2000-2018')
mlo_data = pd.read_csv('./SourceData/OCS__GCMS_flask.txt', delim_whitespace=True, header=1, parse_dates=['yyyymmdd'])
mlo_data = mlo_data.loc[mlo_data['site'] == 'mlo']
mlo_data = constrainDataFrameByYears(mlo_data, 'yyyymmdd', 2000, 2018)
print('success')

#close it
#del [[jfj_data, mlo_data]]
#gc.collect()

# load NOAA sst data
#print('Begin loading Noaa sst data')
#sst_data = xr.open_mfdataset('./SourceData/sst/*.nc', concat_dim='time')

# modify the longitude
#print('Modifying sst_data coordinates')
#sst_data = sst_data.assign_coords(lon=(((sst_data.lon + 180) % 360) - 180))
#sst_data = sst_data.sortby(sst_data.lon)
#sst_mask = regionmask.defined_regions.ar6.ocean.mask(sst_data)

#CAR_index = regionmask.defined_regions.ar6.ocean.map_keys('Caribbean')
#print('Getting CAR sst')
#CAR_sst = sst_data.where(sst_mask == CAR_index)
#print('Getting mean')
#dostuff(CAR_sst)

noaa_2000 = xr.open_dataset('./SourceData/sst/sst.day.mean.2000.nc').load()
noaa_200 = noaa_2000.assign_coords(lon=(((noaa_2000.lon + 180) % 360) - 180))
noaa_2000 = noaa_2000.sortby(noaa_2000.lon)
sst_mask = regionmask.defined_regions.ar6.ocean.mask(noaa_2000)
CAR_index = regionmask.defined_regions.ar6.ocean.map_keys('Caribbean')
CAR_sst = noaa_2000.where(sst_mask == CAR_index)
data_mean = CAR_sst.mean(dim=('lat','lon'))
fig = plt.figure()
data_mean.sst.plot.line(label='mean sst CAR')

#print(type(data_mean['time']))
#decimalDates = data_mean['time']
#cubic_interp = interp1d(decimalDates, data_mean['sst'], kind='cubic')
#plt.plot(decimalDates, cubic_interp(decimalDates), '--', label='interp')

interp = data_mean.sst.interp(time=list(map(str,jfj_data['time'])), method='cubic')

my_data_frame = jfj_data
my_data_frame['CAR_sst'] = np.nan
my_data_frame.reset_index(inplace=True, drop=True)
print(my_data_frame)

interp = interp.to_dataframe()
interp.reset_index(inplace=True, drop=True)
#my_data_frame.join(interp, on='time')
#my_data_frame.combine_first(interp.to_series(), axis=0)
print('--------------------------------------------------------------------')
#print(my_data_frame)
interp.columns= ['CAR_sst']
print(interp)
#interp.plot.line('o', label='cubic')
#interp.plot.line(label='cubic line')

#plt.legend()
#plt.show()

my_data_frame.update(interp)
print(my_data_frame)

# print(type(regionmask.defined_regions.ar6.ocean))

#print(regionmask.defined_regions.ar6.ocean.names)
#for region in regionmask.defined_regions.ar6.ocean:
#    print(region.number)




'''
print('load noaa_2000 data and adjust longitude')
noaa_2000 = xr.open_dataset('./SourceData/sst.day.mean.2000.nc').load()
noaa_2000 = noaa_2000.assign_coords(lon=(((noaa_2000.lon + 180) % 360 ) - 180))
noaa_2000 = noaa_2000.sortby(noaa_2000.lon)
print('success')

print('define regionmask for noaa_2000')
mask = regionmask.defined_regions.ar6.ocean.mask(noaa_2000);
SIO_index = regionmask.defined_regions.ar6.ocean.map_keys('S.Indic-Ocean')
noaa_SIO = noaa_2000.where(mask == SIO_index)

SOO_index= regionmask.defined_regions.ar6.ocean.map_keys('Southern-Ocean')
noaa_SOO = noaa_2000.where(mask == SOO_index)

noaa_SIO.mean(dim=('lat','lon')).sst.plot.line(label='South Indic Ocean')
noaa_SOO.mean(dim=('lat', 'lon')).sst.plot.line(label='Southern-Ocean')
plt.legend()
plt.show()

fig = plt.figure()
noaa_SIO.sst[0].plot()
noaa_SOO.sst[0].plot()
plt.show()
'''


'''
after_2000 = jfj_data['dd-mmm-yyyy'] >= '2000-1-1'
filtered_dates = jfj_data.loc[after_2000]
error = filtered_dates['stdev']
x = filtered_dates['dd-mmm-yyyy']
y = filtered_dates['ppt']

fig = plt.figure(figsize=(16,8))

plt.plot(x,y)
plt.title("Jungfraujoch Mean COS")
plt.ylabel("Mean COS")
plt.xlabel("Date")

plt.savefig('jungfraujoch_mean_cos.png')


plt.close()

fig = plt.figure(figsize=(16,8))

mlo_data = pd.read_csv('./SourceData/OCS__GCMS_flask.txt', delim_whitespace=True, header=1, parse_dates=['yyyymmdd'])
#mlo_data = pd.read_csv('OCS__GCMS_flask.txt', delim_whitespace=True, header=1)
manua_loa_filter = mlo_data['site'] == 'mlo'
manua_loa_dates = mlo_data.loc[manua_loa_filter]
#yerr = manua_loa_dates['OCS__sd']
x = manua_loa_dates['yyyymmdd']
y = manua_loa_dates['OCS_']
#plt.errorbar(x,y, yerr=yerr, fmt='o')
plt.plot(x,y)
plt.title("Manua Loa Mean COS")
plt.ylabel("Mean COS")
plt.xlabel('Date')

#plt.legend(['Jungfraujoch', 'Manua Loa'])
#plt.savefig('mlo_jfj_mean cos.png')

plt.savefig('manua_loa_mean_cos.png')
plt.close()

ds_disk = xr.open_dataset('./SourceData/sst.day.mean.2000.nc').load()
#print(ds_disk)


#fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14,4))
#ds_disk.xc.plot(ax=ax1)
#ds_disk.yc.plot(ax=ax2)
#fig = plt.figure(figsize=(16,8))
#ds_disk.sst[0].plot()
#plt.savefig('sst0.png')

#print(ds_disk)

fig = plt.figure(figsize=(8,16))
sst = ds_disk.sst
#sst.mean(dim='time').plot(vmin=-2,vmax=32)
atlantic_west = sst.sel(lat=slice(-60,60), lon=slice(300, 360))
atlantic_east = sst.sel(lat=slice(-60,60), lon=slice(0, 20))
atlantic_east[0].plot()
plt.figure(figsize=(8,16))
atlantic_west[0].plot()

#atlantic = xr.concat([atlantic_west, atlantic_east],'time')
atlantic_east.coords['lon'] = atlantic_east.coords['lon'] + 360
atlantic = atlantic_west.combine_first(atlantic_east)
plt.figure(figsize=(8,16))
atlantic[0].plot()

plt.figure(figsize=(8,12))

atlantic.mean(dim=('lat', 'lon')).plot()
plt.title('Atlantic Mean SST')
#plt.title('Thingy')
#atlantic_region = sst.sel(lat=slice(-60,60), lon=((sst.lon < 20) | (sst.lon > 300)))
#atlantic_region[0].plot()

#atlantic_east.mean(dim=('lat','lon')).plot()

#plt.figure(figsize=(16,8))
#sst.mean(dim=('time','lon')).plot()
plt.show()
#print(ds_disk.dims)
#a = ds_disk.mean(dim='time')
#fig = plt.figure(figsize=(16,8))
#a.plot.scatter()
#plt.savefig('aplot.png')
#da = ds_disk.to_array()
#print(da.dims)
'''
