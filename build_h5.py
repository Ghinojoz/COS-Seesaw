from matplotlib.transforms import Bbox
import numpy as np
import h5py
import pandas as pd
import datetime
import warnings
import matplotlib.pyplot as plt
import xarray as xr

jfj_data = pd.read_csv('./SourceData/jfj_OCS_trop_VMR_DailyMeans_4Elliott.txt', header=3, delim_whitespace=True, skiprows=[4], parse_dates=['dd-mmm-yyyy'])

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