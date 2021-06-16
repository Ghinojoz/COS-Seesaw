from matplotlib.transforms import Bbox
import numpy as np
import h5py
import pandas as pd
import datetime
import warnings
import matplotlib.pyplot as plt

jfj_data = pd.read_csv('jfj_OCS_trop_VMR_DailyMeans_4Elliott.txt', header=3, delim_whitespace=True, skiprows=[4], parse_dates=['dd-mmm-yyyy'])

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

mlo_data = pd.read_csv('OCS__GCMS_flask.txt', delim_whitespace=True, header=1, parse_dates=['yyyymmdd'])
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
