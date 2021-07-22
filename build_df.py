
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
    



def loadCOSData(file_name, time_column_name, site_name, start, end):
    print(divider)
    print('Load COS data for' + cos_site + ' from ' + str(start) + ' to ' + str(end))
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

    print(divider)
    print('Success!')
    print(divider)

    return cos_data.reset_index(drop=True)



# ---------------------------------------------------------------------------------------------------------------------
# build data frame and save as pickle
# ---------------------------------------------------------------------------------------------------------------------
my_data_frame = loadCOSData(cos_file, 'yyyymmdd', cos_site, year_start, year_end)
regions = buildRegions()




print (my_data_frame)



