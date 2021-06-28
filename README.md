# COS-Seesaw
 
## Overview of repo
Most of the work is done by two scripts within the repo, build_df.py and Seesaw_GA2M.ipynb. If you intend to run the scipts here, be sure to pip install the requirements.txt file, which should include all dependencies. You may also wish to download the COS_Seesaw_dataframe.pkl which is the current dataframe used for the project. This would allow you to run the machine learning portion (Seesaw_GA2M.ipynb) without having to first having to build the data frame using build_df.py, which is currently quite slow

## build_df.py
The first script. build_df.py, is run to generate the dataframe for ebm to work upon. It requires a SourceData subdirectory which is not included in the repo, but can be downloaded from: location coming soon. This directory should be placed in the same directory as build_df.py. Execution of build_df.py is currently quite slow, taking roughly 20 minutes to complete, however improving execution time is a priority and we hope to see improvement soon. Upon completion, the data frame contains oservations of the COS ppt for Jungfraujoch, as well as mean temperature for several oceanic regions corresponding to: https://regionmask.readthedocs.io/en/stable/_images/plotting_ar6_ocean.png

## Seesaw_GA2M.ipynb
This jupyter notebook program trains and explains a GA2M implementation, specifically, it trains an explainable boosting regressor. The documentation for this machine learning model can be found here: https://github.com/interpretml/interpret . The model currently has a high RMSE, which is to be expected at this point in the project


## Note
The environment.yml and requirements.txt are not working properly for non-linux systems at this time. I hope to have this resolved soon.
