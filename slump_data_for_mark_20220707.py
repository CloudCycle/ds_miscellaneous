import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as md
import seaborn as sns
from datetime import datetime, timedelta
from bokeh.plotting import figure,show, gridplot, output_notebook, ColumnDataSource
from bokeh.models import HoverTool, Range1d, BooleanFilter, CDSView, ColumnDataSource
from bokeh.io import export_png
from scipy import stats
from contextlib import contextmanager,redirect_stderr,redirect_stdout
from os import devnull
import re
from data_parsing import CalibrationDataConstructorCC8
from itertools import chain, product, combinations
from scipy import stats
from ds_tools import AWSClient
from utils import PlotWaterAddition as pwa, TruckData
import holoviews as hv
from sklearn.linear_model import LinearRegression
from collections import defaultdict
from utils import TruckData, PlotDatasets
from ccml.app_slump import *
from tqdm import tqdm
from collections import defaultdict
from dateutil import parser
from dask import delayed, compute


pd.set_option('display.max_rows', 200)
redshift_client = AWSClient()


query_augmented_log = """SELECT AL.system_datetime_utc_str::TIMESTAMP, 
                AL.truck_registration, 
                AL.pressure_b_bar_sd, 
                AL.docket_delivered_quantity, 
                AL.truck_id, 
                AL.calibration_gearbox_ratio, 
                AL.pressure_a_bar_data_valid, 
                AL.temperature_module_concrete_temperature_degc, 
                AL.rmc_provider, 
                AL.temperature_module_data_valid, 
                AL.drum_data_valid, 
                AL.pressure_b_bar_mean, 
                AL.pressure_a_bar_mean, 
                AL.pressure_a_bar_sd, 
                AL.prediction_trip_posix_timestamp_start, 
                AL.run_id, 
                AL.calibration_motor_efficiency, 
                AL.pressure_b_bar_data_valid, 
                AL.calibration_motor_displacement_cm3, 
                AL.system_datetime_posix_utc_seconds, 
                AL.imei, 
                AL.drum_speed_mean_rpm,
                AL.docket_datetime_batch_utc,
                AL.docket_id,
                AL.docket_slump_code,
                AL.metrics_distance_travelled_metres,
                AL.prediction_rotation_posix_timestamp_start,
                AL.prediction_slump_mm_current,
                AL.prediction_slump_mm_lastvalid,
                AL.prediction_slump_model_filename_current,
                msp.prediction_slump_mm_uncertainty,
                msp.prediction_slump_mm_unsaturated_uncertainty,
                msp.drum_speed_power_intercept_uncertainty,
                msp.drum_speed_power_intercept,
                msp.prediction_slump_mm_unsaturated
                FROM augmented_log AL 
                LEFT JOIN ml_slump_production msp 
                     ON AL.truck_registration = msp.truck_registration 
                     AND AL.system_datetime_posix_utc_seconds = msp.system_datetime_posix_utc_seconds
                     
    WHERE AL.system_datetime_utc_str BETWEEN '{start_date}' AND '{end_date}'
    ORDER BY AL.truck_registration, AL.system_datetime_posix_utc_seconds
    

"""


if __name__ == '__main__':
     
    start_date = '2022-07-01'
    end_date = '2022-07-08'

    df_aug_log = redshift_client.execute_redshift_query(query_augmented_log.format(start_date=start_date, end_date=end_date))
    
    df_aug_log.to_csv('augmented_log_20220701-20220708.csv')
    
    start_date = '2022-06-23'
    end_date = '2022-06-30'

    df_aug_log = redshift_client.execute_redshift_query(query_augmented_log.format(start_date=start_date, end_date=end_date))
    
    df_aug_log.to_csv('augmented_log_20220623-20220630.csv')
  
