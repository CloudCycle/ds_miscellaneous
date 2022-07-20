import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as md
from scipy import stats
from datetime import datetime
from dask import delayed, compute
from ds_tools import AWSClient
import matplotlib.dates as md
import holoviews as hv
from IPython.display import display

pd.set_option('display.max_rows',500)
hv.extension('matplotlib')
redshift_client = AWSClient()
colors = ['#2E73E2', '#CC9933', '#33CC99', '#9933CC', '#343435', '#000066']





query_aug_log = """SELECT DATE_PART('EPOCH', CAST(system_datetime_utc_str AS TIMESTAMP)) AS epoch, 
                (pressure_a_bar_mean - pressure_b_bar_mean) AS pressure_diff,
                prediction_truckstate_current,
                metrics_distance_travelled_metres,
                drum_speed_mean_rpm,
                imei,
                prediction_trip_posix_timestamp_start,
                pressure_a_bar_mean,
                pressure_b_bar_mean,
                system_datetime_posix_utc_seconds,
                system_datetime_utc_str,
                (pressure_a_bar_mean - pressure_b_bar_mean) * calibration_motor_displacement_cm3 * calibration_motor_efficiency / (20 * PI()) AS torque,
                truck_registration,
                imu_temperature_degc,
                system_cpu_temperature_degc,
                temperature_module_surface_temperature_degc,
                temperature_module_ambient_temperature_degc,
                temperature_module_sunhat_temperature_degc,
                temperature_module_data_valid,
                system_enclosure_temperature_degc
                
                FROM augmented_log
                
    WHERE epoch BETWEEN '{start_date}' AND '{end_date}'
    -- AND truck_registration IN {trucks}
    ORDER BY truck_registration, system_datetime_posix_utc_seconds
    --LIMIT 1000
    """

to_timestamp = lambda x: (x - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')


if __name__ == '__main__':
    
    start_date = np.datetime64("2022-07-11")
    end_date = np.datetime64("2022-07-20")
    
    trucks = ('GF621GH', 'FT110NM', 'GA284CH', 'GH284XK', 'GA675CH')
    df = (redshift_client
          .execute_redshift_query(query_aug_log
                                  .format(start_date=to_timestamp(start_date),
                                          end_date = to_timestamp(end_date),
                                          trucks=trucks)
                                 )
          .assign(time = lambda x: pd.to_datetime(x['system_datetime_utc_str']))
          .set_index(['imei', 'system_datetime_posix_utc_seconds'])
         )
    
    
    
    opts = hv.opts.Histogram(alpha=0.5, 
                             show_grid=True, 
                             show_frame=True, 
                             fig_size=240, 
                             aspect=1.6, 
                             xlabel='Temperature (deg C)',
                             color=hv.Cycle(colors))
    channels = ['system_cpu_temperature_degc', 
               'system_enclosure_temperature_degc',
               'imu_temperature_degc']
    for device, df_trip in df.groupby(level=[0]):
        if (len(df_trip) < 10): continue
        o = []
        
        for channel in channels:
            if not df_trip[channel].isna().all():
                h = hv.Histogram(np.histogram(df_trip[channel].dropna()), label=channel).opts(opts)
                o.append(h)
        
        overlay = hv.Overlay(o).opts(title=f"imei: {device}")
        display(overlay)
        hv.save(overlay, f"figures/hot_weather/{device}.png", dpi=150)
        
        
    df.to_csv("hot_weather_data.csv")
    