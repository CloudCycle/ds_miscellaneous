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
hv.extension('bokeh')
redshift_client = AWSClient()
colors = ['#2E73E2', '#CC9933', '#33CC99', '#9933CC', '#343435', '#000066']



query_aug_log = """SELECT
                (pressure_a_bar_mean - pressure_b_bar_mean) AS pressure_diff,
                prediction_truckstate_current,
                metrics_distance_travelled_metres,
                drum_speed_mean_rpm,
                imei,
                prediction_slump_mm_current,
                prediction_trip_posix_timestamp_start,
                pressure_a_bar_mean,
                pressure_b_bar_mean,
                system_datetime_posix_utc_seconds,
                system_datetime_utc_str,
                (pressure_a_bar_mean - pressure_b_bar_mean) * calibration_motor_displacement_cm3 * calibration_motor_efficiency / (20 * PI()) AS torque,
                truck_registration

                FROM augmented_log
                
    WHERE system_datetime_utc_str BETWEEN '{start_date}' AND '{end_date}'
    AND truck_registration IN {trucks}
    ORDER BY truck_registration, system_datetime_posix_utc_seconds
    --LIMIT 1000
    """

to_timestamp = lambda x: (x - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')


if __name__ == '__main__':
    
    start_date = np.datetime64("2022-07-11")
    end_date = np.datetime64("2022-07-21")
    
    trucks = ('FJ22PXW', 'FJ22PXV', 'FJ22PXU', 'FJ22PXY', 'FJ22PXO', 'FJ22PXP', 'FJ22PXX', 'FJ22PXL')
    df = (redshift_client
          .execute_redshift_query(query_aug_log
                                  .format(start_date=np.datetime_as_string(start_date),
                                          end_date =np.datetime_as_string(end_date),
                                          trucks=trucks)
                                 )
          .assign(time = lambda x: pd.to_datetime(x['system_datetime_utc_str']))
          .set_index(['truck_registration', 'prediction_trip_posix_timestamp_start', 'system_datetime_utc_str'])
         )
    
    
    opts = [hv.opts.Overlay(width=500, height=350, ylim=(-50,250)), hv.opts.Scatter(tools=['hover'])]
    for truck, df_truck in df.groupby(level=[0]):
        df_plot = (df_truck
                   .loc[(truck, slice(None), slice(None))]
                   .assign(rpmx10 = lambda x: x['drum_speed_mean_rpm']*10)
                  )

        f_torque = (hv.Scatter(df_plot, 'time', 'torque', label='torque').opts(color=colors[0], show_grid=True)
                    *hv.Curve(df_plot, 'time', 'torque', label='torque').opts(color=colors[0], alpha=0.5))
        
        f_speed = (hv.Scatter(df_plot, 'time', 'rpmx10', label='rpmx10').opts(color=colors[1])
                    *hv.Curve(df_plot, 'time', 'rpmx10', label='rpmx10').opts(color=colors[1], alpha=0.5))
        
        f_slump = (hv.Scatter(df_plot, 'time', 'prediction_slump_mm_current', label='slump').opts(color=colors[2])
                    *hv.Curve(df_plot, 'time', 'prediction_slump_mm_current', label='slump').opts(color=colors[2], alpha=0.5))
        
        f_distance = (hv.Scatter(df_plot, 'time', 'metrics_distance_travelled_metres', label='inst. distance').opts(color=colors[3])
                    *hv.Curve(df_plot, 'time', 'metrics_distance_travelled_metres', label='inst. distance').opts(color=colors[3], alpha=0.5))
        
        overlay = f_torque * f_speed * f_slump * f_distance
        opts.append(hv.opts.Overlay(title=truck))
        display(overlay.opts(opts))
    
    