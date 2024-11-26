from ds_tools import AWSClient
import pandas as pd
import numpy as np
import holoviews as hv
from IPython.display import display

hv.extension('bokeh')

pd.set_option("display.max_rows",200)
pd.set_option("display.max_columns",100)
redshift_client = AWSClient()
hv.extension('matplotlib')


query_aug_log = """SELECT
                (pressure_a_bar_mean - pressure_b_bar_mean) AS pressure_diff,
                prediction_truckstate_current,
                metrics_distance_travelled_metres,
                drum_speed_mean_rpm,
                drum_data_valid,
                drum_speed_minimum_rpm,
                imei,
                docket_id,
                prediction_slump_mm_current,
                prediction_trip_posix_timestamp_start,
                pressure_a_bar_mean,
                pressure_b_bar_mean,
                system_datetime_utc_str,
                (pressure_a_bar_mean - pressure_b_bar_mean) * calibration_motor_displacement_cm3 * calibration_motor_efficiency / (20 * PI()) AS torque,
                truck_registration

                FROM augmented_log
                
    WHERE system_datetime_utc_str BETWEEN '{start_date}' AND '{end_date}'
    AND truck_registration = '{truck}'
    AND prediction_trip_posix_timestamp_start > 0
    ORDER BY truck_registration, prediction_trip_posix_timestamp_start, system_datetime_utc_str
    --LIMIT 1000
    """

to_timestamp = lambda x: np.array(x, dtype='datetime64[s]').astype(float)
from_timestamp = lambda x: np.array(x, dtype=float).astype('datetime64[s]')



def plot_fundamentals(df):
    """Plot basic channels for signal analysis"""
    df_plot = (df
               .assign(rpmx10 = lambda x: x['drum_speed_mean_rpm']*10)
              )

    f_torque = (hv.Scatter(df_plot, 'system_datetime_utc_str', 'torque', label='torque').opts(color=colors[0], show_grid=True)
                *hv.Curve(df_plot, 'system_datetime_utc_str', 'torque', label='torque').opts(color=colors[0], alpha=0.5))

    f_speed = (hv.Scatter(df_plot, 'system_datetime_utc_str', 'rpmx10', label='rpmx10').opts(color=colors[1])
                *hv.Curve(df_plot, 'system_datetime_utc_str', 'rpmx10', label='rpmx10').opts(color=colors[1], alpha=0.5))

    f_slump = (hv.Scatter(df_plot, 'system_datetime_utc_str', 'prediction_slump_mm_current', label='slump').opts(color=colors[2])
                *hv.Curve(df_plot, 'system_datetime_utc_str', 'prediction_slump_mm_current', label='slump').opts(color=colors[2], alpha=0.5))

    f_distance = (hv.Scatter(df_plot, 'system_datetime_utc_str', 'metrics_distance_travelled_metres', label='inst. distance').opts(color=colors[3])
                *hv.Curve(df_plot, 'system_datetime_utc_str', 'metrics_distance_travelled_metres', label='inst. distance').opts(color=colors[3], alpha=0.5))

    overlay = f_torque * f_speed * f_slump * f_distance

    return overlay


if __name__ == '__main__':
    
    start_date = np.datetime64("2022-09-12")
    end_date = np.datetime64("2022-10-01")
    
    truck = 'RX16WXT'
    df_ = (redshift_client
          .execute_redshift_query(query_aug_log
                                  .format(start_date=np.datetime_as_string(start_date),
                                          end_date =np.datetime_as_string(end_date),
                                          truck=truck)
                                 )
         )

    
    df = (df_
          .assign(system_datetime_utc_str=lambda x: pd.to_datetime(x['system_datetime_utc_str'], utc=True))
          .set_index(['truck_registration', 'prediction_trip_posix_timestamp_start', 'system_datetime_utc_str'])
         )
    
    
#     df_before = df.xs(slice(pd.to_datetime('2022-09-12'),pd.to_datetime('2022-09-17')), level=2, drop_level=False)
    
    
#     o_ = []
#     for (truck, trip_start), df_trip in df_before.groupby(level=[0,1]):
#         time = df_trip.index.get_level_values(2).to_series()
#         time_diff = time.diff().dt.seconds
#         o = hv.Histogram(np.histogram(time_diff.dropna()))
#         display(o.opts(title=f"{truck}, {trip_start}"))
#         print(truck, trip_start)
        
    
    df['date'] = df.index.get_level_values(2).to_series().dt.date.values
    for date in np.arange(start_date, end_date+np.timedelta64(1, 'D')):
        df_day = df.query("date==@date")
        time = df_day.index.get_level_values(2).to_series()
        time_diff = time.diff().dt.seconds
        bins = np.arange(5,305,10)
        o = hv.Histogram(np.histogram(time_diff.dropna(), bins))
        display(o)
        display(date, pd.DataFrame(np.histogram(time_diff.dropna(), bins)))
        
        

    
    