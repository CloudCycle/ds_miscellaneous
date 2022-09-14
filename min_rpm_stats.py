import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as md
from datetime import datetime
# from dask import delayed, compute
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
                drum_data_valid,
                drum_speed_minimum_rpm,
                imei,
                prediction_slump_mm_current,
                prediction_trip_posix_timestamp_start,
                pressure_a_bar_mean,
                pressure_b_bar_mean,
                system_datetime_utc_str,
                (pressure_a_bar_mean - pressure_b_bar_mean) * calibration_motor_displacement_cm3 * calibration_motor_efficiency / (20 * PI()) AS torque,
                truck_registration

                FROM augmented_log
                
    WHERE system_datetime_utc_str BETWEEN '{start_date}' AND '{end_date}'
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

 

def is_cifa(df_trip, df_truck):
    truck_registration = df_trip.index.get_level_values(0).unique()[0]
    
    if truck_registration in df_truck.index:
        truck_attrs = df_truck.loc[truck_registration]
        if truck_attrs['Base'] == 'Segrate':
            return True
        elif isinstance(truck_attrs['Truck Type'], str) and ('cifa' in truck_attrs['Truck Type'].lower()):
            return True
        else:
            return False
        
    else:
        return False

    

def trip_duration_seconds(df_trip):
    """trip duration"""
    system_datetime_utc = df_trip.index.get_level_values('system_datetime_utc_str')
    time_delta = system_datetime_utc.max() - system_datetime_utc.min()
    return time_delta.total_seconds()


def drum_data_valid_seconds(df_trip):
    """Total duration when drum data was valid"""
    system_datetime_utc = (df_trip
                           .index
                           .get_level_values('system_datetime_utc_str')
                          )
    time_delta = np.trapz(df_trip['drum_data_valid']*1.0, system_datetime_utc)
    return time_delta.total_seconds()



def is_short_trip(df_trip):
    """Identify short trips to filter out"""
    if len(df_trip) < 6: # less than 1 min
        return True
    else:
        return False
    
    
    
    
def drum_speed_histogram(df_trip, levels_=np.arange(0,4.25,0.25), 
                         filter_func=lambda x:x['metrics_distance_travelled_metres']>10):
    """Calulate time residency of drum speed at different levels"""
    drum_speed = df_trip.get('drum_speed_mean_rpm')
    system_datetime_utc = df_trip.index.get_level_values('system_datetime_utc_str')
    filter_vector = filter_func(df_trip)*(df_trip['torque']>=0)*1.0

    residency = dict()
    levels = np.append(levels_, 20)
    for i,level in enumerate(levels):
        if i == 0:
            speed_under_level = drum_speed <= level
        else:
            speed_under_level = (levels[i-1] < drum_speed) & (drum_speed <= level)

        time_delta = np.trapz(speed_under_level*filter_vector, system_datetime_utc).total_seconds()
        
        if level == 20: key = '>4'
        else: key = str(level)
        
        residency[key] = time_delta


    return pd.DataFrame([residency])


if __name__ == '__main__':
    
    start_date = np.datetime64("2022-07-01")
    end_date = np.datetime64("2022-08-01")
    
    
    df_ = (redshift_client
          .execute_redshift_query(query_aug_log
                                  .format(start_date=np.datetime_as_string(start_date),
                                          end_date =np.datetime_as_string(end_date))
                                 )
         )

    
    df = (df_
          .assign(system_datetime_utc_str=lambda x: pd.to_datetime(x['system_datetime_utc_str'], utc=True))
          .astype({'drum_data_valid':bool})
          .set_index(['truck_registration', 'prediction_trip_posix_timestamp_start', 'system_datetime_utc_str'])
         )
    
    df_sample = df.iloc[500000:510000]
    

    df_truck = (pd
                .read_excel('v3_Tracker_1663077904.xlsx', skiprows=2, nrows=33)
                .set_index('Registration')
               )
    

    
    df_unfilt = df.join(df.groupby(level=[0,1]).apply(is_short_trip).to_frame(name='is_short_trip'))
    
    g = df_unfilt.query("is_short_trip==False").drop('<unknown>',level=0).groupby(level=[0,1])
    
    df_move = (g
     .apply(drum_speed_histogram)
     .droplevel(2)
     .join(g.apply(trip_duration_seconds).to_frame(name='trip_duration'))
     .join(g.apply(drum_data_valid_seconds).to_frame(name='drum_data_valid_duration'))
     .join(g.apply(is_cifa, df_truck=df_truck).to_frame(name='is_cifa'))
    )

    
    hv.Bars((df_move
             .groupby('is_cifa')
             .sum()
             .drop(['trip_duration','drum_data_valid_duration'], axis=1)
             .loc[False]
             .T
             .to_frame()
             .reset_index()
             .rename({'index':'rpm_bin', False:'time_spent'},axis=1))
           ).opts(width=600, height=400, show_grid=True, title='Non Cifa during truck motion')
    
    
    hv.Bars((df_move
             .groupby('is_cifa')
             .sum()
             .drop(['trip_duration','drum_data_valid_duration'], axis=1)
             .loc[True]
             .T
             .to_frame()
             .reset_index()
             .rename({'index':'rpm_bin', True:'time_spent'},axis=1))
           ).opts(width=600, height=400, show_grid=True, title='Cifa during truck motion')
    
    
    
    df_stop = (g
     .apply(drum_speed_histogram, filter_func=lambda x: x['metrics_distance_travelled_metres']<=10)
     .droplevel(2)
     .join(g.apply(trip_duration_seconds).to_frame(name='trip_duration'))
     .join(g.apply(drum_data_valid_seconds).to_frame(name='drum_data_valid_duration'))
     .join(g.apply(is_cifa, df_truck=df_truck).to_frame(name='is_cifa'))
    )

    hv.Bars((df_stop
             .groupby('is_cifa')
             .sum()
             .drop(['trip_duration','drum_data_valid_duration'], axis=1)
             .loc[False]
             .T
             .to_frame()
             .reset_index()
             .rename({'index':'rpm_bin', False:'time_spent'},axis=1))
           ).opts(width=600, height=400, show_grid=True, title='Non Cifa during truck stationary')
    
    
    hv.Bars((df_stop
             .groupby('is_cifa')
             .sum()
             .drop(['trip_duration','drum_data_valid_duration'], axis=1)
             .loc[True]
             .T
             .to_frame()
             .reset_index()
             .rename({'index':'rpm_bin', True:'time_spent'},axis=1))
           ).opts(width=600, height=400, show_grid=True, title='Cifa during truck stationary')
    