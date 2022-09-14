import pandas as pd
from ds_tools import AWSClient
import holoviews as hv
import numpy as np

pd.set_option('display.max_rows', 200)
hv.extension('bokeh')
redshift_client = AWSClient()

colors = ['#2E73E2', '#CC9933', '#33CC99', '#9933CC', '#343435', '#000066']

to_timestamp = lambda x: np.array(x, 'datetime64[s]').astype(float)
from_timestamp = lambda x: np.array(x).astype('datetime64[s]')



 
query = """SELECT truck_registration, 
                    prediction_trip_posix_timestamp_start, 
                    system_datetime_utc_str, 
                    drum_speed_mean_rpm,
                    metrics_distance_travelled_metres,
                    prediction_slump_mm_current,
                    docket_slump_code,
                    (pressure_a_bar_mean - pressure_b_bar_mean) * calibration_motor_displacement_cm3 * calibration_motor_efficiency / (20 * PI()) AS torque,
                    geofence_type_current,
                    prediction_water_addition_posix_timestamp_start,
                    prediction_water_addition_confidence

FROM augmented_log

WHERE system_datetime_utc_str BETWEEN '{start_date}' AND '{end_date}'
AND truck_registration in {truck_list}

ORDER BY truck_registration, prediction_trip_posix_timestamp_start, system_datetime_utc_str       

"""



def plot_sensor_data(df_trip):
    """Plot fundamental channels"""
    df_plot = (df_trip
               .assign(rpmx10 = lambda x: x['drum_speed_mean_rpm']*10*np.sign(x['torque']))
               .sort_index()
               )
    
    f_torque = (hv.Scatter(df_plot, 'system_datetime_utc_str', 'torque', group='sensor', label='torque').opts(color=colors[0], show_grid=True)
                * hv.Curve(df_plot, 'system_datetime_utc_str', 'torque', group='sensor', label='torque').opts(color=colors[0], alpha=0.5)
               )

    f_speed = (hv.Scatter(df_plot, 'system_datetime_utc_str', 'rpmx10', group='sensor', label='drum speed x10').opts(color=colors[1])
               * hv.Curve(df_plot, 'system_datetime_utc_str', 'rpmx10', group='sensor', label='drum speed x10').opts(color=colors[1], alpha=0.5)
              )
    
    f_distance = (hv.Scatter(df_plot, 'system_datetime_utc_str', 'metrics_distance_travelled_metres', group='sensor', label='inst. distance (m)').opts(color=colors[2])
                  * hv.Curve(df_plot, 'system_datetime_utc_str', 'metrics_distance_travelled_metres', group='sensor', label='inst. distance (m)').opts(color=colors[2], alpha=0.5)
                 )
    
    overlay = f_torque*f_speed*f_distance
    return overlay




if __name__ == '__main__':
    
    truck_list = ('FJ22PXX', 'FJ22PXV', 'FJ22PXO', 'FJ22PXP', 'FJ22PXW')
    start_date = np.datetime64("2022-08-29")
    end_date = np.datetime64("2022-09-09")

    df_ = (redshift_client
          .execute_redshift_query(query
                                  .format(start_date=np.datetime_as_string(start_date),
                                          end_date=np.datetime_as_string(end_date),
                                         truck_list = truck_list)
                                 )
          )

    df = (df_
          .assign(system_datetime_utc_str = lambda x: pd.to_datetime(x['system_datetime_utc_str'], utc=True))
          .set_index(['truck_registration', 'prediction_trip_posix_timestamp_start', 'system_datetime_utc_str'])
         )
    
    df_grouped = df.drop(-99999.0, level=1).groupby(level=[0,1])
    
    opts = [hv.opts.Overlay(width=650, height=400),
           hv.opts.Scatter('sensor', tools=['hover'])]
    
    for (truck, trip_start), df_trip in df_grouped:
        if (trip_start >= to_timestamp('2022-09-07')): 
            o = plot_sensor_data(df_trip).opts(opts)
            display(o)
            display((truck, trip_start), from_timestamp(trip_start), df_trip['docket_slump_code'].unique())
   
    
    # Observations
    # Normal behavior under discharge and drum spin up
    # Ocassional speed sensor bottoming out for couple of samples
    # FJ22PXW not sending speed data since mid day 2022-09-06