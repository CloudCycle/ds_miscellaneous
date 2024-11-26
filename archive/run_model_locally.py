import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as md
from scipy import stats
from ccml.app_slump import *
from datetime import datetime
from dask import delayed, compute
from ds_tools import AWSClient
import matplotlib.dates as md
import holoviews as hv

hv.extension('bokeh')
redshift_client = AWSClient()
colors = ['#2E73E2', '#CC9933', '#33CC99', '#9933CC', '#343435', '#000066']



def generate_mapper(data):
    """Generate a Local model to make predictions"""
    MODEL_FILEPATH_MAPPING = "{'default': 's3://cloudcycle-ml-prediction-models/slump/production/model_objects/model_clsCloudCycleSlumpModel_v3_1_0.pkl'}"
	
    print("Local copy of model being used: model_clsCloudCycleSlumpModel_v3_1_0.pkl")
    MODEL_FILEPATH_MAPPING = "{'default':'model_clsCloudCycleSlumpModel_v3_1_0.pkl'}"

    model_mapper = clsModelMapper(
        model_filepath_mapping=ast.literal_eval(MODEL_FILEPATH_MAPPING),
        is_production=(IS_PRODUCTION == "1"),
        s3_cache_output_write=True,
        redshift_secret_name=REDSHIFT_SECRET_NAME
)
    ml = LocalMLAPI(model_mapper, DYNAMODBTABLE, data=data)

    return ml


def create_event(data, run_type='not-live'):
    """Create a payload for making local predictions"""
    input_json = {'str_request_type': 'predict',
                  'dynamodb_table': 'Production-DataStorage-Stack-augmentedlog6B4846BA-15J9I7UD422OI',
                  'imei': ['359206106340219'],
                  'run_id': [run_type],
                  'system_datetime_utc_str_start': ['2022-06-28T10:05:00.000Z'],
                  'system_datetime_utc_str_end': ['2022-06-28T11:00:00.000Z']}
        
    input_json['system_datetime_utc_str_start'] = [data['system_datetime_utc_str'].iloc[0]]
    input_json['system_datetime_utc_str_end'] = [data['system_datetime_utc_str'].iloc[-1]]
    input_json['imei'] = [data['imei'].iloc[0]]

    event = {"Records": [{"Sns": {"Message": json.dumps(input_json)}}]}

    return event


def local_predict(data:pd.DataFrame, ml:LocalMLAPI)->dict:
    """Run local predictions on a dataframe"""
    event = create_event(data)
    
    input_json = ml.gather_records(event)

    ml.data = data
    
    output = ml.predict(input_json)

    null = np.nan
    pred = eval(output)
    return pred



def batch_predict_pipeline(df_batch:pd.DataFrame) -> pd.DataFrame:
    """Generate predictions for a batch of data after cleaning rows with negative torque.
    Output dataframe may be shorted than input since rows with negative torque values are dropped"""
 
    df_calc = (df_batch
               .query("torque >= 0")
               .sort_values('system_datetime_utc_str')
               .reset_index()
              )
    
    ml = generate_mapper(df_calc)

    pred_ = local_predict(df_calc, ml)
    pred = pd.DataFrame(pred_)

    return pred


cols = ['calibration_gearbox_ratio',
         'calibration_motor_displacement_cm3',
         'calibration_motor_efficiency',
         'drum_data_valid',
         'drum_speed_mean_rpm',
         'imei',
         'prediction_trip_posix_timestamp_start',
         'pressure_a_bar_data_valid',
         'pressure_a_bar_mean',
         'pressure_a_bar_sd',
         'pressure_b_bar_data_valid',
         'pressure_b_bar_mean',
         'pressure_b_bar_sd',
         'rmc_provider',
         'run_id',
         'system_datetime_posix_utc_seconds',
         'system_datetime_utc_str',
         'torque',
         'truck_id',
         'truck_registration',
         'docket_delivered_quantity'
       ]


query_aug_log = """SELECT DATE_PART('EPOCH', CAST(system_datetime_utc_str AS TIMESTAMP)) AS epoch, 
                CAST(system_datetime_utc_str AS TIMESTAMP) AS time,
                (pressure_a_bar_mean - pressure_b_bar_mean) AS pressure_diff,
                prediction_slump_mm_current,
                prediction_truckstate_current,
                metrics_distance_travelled_metres,
                (10*drum_speed_mean_rpm) AS rpmx10,
                prediction_water_addition_confidence,
                prediction_water_addition_posix_timestamp_start,
                calibration_gearbox_ratio,
                calibration_motor_displacement_cm3,
                calibration_motor_efficiency,
                drum_data_valid,
                drum_speed_mean_rpm,
                imei,
                prediction_trip_posix_timestamp_start,
                pressure_a_bar_data_valid,
                pressure_a_bar_mean,
                pressure_a_bar_sd,
                pressure_b_bar_data_valid,
                pressure_b_bar_mean,
                pressure_b_bar_sd,
                run_id,
                system_datetime_posix_utc_seconds,
                system_datetime_utc_str,
                (pressure_a_bar_mean - pressure_b_bar_mean) * calibration_motor_displacement_cm3 * calibration_motor_efficiency / (20 * PI()) AS torque,
                1 as truck_id,
                'Holcim' as rmc_provider,
                truck_registration,
                docket_delivered_quantity
                
                FROM augmented_log
                
    WHERE epoch BETWEEN '{start_date}' AND '{end_date}'
    AND truck_registration IN {trucks}
    ORDER BY truck_registration, time
    --LIMIT 10
    """


to_timestamp = lambda x: (x - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')

def predict_and_plot_offline(df_batch):
    df_res = (batch_predict_pipeline(df_batch)
              .set_index(['truck_registration', 
                          'prediction_trip_posix_timestamp_start', 
                          'system_datetime_posix_utc_seconds'])
              .assign(time = lambda x: pd.to_datetime(x['system_datetime_utc_str']),
                     prediction_slump_mm_current= lambda x: x['prediction_slump_mm'])
             )

    calculated_slump = (hv
                        .Curve(df_res, 'time', 'prediction_slump_mm_current', label='offline slump')
                       )
    return calculated_slump
    
    

def plot_speed_torque_slump(df_batch):
    torque = hv.Curve(df_batch, 'time', 'torque', label='torque')
    speed = (hv.Curve(df_batch
                      .assign(speed=lambda x: x['drum_speed_mean_rpm']*((df_batch['torque']>=0) + -1*((df_batch['torque']<0)))*10), 
                      'time', 'speed', label='drum speed x10')
             )

    system_slump = (hv
                    .Curve(df_batch, 'time', 'prediction_slump_mm_current', label='slump')
                   )
    overlay = torque*system_slump*speed    
    return overlay
    

def plot_slumps(df, offline=False, opts=None, backend='bokeh', save=False):
    hv.extension(backend)
    for (truck_registration, trip_start), df_batch in df.groupby(level=[0,1]):
        overlay = plot_speed_torque_slump(df_batch)
        
        if offline and (trip_start > 0): # Calculate and plot offline slumps
            calculated_slump = predict_and_plot_offline(df_batch)
            overlay = overlay * calculated_slump
        
        if opts: overlay.opts(opts)
        trip_start_datetime = np.datetime_as_string(np.array(trip_start, dtype='datetime64[s]'))
        display(overlay.opts(title=f'Truck: {truck_registration}, trip start: {trip_start_datetime}', ylim=(-150,300)))
        
        if save and opts: 
            trip_start_datetime = trip_start_datetime.replace(":", "_")
            hv.save(overlay, 
                    filename=f'figures/milan_history_july/{truck_registration}_{trip_start_datetime}.png',
                   backend='matplotlib', dpi=150)
        # print(truck_registration, trip_start)
        
    return None
    
    
    
if __name__ == '__main__':
    
    
    start_date = np.datetime64("2022-07-04")
    end_date = np.datetime64("2022-07-15")
    
    trucks = ('GF621GH', 'FT110NM', 'GA284CH', 'GH284XK', 'GA675CH')
    df = (redshift_client
          .execute_redshift_query(query_aug_log
                                  .format(start_date=to_timestamp(start_date),
                                          end_date = to_timestamp(end_date),
                                          trucks=trucks)
                                 )
          .assign(run_id="not-live",
                  pressure_a_bar_data_valid = lambda x: x['pressure_a_bar_data_valid'].map({'true':True, 'false':False}),
                  pressure_b_bar_data_valid = lambda x: x['pressure_b_bar_data_valid'].map({'true':True, 'false':False}),
                  drum_data_valid = lambda x: x['drum_data_valid'].map({'true':True, 'false':False}),
                  time = lambda x: pd.to_datetime(x['system_datetime_utc_str'])
                )
          .set_index(['truck_registration', 'prediction_trip_posix_timestamp_start', 'system_datetime_posix_utc_seconds'])
         )
        
    df.to_pickle('figures/milan_history_july/milan_trucks_data_20220704-20220715.pickle')
    
    plot_slumps(df, offline=True, backend='bokeh')
    
    
    hv.extension('matplotlib')
    opts = [hv.opts.Curve(show_grid=True, color=hv.Cycle(colors), xformatter=md.DateFormatter('%H:%M')), 
            hv.opts.Overlay(show_frame=True, 
                            fig_size=240, 
                            aspect=1.6, 
                            xlabel='Time (UTC)', 
                            ylabel='values')]
    
    plot_slumps(df, offline=False, opts=opts, backend='matplotlib', save=True)


    