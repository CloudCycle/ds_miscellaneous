from ds_tools import AWS_Storage_client, AWSClient
import joblib
import pandas as pd 
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
from tqdm import tqdm


pd.set_option('display.max_rows',200)

redshift_client = AWSClient()

def query_s3_bucket(days, hour=None, bucket_name='production-datastorage-stack-v4rawf6afa338-159ye6v4addzx'):
    s3_client = AWS_Storage_client(bucket_name=bucket_name)
    return [s3_client.execute_bucket_query(day, hour=hour, ) for day in days]


    
buckets = ["production-datastorage-stac-rawdevicejson568138dc-1djhnp70ebsko", 
           "production-datastorage-stack-rawavro410deda0971-16a7clzjn1491"
          ]


aug_log_query = """SELECT * 
                FROM augmented_log
                WHERE system_datetime_utc_str BETWEEN '{start_date}' AND '{end_date}'
                AND
                truck_registration = '{truck}'
                ORDER BY IMEI, system_datetime_utc_str
                LIMIT 100
                
"""

if __name__ == '__main__':
    
    truck = 'KX67RKJ'
    start_date = datetime(2022,10,18)
    end_date = datetime(2022,10,19)
    results = []
    errors = dict()
    for dt_obj in tqdm(pd.date_range(start_date, end_date)):
        try:
            results.append(query_s3_bucket([dt_obj.strftime('%Y/%m/%d')], bucket_name=buckets[0])[0])
        except Exception as e:
            errors[dt_obj] = e

    df_packet = pd.concat(results)
    
    df_aug_log_ = (redshift_client
                  .execute_redshift_query(aug_log_query.format(start_date=start_date.strftime("%Y-%m-%d"), 
                                                               end_date=end_date.strftime("%Y-%m-%d"),
                                                              truck=truck))
                 )

    df_aug_log = (df_aug_log_
                  .rename({'water_cumu_vol':'water_flowmeter_total_volume_m3',
                          'water_flow_rate':'water_flowmeter_flow_rate_m3/hr',
                          'water_temp':'water_flowmeter_temperature_degc',
                          'water_valid':'water_flowmeter_data_valid',
                          'pressure_a_temperature_degc':'pressure_a_bar_temperature_degc',
                          'pressure_b_temperature_degc':'pressure_b_bar_temperature_degc',
                          }, 
                          axis='columns')
                 )

    
    df_packet.columns.str.lower().difference(df_aug_log.columns)
    
    
    # drum_speed_values_used , pressure_a_bar_values_used 
    
    