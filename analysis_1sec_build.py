import pandas as pd
import numpy as np
from ds_tools import AWSClient
import holoviews as hv
from IPython.display import display
from ccml.app_slump import *
import json
from read_packet_data import *


pd.set_option("display.max_rows", 500)
hv.extension("bokeh")
redshift_client = AWSClient()

colors = ["#2E73E2", "#CC9933", "#33CC99", "#9933CC", "#343435", "#000066"]

MODEL_FILEPATH_MAPPING = "{'default':'model_clsCloudCycleSlumpModel_v3_2_4.pkl'}"


def generate_mapper(data, MODEL_FILEPATH_MAPPING):

    model_mapper = clsModelMapper(
        model_filepath_mapping=ast.literal_eval(MODEL_FILEPATH_MAPPING),
        is_production=(IS_PRODUCTION == "1"),
        s3_cache_output_write=True,
        redshift_secret_name=REDSHIFT_SECRET_NAME,
    )
    ml = LocalMLAPI(model_mapper, DYNAMODBTABLE, data=data)

    return ml


def create_event(data, run_type="not-live"):
    input_json = {
        "str_request_type": "predict",
        "dynamodb_table": "Production-DataStorage-Stack-augmentedlog6B4846BA-15J9I7UD422OI",
        "imei": ["359206106341837"],
        "run_id": [run_type],
        "system_datetime_utc_str_start": ["2021-11-01T05:48:46.789Z"],
        "system_datetime_utc_str_end": ["2022-08-21T11:00:13.000Z"],
    }

    input_json["system_datetime_utc_str_start"] = [
        data["system_datetime_utc_str"].min()
    ]
    input_json["system_datetime_utc_str_end"] = [data["system_datetime_utc_str"].max()]
    input_json["imei"] = [data["imei"].iloc[0]]

    event = {"Records": [{"Sns": {"Message": json.dumps(input_json)}}]}

    return event


def local_predict(data):

    event = create_event(data)
    ml = generate_mapper(data, MODEL_FILEPATH_MAPPING)

    input_json = ml.gather_records(event)
    output = ml.predict(input_json)

    null = np.nan
    pred = eval(output)
    return pred


cols = [
    "calibration_gearbox_ratio",
    "calibration_motor_displacement_cm3",
    "calibration_motor_efficiency",
    "drum_data_valid",
    "drum_speed_mean_rpm",
    "imei",
    "prediction_trip_posix_timestamp_start",
    "pressure_a_bar_data_valid",
    "pressure_a_bar_mean",
    "pressure_a_bar_sd",
    "pressure_b_bar_data_valid",
    "pressure_b_bar_mean",
    "pressure_b_bar_sd",
    "rmc_provider",
    "run_id",
    "system_datetime_posix_utc_seconds",
    "system_datetime_utc_str",
    "torque",
    "truck_id",
    "truck_registration",
    "docket_delivered_quantity",
    "truck_project",
    "geofence_type_lastvalid",
]


def add_geofence_base(df_batch, time_cutoff_min=15):
    """Add geofence type BASE until a point where speed comes belov 5 rpm and is stable"""
    if not "geofence_type_lastvalid" in df_batch.columns:
        df_batch["geofence_type_lastvalid"] = "<empty>"

    time_since_start = (
        df_batch["system_datetime_utc_str"] - df_batch["system_datetime_utc_str"].min()
    ).dt.seconds
    speed_drop_time = df_batch["system_datetime_utc_str"].loc[
        (df_batch["drum_speed_mean_rpm"] < 5)
        & (df_batch["drum_speed_mean_rpm"] >= 1)
        & (time_since_start <= time_cutoff_min * 60)
    ]

    for t_start in speed_drop_time:
        t_end = t_start + pd.Timedelta(1, "m")
        df_1_min = df_batch.query(
            "(system_datetime_utc_str >= @t_start) & (system_datetime_utc_str <= @t_end)"
        )
        if df_1_min["drum_speed_mean_rpm"].std() <= 0.15:
            df_batch.loc[
                df_batch["system_datetime_utc_str"] <= t_start, "geofence_type_lastvalid"
            ] = "BASE"
            return df_batch
        else:
            continue

    return df_batch


def batch_predict_pipeline_trips(df_batch: pd.DataFrame) -> pd.DataFrame:
    """Generate predictions for a batch of data after cleaning rows with negative torque.
    Output dataframe may be shorted than input since rows with negative torque values are dropped"""

    data = (
        df_batch.sort_values("system_datetime_utc_str")
        .assign(
            run_id="not-live",
            system_datetime_posix_utc_seconds=lambda x: x[
                "system_datetime_utc_str"
            ].values.astype(np.int64)
            // 10**9,
            truck_project="Cemex",
        )
        .pipe(add_geofence_base)
        .get(cols)
        # .query("(torque >= 0) and (drum_speed_mean_rpm >= 1)")
        .sort_values("system_datetime_utc_str")
        .astype({"system_datetime_utc_str": str})
        .reset_index(drop=True)
    )

    pred = local_predict(data)
    df_pred = pd.DataFrame(pred)

    return df_pred


query_augmented_log = """SELECT
        AL.prediction_rotation_posix_timestamp_start,
        AL.docket_plant_code,
        AL.docket_jobsite_description,
        AL.docket_id,
        AL.docket_datetime_batch_utc,
        AL.prediction_water_addition_posix_timestamp_start,
        AL.prediction_water_addition_confidence,
        AL.prediction_slump_mm_current,
        AL.geofence_type_current,
        AL.geofence_name_current,
        AL.geofence_type_lastvalid,
        AL.geofence_name_lastvalid,
        AL.metrics_distance_travelled_metres,
        AL.pressure_a_bar_mean - AL.pressure_b_bar_mean AS pressure_diff,
        AL.calibration_gearbox_ratio,
        AL.calibration_motor_displacement_cm3,
        AL.calibration_motor_efficiency,
        AL.drum_speed_mean_rpm,
        AL.drum_data_valid,
        AL.imei,
        AL.pressure_a_bar_mean,
        AL.pressure_a_bar_data_valid,
        AL.pressure_a_bar_sd,
        AL.pressure_b_bar_mean,
        AL.pressure_b_bar_data_valid,
        AL.pressure_b_bar_sd,
        AL.run_id,
        AL.system_datetime_posix_utc_seconds,
        AL.system_datetime_utc_str,
        AL.prediction_trip_posix_timestamp_start,
        AL.prediction_truckstate_current,
        (pressure_a_bar_mean - pressure_b_bar_mean) * calibration_motor_displacement_cm3 * calibration_motor_efficiency / (20 * PI()) AS torque,
        1 as truck_id,
        AL.truck_registration,
        AL.docket_slump_code,
        'Cemex' as rmc_provider,
        AL.docket_delivered_quantity

FROM augmented_log AL
WHERE AL.system_datetime_utc_str BETWEEN '{start_date}' AND '{end_date}'
AND truck_registration = '{truck}'
ORDER BY AL.truck_registration, AL.prediction_trip_posix_timestamp_start, AL.system_datetime_utc_str
"""


to_timestamp = lambda x: x.astype("datetime64[s]").astype(float)
from_timestamp = lambda x: np.array(x).astype("datetime64[s]")


def plot_fundamentals(df):
    """Plot basic channels for signal analysis"""
    df_plot = (df
               .assign(rpmx10=lambda x: x["drum_speed_mean_rpm"] * 10)
              )

    f_torque = hv.Scatter(
        df_plot, "system_datetime_utc_str", "torque", label="torque"
    ).opts(color=colors[0], show_grid=True) * hv.Curve(
        df_plot, "system_datetime_utc_str", "torque", label="torque"
    ).opts(
        color=colors[0], alpha=0.5
    )

    f_speed = hv.Scatter(
        df_plot, "system_datetime_utc_str", "rpmx10", label="rpmx10"
    ).opts(color=colors[1]) * hv.Curve(
        df_plot, "system_datetime_utc_str", "rpmx10", label="rpmx10"
    ).opts(
        color=colors[1], alpha=0.5
    )

    f_slump = hv.Scatter(
        df_plot, "system_datetime_utc_str", "prediction_slump_mm_current", label="slump"
    ).opts(color=colors[3]) * hv.Curve(
        df_plot, "system_datetime_utc_str", "prediction_slump_mm_current", label="slump"
    ).opts(
        color=colors[3],alpha=0.5
    )

    f_rug = hv.Scatter(
        df_plot.assign(rug=0), "system_datetime_utc_str", "rug", label=""
    ).opts(color="black", marker="dash", angle=90, size=5)

    overlay = f_torque * f_speed * f_slump * f_rug

    return overlay



def extend_trip_start(df_trip, df_reference, time_sec=600):
    """Extend start of trip by time_sec duration to see batching pattern clearly"""
    trip_id = df_trip.index.values[0][:2]
    trip_start = df_trip.index.values[0][2]
    trip_end = df_trip.index.values[-1][2]
    required_start = trip_start - pd.Timedelta(time_sec, "s")
    df_trip_extended = (
        df_reference.xs(
            (trip_id[0], slice(required_start, trip_end)),
            level=[0, 2],
            drop_level=False,
        )
        .sort_index(level=2)
        .reset_index()
        .assign(prediction_trip_posix_timestamp_start=trip_id[1])
        .set_index(
            [
                "truck_registration",
                "prediction_trip_posix_timestamp_start",
                "system_datetime_utc_str",
            ]
        )
    )
    return df_trip_extended



def moving_average_missing(df_chunk, time_window_sec=20):
    ts = df_chunk.index.get_level_values('system_datetime_utc_str').values[-1]
    df_t_window = df_chunk.xs(slice(ts-pd.Timedelta(20,'s'),ts), drop_level=False)
    if len(df_t_window) > 2:
        return df_chunk.mean()
    else:
        return df_chunk.xs(ts, drop_level=False)
    


def downsample_trip(df_trip):
    numeric_columns = ['pressure_a_bar_mean', 'pressure_b_bar_mean', 'drum_speed_mean_rpm']

    df_trip_resampled = (df_trip
                         .reset_index()
                         .set_index('system_datetime_utc_str')
                         .resample('1S').ffill()
                         .drop(numeric_columns,axis=1)
                         .resample('10S')
                         .first()
                        )

    df_resampled_numeric = (df_trip
                            .reset_index()
                            .set_index('system_datetime_utc_str')
                            .get(numeric_columns)
                            .resample('1S').ffill()
                            .rolling(20)
                            .mean()
                            .resample('10S')
                            .first()
                           )

    df_trip_resampled = df_trip_resampled.join(df_resampled_numeric, how='left')
    df_trip_resampled['torque'] = ((df_trip_resampled['pressure_a_bar_mean'] - df_trip_resampled['pressure_b_bar_mean']) 
                                   * df_trip_resampled['calibration_motor_displacement_cm3'] 
                                   * df_trip_resampled['calibration_motor_efficiency'] / (20 * np.pi)
                                  )
    
    return (df_trip_resampled
            .reset_index()
            .set_index([
                "truck_registration",
                "prediction_trip_posix_timestamp_start",
                "system_datetime_utc_str",
            ])
            .sort_index()
           )
    
    
    
    
if __name__ == "__main__":

    start_date = np.datetime64("2022-10-17") # np.datetime64("2022-10-17")
    end_date = np.datetime64("2022-10-20") # np.datetime64("2022-10-20")
    truck = "KX67RKJ"

    df_ = redshift_client.execute_redshift_query(
        query_augmented_log.format(
            start_date=np.datetime_as_string(start_date),
            end_date=np.datetime_as_string(end_date),
            truck=truck,
        )
    )

    df = (
        df_.assign(
            system_datetime_utc_str=lambda x: pd.to_datetime(
                x["system_datetime_utc_str"], utc=True
            ),
            date=lambda x: x["system_datetime_utc_str"].dt.date,
        )
        .set_index(
            [
                "truck_registration",
                "prediction_trip_posix_timestamp_start",
                "system_datetime_utc_str",
            ]
        )
        .sort_index()
    )

    #%%
    for date, df_day in df.groupby("date"):
        overlay = plot_fundamentals(df_day.sort_index(level=2))
        for trip_start in (
            df_day.drop(-99999, level=1).index.get_level_values(1).unique()
        ):
            overlay = overlay * hv.VLine(pd.to_datetime(trip_start, unit="s")).opts(
                color="red", line_dash="dashed"
            )

        title = f"{str(date)}"
        opts = [
            hv.opts.Overlay(
                width=500, height=350, ylim=(-50, 250), show_grid=True, title=title
            ),
            hv.opts.Scatter(tools=["hover"]),
        ]
        display(overlay.opts(opts))



    #%%
    plot_only = {1665996610.0: '10_sec_trip_1', 1666089310.0:'1_sec_trip_1', 1666015734.0:'10_sec_trip_2', 1666164274.0:'1_sec_trip_2'}  # 1666079640.0,
    res = dict()
    merged_data = dict()
    merged_downsampled = dict()
    for (truck, trip_start), df_trip_ in df.drop(-99999.0, level=1).groupby(
        level=[0, 1]
    ):

        if trip_start not in plot_only:
            continue
        df_trip = extend_trip_start(df_trip_, df, 600)
        # overlay = plot_fundamentals(df_trip) * hv.VLine(
        #     pd.to_datetime(trip_start, unit="s", utc=True)
        # )

        slump_code = [
            i for i in df_trip["docket_slump_code"].unique() if "<empty>" not in i
        ]
        if len(slump_code):
            sc = slump_code[0]
        else:
            sc = "<empty>"

        title = f"{truck}, {str(from_timestamp(trip_start))}, {sc}"
        opts = [
            hv.opts.Overlay(
                width=600, height=400, ylim=(-50, 250), show_grid=True, title=title
            ),
            hv.opts.Scatter(tools=["hover"]),
        ]

        # display(overlay.opts(opts))
        # display(truck, trip_start)
        # -------------------------------------- #

        df_trip = df_trip_.copy()
        df_pred = batch_predict_pipeline_trips(
            df_trip.reset_index().assign(geofence_type_lastvalid="<empty>")
        )
        df_pred = df_pred.assign(
            system_datetime_utc_str=lambda x: pd.to_datetime(
                x["system_datetime_utc_str"], utc=True
            )
        ).set_index(
            [
                "truck_registration",
                "prediction_trip_posix_timestamp_start",
                "system_datetime_utc_str",
            ]
        )
        df_merged = (
            df_trip.join(df_pred, how="left", rsuffix="_pred")
            .assign(
                prediction_slump_mm_current=lambda x: np.maximum(
                    -10, x["prediction_slump_mm"]
                )
            )
            .sort_index()
        )

        
        if '1_sec_' in plot_only[float(trip_start)]:
            overlay_ = plot_fundamentals(df_merged)
            df_plot = (df_merged
               .assign(rpmx10=lambda x: x["drum_speed_mean_rpm"] * 10,
                      drum_speed_power_intercept = lambda x: x['drum_speed_power_intercept']*100)
              )
            f_torque = hv.Scatter(
                    df_plot, "system_datetime_utc_str", "torque", label="torque"
                ).opts(color=colors[0], show_grid=True) * hv.Curve(
                    df_plot, "system_datetime_utc_str", "torque", label="torque"
                ).opts(
                    color=colors[0], alpha=0.5
                )
            
            f_speed = hv.Scatter(df_plot, "system_datetime_utc_str", "rpmx10", label="rpmx10"
                                ).opts(color=colors[0]) * hv.Curve(
                                    df_plot, "system_datetime_utc_str", "rpmx10", label="rpmx10"
                                ).opts(
                                    color=colors[0], alpha=0.5
                                )

            f_slump = hv.Scatter(
                df_plot, "system_datetime_utc_str", "drum_speed_power_intercept", label="interceptx100"
            ).opts(color=colors[0], marker='s') * hv.Scatter(
                df_plot, "system_datetime_utc_str", "prediction_slump_mm_current", label="slump"
            ).opts(
                color=colors[0],
            )

            df_trip_downsampled = downsample_trip(df_trip_)
            df_pred_downsampled = batch_predict_pipeline_trips(
                df_trip_downsampled.reset_index().assign(geofence_type_lastvalid="<empty>")
            )

            df_pred_downsampled = (df_pred_downsampled
                                   .assign(system_datetime_utc_str=lambda x: pd.to_datetime(x["system_datetime_utc_str"], utc=True))
                                   .set_index(["truck_registration","prediction_trip_posix_timestamp_start","system_datetime_utc_str"])
                                   )
            df_merged_downsampled = (
                df_trip_downsampled.join(df_pred_downsampled, how="left", rsuffix="_pred")
                .assign(
                    prediction_slump_mm_current=lambda x: np.maximum(
                        -10, x["prediction_slump_mm"]
                    )
                )
                .sort_index()
            )

            df_plot = (df_merged_downsampled
               .assign(rpmx10=lambda x: x["drum_speed_mean_rpm"] * 10,
                      drum_speed_power_intercept = lambda x: x['drum_speed_power_intercept']*100)
              )
            
            f_torque_down = hv.Scatter(
                    df_plot, "system_datetime_utc_str", "torque", label="torque 10s"
                ).opts(color=colors[1], show_grid=True) * hv.Curve(
                    df_plot, "system_datetime_utc_str", "torque", label="torque 10s"
                ).opts(
                    color=colors[1], alpha=0.5
                )
            
            f_speed_down = hv.Scatter(df_plot, "system_datetime_utc_str", "rpmx10", label="rpmx10 10s"
                    ).opts(color=colors[1]) * hv.Curve(
                        df_plot, "system_datetime_utc_str", "rpmx10", label="rpmx10 10s"
                    ).opts(
                        color=colors[1], alpha=0.5
                    )

            f_slump_down = hv.Scatter(
                df_plot, "system_datetime_utc_str", "drum_speed_power_intercept", label="interceptx100 10s"
            ).opts(color=colors[1], marker='s') * hv.Scatter(
                df_plot, "system_datetime_utc_str", "prediction_slump_mm_current", label="slump 10s"
            ).opts(
                color=colors[1],
            )

            ot = f_torque * f_torque_down
            os = f_speed * f_speed_down
            osl = f_slump * f_slump_down
            overlay = hv.Layout([ot.opts(opts), os.opts(opts), osl.opts(opts)]).cols(1)
            
            
            merged_downsampled[trip_start] = df_merged_downsampled
        else:
            overlay = plot_fundamentals(df_merged).opts(opts)
            
            
        display(overlay)
        
        merged_data[trip_start] = df_merged
        res[trip_start] = overlay
        # display(truck, trip_start)

        
            
        
    for trip_start in plot_only:
        display(res[trip_start].opts(opts+[hv.opts.Overlay(height=600,width=800)]))

    
    
    # ------------------Missing data-------------------- #
    results = []
    errors = dict()
    for dt_obj in tqdm(pd.date_range(start_date, end_date)[:-1]):
        try:
            results.append(query_s3_bucket([dt_obj.strftime('%Y/%m/%d')], bucket_name=buckets[0])[0])
        except Exception as e:
            errors[dt_obj] = e

    df_packet = pd.concat(results)
    

    imei = int(df['imei'].unique()[0])
    calibration_motor_displacement_cm3 = df['calibration_motor_displacement_cm3'].unique()[0]
    calibration_motor_efficiency = df['calibration_motor_efficiency'].unique()[0]
    
    df_p = (df_packet
    .query("IMEI==@imei")
    .assign(
        system_datetime_utc_str=lambda x: pd.to_datetime(
            x["system_datetime_posix_utc_seconds"], unit='s', utc=True
        ),
        date=lambda x: x["system_datetime_utc_str"].dt.date,
        truck_registration = truck,
        prediction_slump_mm_current = -10,
        torque = lambda x: (x['pressure_a_bar_mean'] - x['pressure_b_bar_mean'])*calibration_motor_displacement_cm3*calibration_motor_efficiency/(20*np.pi)
    )
    .set_index(
        [
            "truck_registration",
            "system_datetime_utc_str",
        ]
    )
    .sort_index()
    )
    
    
    
    for date in pd.date_range(start_date, end_date)[:-1]:
        title = f"{truck}, {date.strftime('%F')} Redshift data"
        opts = [
            hv.opts.Overlay(ylabel='value', xlabel='UTC time',
                width=600, height=400, ylim=(-50, 250), show_grid=True, title=title
            ),
            hv.opts.Scatter(tools=["hover"]),
        ]
        o = plot_fundamentals(df.query("date == @date").sort_index(level=2))
        display(o.opts(opts))
        
        title = f"{truck}, {date.strftime('%F')} S3 data"
        o = plot_fundamentals(df_p.query("date == @date"))
        display(o.opts(opts + [hv.opts.Overlay(title=title)]))
        
        
    # Run on merged data
    merged_data = (df_p
                   .join(df.reset_index().set_index(['truck_registration', 'system_datetime_utc_str']), lsuffix='', rsuffix='_rs')
                   .ffill()
                   .drop_duplicates()
                   .reset_index()
                   .set_index(["truck_registration","prediction_trip_posix_timestamp_start","system_datetime_utc_str"])
                  )
    
    
    df = merged_data
    #%%
    plot_only = {1665996610.0: '10_sec_trip_1', 1666089310.0:'1_sec_trip_1', 1666015734.0:'10_sec_trip_2', 1666164274.0:'1_sec_trip_2'}  # 1666079640.0,
    res = dict()
    merged_data = dict()
    merged_downsampled = dict()
    for (truck, trip_start), df_trip_ in df.drop(-99999.0, level=1).groupby(
        level=[0, 1]
    ):
        df_trip_ = df_trip_.drop_duplicates()
        if trip_start not in plot_only:
            continue
        df_trip = extend_trip_start(df_trip_, df, 600)
        # overlay = plot_fundamentals(df_trip) * hv.VLine(
        #     pd.to_datetime(trip_start, unit="s", utc=True)
        # )

        slump_code = [
            i for i in df_trip["docket_slump_code"].unique() if "<empty>" not in i
        ]
        if len(slump_code):
            sc = slump_code[0]
        else:
            sc = "<empty>"

        title = f"{truck}, {str(from_timestamp(trip_start))}, {sc}"
        opts = [
            hv.opts.Overlay(
                width=600, height=400, ylim=(-50, 250), show_grid=True, title=title
            ),
            hv.opts.Scatter(tools=["hover"]),
        ]

        # display(overlay.opts(opts))
        # display(truck, trip_start)
        # -------------------------------------- #

        df_trip = df_trip_.copy().drop_duplicates()
        df_pred = batch_predict_pipeline_trips(
            df_trip.reset_index().assign(geofence_type_lastvalid="<empty>")
        )
        df_pred = df_pred.assign(
            system_datetime_utc_str=lambda x: pd.to_datetime(
                x["system_datetime_utc_str"], utc=True
            )
        ).set_index(
            [
                "truck_registration",
                "prediction_trip_posix_timestamp_start",
                "system_datetime_utc_str",
            ]
        )
        df_merged = (
            df_trip.join(df_pred, how="left", rsuffix="_pred")
            .assign(
                prediction_slump_mm_current=lambda x: np.maximum(
                    -10, x["prediction_slump_mm"]
                )
            )
            .sort_index()
        )

        
        if '1_sec_' in plot_only[float(trip_start)]:
            overlay_ = plot_fundamentals(df_merged)
            df_plot = (df_merged
               .assign(rpmx10=lambda x: x["drum_speed_mean_rpm"] * 10,
                      drum_speed_power_intercept = lambda x: x['drum_speed_power_intercept']*100)
              )
            f_torque = hv.Scatter(
                    df_plot, "system_datetime_utc_str", "torque", label="torque"
                ).opts(color=colors[0], show_grid=True) * hv.Curve(
                    df_plot, "system_datetime_utc_str", "torque", label="torque"
                ).opts(
                    color=colors[0], alpha=0.5
                )
            
            f_speed = hv.Scatter(df_plot, "system_datetime_utc_str", "rpmx10", label="rpmx10"
                                ).opts(color=colors[0]) * hv.Curve(
                                    df_plot, "system_datetime_utc_str", "rpmx10", label="rpmx10"
                                ).opts(
                                    color=colors[0], alpha=0.5
                                )

            f_slump = hv.Scatter(
                df_plot, "system_datetime_utc_str", "drum_speed_power_intercept", label="interceptx100"
            ).opts(color=colors[0], marker='s') * hv.Scatter(
                df_plot, "system_datetime_utc_str", "prediction_slump_mm_current", label="slump"
            ).opts(
                color=colors[0],
            )

            df_trip_downsampled = downsample_trip(df_trip_)
            df_pred_downsampled = batch_predict_pipeline_trips(
                df_trip_downsampled.reset_index().assign(geofence_type_lastvalid="<empty>")
            )

            df_pred_downsampled = (df_pred_downsampled
                                   .assign(system_datetime_utc_str=lambda x: pd.to_datetime(x["system_datetime_utc_str"], utc=True))
                                   .set_index(["truck_registration","prediction_trip_posix_timestamp_start","system_datetime_utc_str"])
                                   )
            df_merged_downsampled = (
                df_trip_downsampled.join(df_pred_downsampled, how="left", rsuffix="_pred")
                .assign(
                    prediction_slump_mm_current=lambda x: np.maximum(
                        -10, x["prediction_slump_mm"]
                    )
                )
                .sort_index()
            )

            df_plot = (df_merged_downsampled
               .assign(rpmx10=lambda x: x["drum_speed_mean_rpm"] * 10,
                      drum_speed_power_intercept = lambda x: x['drum_speed_power_intercept']*100)
              )
            
            f_torque_down = hv.Scatter(
                    df_plot, "system_datetime_utc_str", "torque", label="torque 10s"
                ).opts(color=colors[1], show_grid=True) * hv.Curve(
                    df_plot, "system_datetime_utc_str", "torque", label="torque 10s"
                ).opts(
                    color=colors[1], alpha=0.5
                )
            
            f_speed_down = hv.Scatter(df_plot, "system_datetime_utc_str", "rpmx10", label="rpmx10 10s"
                    ).opts(color=colors[1]) * hv.Curve(
                        df_plot, "system_datetime_utc_str", "rpmx10", label="rpmx10 10s"
                    ).opts(
                        color=colors[1], alpha=0.5
                    )

            f_slump_down = hv.Scatter(
                df_plot, "system_datetime_utc_str", "drum_speed_power_intercept", label="interceptx100 10s"
            ).opts(color=colors[1], marker='s') * hv.Scatter(
                df_plot, "system_datetime_utc_str", "prediction_slump_mm_current", label="slump 10s"
            ).opts(
                color=colors[1],
            )

            ot = f_torque * f_torque_down
            os = f_speed * f_speed_down
            osl = f_slump * f_slump_down
            overlay = hv.Layout([ot.opts(opts), os.opts(opts), osl.opts(opts)]).cols(1)
            
            
            merged_downsampled[trip_start] = df_merged_downsampled
        else:
            overlay = plot_fundamentals(df_merged).opts(opts)
            
            
        display(overlay)
        
        merged_data[trip_start] = df_merged
        res[trip_start] = overlay
        # display(truck, trip_start)
