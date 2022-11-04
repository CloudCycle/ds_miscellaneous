import pandas as pd
import numpy as np
from ds_tools import AWSClient
import holoviews as hv
from IPython.display import display
from ccml.app_slump import *

pd.set_option("display.max_rows", 500)
hv.extension("bokeh")
redshift_client = AWSClient()

colors = ["#2E73E2", "#CC9933", "#33CC99", "#9933CC", "#343435", "#000066"]

# MODEL_FILEPATH_MAPPING = "{'default': 's3://cloudcycle-ml-prediction-models/slump/production/model_objects/model_clsCloudCycleSlumpModel_v3_2_4.pkl'}"

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
        df_batch["datetime_utc"] - df_batch["datetime_utc"].min()
    ).dt.seconds
    speed_drop_time = df_batch["datetime_utc"].loc[
        (df_batch["drum_speed_mean_rpm"] < 5)
        & (df_batch["drum_speed_mean_rpm"] >= 1)
        & (time_since_start <= time_cutoff_min * 60)
    ]

    for t_start in speed_drop_time:
        t_end = t_start + pd.Timedelta(1, "m")
        df_1_min = df_batch.query(
            "(datetime_utc >= @t_start) & (datetime_utc <= @t_end)"
        )
        if df_1_min["drum_speed_mean_rpm"].std() <= 0.15:
            df_batch.loc[
                df_batch["datetime_utc"] <= t_start, "geofence_type_lastvalid"
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
        # .pipe(add_geofence_base)
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
    df_plot = df.assign(rpmx10=lambda x: x["drum_speed_mean_rpm"] * 10)

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
    ).opts(color=colors[2]) * hv.Curve(
        df_plot, "system_datetime_utc_str", "prediction_slump_mm_current", label="slump"
    ).opts(
        color=colors[2], alpha=0.5
    )

    f_rug = hv.Scatter(
        df_plot.assign(rug=0), "system_datetime_utc_str", "rug", label=""
    ).opts(color="black", marker="dash", angle=90, size=5)

    overlay = f_torque * f_speed * f_slump * f_rug

    return overlay


def moving_mean(signal, n_samples=10):
    kernel = np.ones((n_samples,)) / n_samples
    y = np.convolve(signal, kernel, mode="same")

    return y


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


if __name__ == "__main__":

    #%%
    start_date = np.datetime64("2022-10-17")
    end_date = np.datetime64("2022-10-20")

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
    plot_only = [1665996610, 1666089310.0, 1666015734.0, 1666164274]  # 1666079640.0,
    res = dict()
    for (truck, trip_start), df_trip in df.drop(-99999.0, level=1).groupby(
        level=[0, 1]
    ):

        if trip_start not in plot_only:
            continue
        df_trip = extend_trip_start(df_trip, df, 600)
        overlay = plot_fundamentals(df_trip) * hv.VLine(
            pd.to_datetime(trip_start, unit="s", utc=True)
        )

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
                width=500, height=350, ylim=(-50, 250), show_grid=True, title=title
            ),
            hv.opts.Scatter(tools=["hover"]),
        ]

        # display(overlay.opts(opts))
        display(truck, trip_start)
        # -------------------------------------- #

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

        overlay = plot_fundamentals(df_merged).opts(opts)
        # display(overlay)
        res[trip_start] = overlay
        # display(truck, trip_start)

    for trip_start in plot_only:
        display(res[trip_start])

