import json

import numpy as np
import pandas as pd


def load_recording(path):
    accel = pd.read_csv(f"{path}/accel.csv") # timestamp, x, y, z
    gyro = pd.read_csv(f"{path}/gyro.csv") # timestamp, x, y, z
    classification = pd.read_csv(f"{path}/classification.csv") # timestamp,prediction_float,prediction_binary,start_accel_ts,end_accel_ts,start_gyro_ts,end_gyro_ts,window_data
    with open(f"{path}/annotations.json") as f:
        annotations = json.load(f)
        # annotations is a dict with keys "phoneStartTimestamp", "phoneEndTimestamp", "events": ["name", "start_time", "end_time"]

    phone_start_timestamp = annotations["phoneStartTimestamp"]
    phone_end_timestamp = annotations["phoneEndTimestamp"]

    # read watch info file info.txt which has 2 lines 
    # start_time=1776349856154
    # phone_timestamp=1776349855769
    with open(f"{path}/info.txt") as f:
        info = f.read().splitlines()
        watch_start_timestamp = int(info[0].split('=')[1])
        watch_end_timestamp = int(info[2].split('=')[1])
    
    return accel, gyro, classification, annotations, info

def combine_accel_gyro(accel, gyro):
    # assume accel and gyro are sorted by timestamp and have same number of rows
    # take accel timestamp
    combined = pd.DataFrame()
    combined['timestamp'] = accel['timestamp']
    combined['accel_x'] = accel['x']
    combined['accel_y'] = accel['y']
    combined['accel_z'] = accel['z']
    combined['gyro_x'] = gyro['x']
    combined['gyro_y'] = gyro['y']
    combined['gyro_z'] = gyro['z']
    return combined

def add_annotations_to_combined(combined_accel_gyro, annotations, info):
    # Convert events to watch time
    events = convert_events_to_imu_time(annotations, info, combined_accel_gyro)

    # Add event annotations to your combined dataframe
    combined_accel_gyro = add_events_to_combined(combined_accel_gyro, events)

    return combined_accel_gyro, events

def convert_events_to_imu_time(annotations, info, combined, linear=False):
    phone_start = annotations["phoneStartTimestamp"]
    phone_end = annotations["phoneEndTimestamp"]

    watch_start = int(info[0].split('=')[1])
    watch_end = int(info[2].split('=')[1])

    imu_start = combined['timestamp'].iloc[0]  # nanoseconds

    if linear:
        # phone to watch (linear)
        a = (watch_end - watch_start) / (phone_end - phone_start)
        b = watch_start - a * phone_start
    else:
        offset = phone_start - watch_start

    events_imu = []
    for event in annotations["events"]:
        # phone to watch wall time
        if linear:
            start_watch = a * event["startTime"] + b
            end_watch = a * event["endTime"] + b
        else:
            start_watch = event['startTime'] - offset
            end_watch = event['endTime'] - offset

        # watch wall to imu boot time
        start_imu = imu_start + (start_watch - watch_start) * 1_000_000
        end_imu = imu_start + (end_watch - watch_start) * 1_000_000

        events_imu.append({
            "name": event["name"],
            "start_time": start_imu,
            "end_time": end_imu
        })

    events_imu.sort(key=lambda x: x["start_time"])

    return events_imu

def convert_events_to_watch_time(annotations, info, linear=False):
    phone_start = annotations["phoneStartTimestamp"]
    phone_end = annotations["phoneEndTimestamp"]

    watch_start = int(info[0].split('=')[1])
    watch_end = int(info[2].split('=')[1])

    if linear:
        # Compute linear mapping parameters
        a = (watch_end - watch_start) / (phone_end - phone_start)
        b = watch_start - a * phone_start
    else:
        offset = phone_start - watch_start

    events_watch = []
    for event in annotations["events"]:
        if linear:
            start_watch = a * event["startTime"] + b
            end_watch = a * event["endTime"] + b
        else:
            start_watch = event['startTime'] - offset
            end_watch = event['endTime'] - offset

        events_watch.append({
            "name": event["name"],
            "start_watch": start_watch,
            "end_watch": end_watch
        })

    events_watch.sort(key=lambda x: x["start_watch"])

    return events_watch

def add_events_to_combined(combined, events_watch):
    combined['event'] = None

    for event in events_watch:
        mask = (
            (combined['timestamp'] >= event['start_time']) &
            (combined['timestamp'] <= event['end_time'])
        )
        combined.loc[mask, 'event'] = event['name']

    return combined

def add_predictions_to_combined(combined, classification):
    # problem this creates: At time 2.5s is displayed a prediction that was only computed at 4s
    # This shows model centric view

    combined['prediction_float'] = np.nan
    combined['prediction_binary'] = np.nan

    for idx, row in classification.iterrows():
        accel_timestamps = set(int(ts) for ts in row['accel_timestamps'].split(';'))
        mask = combined['timestamp'].isin(accel_timestamps)
        combined.loc[mask, 'prediction_float'] = row['prediction_float']
        combined.loc[mask, 'prediction_binary'] = row['prediction_binary']

    return combined

# def add_prediction_signals(combined, classification):
    

def save(dataframe, path):
    dataframe.to_csv(f"{path}/combined.csv", index=False)

def main():
    path = "session_1776938763711"

    accel, gyro, classification, annotations, info = load_recording(path)
    combined_accel_gyro = combine_accel_gyro(accel, gyro)
    combined_with_annotations, events_synthetic = add_annotations_to_combined(combined_accel_gyro, annotations, info)
    combined_with_predictions = add_predictions_to_combined(combined_with_annotations, classification)

    save(combined_with_predictions, path)

if __name__ == "__main__":
    main()