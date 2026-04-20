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
    # 1. Create synthetic time for IMU (assuming 50Hz = 20ms intervals)
    synthetic_time_ms = np.arange(len(combined_accel_gyro)) * 20
    combined_accel_gyro['synthetic_time_ms'] = synthetic_time_ms

    # 2. Calculate offset between phone and watch start times
    phone_start_timestamp = annotations["phoneStartTimestamp"]
    watch_start_timestamp = int(info[0].split('=')[1])
    phone_watch_offset_ms = round((phone_start_timestamp - watch_start_timestamp) / 20) * 20

    # 3. Convert events to synthetic time
    events_synthetic = []
    for event in annotations["events"]:
        # Get event times relative to phone start
        event_start_rel = round((event['startTime'] - phone_start_timestamp) / 20) * 20
        event_end_rel = round((event['endTime'] - phone_start_timestamp) / 20) * 20
        
        # Adjust for phone-watch offset
        # If phone started before watch (offset > 0), subtract it
        event_start_synthetic = event_start_rel - phone_watch_offset_ms
        event_end_synthetic = event_end_rel - phone_watch_offset_ms
        
        events_synthetic.append({
            'name': event['name'],
            'start_time_synthetic': event_start_synthetic,
            'end_time_synthetic': event_end_synthetic,
            'original_start': event['startTime'],
            'original_end': event['endTime']
        })

    # sort events by synthetic start time
    events_synthetic.sort(key=lambda x: x['start_time_synthetic'])

    # 4. Add event annotations to your combined dataframe
    combined_accel_gyro['event'] = None
    for event in events_synthetic:
        mask = (combined_accel_gyro['synthetic_time_ms'] >= event['start_time_synthetic']) & \
            (combined_accel_gyro['synthetic_time_ms'] <= event['end_time_synthetic'])
        combined_accel_gyro.loc[mask, 'event'] = event['name']

    return combined_accel_gyro, events_synthetic

def add_predictions_to_combined(combined, classification):
    combined['prediction_float'] = np.nan
    combined['prediction_binary'] = np.nan

    for idx, row in classification.iterrows():
        accel_timestamps = set(int(ts) for ts in row['accel_timestamps'].split(';'))
        mask = combined['timestamp'].isin(accel_timestamps)
        combined.loc[mask, 'prediction_float'] = row['prediction_float']
        combined.loc[mask, 'prediction_binary'] = row['prediction_binary']

    return combined

def save(dataframe, path):
    dataframe.to_csv(f"{path}/combined.csv", index=False)

def main():
    path = "session_1776670910347"

    accel, gyro, classification, annotations, info = load_recording(path)
    combined_accel_gyro = combine_accel_gyro(accel, gyro)
    combined_with_annotations, events_synthetic = add_annotations_to_combined(combined_accel_gyro, annotations, info)
    combined_with_predictions = add_predictions_to_combined(combined_with_annotations, classification)

    save(combined_with_predictions, path)

if __name__ == "__main__":
    main()