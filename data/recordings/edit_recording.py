import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json 

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
    combined_accel_gyro = combine_accel_gyro_events(combined_accel_gyro, events_synthetic)

    return combined_accel_gyro, events_synthetic

def combine_accel_gyro_events(accel_gyro, events):
    accel_gyro['event'] = None
    for event in events:
        mask = (accel_gyro['synthetic_time_ms'] >= event['start_time_synthetic']) & \
            (accel_gyro['synthetic_time_ms'] <= event['end_time_synthetic'])
        accel_gyro.loc[mask, 'event'] = event['name']
    return accel_gyro

def plot_recording(data, events):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

    # Convert time to seconds
    synthetic_s = data['synthetic_time_ms']

    # Add event regions with different colors
    unique_events = [e['name'] for e in events]
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_events)))

    # Plot 1: Accelerometer data
    ax1.plot(synthetic_s, data['accel_x'], label='accel_x', alpha=0.7)
    ax1.plot(synthetic_s, data['accel_y'], label='accel_y', alpha=0.7)
    ax1.plot(synthetic_s, data['accel_z'], label='accel_z', alpha=0.7)

    for event in events:
        start_s = event['start_time_synthetic']
        end_s = event['end_time_synthetic']
        color_idx = unique_events.index(event['name'])
        ax1.axvspan(start_s, end_s, alpha=0.2, color=colors[color_idx], label=event['name'])

    # Set better y-axis limits based on percentiles (removes extreme outliers)
    accel_data = np.concatenate([data['accel_x'].values, 
                                data['accel_y'].values,
                                data['accel_z'].values])
    accel_min, accel_max = np.percentile(accel_data, [1, 99])
    accel_range = accel_max - accel_min
    ax1.set_ylim([accel_min - 0.8*accel_range, accel_max + 0.8*accel_range])

    ax1.set_ylabel('Acceleration (m/s²)')
    ax1.set_title('Accelerometer Data')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Gyroscope data
    ax2.plot(synthetic_s, data['gyro_x'], label='gyro_x', alpha=0.7)
    ax2.plot(synthetic_s, data['gyro_y'], label='gyro_y', alpha=0.7)
    ax2.plot(synthetic_s, data['gyro_z'], label='gyro_z', alpha=0.7)

    for event in events:
        start_s = event['start_time_synthetic']
        end_s = event['end_time_synthetic']
        color_idx = unique_events.index(event['name'])
        ax2.axvspan(start_s, end_s, alpha=0.2, color=colors[color_idx])

    # Set better y-axis limits based on percentiles (removes extreme outliers)
    gyro_data = np.concatenate([data['gyro_x'].values,
                                data['gyro_y'].values,
                                data['gyro_z'].values])
    gyro_min, gyro_max = np.percentile(gyro_data, [1, 99])
    gyro_range = gyro_max - gyro_min
    ax2.set_ylim([gyro_min - 0.5*gyro_range, gyro_max + 0.5*gyro_range])

    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Gyroscope (rad/s)')
    ax2.set_title('Gyroscope Data')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print event times for verification
    print("Event synthetic times (ms):")
    for event in events:
        print(f"{event['name']}: {event['start_time_synthetic']:.0f}ms - {event['end_time_synthetic']:.0f}ms")

def edit_event_times(events):
    print("Available events:")
    for i, event in enumerate(events):
        print(f"{i}: {event['name']} (current: {event['start_time_synthetic']}ms - {event['end_time_synthetic']}ms)")

    try:
        selected_index = int(input("Enter the index of the event to edit: "))
        if 0 <= selected_index < len(events):
            selected_event = events[selected_index]
        else:
            print("Invalid index.")
            return
    except ValueError:
        print("Invalid input. Please enter a number.")
        return

    new_start = input(f"Enter new start time in ms (relative to synthetic time) for event '{selected_event['name']}' (current: {selected_event['start_time_synthetic']}ms): ")
    new_end = input(f"Enter new end time in ms (relative to synthetic time) for event '{selected_event['name']}' (current: {selected_event['end_time_synthetic']}ms): ")

    if new_start.strip() != "":
        try:
            new_start_ms = int(new_start)
            if new_start_ms % 20 != 0:
                print("Start time must be a multiple of 20ms.")
                return
            selected_event['start_time_synthetic'] = new_start_ms
        except ValueError:
            print("Invalid input for start time. Please enter a number.")
            return

    if new_end.strip() != "":
        try:
            new_end_ms = int(new_end)
            if new_end_ms % 20 != 0:
                print("End time must be a multiple of 20ms.")
                return
            selected_event['end_time_synthetic'] = new_end_ms
        except ValueError:
            print("Invalid input for end time. Please enter a number.")
            return
        
    # replace event in events list with updated event
    events[selected_index] = selected_event

    return events

def main():
    path = "session_1776670910347"
    accel, gyro, classification, annotations, info = load_recording(path)
    combined_accel_gyro = combine_accel_gyro(accel, gyro)
    combined_with_annotations, events_synthetic = add_annotations_to_combined(combined_accel_gyro, annotations, info)
    plot_recording(combined_with_annotations, events_synthetic)

    while True:
        user_input = input("Do you want to edit event times? (y/n): ")
        if user_input.lower() == 'y':
            # edit event times
            updated_events = edit_event_times(events_synthetic)

            # replot with new event times
            new_combined = combine_accel_gyro_events(combined_accel_gyro, updated_events)
            plot_recording(new_combined, updated_events)

            # save dataframe with annotations to csv
            new_combined.to_csv(f"{path}/combined_edited.csv", index=False)
        elif user_input.lower() == 'n':
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")


if __name__ == "__main__":
    main()