from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def plot_recording_with_predictions(data, path, linear=False):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    alignment = load_time_alignment(path)
    imu_start = data["timestamp"].iloc[0]

    time = data["timestamp"]

    # --- Plot 1: Accelerometer data ---
    ax1.plot(time, data['accel_x'], label='accel_x', alpha=0.7)
    ax1.plot(time, data['accel_y'], label='accel_y', alpha=0.7)
    ax1.plot(time, data['accel_z'], label='accel_z', alpha=0.7)

    accel_data = data[['accel_x', 'accel_y', 'accel_z']].values
    accel_min, accel_max = np.percentile(accel_data, [1, 99])
    accel_range = accel_max - accel_min
    ax1.set_ylim([accel_min - 0.8 * accel_range, accel_max + 0.8 * accel_range])
    ax1.set_ylabel('Acceleration (m/s²)')
    ax1.set_title('Accelerometer Data')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Gyroscope data ---
    ax2.plot(time, data['gyro_x'], label='gyro_x', alpha=0.7)
    ax2.plot(time, data['gyro_y'], label='gyro_y', alpha=0.7)
    ax2.plot(time, data['gyro_z'], label='gyro_z', alpha=0.7)

    gyro_data = data[['gyro_x', 'gyro_y', 'gyro_z']].values
    gyro_min, gyro_max = np.percentile(gyro_data, [1, 99])
    gyro_range = gyro_max - gyro_min
    ax2.set_ylim([gyro_min - 0.8 * gyro_range, gyro_max + 0.8 * gyro_range])
    ax2.set_ylabel('Gyroscope (rad/s)')
    ax2.set_title('Gyroscope Data')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # --- Plot 3: Predictions ---
    ax3.plot(time, data['prediction_float'], label='Predictions', color='black', linewidth=1.5, alpha=0.7)
    ax3.plot(time, data['prediction_binary'], label='>=0.5 = Exercise', color='red', linewidth=1.5, alpha=0.4)
    ax3.axhline(y=0.5, color='red', linestyle='--', label='Threshold = 0.5')
    ax3.set_ylabel('Probability')
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylim([-0.1, 1.1])
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # --- Overlay Event Durations ---
    data = data.copy()

    event_col = data['event'].fillna('None')
    event_changes = event_col != event_col.shift()
    data['event_group'] = event_changes.cumsum()

    unique_events = [e for e in event_col.unique() if e != 'None']
    cmap = plt.get_cmap('tab10')
    event_to_color = {name: cmap(i % 10) for i, name in enumerate(unique_events)}

    event_loc_up = True

    for _, group in data.groupby('event_group'):
        evt_name = group['event'].iloc[0]

        if pd.isna(evt_name) or evt_name == 'None':
            continue

        t_start = group['timestamp'].iloc[0]
        t_end = group['timestamp'].iloc[-1]
        color = event_to_color[evt_name]

        for ax in [ax1, ax2, ax3]:
            ax.axvspan(t_start, t_end, color=color, alpha=0.15)

            if ax == ax1:
                if event_loc_up:
                    loc = 0
                    va = 'bottom'
                    event_loc_up = False
                else:
                    loc = 1
                    va = 'top'
                    event_loc_up = True

                ax1.text(
                    (t_start + t_end) / 2,
                    ax1.get_ylim()[loc],
                    evt_name,
                    ha='center',
                    va=va,
                    color=color,
                    fontweight='bold',
                    fontsize=9
                )

    # --- Click timestamp helper ---
    click_lines = []

    def on_click(event):
        if event.inaxes not in [ax1, ax2, ax3]:
            return

        if event.xdata is None:
            return

        clicked_imu_timestamp = event.xdata

        nearest_idx = (data["timestamp"] - clicked_imu_timestamp).abs().idxmin()
        nearest_imu_timestamp = data.loc[nearest_idx, "timestamp"]

        phone_timestamp = imu_to_phone_time(
            nearest_imu_timestamp,
            imu_start,
            alignment,
            linear=linear
        )

        print("Clicked point")
        print(f"IMU timestamp:   {nearest_imu_timestamp:.0f} ns")
        print(f"Phone timestamp: {phone_timestamp:.0f} ms")
        print(f"Row index:       {nearest_idx}")
        print()

        for line in click_lines:
            line.remove()
        click_lines.clear()

        for ax in [ax1, ax2, ax3]:
            line = ax.axvline(nearest_imu_timestamp, color="purple", linestyle="--", linewidth=1.5)
            click_lines.append(line)

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)

    plt.tight_layout()
    plt.show()

def load_time_alignment(path):
    """
    Loads the values needed to convert IMU timestamps back to phone timestamps.
    """
    import json

    with open(f"{path}/annotations.json") as f:
        annotations = json.load(f)

    with open(f"{path}/info.txt") as f:
        info = f.read().splitlines()

    phone_start = annotations["phoneStartTimestamp"]
    phone_end = annotations["phoneEndTimestamp"]

    watch_start = int(info[0].split("=")[1])
    watch_end = int(info[2].split("=")[1])

    return {
        "phone_start": phone_start,
        "phone_end": phone_end,
        "watch_start": watch_start,
        "watch_end": watch_end,
    }


def imu_to_phone_time(imu_timestamp, imu_start, alignment, linear=False):
    """
    Converts an IMU timestamp from combined.csv back to phone Unix time in ms.

    imu_timestamp: timestamp from combined.csv, nanoseconds since boot
    imu_start: first timestamp in combined.csv, nanoseconds since boot
    alignment: output of load_time_alignment(path)
    linear: use linear phone-watch drift correction if wanted
    """
    phone_start = alignment["phone_start"]
    phone_end = alignment["phone_end"]
    watch_start = alignment["watch_start"]
    watch_end = alignment["watch_end"]

    # IMU boot-time ns -> watch wall-time ms
    watch_timestamp = watch_start + (imu_timestamp - imu_start) / 1_000_000

    if linear:
        # phone -> watch was:
        # watch = a * phone + b
        # so phone = (watch - b) / a
        a = (watch_end - watch_start) / (phone_end - phone_start)
        b = watch_start - a * phone_start
        phone_timestamp = (watch_timestamp - b) / a
    else:
        # Your original offset method:
        # watch = phone - offset
        # offset = phone_start - watch_start
        # therefore phone = watch + offset
        offset = phone_start - watch_start
        phone_timestamp = watch_timestamp + offset

    return phone_timestamp

def main():
    path = "session_1779608056714"

    df = pd.read_csv(f"{path}/combined.csv")

    plot_recording_with_predictions(df, path, linear=False)


if __name__ == "__main__":
    main()