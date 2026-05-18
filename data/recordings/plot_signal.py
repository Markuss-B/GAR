import pandas as pd
import matplotlib.pyplot as plt
from process_and_save_recording import load_recording, convert_events_to_watch_time


import numpy as np

def plot_signal(classification, events):
    df_class = classification
    df_events = pd.DataFrame(events)

    # 3. Plotting
    plt.figure(figsize=(15, 4))

    # Plot the classification signal
    # Use prediction_float for a continuous signal or prediction_binary for a step plot
    plt.step(df_class['timestamp'], df_class['prediction_float'], where='post', 
            label='Prediction Probability', color='black', alpha=0.7, linewidth=1.5)
    
    plt.step(df_class['timestamp'], df_class['prediction_binary'], where='post', label='>=0.5 = Exercise', color='red', linewidth=1.5, alpha=0.4)
    plt.axhline(y=0.5, color='red', linestyle='--', label='Threshold = 0.5')

    # Define a color map for different event types
    unique_events = df_events['name'].unique()
    colors = plt.get_cmap('tab20', len(unique_events))
    event_color_map = {name: colors(i) for i, name in enumerate(unique_events)}

    # Overlay events as vertical spans (axvspan)
    for _, event in df_events.iterrows():
        plt.axvspan(event['start_watch'], event['end_watch'], 
                    color=event_color_map[event['name']], 
                    alpha=0.3, 
                    label=event['name'])

    # Handle Legend (avoid duplicate labels for same event types)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1, 1))

    plt.title('Classification Signal with Event Overlays')
    plt.xlabel('Timestamp (ms)')
    plt.ylabel('Prediction Value')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    plt.show()

def main():
    path = "session_1777462103399"
    accel, gyro, classification, annotation, info = load_recording(path)
    events_watch_time = convert_events_to_watch_time(annotation, info)
    #classification['prediction_binary'] = apply_state_machine(classification, start_buffer=2, end_buffer=3)
    plot_signal(classification, events_watch_time)

if __name__ == "__main__":
    main()