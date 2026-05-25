from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from edit_recording import load_recording
from process_and_save_recording import convert_events_to_watch_time
from plot_signal import plot_signal

def apply_smoothing_non_recursive(classification, window_size=3):
    df = classification.copy()
    
    df['prediction_binary_median'] = df['prediction_binary'].rolling(window=window_size).median().fillna(0).astype(int)
    
    return df

def apply_state_machine(df, start_buffer=3, end_buffer=5):
    raw_binary = df['prediction_binary'].values
    smoothed = np.zeros_like(raw_binary)
    current_state = 0
    counter = 0

    for i in range(len(raw_binary)):
        if current_state == 0:
            if raw_binary[i] == 1:
                counter += 1
            else:
                counter = 0
            if counter >= start_buffer:
                current_state = 1
                counter = 0
        else:
            if raw_binary[i] == 0:
                counter += 1
            else:
                counter = 0
            if counter >= end_buffer:
                current_state = 0
                counter = 0
        
        smoothed[i] = current_state
    return smoothed

def apply_accumulator_aggregation(
    df,
    threshold=3
):
    preds = df['prediction_binary'].values
    aggregated = []

    
    state = 0
    accumulator = 0
    for pred in preds:
        if pred == 1:
            accumulator += 1
        else:
            accumulator -= 1

        accumulator = max(0, min(threshold, accumulator))

        if state == 0 and accumulator >= threshold:
            state = 1
        elif state == 1 and accumulator <= 0:
            state = 0

        aggregated.append(state)

    return aggregated

def apply_inertia_strategy(clf, entry_threshold=2, max_inertia=15):
    preds = clf['prediction_binary'].values
    smoothed = np.zeros_like(preds)
    
    inertia = 0
    is_active = False
    
    for i in range(len(preds)):
        if preds[i] == 1:
            inertia = min(inertia + 1, max_inertia)
        else:
            inertia = max(inertia - 1, 0)
            
        # Entry Logic
        if not is_active and inertia >= entry_threshold:
            is_active = True
        
        # Exit Logic: Must hit 0 to exit
        if is_active and inertia == 0:
            is_active = False
            
        smoothed[i] = 1 if is_active else 0
        
    return smoothed

def plot_signal_on_axis(ax, classification, events, title=""):
    df_class = classification
    df_events = pd.DataFrame(events)

    ax.step(df_class['timestamp'], df_class['prediction_float'], where='post',
            label='Prediction Probability', color='red', alpha=0.1, linewidth=1.5)

    ax.step(df_class['timestamp'], df_class['prediction_binary'], where='post',
            label='>=0.5 = Exercise', color='black', linewidth=1.5, alpha=0.7)

    # Event colors
    unique_events = df_events['name'].unique()
    colors = plt.get_cmap('tab20', len(unique_events))
    event_color_map = {name: colors(i) for i, name in enumerate(unique_events)}

    seen = set()

    for _, event in df_events.iterrows():
        label = event['name'] if event['name'] not in seen else None
        seen.add(event['name'])

        ax.axvspan(
            event['start_watch'],
            event['end_watch'],
            color=event_color_map[event['name']],
            alpha=0.3,
            label=label
        )

    ax.set_title(title)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

def plot_multiple(strategies, events):
    fig, axs = plt.subplots(4, 1, figsize=(15, 10), sharex=True)

    for ax, (name, df) in zip(axs, strategies):
        plot_signal_on_axis(ax, df, events, title=name)

    handles, labels = axs[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    fig.legend(by_label.values(), by_label.keys(),
            loc='upper right', bbox_to_anchor=(1, 1))

    # Only bottom plot gets x-label
    axs[-1].set_xlabel('Timestamp (ms)')

    # Shared y-label (optional)
    fig.text(0.01, 0.5, 'Prediction Value', va='center', rotation='vertical')

    plt.ylabel('  ')

    plt.tight_layout()
    plt.show()

def main():
    path = "session_1776938763711"
    id = 3
    accel, gyro, classification, annotation, info = load_recording(path)

    events_watch_time = convert_events_to_watch_time(annotation, info)

    # ignore treadmill running
    ignore_list = ['Treadmill Running', 'Side Crunches',
                #'Chest Fly Machine', 'Dumbbell Overhead Press', 'Cable Rows'
                ]
    filtered_events = [e for e in events_watch_time if e['name'] not in ignore_list]

    raw_df = classification.copy()

    acc = classification.copy()
    acc["prediction_binary"] = apply_accumulator_aggregation(classification)

    preds = pd.read_csv("prepared_sessions/fine_tuned_test_predictions.csv")
    preds = preds[preds["session_id"] == id]

    finetuned = classification.copy()
    finetuned["prediction_float"] = preds["prediction_finetuned_float"].values
    finetuned["prediction_binary"] = preds["prediction_finetuned_binary"].values

    finetuned_acc3 = finetuned.copy()
    finetuned_acc3["prediction_binary"] = apply_accumulator_aggregation(finetuned_acc3)
    
    # median_df = apply_smoothing_non_recursive(classification, window_size=3).copy()
    # median_df['prediction_binary'] = median_df['prediction_binary_median']
    
    # hyst_df_2_3 = classification.copy()
    # hyst_df_2_3['prediction_binary'] = apply_state_machine(classification, start_buffer=2, end_buffer=3)

    # hyst_df_1_3 = classification.copy()
    # hyst_df_1_3['prediction_binary'] = apply_state_machine(classification, start_buffer=1, end_buffer=3)

    # inertia = classification.copy()
    # inertia["prediction_binary"] = apply_inertia_strategy(inertia, entry_threshold=2, max_inertia=15)

    strategies = [
        ("raw", raw_df),
        ("acc 3", acc),
        ("finetuned", finetuned),
        ("finetuned_acc3", finetuned_acc3)
        #("median", median_df),
        #("state_machine s=2 e=3", hyst_df_2_3),
        #("state_machine s=1 e=3", hyst_df_1_3),
        #("inertia", inertia)
    ]

    plot_multiple(strategies, filtered_events)

if __name__ == "__main__":
    main()



    