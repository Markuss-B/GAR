import sys
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np

# Add the data loaders directory to the path

from data.loaders import MyoGymLoader, RecoFitLoader

def plot_session(df, activity_mapping, trainer_id=1, df_name="MyoGym", figsize=(14, 8), transform_units=False):
    """
    Plot a session's accelerometer and gyroscope data with activity labels.
    
    Args:
        trainer_id: The trainer ID to plot (default: 1)
        figsize: Figure size (width, height) in inches
        transform_units: Whether to transform units (default: False)
    """
    
    # Filter data for the specified trainer
    session_data = df[df['trainer'] == trainer_id].reset_index(drop=True)
    
    if session_data.empty:
        print(f"No data found for trainer {trainer_id}")
        print(f"Available trainers: {df['trainer'].unique()}")
        return
    
    print(f"Plotting session for trainer {trainer_id}")
    print(f"Session duration: {session_data['time'].max():.2f} seconds")
    print(f"Data points: {len(session_data)}")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    fig.suptitle(f'{df_name} Session: Trainer {trainer_id}', fontsize=14, fontweight='bold')
    
    # Time formatter for mm:ss:ms
    def time_formatter(x, pos):
        minutes = int(x // 60)
        seconds = int(x % 60)
        ms = int((x % 1) * 1000)
        return f"{x:.2f}s\n{minutes:02d}:{seconds:02d}:{ms:03d}"
    
    # Apply formatter to x-axis
    for ax in axes:
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(time_formatter))
    
    time = session_data['time'].values
    
    # Plot accelerometer data
    ax_acc = axes[0]
    ax_acc.plot(time, session_data['acc_x'], label='Acc X', linewidth=1.5)
    ax_acc.plot(time, session_data['acc_y'], label='Acc Y', linewidth=1.5)
    ax_acc.plot(time, session_data['acc_z'], label='Acc Z', linewidth=1.5)
    if transform_units:
        ax_acc.set_ylabel('Acceleration (m/s²)', fontsize=11)
    else:
        ax_acc.set_ylabel('Acceleration (g)', fontsize=11)
    ax_acc.set_title('Accelerometer Data', fontsize=12, fontweight='bold')
    ax_acc.legend(loc='upper right', fontsize=9)
    ax_acc.grid(True, alpha=0.3)
    
    # Plot gyroscope data
    ax_gyr = axes[1]
    ax_gyr.plot(time, session_data['gyr_x'], label='Gyr X', linewidth=1.5)
    ax_gyr.plot(time, session_data['gyr_y'], label='Gyr Y', linewidth=1.5)
    ax_gyr.plot(time, session_data['gyr_z'], label='Gyr Z', linewidth=1.5)
    if transform_units:
        ax_gyr.set_ylabel('Angular Velocity (rad/s)', fontsize=11)
    else:
        ax_gyr.set_ylabel('Angular Velocity (dps)', fontsize=11)
    ax_gyr.set_xlabel('Time (mm:ss:ms)', fontsize=11)
    ax_gyr.set_title('Gyroscope Data', fontsize=12, fontweight='bold')
    ax_gyr.legend(loc='upper right', fontsize=9)
    ax_gyr.grid(True, alpha=0.3)
    
    # Add activity background colors
    activities = session_data['activity'].values
    unique_activities = np.unique(activities)
    
    # Create color map for activities
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_activities)))
    activity_colors = {act: colors[i] for i, act in enumerate(sorted(unique_activities))}
    
    # Add shaded regions for activities
    current_activity = activities[0]
    start_idx = 0
    for i in range(1, len(activities)):
        if activities[i] != current_activity:
            end_time = time[i]
            start_time = time[start_idx]
            color = activity_colors[current_activity]
            for ax in axes:
                ax.axvspan(start_time, end_time, alpha=0.15, color=color)
            current_activity = activities[i]
            start_idx = i
    
    # Add final activity region
    color = activity_colors[current_activity]
    for ax in axes:
        ax.axvspan(time[start_idx], time[-1], alpha=0.15, color=color)
    
    # Build activity segments for click detection
    activity_segments = []
    current_activity = activities[0]
    start_idx = 0
    for i in range(1, len(activities)):
        if activities[i] != current_activity:
            end_time = time[i]
            start_time = time[start_idx]
            activity_segments.append((start_time, end_time, current_activity))
            current_activity = activities[i]
            start_idx = i
    # Add final segment
    activity_segments.append((time[start_idx], time[-1], current_activity))
    
    # Create legend for activities
    # legend_patches = [
    #     mpatches.Patch(color=activity_colors[act], alpha=0.5, 
    #                   label=f"{activity_mapping.get(act, f'Activity {act}')}")
    #     for act in sorted(unique_activities) if act != 0
    # ]
    # if 0 in unique_activities:
    #     legend_patches.append(
    #         mpatches.Patch(color=activity_colors[0], alpha=0.5, 
    #                       label=activity_mapping.get(0, 'No Activity'))
    #     )
    
    # fig.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(0.12, 0.98), 
    #           fontsize=9, title='Activities', title_fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Add click event to show activity details as legend
    def on_click(event):
        if event.inaxes is not None:
            x_click = event.xdata
            for start_t, end_t, act_id in activity_segments:
                if start_t <= x_click < end_t:
                    activity_name = activity_mapping.get(act_id, f'Activity {act_id}')
                    duration = end_t - start_t
                    # Remove previous legend if exists
                    if hasattr(fig, '_activity_legend'):
                        fig._activity_legend.remove()
                    # Create new legend patch
                    patch = mpatches.Patch(color=activity_colors[act_id], alpha=0.5, 
                                          label=f"{activity_name}\nStart: {time_formatter(start_t, None)}\nEnd: {time_formatter(end_t, None)}\nDuration: {time_formatter(duration, None)}")
                    fig._activity_legend = fig.legend([patch], [patch.get_label()], 
                                                     loc='upper left', bbox_to_anchor=(0.12, 0.98), 
                                                     fontsize=9, title='Clicked Activity', title_fontsize=10)
                    fig.canvas.draw()
                    break
    
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()


if __name__ == "__main__":
    # Plot the first trainer's session
    # Load data
    dataset = int(input("MyoGym(1) or RecoFit(2): "))
    trainer_id = input("Trainer id: ")
    transform_units = input("Transform units? (y/n): ").lower() == 'y'

    if dataset == 1:
        name = "MyoGym"
        loader = MyoGymLoader("../data/datasets/MyoGym.mat", transform_units=transform_units)
        df, activity_mapping = loader.load_data()
    else:
        name = "RecoFit"
        loader = RecoFitLoader("../data/datasets/RecoFit", transform_units=transform_units)
        df, activity_mapping = loader.load_data()

    plot_session(df, activity_mapping, trainer_id=int(trainer_id), df_name=name, transform_units=transform_units)
