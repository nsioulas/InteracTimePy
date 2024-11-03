import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from ipywidgets import Output, VBox, HBox, Button, ToggleButton, Textarea
import pickle
import os
from IPython.display import display

def interactive_timeseries_plot(dfs, save_path, plot_config=None, width=900, height_per_subplot=300, resample_rule=None, default_freq = '1T'):
    """
    Generate an interactive, multi-panel overview plot of timeseries data from given DataFrames using Plotly FigureWidget.
    The function enables users to:
    - Zoom into specific parts of the timeseries, synchronizing all subplots.
    - Select time intervals by enabling selection mode and using two single clicks to mark the start and end.
    - Display the interval with a vertical shaded area (axvspan) in real-time across all subplots.
    - Compute and store the interval's mean, median, and standard deviation.
    - Add comments to each selected interval.
    - Delete selected intervals along with the corresponding vertical shading.
    - Display information about the functionalities.
    - Save the results to a specified path.
    - Optionally, plot multiple columns in the same subplot with axis type and twin axes options.
    - Resample the timeseries data if a resampling rule is provided.
    - Accept multiple DataFrames as input, merge them into one with the cadence of the first DataFrame.
    
    Parameters:
    - dfs: List of pandas DataFrames with timeseries data. Index should be datetime.
    - save_path: Path to the folder where the results will be saved.
    - plot_config: List of dictionaries specifying plot configuration for each subplot.
    - width: Width of the figure in pixels.
    - height_per_subplot: Height per subplot in pixels.
    - resample_rule: Resampling rule (e.g., '1H' for hourly resampling). If None, frequency is inferred.
    """
    # Ensure that dfs is a list
    if not isinstance(dfs, list):
        dfs = [dfs]
    
    # Ensure all DataFrames have datetime index
    dfs = [df.copy() for df in dfs]
    for df in dfs:
        df.index = pd.to_datetime(df.index)
    
    # Determine the frequency
    if resample_rule is not None:
        freq = resample_rule
    else:
        freq = pd.infer_freq(dfs[0].index)
        if freq is None:
            freq = default_freq  # default frequency

    # Resample all DataFrames
    resampled_dfs = []
    for df in dfs:
        df_resampled = df.resample(freq).mean()
        resampled_dfs.append(df_resampled)
    
    # Concatenate all DataFrames
    df = pd.concat(resampled_dfs, axis=1)
    
    # Optionally, fill missing values (interpolate or forward fill)
    df.interpolate(method='time', inplace=True)

    # Handle plot_config
    if plot_config is None:
        # Default: Each column in a separate subplot
        plot_config = [{'columns': [col], 'axis_type': 'linear', 'twin_axes': False} for col in df.columns]
    
    num_subplots = len(plot_config)
    
    # Determine specs for subplots
    specs = []
    for config in plot_config:
        if config.get('twin_axes', False):
            specs.append([{'secondary_y': True}])
        else:
            specs.append([{}])
    
    # Create subplots with visible boundaries
    fig = make_subplots(
        rows=num_subplots,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        specs=specs
    )
    
    # Add traces
    for i, config in enumerate(plot_config):
        row = i + 1
        cols = config['columns']
        axis_type = config.get('axis_type', 'linear')
        twin_axes = config.get('twin_axes', False)
        y_label = config.get('y_label', None)
        if y_label is None:
            y_label = cols  # Use column names if y_label not specified
        elif isinstance(y_label, str):
            y_label = [y_label] * len(cols)
    
        if twin_axes and len(cols) >= 2:
            first_col = cols[0]
            other_cols = cols[1:]
            # Handle y_labels
            first_label = y_label[0] if len(y_label) > 0 else first_col
            other_labels = y_label[1:] if len(y_label) > 1 else other_cols
            # Handle axis_type
            if isinstance(axis_type, list) and len(axis_type) >= 2:
                primary_axis_type = axis_type[0]
                secondary_axis_type = axis_type[1]
            else:
                primary_axis_type = axis_type if isinstance(axis_type, str) else 'linear'
                secondary_axis_type = primary_axis_type
            # Add first trace on primary y-axis
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df[first_col], mode='lines', name=first_col, showlegend=False
                ),
                row=row, col=1, secondary_y=False
            )
            # Add other traces on secondary y-axis
            for idx, col in enumerate(other_cols):
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=df[col], mode='lines', name=col, showlegend=False
                    ),
                    row=row, col=1, secondary_y=True
                )
            # Set y-axis types and labels
            fig.update_yaxes(type=primary_axis_type, title_text=first_label, row=row, col=1, secondary_y=False)
            fig.update_yaxes(type=secondary_axis_type, title_text=', '.join(other_labels), row=row, col=1, secondary_y=True)
        else:
            # All traces on primary y-axis
            for idx, col in enumerate(cols):
                label = y_label[idx] if idx < len(y_label) else col
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=df[col], mode='lines', name=col, showlegend=False
                    ),
                    row=row, col=1
                )
            # Set y-axis type and label
            fig.update_yaxes(type=axis_type, title_text=', '.join(y_label), row=row, col=1)
    
    # Create FigureWidget
    fig_widget = go.FigureWidget(fig)
    fig_widget.update_layout(
        height=height_per_subplot*num_subplots,
        width=width,
        dragmode='zoom',
        template='plotly_white',  # Use white background
        showlegend=False
    )
    
    # Make subplots boundaries visible
    for i in range(num_subplots):
        fig_widget.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, row=i+1, col=1)
        fig_widget.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, row=i+1, col=1)
    
    # Initialize shapes
    fig_widget.layout.shapes = []
    
    # Widgets and variables
    selection_toggle = ToggleButton(description='Selection Mode', value=False)
    delete_toggle = ToggleButton(description='Delete Mode', value=False)
    info_button = Button(description='Info')
    save_button = Button(description='Save Results')
    stats_output = Output()
    save_output = Output()
    comment_box = Textarea(description='Add comment:', layout={'width': '400px', 'height': '100px'})
    submit_comment_button = Button(description='Submit Comment')
    comment_output = Output()
    current_event_name = None
    selection_points = []
    event_data = {}
    event_counter = 1
    event_shapes = {}  # Map event names to shapes
    
    # Click function
    def click_fn(trace, points, state):
        nonlocal selection_points, event_counter, current_event_name, event_shapes, event_data
    
        if selection_toggle.value:
            # In Selection Mode
            if len(points.point_inds) == 0:
                return  # No point was clicked
            # Get the x-value of the clicked point
            clicked_index = points.point_inds[0]
            x_value = trace.x[clicked_index]
            selection_points.append(x_value)
            if len(selection_points) == 2:
                # Two points have been selected; process the interval
                start_time, end_time = sorted(selection_points)
                interval_df = df.loc[start_time:end_time]
                if interval_df.empty:
                    selection_points = []
                    return
                # Compute statistics
                stats = {
                    'start': str(start_time),
                    'end': str(end_time),
                    'mean': interval_df.mean().to_dict(),
                    'median': interval_df.median().to_dict(),
                    'std': interval_df.std().to_dict(),
                }
                event_name = f'ev_{event_counter}'
                event_data[event_name] = stats
                event_counter += 1
                current_event_name = event_name  # Set the current event
                # Highlight the selected interval across all subplots
                new_shape = dict(
                    type='rect',
                    xref='x',
                    x0=start_time,
                    x1=end_time,
                    yref='paper',
                    y0=0,
                    y1=1,
                    fillcolor='rgba(200, 200, 200, 0.5)',  # Light gray with 50% opacity
                    layer='below',
                    line_width=0,
                )
                fig_widget.add_shape(new_shape)
                # Get the actual shape object and store it
                added_shape = fig_widget.layout.shapes[-1]
                event_shapes[event_name] = added_shape
                # Update the stats output
                with stats_output:
                    print(f"{event_name}:")
                    print(f"Start: {stats['start']}")
                    print(f"End: {stats['end']}")
                    print(f"Mean: {stats['mean']}")
                    print(f"Median: {stats['median']}")
                    print(f"Std: {stats['std']}")
                    print('-' * 40)
                # Display the comment box and submit comment button
                display(HBox([comment_box, submit_comment_button]))
                # Reset selection points
                selection_points = []
        elif delete_toggle.value:
            # In Delete Mode
            if len(points.point_inds) == 0:
                return  # No point was clicked
            # Get the x-value of the clicked point
            clicked_index = points.point_inds[0]
            x_value = pd.to_datetime(trace.x[clicked_index])
            # Remove shapes corresponding to intervals that include x_value
            events_to_delete = []
            for event_name, shape in event_shapes.items():
                x0 = pd.to_datetime(shape['x0'])
                x1 = pd.to_datetime(shape['x1'])
                if x0 <= x_value <= x1:
                    events_to_delete.append(event_name)
                    if event_name in event_data:
                        del event_data[event_name]
                        with stats_output:
                            print(f"Deleted {event_name}")
            # Remove shapes from the figure
            shapes_to_keep = []
            for shape in fig_widget.layout.shapes:
                if shape not in [event_shapes[event_name] for event_name in events_to_delete]:
                    shapes_to_keep.append(shape)
            fig_widget.layout.shapes = tuple(shapes_to_keep)
            # Remove events from event_shapes
            for event_name in events_to_delete:
                del event_shapes[event_name]
        else:
            # Do nothing
            pass
    
    # Attach click handler to all traces
    for trace in fig_widget.data:
        trace.on_click(click_fn)
    
    # Submit comment function
    def submit_comment(b):
        nonlocal current_event_name
        if current_event_name is not None:
            comment = comment_box.value
            event_data[current_event_name]['comment'] = comment
            with comment_output:
                print(f"Comment added to {current_event_name}: {comment}")
            # Clear the comment box
            comment_box.value = ''
            current_event_name = None  # Reset current event
    
    submit_comment_button.on_click(submit_comment)
    
    # Sync zoom function using observe
    def sync_zoom(change):
        if 'xaxis.range' in change['new']:
            x_range = change['new']['xaxis.range']
            if x_range:
                x0, x1 = x_range
                with fig_widget.batch_update():
                    for i in range(num_subplots):
                        axis_name = 'xaxis{}'.format(i+1) if i > 0 else 'xaxis'
                        fig_widget.layout[axis_name].range = [x0, x1]
    
    fig_widget.observe(sync_zoom, names='layout')
    
    # Function to save results
    def save_results(b):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        start_timestamp = df.index[0]
        end_timestamp = df.index[-1]
        filename = f"{start_timestamp.strftime('%H_%M_%d_%m_%Y')}_{end_timestamp.strftime('%H_%M_%d_%m_%Y')}.pkl"
        filepath = os.path.join(save_path, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(event_data, f)
        with save_output:
            print(f'Data saved to {filepath}')
    
    save_button.on_click(save_results)
    
    # Function to handle selection toggle change
    def on_selection_toggle_change(change):
        if change.new:
            # Selection Mode activated
            delete_toggle.value = False  # Ensure Delete Mode is off
        else:
            # Selection Mode turned off
            pass  # Do nothing
    
    selection_toggle.observe(on_selection_toggle_change, names='value')
    
    # Function to handle delete toggle change
    def on_delete_toggle_change(change):
        if change.new:
            # Delete Mode activated
            selection_toggle.value = False  # Ensure Selection Mode is off
        else:
            # Delete Mode turned off
            pass  # Do nothing
    
    delete_toggle.observe(on_delete_toggle_change, names='value')
    
    # Info button function
    def show_info(b):
        info_text = """
        **Functionality Guide:**
    
        - **Zooming:** Use mouse scroll or drag to zoom into specific parts of the timeseries. All subplots will synchronize.
    
        - **Selection Mode:** Click the 'Selection Mode' toggle button to enable interval selection.
          - In Selection Mode, click once to mark the **start** of the interval.
          - Click again to mark the **end** of the interval.
          - The selected interval will be highlighted across all subplots.
          - Statistics for the interval will be computed and displayed.
          - You can add a comment to the interval after selecting.
    
        - **Delete Mode:** Click the 'Delete Mode' toggle button to enable deletion of intervals.
          - In Delete Mode, click on a previously selected interval to delete it.
          - The interval and its corresponding shading will be removed from the plot and the event data.
    
        - **Save Results:** Click the 'Save Results' button to save the event data to a file.
    
        """
        with stats_output:
            print(info_text)
    
    info_button.on_click(show_info)
    
    # Display the widgets and the figure
    display(VBox([
        fig_widget,
        HBox([selection_toggle, delete_toggle, info_button, save_button, save_output]),
        stats_output,
        comment_output
    ]))
