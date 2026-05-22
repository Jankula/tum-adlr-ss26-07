import os
import glob
import plotly.graph_objects as go
from tensorboard.backend.event_processing import event_accumulator

def plot_interactive_epochs(config):
    """
    Parses logged TensorBoard data for the current experiment and generates
    an interactive HTML plot overlaying the timestep loss curves for all epochs.
    """
    # 1. Resolve paths exactly how they are structured in your main function
    log_path = f"logs/diffusion2obj/{config['experiment_name']}"
    event_files = glob.glob(os.path.join(log_path, "events.out.tfevents.*"))
    
    if not event_files:
        # Recursive fallback search just in case
        event_files = glob.glob(os.path.join(log_path, "**", "events.out.tfevents.*"), recursive=True)
        
    if not event_files:
        print(f"Error: No TensorBoard log files discovered in: {log_path}")
        return

    print(f"Parsing log stream: {os.path.basename(event_files[0])}...")
    
    # 2. Extract scalar events (size_guidance={...: 0} forces loading ALL logs)
    ea = event_accumulator.EventAccumulator(event_files[0], size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    
    # 3. Filter and sort your specific epoch charts
    target_prefix = "MSE Timestep Error/Epoch_"
    epoch_tags = sorted([t for t in ea.Tags()['scalars'] if t.startswith(target_prefix)])
    
    if not epoch_tags:
        print(f"Error: No charts found matching prefix: '{target_prefix}'")
        return

    # 4. Build the interactive Plotly figure
    fig = go.Figure()
    
    for tag in epoch_tags:
        epoch_label = tag.split("_")[-1]  # Extracts '000', '001', etc.
        
        events = ea.Scalars(tag)
        timesteps = [e.step for e in events]
        loss_values = [e.value for e in events]
        
        fig.add_trace(go.Scatter(
            x=timesteps,
            y=loss_values,
            mode='lines',
            name=f"Epoch {epoch_label}",
            opacity=0.75,
            line=dict(width=1.5),
            hovertemplate="Timestep: %{x}<br>MSE Error: %{y:.5f}<extra></extra>"
        ))

    # 5. Apply layout settings for clean visualization
    fig.update_layout(
        title=dict(
            text=f"Timestep Loss Evolution Overlap — {config['experiment_name']}",
            x=0.5
        ),
        xaxis_title="Diffusion Timestep (t)",
        yaxis_title="Average MSE Error",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(
            title="Interactive Legend<br><sup>(Click to toggle | Double-click to isolate)</sup>",
            traceorder="normal"
        )
    )

    # 6. Save and automatically open in your default Ubuntu browser window
    output_html = f"overlay_{config['experiment_name']}.html"
    fig.write_html(output_html, auto_open=True)
    print(f"Interactive canvas successfully rendered: {output_html}")