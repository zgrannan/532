import pandas as pd 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from trl import SFTTrainer
from typing import Union


def create_training_plots(data: Union[pd.DataFrame, SFTTrainer]) -> go.Figure:
    """Create interactive training visualization using Plotly."""

    # Get training history
    if isinstance(data, pd.DataFrame):
        history = data
        num_epoch = int(np.round(data.epoch.max(), 1))
    elif isinstance(data, SFTTrainer):
        history = pd.DataFrame(data.state.log_history)
        num_epoch = data.args.num_train_epochs

    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=('Training Metrics',),
        specs=[[{"secondary_y": True}]]
    )

    # Add traces for loss and grad_norm
    fig.add_trace(
        go.Scatter(
            x=history['step'],
            y=history['loss'],
            name="Loss",
            line=dict(color='blue'),
            mode='lines'
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            x=history['step'],
            y=history['grad_norm'],
            name="Gradient Norm",
            line=dict(color='red'),
            mode='lines'
        ),
        secondary_y=True
    )

    # Calculate epoch boundaries
    steps_per_epoch = len(history) // num_epoch
    epoch_boundaries = np.arange(0, len(history), steps_per_epoch)

    # Add epoch markers
    for epoch, step in enumerate(epoch_boundaries):
        fig.add_vline(
            x=step, 
            line_dash="dash", 
            line_color="gray",
            opacity=0.5
        )
        # Add epoch number annotation below the graph
        fig.add_annotation(
            x=step,
            y=-0.15,  # Position below the graph
            text=f"Epoch {epoch}",
            showarrow=False,
            yref='paper',  # Reference to the paper coordinates
            yshift=30,
            font=dict(size=10)
        )

    # Update layout
    fig.update_layout(
        height=600,  # Increased height
        width=1000,  # Increased width
        title_text="Training Progress Overview",
        showlegend=True,
        hovermode='x unified',
        template='plotly_white',
        margin=dict(b=100),  # Increased bottom margin for epoch labels
    )

    # Update y-axes labels
    fig.update_yaxes(title_text="Loss", secondary_y=False)
    fig.update_yaxes(title_text="Gradient Norm", secondary_y=True)
    
    # Update x-axis label
    fig.update_xaxes(title_text="Steps")

    # Add range slider
    fig.update_xaxes(rangeslider_visible=False)

    # Add helpful hover template
    for trace in fig.data:
        trace.update(
            hovertemplate=(
                "<b>Step</b>: %{x}<br>" +
                "<b>%{fullData.name}</b>: %{y:.6f}<br>" +
                "<extra></extra>"  # This removes the secondary box
            )
        )

    return fig