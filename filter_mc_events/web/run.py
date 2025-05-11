import dash
from dash import dcc, html, Input, Output
import h5py
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# Load HDF5 data once
with h5py.File('many_muon_hits.hdf5', 'r') as f:
    hits = f['hits'][:]

# Convert to DataFrame for ease of filtering
df = pd.DataFrame(hits)
df['event_id'] = df['event_id'].astype(int)
df['io_group'] = df['io_group'].astype(int)

# Get unique event_ids
event_ids = sorted(df['event_id'].unique())

# Build Dash app
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H2("Muon Event Display"),
    dcc.Dropdown(
        id='event-dropdown',
        options=[{'label': str(eid), 'value': eid} for eid in event_ids],
        value=event_ids[0]
    ),
    dcc.Graph(
        id='event-graph',
        style={'height': '800px', 'width': '100%'}
    )
])

from dash import State

app.layout = html.Div([
    html.H2("Muon Event Display"),
    html.Div([
        html.Button('Previous', id='prev-button', n_clicks=0),
        dcc.Dropdown(
            id='event-dropdown',
            options=[{'label': str(eid), 'value': eid} for eid in event_ids],
            value=event_ids[0],
            style={'width': '200px', 'display': 'inline-block', 'margin': '0 10px'}
        ),
        html.Button('Next', id='next-button', n_clicks=0),
    ], style={'display': 'flex', 'alignItems': 'center'}),
    dcc.Graph(id='event-graph')
])

@app.callback(
    Output('event-dropdown', 'value'),
    Input('prev-button', 'n_clicks'),
    Input('next-button', 'n_clicks'),
    State('event-dropdown', 'value')
)
def change_event(prev_clicks, next_clicks, current_value):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_value
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    idx = event_ids.index(current_value)
    if button_id == 'prev-button':
        idx = max(idx - 1, 0)
    elif button_id == 'next-button':
        idx = min(idx + 1, len(event_ids) - 1)
    return event_ids[idx]

@app.callback(
    Output('event-graph', 'figure'),
    Input('event-dropdown', 'value')
)
    # fig.update_layout(height=800)  # ? this line makes the plot taller
def update_display(event_id):
    subdf = df[df['event_id'] == event_id]
    fig = go.Figure()

    # for group in np.unique(subdf['io_group']):
    #     group_df = subdf[subdf['io_group'] == group]
    #     fig.add_trace(go.Scatter3d(
    #         x=group_df['x'], y=group_df['y'], z=group_df['z'],
    #         mode='markers',
    #         marker=dict(
    #             size=3,
    #             color=group_df['Q'],
    #             coloraxis='coloraxis'  # Shared color scale
    #         ),
    #         name=f'io_group {group}'
    #     ))
    fig.add_trace(go.Scatter3d(
        x=subdf['x'], y=subdf['y'], z=subdf['z'],
        mode='markers',
        marker=dict(
            size=1,
            color=subdf['Q'],
            coloraxis='coloraxis'  # Shared color scale
        ),
        name='all hits'
    ))

    fig.update_layout(
        title=f'Event ID {event_id}',
        height=800,
        coloraxis=dict(
            colorscale='Viridis',
            colorbar=dict(title='Q')
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.2,
            xanchor='center',
            x=0.5
        )
    )
    return fig

if __name__ == '__main__':
    app.run(debug=True)

