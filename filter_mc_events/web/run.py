import dash
from dash import dcc, html, Input, Output
import h5py
import numpy as np
import pandas as pd
import plotly.express as px

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
def update_display(event_id):
    subdf = df[df['event_id'] == event_id]
    fig = px.scatter_3d(
        subdf, x='x', y='y', z='z', color='Q',
        title=f'Event ID {event_id}',
        labels={'Q': 'Charge'}
    )
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(height=800)  # ? this line makes the plot taller
    return fig

if __name__ == '__main__':
    app.run(debug=True)

