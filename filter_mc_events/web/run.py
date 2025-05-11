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
from dash import State

app.layout = html.Div([
    html.H2("Event Display"),
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
    html.Div([
        html.Button('Pick', id='pick-button', n_clicks=0),
        html.Button('Reset', id='reset-button', n_clicks=0),
        html.H4("Picked Points (x, y, z):"),
        html.Pre(id='picked-points', children="[]"),
        dcc.Store(id='picked-store', data=[]),
    ]),
    html.Div([
        html.Button('Draw Line', id='line-button', n_clicks=0),
        dcc.Store(id='line-store', data=[]),
    ]),

    dcc.Graph(
        id='event-graph',
        style={'height': '800px', 'width': '100%'}
    )
])

@app.callback(
    Output('picked-store', 'data'),
    Input('pick-button', 'n_clicks'),
    Input('reset-button', 'n_clicks'),
    Input('event-dropdown', 'value'),
    State('event-graph', 'clickData'),
    State('picked-store', 'data'),
    prevent_initial_call=True
)
def update_picked(pick_n, reset_n, event_id, clickData, picked):
    trigger_id = dash.callback_context.triggered_id

    if trigger_id in ['reset-button', 'event-dropdown']:
        return []

    if trigger_id == 'pick-button' and clickData:
        point = clickData['points'][0]
        x, y, z = point['x'], point['y'], point['z']
        q = point.get('marker.color', None)
        picked.append((x, y, z))
    return picked

@app.callback(
    Output('picked-points', 'children'),
    Input('picked-store', 'data')
)
def display_picked(data):
    return str(data)


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
    Output('line-store', 'data'),
    Input('line-button', 'n_clicks'),
    Input('event-dropdown', 'value'),
    State('picked-store', 'data'),
    prevent_initial_call=True
)
def update_line(n_clicks, event_id, picked):
    ctx = dash.callback_context
    trigger_id = ctx.triggered_id

    if trigger_id == 'event-dropdown':
        return []  # clear line on event change

    if trigger_id == 'line-button' and len(picked) >= 2:
        print(trigger_id, picked)
        return picked[:2]

    return []

@app.callback(
    Output('event-graph', 'figure'),
    Input('event-dropdown', 'value'),
    Input('line-store', 'data'),
)
def update_display(event_id, line_data):
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
    print('line-store', line_data)

    # Draw line if present
    if line_data and len(line_data) == 2:
        x, y, z = zip(*line_data)
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines+markers',
            line=dict(color='red', width=5),
            marker=dict(size=5, color='red'),
            name='Line'
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

