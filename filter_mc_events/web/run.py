import dash
from dash import dcc, html, Input, Output, State
import h5py
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys

# Load HDF5 data once
df = None
event_ids = []

# Build Dash app
app = dash.Dash(__name__)

@app.callback(
    Output('h5f-load-status', 'children'),
    Output('event-dropdown', 'value', allow_duplicate=True),
    Output('event-dropdown', 'options'),
    Input('load-button', 'n_clicks'),
    State('h5f-path-input', 'value'),
    prevent_initial_call=True,
)
def load_data(n_clicks, path):
    global df
    global event_ids
    source_file = path
    istred = False
    isnd = False
    with h5py.File(source_file, 'r') as f:
        if '/hits' in f:
            hits = f['hits'][:]
            istred = True
        elif 'charge/calib_prompt_hits' in f and 'charge/events/' in f:
            isnd = True
            hits = f['charge/calib_prompt_hits/data'][:]
            hits = hits[['id', 'x', 'y', 'z', 't_drift', 'ts_pps',
                         'io_group', 'io_channel', 'chip_id', 'channel_id', 'Q', 'E']]
            event_id = f['/charge/events/ref/charge/calib_prompt_hits/ref'][:,0][np.argsort(f['/charge/events/ref/charge/calib_prompt_hits/ref'][:,1])]
        else:
            return f"Not found valid data in {path}", None, []

    # Convert to DataFrame for ease of filtering
    if istred:
        df = pd.DataFrame(hits)
        df['event_id'] = df['event_id'].astype(int)
        df['io_group'] = df['io_group'].astype(int)
    elif isnd:
        df = pd.DataFrame.from_records(hits)
        df['event_id'] = event_id
        df['io_group'] = df['io_group'].astype(int)
    # Get unique event_ids
    event_ids = sorted(df['event_id'].unique())
    return f"Loaded hits from {path}", event_ids[0], event_ids

app.layout = html.Div([
    html.H2("Event Display"),
    html.Div([
        html.Label("Enter HDF5 file path:"),
        dcc.Input(
            id='h5f-path-input',
            type='text',
            placeholder='/group1/my_dataset',
            debounce=True,  # update only after user finishes typing
            style={'width': '300px', 'margin-right': '10px'}
        ),
        html.Button("Load File", id='load-button', n_clicks=0),
    ]),
    html.Div(id='h5f-load-status', style={'margin-top': '15px'}),
    html.Div([
        html.Button('Previous', id='prev-button', n_clicks=0),
        dcc.Dropdown(
            id='event-dropdown',
            options=[{'label': str(eid), 'value': eid} for eid in event_ids],
            value = event_ids[0] if len(event_ids) else 0,
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
    ]),
    dcc.Store(id='line-store', data=[]),

    html.Div([
        html.Button('Select Nearby Points', id='select-nearby-button', n_clicks=0),
        dcc.Input(
            id='distance-threshold',
            type='number',
            value=3,
            step=0.1,
            style={'width': '100px', 'marginLeft': '10px'}
        ),
    ], style={'marginTop': '10px'}),
    dcc.Store(id='nearby-store', data=[]),
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

    if current_value:
        idx = event_ids.index(current_value)
    else:
        idx = event_ids[0]
    if button_id == 'prev-button':
        idx = max(idx - 1, 0)
    elif button_id == 'next-button':
        idx = min(idx + 1, len(event_ids) - 1)
    return event_ids[idx]

@app.callback(
    Output('nearby-store', 'data'),
    Input('select-nearby-button', 'n_clicks'),
    State('line-store', 'data'),
    State('event-dropdown', 'value'),
    State('distance-threshold', 'value'),
    State('h5f-path-input', 'value'),
    prevent_initial_call=True
)
def compute_nearby(n_clicks, line_data, event_id, threshold, source_file):
    if not line_data or len(line_data) < 2:
        return {'in': [], 'out': []}

    p0, p1 = np.array(line_data[0]), np.array(line_data[1])
    subdf = df[df['event_id'] == event_id]

    # Compute distances
    seg_vec = p1 - p0
    seg_len2 = np.dot(seg_vec, seg_vec)
    points = subdf[['x', 'y', 'z']].to_numpy()
    vecs = points - p0
    t = np.dot(vecs, seg_vec) / seg_len2
    t = np.clip(t, 0, 1)
    proj = p0 + np.outer(t, seg_vec)
    dists = np.linalg.norm(points - proj, axis=1)

    in_mask = dists <= threshold
    df_in = subdf[in_mask]
    df_out = subdf[~in_mask]

    # Save to HDF5
    output_file = f"{source_file.replace('.hdf5', '_evt')}{event_id}.hdf5"
    direction = seg_vec / np.linalg.norm(seg_vec)

    with h5py.File(output_file, 'w') as f:
        f.create_dataset('hits/selected/data', data=df_in.to_records(index=False))
        f.create_dataset('hits/deselected/data', data=df_out.to_records(index=False))
        f.create_dataset('picked/points/data', data=np.array([p0, p1]))
        f.create_dataset('picked/direction/data', data=direction)
        f.create_dataset('source_file', data=np.bytes_(source_file))

    return {
        'in': df_in.to_dict('records'),
        'out': df_out.to_dict('records')
    }

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
        return picked[:2]

    return []

@app.callback(
    Output('picked-store', 'data', allow_duplicate=True),
    Output('line-store', 'data', allow_duplicate=True),
    Output('nearby-store', 'data', allow_duplicate=True),
    Input('event-dropdown', 'value'),
    prevent_initial_call=True
)
def clear_on_event_change(event_id):
    return [], [], {'in': [], 'out': []}

@app.callback(
    Output('event-graph', 'figure'),
    Input('event-dropdown', 'value'),
    Input('line-store', 'data'),
    Input('nearby-store', 'data'),
)
def update_display(event_id, line_data, nearby_data):
    if event_id is not None:
        subdf = df[df['event_id'] == event_id]
    else:
        return go.Figure()
    fig = go.Figure()

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
    xmin, xmax = -2+np.min(subdf["x"]), 2+np.max(subdf["x"])
    ymin, ymax = -2+np.min(subdf["y"]), 2+np.max(subdf["y"])
    zmin, zmax = -2+np.min(subdf["z"]), 2+np.max(subdf["z"])

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

    # Plot nearby points
    if nearby_data:
        near_in = pd.DataFrame(nearby_data['in'])
        near_out = pd.DataFrame(nearby_data['out'])

        if not near_out.empty:
            fig.add_trace(go.Scatter3d(
                x=near_out['x'], y=near_out['y'], z=near_out['z'],
                mode='markers',
                marker=dict(size=2, color='lightgray'),
                name='Outside Range'
            ))

        if not near_in.empty:
            fig.add_trace(go.Scatter3d(
                x=near_in['x'], y=near_in['y'], z=near_in['z'],
                mode='markers',
                marker=dict(size=4, color='blue'),
                name='Within Range'
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
        ),
        scene=dict(
            # xaxis=dict(range=[-80, 80], title='x'),
            # yaxis=dict(range=[-80, 80], title='y'),
            # zaxis=dict(range=[-80, 80], title='z')
            xaxis=dict(range=[xmin, xmax], title='x'),
            yaxis=dict(range=[ymin, ymax], title='y'),
            zaxis=dict(range=[zmin, zmax], title='z')
        )
    )
    return fig

if __name__ == '__main__':
    try:
        port = sys.argv[1]
    except:
        port = 8050
    app.run(debug=True, port=port)

