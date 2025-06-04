import dash
from dash import dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import h5py
from collections import defaultdict
from sklearn.cluster import DBSCAN
import os

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = dbc.Container([
    html.H2("Interactive Projection Plots from HDF5"),
    dbc.Row([
        dbc.Col([dcc.Upload(
            id='upload-data',
            children=html.Div(['Drag and Drop or ', html.A('Select a HDF5 File')]),
            style={
                'width': '100%', 'height': '60px', 'lineHeight': '60px',
                'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                'textAlign': 'center', 'margin': '10px'
            },
            multiple=False
        )], width=6),
        dbc.Col([
            dbc.Input(id='qmin', type='number', placeholder='Q min', value=25),
            dbc.Input(id='qmax', type='number', placeholder='Q max', value=100),
            dbc.Button("Update Plot", id="update-button", className="mt-2", color="primary")
        ])
    ]),
    html.Div(id='filepath-display', style={'marginTop': '10px', 'fontStyle': 'italic'}),
    html.Div([
        html.Span("Hit Color", style={"margin-right": "10px", "align-self": "center"}),
        dcc.RadioItems(
            id='hit-color-style',
            options=[
                {'label': 'Category', 'value': 'category'},
                {'label': 'totQ', 'value': 'totQ'}
            ],
            value='category',  # initial selection
            labelStyle={'display': 'inline-block', 'margin-left': '15px'}
        ),
    ], style={"display": "flex", "align-items": "center", "margin-bottom": "20px"}),
    dcc.Store(id='memory-path'),
    dcc.Graph(id='yz-plot'),
    dcc.Graph(id='xz-plot'),
])


def load_hits_from_hdf5(filepath):
    with h5py.File(filepath, 'r') as f:
        selected = f['/hits/selected/data'][:]
        deselected = f['/hits/deselected/data'][:]
    return selected, deselected


def reduce_hits(hits):
    y = np.round(hits['y'], 3)
    z = np.round(hits['z'], 3)
    yz = np.stack((y, z), axis=1)
    if len(yz) <= 1:
        return []
    # clustering = DBSCAN(eps=0.001, min_samples=1).fit(yz)
    # labels = clustering.labels_
    grouped = defaultdict(lambda: {'totQ': 0.0, 'totN': 0, 'x' : 0})

    for i, yzval in enumerate(yz):
        key = tuple(yz[i])
        grouped[key]['totQ'] += hits['Q'][i]
        grouped[key]['totN'] += 1
        grouped[key]['x'] += hits['x'][i]
    for k,v in grouped.items():
        grouped[k]['x'] = v['x']/v['totN']

    return [
        {"x" : float(v['x']), "y": float(k[0]), "z": float(k[1]), "totQ": float(v['totQ']), "totN": int(v['totN'])}
        for k, v in grouped.items()
    ]


def filter_marked(hits, qmin, qmax):
    return [h for h in hits if qmin <= h['totQ'] <= qmax], [h for h in hits if qmin >= h['totQ'] or h['totQ'] >  qmax]


@app.callback(
    Output('memory-path', 'data'),
    Output('filepath-display', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def save_uploaded_file(contents, filename):
    if contents is None:
        return dash.no_update, ""
    content_type, content_string = contents.split(',')
    import base64
    import io
    decoded = base64.b64decode(content_string)
    filepath = os.path.join('uploads', filename)
    os.makedirs('uploads', exist_ok=True)
    with open(filepath, 'wb') as f:
        f.write(decoded)
    return filepath, f"Loaded file: {filepath}"


@app.callback(
    Output('yz-plot', 'figure'),
    Output('xz-plot', 'figure'),
    Input('update-button', 'n_clicks'),
    State('qmin', 'value'),
    State('qmax', 'value'),
    State('memory-path', 'data'),
    Input('hit-color-style', 'value')
)
def update_plot(n_clicks, qmin, qmax, filepath, hit_color_style):
    if not filepath or not os.path.exists(filepath):
        return go.Figure(), go.Figure()

    selected, deselected = load_hits_from_hdf5(filepath)
    sel_red = reduce_hits(selected)
    desel_red = reduce_hits(deselected)
    marked, unmarked = filter_marked(sel_red, qmin, qmax)

    all_q = [d['totQ'] for d in sel_red + desel_red ]
    qmin_val, qmax_val = min(all_q), max(all_q)

    def has_data(data):
        return len(data) > 0
    colorbar_label = next((name for name, data in zip(['selected-marked', 'selected-unmarked', 'deselected'], [marked, unmarked, deselected]) if has_data(data)), None)
    colors = {'deselected': 'grey', 'selected-marked' : 'red', 'selected-unmarked' : 'blue'}

    def make_trace(name, data):
        if 'totQ' in hit_color_style:
            marker = dict(
                size=[5+2*d['totN'] for d in data],
                color=[d['totQ'] for d in data],
                colorscale='Viridis',
                cmin=qmin_val,
                cmax=qmax_val,
                showscale = True,
                colorbar = dict(title='totQ')
            )
        else:
            marker = dict(
                size=[5+2*d['totN'] for d in data],
                color = colors[name],
            )
        return go.Scatter(
            x=[d['y'] for d in data],
            y=[d['z'] for d in data],
            mode='markers',
            marker = marker,
            text=[f"totQ: {d['totQ']}<br>totN: {d['totN']}" for d in data],
            hovertemplate="%{text}<br>Y: %{x}<br>Z: %{y}<extra></extra>",
            name=name
        )

    fig = go.Figure([
        # make_trace("selected", sel_red),
        make_trace("deselected", desel_red),
        make_trace("selected-unmarked", unmarked),
        make_trace("selected-marked", marked),
    ])

    all_x = [d['x'] for d in sel_red + desel_red ]
    all_y = [d['y'] for d in sel_red + desel_red ]
    all_z = [d['z'] for d in sel_red + desel_red ]

    x_min, x_max = -3+min(all_x), 3+max(all_x)
    y_min, y_max = -3+min(all_y), 3+max(all_y)
    z_min, z_max = -3+min(all_z), 3+max(all_z)

    fig.update_layout(
        title="YZ Projection: size ~ totN, color ~ totQ",
        xaxis=dict(title="Y", range=[y_min, y_max]),
        yaxis=dict(title="Z", range=[z_min, z_max]),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        legend_title="Hit Types",
        height=800,
        width=800
    )

    def make_trace_xz(name, data):
        if 'totQ' in hit_color_style:
            marker = dict(
                size=[5+2*d['totN'] for d in data],
                color=[d['totQ'] for d in data],
                colorscale='Viridis',
                cmin=qmin_val,
                cmax=qmax_val,
                showscale = True,
                colorbar = dict(title='totQ')
            )
        else:
            marker = dict(
                size=[5+2*d['totN'] for d in data],
                color = colors[name],
            )
        return go.Scatter(
            x=[d['x'] for d in data],
            y=[d['z'] for d in data],
            mode='markers',
            marker = marker,
            text=[f"totQ: {d['totQ']}<br>totN: {d['totN']}" for d in data],
            hovertemplate="%{text}<br>X: %{x}<br>Z: %{y}<extra></extra>",
            name=name
        )

    figxz = go.Figure([
        make_trace_xz("deselected", desel_red),
        make_trace_xz("selected-unmarked", unmarked),
        make_trace_xz("selected-marked", marked),
    ])

    figxz.update_layout(
        title="XZ Projection: size ~ totN, color ~ totQ",
        xaxis=dict(title="X", range=[x_min, x_max]),
        yaxis=dict(title="Z", range=[z_min, z_max]),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        legend_title="Hit Types",
        height=800,
        width=800
    )
    return fig, figxz


if __name__ == '__main__':
    app.run(debug=True, port=8052)


