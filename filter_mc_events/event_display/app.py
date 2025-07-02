################################################################################
# app.py ? simple ROOT event viewer (MVC, interactive)
################################################################################
import numpy as np
import pandas as pd
import uproot
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, callback_context


# ????????????????????????? Model ?????????????????????????
class DataModel:
    def __init__(self):
        self.df = None
        self.event_ids = []

    def load(self, path, tree):
        self.df = uproot.open(path)[tree].arrays(library="pd")
        self.event_ids = sorted(self.df["event_id"].unique())

    def slice(self, event_id, filt):
        if self.df is None or event_id is None:
            return pd.DataFrame()
        df = self.df.query("event_id == @event_id").copy()
        if filt.strip():
            try:
                df = df.query(filt, engine="python")
            except Exception as e:
                print("bad filter:", e)
        return df

model = DataModel()

# ????????????????????????? View ??????????????????????????
app = Dash(__name__, title="ROOT event viewer")
app.layout = html.Div([
    html.H3("ROOT Event Viewer ? 3×2-D + 3-D"),

    html.Div([
        dcc.Input(id="file-path",  placeholder="path/to/file.root", style={"width":"30%"}),
        dcc.Input(id="tree-name",  placeholder="TTree name",        style={"width":"20%"}),
        html.Button("Load", id="load-btn"),
        html.Span(id="load-msg", style={"marginLeft":"8px"})
    ]),

    html.Br(),
    html.Div([
        html.Label("event_id"),
        dcc.Dropdown(id="event-dd", style={"width":"8em"}),
        html.Label("filter"),
        dcc.Input(id="filter-str", debounce=True, style={"width":"40%"}),
        html.Label("mode"),
        dcc.RadioItems(
            id="mode",
            options=[
                {"label":"size=thres | col=Q",     "value":"szT_colQ"},
                {"label":"size=fixed | col=Q",     "value":"szF_colQ"},
                {"label":"size=fixed | col=thres", "value":"szF_colT"},
            ],
            value="szT_colQ",
            inline=True)
    ], style={"display":"flex","gap":"10px","alignItems":"center"}),

    html.Div([
        html.Div("x-range"), dcc.RangeSlider(id="x-range", allowCross=False),
        html.Div("y-range"), dcc.RangeSlider(id="y-range", allowCross=False),
        html.Div("z-range"), dcc.RangeSlider(id="z-range", allowCross=False),
    ], style={"width":"94%"}),

    html.Br(),
    html.Div([
        dcc.Graph(id="xy"),
        dcc.Graph(id="xz"),
        dcc.Graph(id="yz"),
        dcc.Graph(id="xyz"),
    ], style={"display":"grid","gridTemplateColumns":"repeat(2,1fr)","gap":"8px"})
])

# ??????????????????? Controller (1) load file ???????????????????
@app.callback(
    Output("load-msg","children"),
    Output("event-dd","options"),
    Output("event-dd","value"),
    Output("x-range","min"), Output("x-range","max"), Output("x-range","value"),
    Output("y-range","min"), Output("y-range","max"), Output("y-range","value"),
    Output("z-range","min"), Output("z-range","max"), Output("z-range","value"),
    Input("load-btn","n_clicks"),
    State("file-path","value"),
    State("tree-name","value"),
    prevent_initial_call=True
)
def load_file(_, path, tree):
    if not (path and tree):
        return "? enter path & tree", [], None, *([0,0,[0,0]]*3)
    try:
        model.load(path, tree)
    except Exception as e:
        return f"? {e}", [], None, *([0,0,[0,0]]*3)

    df = model.df
    ranges = (
        df.x.min(), df.x.max(), [df.x.min(), df.x.max()],
        df.y.min(), df.y.max(), [df.y.min(), df.y.max()],
        df.z.min(), df.z.max(), [df.z.min(), df.z.max()],
    )
    ranges = (
        -70, 70, [-70, 70],
        -80, 80, [-80, 80],
        -70, 70, [-70, 70],
    )
    opts = [{"label":eid, "value":eid} for eid in model.event_ids]
    return f"? {len(df)} rows", opts, model.event_ids[0], *ranges

# ???????????????? Controller (2) draw / update plots ?????????????
@app.callback(
    Output("xy","figure"),  Output("xz","figure"),
    Output("yz","figure"),  Output("xyz","figure"),
    Input("event-dd","value"),
    Input("filter-str","value"),
    Input("mode","value"),
    Input("x-range","value"),
    Input("y-range","value"),
    Input("z-range","value"),
    allow_duplicate=True  # Dash 2.15+; keeps callback separate
)
def update_plots(eid, filt, mode, xr, yr, zr):
    df = model.slice(eid, filt or "")
    if df.empty:
        return [go.Figure()] * 4

    # ---------- size & colour (robust) ----------
    if mode == "szT_colQ":
        size_arr  = (10 * (df.thres / df.thres.max()) + 4).to_numpy()
        color_arr = df.Q.to_numpy()
    elif mode == "szF_colQ":
        size_arr  = np.full(len(df), 6)        # constant size
        color_arr = df.Q.to_numpy()
    else:                                      # szF_colT
        size_arr  = np.full(len(df), 6)
        color_arr = df.thres.to_numpy()

    mk = dict(size=size_arr,
          color=color_arr,
          colorscale="Viridis",
          showscale=True)

    fig_xy = go.Figure(go.Scattergl(x=df.x, y=df.y, mode="markers", marker=mk))
    fig_xz = go.Figure(go.Scattergl(x=df.x, y=df.z, mode="markers", marker=mk))
    fig_yz = go.Figure(go.Scattergl(x=df.y, y=df.z, mode="markers", marker=mk))
    fig_3d = go.Figure(go.Scatter3d(x=df.x, y=df.y, z=df.z, mode="markers", marker=mk))

    # respect sliders
    if xr and yr and zr:
        for f in (fig_xy, fig_xz):   f.update_xaxes(range=xr)
        fig_xy.update_yaxes(range=yr)
        fig_xz.update_yaxes(range=zr)
        fig_yz.update_xaxes(range=yr).update_yaxes(range=zr)
        fig_3d.update_layout(scene=dict(xaxis_range=xr, yaxis_range=yr, zaxis_range=zr))
        fig_xy.update_layout(xaxis_title="x", yaxis_title="y")
        fig_xz.update_layout(xaxis_title="x", yaxis_title="z")
        fig_yz.update_layout(xaxis_title="y", yaxis_title="z")
        fig_3d.update_layout(scene=dict(xaxis_title="x",
                                yaxis_title="y",
                                zaxis_title="z"))


    return fig_xy, fig_xz, fig_yz, fig_3d

# ??????????????????????????? run ??????????????????????????
if __name__ == "__main__":
    app.run(debug=True)

