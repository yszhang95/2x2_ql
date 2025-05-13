import streamlit as st
import uproot
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("ROOT TTree Viewer (totN Categories)")

# --- Sidebar inputs ---
with st.sidebar:
    filename = st.text_input("ROOT File", value="many_muons.root")
    treename = st.text_input("TTree Name", value="mu_ndlar/hits")

    load_btn = st.button("Load Data")

# Session state to cache data
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

if load_btn:
    try:
        file = uproot.open(filename)
        tree = file[treename]
        arrays = tree.arrays(["event_id", "x", "y", "z", "totN", "Q", "totQ", "tindex", "io_group"], library="np")

        st.session_state.arrays = arrays
        st.session_state.unique_event_ids = np.unique(arrays["event_id"])
        st.session_state.unique_tpc_ids = np.unique(arrays["io_group"])
        st.session_state.data_loaded = True
        st.success("Data loaded successfully.")
    except Exception as e:
        st.session_state.data_loaded = False
        st.error(f"Failed to load ROOT data: {e}")


def create_scatter(xdata, ydata, Q, masks, xlabel, ylabel, Q_min, Q_max,
                   show_colorbar=False):
    fig = go.Figure()
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    for label, (mask, size) in masks.items():
        fig.add_trace(go.Scattergl(
            x=xdata[mask],
            y=ydata[mask],
            mode="markers",
            marker=dict(
                size=size,
                color=Q[mask],
                colorscale="Viridis",
                cmin=Q_min, cmax=Q_max,
                colorbar=dict(title="Charge (Q)"),
                showscale=show_colorbar
            ),
            name=label
        ))
    fig.update_layout(
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        legend_title="totN Category",
        height=500,
        legend=dict(
            orientation="h",      # Horizontal legend
            yanchor="bottom",
            y=-0.3,               # Below plot area
            xanchor="center",
            x=0.5
        )
    )
    return fig

# Show interface if data is loaded
if st.session_state.data_loaded:
    arrays = st.session_state.arrays
    event_ids = st.session_state.unique_event_ids
    tpc_ids= st.session_state.unique_tpc_ids

    selected_event = st.selectbox("Select Event ID", event_ids.astype(str))
    selected_tpc = st.selectbox("Select TPC", tpc_ids.astype(str))
    event_id = int(selected_event)
    tpc_id = int(selected_tpc)
    drift = 1 if tpc_id % 2 else -1

    tpcs = np.unique(arrays["io_group"][arrays["event_id"] == event_id])
    st.code(f"Available tpcs for this event: {tpcs}", language="text")
    st.code(f"Drift direction in this TPC: {drift}", language="text")

    filtered = (arrays["event_id"] == event_id) & (arrays["io_group"] == tpc_id)
    x, y, z, totN = arrays["x"][filtered], arrays["y"][filtered], arrays["z"][filtered], arrays["totN"][filtered]
    Q = arrays["Q"][filtered]


    # Marker controls
    st.markdown("### Marker Style")
    col1, col2, col3 = st.columns(3)
    with col1:
        # color1 = st.color_picker("Color for totN=1", "#1f77b4")
        size1 = st.slider("Size for totN=1", 1, 100, 20)
    with col2:
        # color2 = st.color_picker("Color for totN=2", "#d62728")
        size2 = st.slider("Size for totN=2", 1, 100, 5)
    with col3:
        # color3 = st.color_picker("Color for totN>2", "#2ca02c")
        size3 = st.slider("Size for totN>2", 1, 100, 1)

    # Categories
    masks = {
        "totN=1": (totN == 1, size1),
        "totN=2": (totN == 2, size2),
        "totN>2": (totN > 2, size3),
    }

    if len(Q):
        Q_min, Q_max = np.min(Q), np.max(Q)
    else:
        Q_min, Q_max = 0, 100
    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(create_scatter(x, y, Q, masks, "X", "Y", Q_min, Q_max), use_container_width=True, show_colorbar=True)
    with col2:
        st.plotly_chart(create_scatter(z, y, Q, masks, "Z", "Y", Q_min, Q_max), use_container_width=True)
    with col3:
        st.plotly_chart(create_scatter(x, z, Q, masks, "X", "Z", Q_min, Q_max), use_container_width=True)
