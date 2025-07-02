import streamlit as st
import uproot
import numpy as np
import plotly.express as px

st.set_page_config(layout="wide")
st.title("ROOT TTree Histogram Explorer")

# --- Sidebar inputs ---
with st.sidebar:
    filename = st.text_input("ROOT File", value="many_muons.root")
    treename = st.text_input("TTree Name", value="mu_ndlar/hits")
    load_btn = st.button("Load Tree")

# --- Load data ---
if load_btn:
    try:
        tree = uproot.open(filename)[treename]
        arrays = tree.arrays(["totN", "totQ", "tindex", "Q", "dx"], library="np")

        st.session_state.arrays = arrays
        st.success("Data loaded successfully.")
    except Exception as e:
        st.error(f"Failed to load: {e}")

if "arrays" in st.session_state:
    arrays = st.session_state.arrays
    totN = arrays["totN"]
    totQ = arrays["totQ"]
    tindex = arrays["tindex"]

    # Create masks
    mask_t0       = tindex == 0
    mask_t0_n1    = (tindex == 0) & (totN == 1)
    mask_t0_n2    = (tindex == 0) & (totN == 2)

    st.markdown("### Histogram Settings")

    # Binning controls
    with st.expander("Configure Binning & Ranges", expanded=True):
        st.markdown("#### totN (tindex == 0)")
        c1, c2, c3 = st.columns(3)
        with c1:
            bins_totN = st.number_input("Bins", min_value=0, max_value=200, value=30, key="bins_totN")
        with c2:
            range_totN_min = st.number_input("Range min", value=int(np.min(totN[mask_t0])), key="range_min_totN")
        with c3:
            range_totN_max = st.number_input("Range max", value=int(np.max(totN[mask_t0])), key="range_max_totN")

        st.markdown("#### totQ (all 3 totQ plots)")
        c4, c5, c6 = st.columns(3)
        with c4:
            bins_totQ = st.number_input("Bins", min_value=0, max_value=200, value=30, key="bins_totQ")
        with c5:
            # range_totQ_min = st.number_input("Range min", value=float(np.min(totQ[mask_t0])), key="range_min_totQ")
            range_totQ_min = st.number_input("Range min", value=float(0), key="range_min_totQ")
        with c6:
            # range_totQ_max = st.number_input("Range max", value=float(np.max(totQ[mask_t0])), key="range_max_totQ")
            range_totQ_max = st.number_input("Range max", value=float(50), key="range_max_totQ")

    # --- Plot 1: totN with tindex==0 ---
    col1, col2 = st.columns(2)
    with col1:
        data = totN[mask_t0]
        filtered = data[(data >= range_totN_min) & (data <= range_totN_max)]
        fig1 = px.histogram(filtered, nbins=bins_totN,
                            range_x=[range_totN_min, range_totN_max],
                            title="totN (tindex == 0)")

        st.plotly_chart(fig1, use_container_width=True)

    # --- Plot 2: totQ with tindex==0 ---
    with col2:
        data = totQ[mask_t0]
        filtered = data[(data >= range_totQ_min) & (data <= range_totQ_max)]
        fig2 = px.histogram(
            filtered, nbins=bins_totQ,
            range_x=[range_totQ_min, range_totQ_max],
            title="totQ (tindex == 0)"
        )
        st.plotly_chart(fig2, use_container_width=True)

    # --- Plot 3: totQ with tindex==0 & totN==1 ---
    col3, col4 = st.columns(2)
    with col3:
        data = totQ[mask_t0_n1]
        filtered = data[(data >= range_totQ_min) & (data <= range_totQ_max)]
        fig3 = px.histogram(
            filtered, nbins=bins_totQ,
            range_x=[range_totQ_min, range_totQ_max],
            title="totQ (tindex == 0 & totN == 1)"
        )
        st.plotly_chart(fig3, use_container_width=True)

    # --- Plot 4: totQ with tindex==0 & totN==2 ---
    with col4:
        data = totQ[mask_t0_n2]
        filtered = data[(data >= range_totQ_min) & (data <= range_totQ_max)]
        fig4 = px.histogram(
            filtered, nbins=bins_totQ,
            range_x=[range_totQ_min, range_totQ_max],
            title="totQ (tindex == 0 & totN == 2)"
        )
        st.plotly_chart(fig4, use_container_width=True)


    col5, col6 = st.columns(2)
    with col5:
        st.markdown("### 2D Histogram: totQ - Q vs Q (totN==2 & tindex==0)")
        with st.expander("2D Histogram Settings (totQ - Q vs Q)", expanded=True):

            colx, coly = st.columns(2)

            with colx:
                nbinsx = st.number_input("Number of bins (X axis)", min_value=1, max_value=500, value=60)
                # x_min = st.number_input("X min (Q)", value=float(np.min(x_2d)))
                x_min = st.number_input("X min (Q)", value=float(0))
                # x_max = st.number_input("X max (Q)", value=float(np.max(x_2d)))
                x_max = st.number_input("X max (Q)", value=float(20))

            with coly:
                nbinsy = st.number_input("Number of bins (Y axis)", min_value=1, max_value=500, value=60)
                # y_min = st.number_input("Y min (totQ - Q)", value=float(np.min(y_2d)))
                y_min = st.number_input("Y min (totQ - Q)", value=float(0))
                # y_max = st.number_input("Y max (totQ - Q)", value=float(np.max(y_2d)))
                y_max = st.number_input("Y max (totQ - Q)", value=float(20))

        mask_2d = (totN == 2) & (tindex == 0)
        x_2d = arrays["Q"][mask_2d]
        y_2d = arrays["totQ"][mask_2d] - arrays["Q"][mask_2d]

        # Apply range filters
        mask_range = (x_2d >= x_min) & (x_2d <= x_max) & (y_2d >= y_min) & (y_2d <= y_max)
        x_filtered = x_2d[mask_range]
        y_filtered = y_2d[mask_range]

        # Create 2D heatmap
        fig5 = px.density_heatmap(
            x=x_filtered,
            y=y_filtered,
            nbinsx=nbinsx,
            nbinsy=nbinsy,
            color_continuous_scale="Viridis",
            labels={"x": "Q", "y": "totQ - Q"},
            title="2D Histogram: totQ - Q vs Q"
        )
        fig5.update_layout(height=500)
        st.plotly_chart(fig5, use_container_width=True)
