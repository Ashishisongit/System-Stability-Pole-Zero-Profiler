import numpy as np
import plotly.graph_objects as go
from scipy import signal
import streamlit as st

st.set_page_config(
    page_title="Pole-Zero Profiler", page_icon="📈", layout="wide"
)

st.title("System Stability & Pole-Zero Profiler")

# Sidebar inputs
st.sidebar.header("Transfer Function Coefficients")
num_str = st.sidebar.text_input("Numerator Coefficients N(s)", "1")
den_str = st.sidebar.text_input("Denominator Coefficients D(s)", "1, 3, 2")

try:
    num = [float(x.strip()) for x in num_str.split(",") if x.strip()]
    den = [float(x.strip()) for x in den_str.split(",") if x.strip()]

    if not num or not den:
        st.warning("Please enter valid comma-separated numeric coefficients.")
    else:
        # Core Calculations
        zeros = np.roots(num)
        poles = np.roots(den)

        # Stability Logic
        if any(p.real > 0 for p in poles):
            st.error("🚨 System Status: UNSTABLE (Poles in RHP)")
        elif any(np.isclose(p.real, 0) for p in poles):
            st.warning(
                "⚠️ System Status: MARGINALLY STABLE (Poles on Imaginary Axis)"
            )
        else:
            st.success("✅ System Status: STABLE (All Poles in LHP)")

        col1, col2 = st.columns(2)

        # s-Plane Plot
        with col1:
            fig_splane = go.Figure()
            fig_splane.add_trace(
                go.Scatter(
                    x=poles.real,
                    y=poles.imag,
                    mode="markers",
                    marker=dict(symbol="x", size=14, color="red"),
                    name="Poles",
                )
            )
            fig_splane.add_trace(
                go.Scatter(
                    x=zeros.real,
                    y=zeros.imag,
                    mode="markers",
                    marker=dict(symbol="circle-open", size=14, color="blue"),
                    name="Zeros",
                )
            )
            fig_splane.add_vline(x=0, line_dash="dash", line_color="gray")
            fig_splane.update_layout(
                title="s-Plane Pole-Zero Plot",
                xaxis_title="Real (σ)",
                yaxis_title="Imaginary (jω)",
            )
            st.plotly_chart(fig_splane, use_container_width=True)

        # Step Response
        with col2:
            sys = signal.TransferFunction(num, den)
            t, y = signal.step(sys)
            fig_step = go.Figure()
            fig_step.add_trace(
                go.Scatter(x=t, y=y, mode="lines", name="Response")
            )
            fig_step.update_layout(
                title="Step Response y(t)",
                xaxis_title="Time (t)",
                yaxis_title="y(t)",
            )
            st.plotly_chart(fig_step, use_container_width=True)

except Exception as e:
    st.error(f"Error parsing inputs or calculating roots: {e}")
