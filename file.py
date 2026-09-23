import streamlit as st
import numpy as np
import scipy.signal as signal
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

st.set_page_config(
    page_title="System Stability & Pole-Zero Profiler",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .badge-stable {
        background-color: #DCFCE7;
        color: #166534;
        padding: 0.4rem 1rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.05rem;
        display: inline-block;
        border: 1px solid #86EFAC;
    }
    .badge-unstable {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.4rem 1rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.05rem;
        display: inline-block;
        border: 1px solid #FCA5A5;
    }
    .badge-marginal {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 0.4rem 1rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.05rem;
        display: inline-block;
        border: 1px solid #FCD34D;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 0.75rem;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

SYSTEM_PRESETS = {
    "Cloud Server Auto-Scaling (Stable Control)": {
        "num": [2.5, 1.0],
        "den": [1.0, 3.5, 4.0, 1.5],
        "description": "Auto-scaling loop with moderate proportional-integral (PI) gain and cooling delay. Poles reside strictly in the Left-Half Plane (LHP), ensuring smooth convergence to target server CPU utilization.",
        "category": "IT / Cloud Computing"
    },
    "Server Load Thrashing (Unstable Lag)": {
        "num": [4.0],
        "den": [1.0, -0.6, 5.0],
        "description": "High latency in container provisioning causes feedback lag. The system over-provisions and de-provisions wildly; poles enter the Right-Half Plane (RHP), creating explosive oscillations.",
        "category": "IT / Cloud Computing"
    },
    "Audio Amplifier / 2nd-Order RLC Filter": {
        "num": [10.0],
        "den": [1.0, 1.414, 10.0],
        "description": "Under-damped low-pass acoustic filtering stage (Butterworth / RLC circuit). Exhibits oscillatory transient ringing before settling into steady state.",
        "category": "Audio / Hardware"
    },
    "Pure Harmonic Oscillator (Marginally Stable)": {
        "num": [9.0],
        "den": [1.0, 0.0, 9.0],
        "description": "Zero damping ($D(s) = s^2 + 9$). Non-repeated complex poles lie directly on the imaginary axis ($s = \pm j3$). Sustains permanent sinusoidal oscillation without decay or blowup.",
        "category": "Control Theory"
    },
    "Custom Polynomial Input": {
        "num": [1.0],
        "den": [1.0, 2.0, 2.0],
        "description": "Specify custom numerator and denominator polynomial coefficients to analyze any arbitrary s-domain transfer function.",
        "category": "Custom"
    }
}

st.markdown('<div class="main-title">⚡ System Stability & Pole-Zero Profiler</div>', unsafe_allow_html=True)
# st.markdown('<div class="sub-title">Vidyavardhini\'s College of Eng. & Tech. | Applied Mathematics Thinking-I (2343111) — Group 2</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ System Configuration")
    preset_choice = st.selectbox(
        "Choose System Architecture / Preset:",
        list(SYSTEM_PRESETS.keys())
    )
    preset_info = SYSTEM_PRESETS[preset_choice]
    st.caption(f"**Domain:** {preset_info['category']}")
    st.info(preset_info["description"])

    if preset_choice == "Custom Polynomial Input":
        st.subheader("Polynomial Coefficients (Descending Powers of s)")
        num_str = st.text_input("Numerator N(s) [e.g. 1.0 or 2, 1]:", value="1.0")
        den_str = st.text_input("Denominator D(s) [e.g. 1, 2, 2]:", value="1.0, 2.0, 2.0")
        try:
            num_coeffs = [float(x.strip()) for x in num_str.replace(" ", ",").split(",") if x.strip()]
            den_coeffs = [float(x.strip()) for x in den_str.replace(" ", ",").split(",") if x.strip()]
        except ValueError:
            st.error("Please enter valid comma or space separated numbers.")
            num_coeffs, den_coeffs = [1.0], [1.0, 2.0, 2.0]
    else:
        num_coeffs = preset_info["num"]
        den_coeffs = preset_info["den"]

    st.divider()
    st.subheader("⏱️ Simulation Settings")
    t_max = st.slider("Simulation Time Horizon (seconds):", min_value=2.0, max_value=50.0, value=15.0, step=1.0)
    damping_grid = st.checkbox("Show Damping & Natural Frequency Grid", value=True)

try:
    # Remove leading zeros to avoid rank degeneracy
    num_clean = np.trim_zeros(np.array(num_coeffs, dtype=float), 'f')
    den_clean = np.trim_zeros(np.array(den_coeffs, dtype=float), 'f')
    if len(num_clean) == 0:
        num_clean = np.array([0.0])
    if len(den_clean) == 0:
        den_clean = np.array([1.0])

    # Construct SciPy continuous-time LTI Transfer Function
    sys_tf = signal.TransferFunction(num_clean, den_clean)
    poles = sys_tf.poles
    zeros = sys_tf.zeros

except Exception as err:
    st.error(f"Error computing transfer function: {err}")
    st.stop()

def evaluate_stability(poles_list, tolerance=1e-5):
    """
    Evaluates Bounded-Input Bounded-Output (BIBO) stability:
    1. Unstable: Any pole has Re(p) > tolerance OR repeated poles on Re(p) == 0.
    2. Marginally Stable: Non-repeated poles on Re(p) == 0, all others Re(p) < -tolerance.
    3. Stable: All poles have Re(p) < -tolerance.
    """
    if len(poles_list) == 0:
        return "STABLE", "badge-stable", "No dynamic poles found; pure algebraic gain."

    real_parts = np.real(poles_list)
    imag_parts = np.imag(poles_list)

    # Check for poles strictly in the Right-Half Plane (RHP)
    if np.any(real_parts > tolerance):
        rhp_count = np.sum(real_parts > tolerance)
        return (
            "UNSTABLE",
            "badge-unstable",
            f"Found {rhp_count} pole(s) in the Right-Half Plane (Re(s) > 0). System produces unbounded exponential blowup."
        )

    # Check for poles on the imaginary axis (Re(s) approx 0)
    on_j_axis = np.abs(real_parts) <= tolerance
    if np.any(on_j_axis):
        j_poles = poles_list[on_j_axis]
        # Check for repeated roots on imaginary axis
        has_multiplicity = False
        for i in range(len(j_poles)):
            for j in range(i + 1, len(j_poles)):
                if np.isclose(j_poles[i], j_poles[j], atol=1e-4):
                    has_multiplicity = True
                    break
        if has_multiplicity:
            return (
                "UNSTABLE",
                "badge-unstable",
                "System contains repeated poles on the imaginary axis (jω). Resonance causes linear/polynomial blowup."
            )
        else:
            return (
                "MARGINALLY STABLE",
                "badge-marginal",
                "Non-repeated poles lie on the imaginary axis (Re(s) = 0). System sustains continuous sinusoidal oscillations."
            )

    return (
        "STABLE",
        "badge-stable",
        "All poles strictly reside in the Open Left-Half Plane (Re(s) < 0). System exhibits asymptotic BIBO stability."
    )

status, badge_class, explanation = evaluate_stability(poles)

col_stat1, col_stat2 = st.columns([1, 2])

with col_stat1:
    st.markdown(f'<span class="{badge_class}">STATUS: {status}</span>', unsafe_allow_html=True)
    st.markdown(f"<p style='margin-top: 0.5rem; color: #475569;'>{explanation}</p>", unsafe_allow_html=True)

with col_stat2:
    # Format polynomials for display in LaTeX
    def poly_to_latex(coeffs):
        degree = len(coeffs) - 1
        terms = []
        for i, c in enumerate(coeffs):
            p = degree - i
            c_rounded = round(float(c), 3)
            if math.isclose(c_rounded, 0.0) and degree > 0:
                continue
            if p == 0:
                terms.append(f"{c_rounded}")
            elif p == 1:
                terms.append(f"{c_rounded}s" if c_rounded != 1 else "s")
            else:
                terms.append(f"{c_rounded}s^{{{p}}}" if c_rounded != 1 else f"s^{{{p}}}")
        return " + ".join(terms).replace("+ -", "- ") if terms else "0"

    num_latex = poly_to_latex(num_clean)
    den_latex = poly_to_latex(den_clean)
    st.latex(rf"H(s) = \frac{{N(s)}}{{D(s)}} = \frac{{{num_latex}}}{{{den_latex}}}")

st.divider()
metric_c1, metric_c2, metric_c3, metric_c4 = st.columns(4)

with metric_c1:
    st.metric(label="Total Number of Poles", value=len(poles))
with metric_c2:
    st.metric(label="Total Number of Zeros", value=len(zeros))
with metric_c3:
    max_real = np.max(np.real(poles)) if len(poles) > 0 else 0.0
    st.metric(label="Max Real Part (σ_max)", value=f"{max_real:.4f}", delta="LHP" if max_real < 0 else "RHP/Axis", delta_color="normal" if max_real < 0 else "inverse")
with metric_c4:
    # Calculate dominant damping ratio if complex conjugate poles exist
    dominant_damping = "N/A"
    if len(poles) > 0:
        # find pole closest to imaginary axis
        dom_pole = poles[np.argmax(np.real(poles))]
        wn = np.abs(dom_pole)
        if wn > 1e-6:
            zeta = -np.real(dom_pole) / wn
            dominant_damping = f"{zeta:.3f}"
    st.metric(label="Dominant Damping (ζ)", value=dominant_damping)

tab1, tab2, tab3 = st.tabs(["📍 Complex s-Plane (Poles & Zeros)", "📈 Time-Domain Transient Response", "📖 Mathematical Foundations & IT Modeling"])

with tab1:
    st.subheader("Complex Frequency Domain: s-Plane Mapping (s = σ + jω)")
    
    # Calculate plot limits dynamically
    all_points = list(poles) + list(zeros) + [0.0]
    real_vals = [np.real(p) for p in all_points]
    imag_vals = [np.imag(p) for p in all_points]
    
    r_max = max(max(map(abs, real_vals)), max(map(abs, imag_vals)), 3.0) * 1.35
    
    fig_splane = go.Figure()

    # Shade the Left Half-Plane (Stable zone)
    fig_splane.add_shape(
        type="rect",
        x0=-r_max, y0=-r_max, x1=0, y1=r_max,
        fillcolor="rgba(34, 197, 94, 0.12)",
        line=dict(width=0),
        layer="below"
    )
    # Shade the Right Half-Plane (Unstable zone)
    fig_splane.add_shape(
        type="rect",
        x0=0, y0=-r_max, x1=r_max, y1=r_max,
        fillcolor="rgba(239, 68, 68, 0.12)",
        line=dict(width=0),
        layer="below"
    )

    # Optional: Natural frequency radial arcs and damping lines
    if damping_grid:
        for wn_val in np.linspace(r_max / 4, r_max * 0.85, 4):
            theta = np.linspace(0, 2 * np.pi, 100)
            fig_splane.add_trace(go.Scatter(
                x=wn_val * np.cos(theta),
                y=wn_val * np.sin(theta),
                mode='lines',
                line=dict(color='rgba(148, 163, 184, 0.3)', dash='dot', width=1),
                showlegend=False,
                hoverinfo='skip'
            ))

    # Real and Imaginary Reference Axes
    fig_splane.add_hline(y=0, line=dict(color="#475569", width=1.5))
    fig_splane.add_vline(x=0, line=dict(color="#DC2626", width=2, dash="dash"), annotation_text="Imaginary Axis (jω)", annotation_position="top left")

    # Plot Poles as 'X'
    if len(poles) > 0:
        p_re = [np.real(p) for p in poles]
        p_im = [np.imag(p) for p in poles]
        p_labels = [f"Pole: {p.real:+.3f} {p.imag:+.3f}j" for p in poles]
        p_colors = ["#EF4444" if p.real > 1e-4 else "#2563EB" for p in poles]
        
        fig_splane.add_trace(go.Scatter(
            x=p_re,
            y=p_im,
            mode='markers+text',
            name='Poles (x)',
            marker=dict(symbol='x', size=14, color=p_colors, line=dict(width=3, color=p_colors)),
            text=[f"P{i+1}" for i in range(len(poles))],
            textposition="top center",
            hovertext=p_labels,
            hoverinfo='text'
        ))

    # Plot Zeros as 'O'
    if len(zeros) > 0:
        z_re = [np.real(z) for z in zeros]
        z_im = [np.imag(z) for z in zeros]
        z_labels = [f"Zero: {z.real:+.3f} {z.imag:+.3f}j" for z in zeros]
        
        fig_splane.add_trace(go.Scatter(
            x=z_re,
            y=z_im,
            mode='markers+text',
            name='Zeros (o)',
            marker=dict(symbol='circle-open', size=13, color='#059669', line=dict(width=2.5)),
            text=[f"Z{i+1}" for i in range(len(zeros))],
            textposition="top center",
            hovertext=z_labels,
            hoverinfo='text'
        ))

    fig_splane.update_layout(
        title="Complex Frequency Plane Mapping (s = σ + jω)",
        xaxis_title="Real Part (σ - Damping / Decay Rate) [sec⁻¹]",
        yaxis_title="Imaginary Part (jω - Oscillation Frequency) [rad/sec]",
        xaxis=dict(range=[-r_max, r_max], zeroline=False, gridcolor='#E2E8F0'),
        yaxis=dict(range=[-r_max, r_max], zeroline=False, gridcolor='#E2E8F0', scaleanchor="x", scaleratio=1),
        width=750,
        height=580,
        plot_bgcolor='white',
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02, bgcolor="rgba(255,255,255,0.85)")
    )
    st.plotly_chart(fig_splane, use_container_width=True)

    # Coordinates readout table
    col_p, col_z = st.columns(2)
    with col_p:
        st.markdown("**Computed Poles & Natural Modes:**")
        if len(poles) > 0:
            pole_table = [{"Index": f"p{i+1}", "Real (σ)": f"{p.real:.4f}", "Imag (ω)": f"{p.imag:.4f}", "Damping Type": "Stable Decay" if p.real < 0 else ("Explosive Growth" if p.real > 0 else "Pure Oscillation")} for i, p in enumerate(poles)]
            st.dataframe(pole_table, use_container_width=True)
        else:
            st.info("No poles found.")
            
    with col_z:
        st.markdown("**Computed Zeros:**")
        if len(zeros) > 0:
            zero_table = [{"Index": f"z{i+1}", "Real (σ)": f"{z.real:.4f}", "Imag (ω)": f"{z.imag:.4f}"} for i, z in enumerate(zeros)]
            st.dataframe(zero_table, use_container_width=True)
        else:
            st.info("System has no finite zeros (All-pole system).")

with tab2:
    st.subheader("Time-Domain Verification: y(t) = L⁻¹{H(s) · X(s)}")
    
    t_span = np.linspace(0, t_max, 1000)
    
    try:
        # Step response simulation
        t_step, y_step = signal.step(sys_tf, T=t_span)
        # Impulse response simulation
        t_imp, y_imp = signal.impulse(sys_tf, T=t_span)

        # Detect numerical overflows if unstable
        if np.any(np.isnan(y_step)) or np.any(np.abs(y_step) > 1e6):
            clipped_step = np.clip(y_step, -1e5, 1e5)
            is_overflow = True
        else:
            clipped_step = y_step
            is_overflow = False

        fig_time = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=("Unit Step Response: y_step(t) [System input = u(t)]", "Impulse Response: h(t) [System input = δ(t)]"),
            vertical_spacing=0.12
        )

        fig_time.add_trace(
            go.Scatter(x=t_step, y=clipped_step, mode='lines', name='Step Response', line=dict(color='#2563EB', width=2.5)),
            row=1, col=1
        )
        # Reference unit line
        fig_time.add_hline(y=1.0, line=dict(color='#64748B', dash='dot', width=1.5), row=1, col=1)

        fig_time.add_trace(
            go.Scatter(x=t_imp, y=np.clip(y_imp, -1e5, 1e5), mode='lines', name='Impulse Response', line=dict(color='#7C3AED', width=2)),
            row=2, col=1
        )

        fig_time.update_layout(
            height=580,
            plot_bgcolor='white',
            showlegend=True,
            xaxis2_title="Time t (seconds)",
            yaxis_title="Output Amplitude y(t)",
            yaxis2_title="Output Amplitude h(t)"
        )
        fig_time.update_xaxes(gridcolor='#F1F5F9')
        fig_time.update_yaxes(gridcolor='#F1F5F9')

        st.plotly_chart(fig_time, use_container_width=True)

        if is_overflow:
            st.warning("⚠️ Notice: The response amplitude grows exponentially because the system has poles in the Right-Half Plane (RHP). Plotted values are clipped to avoid visual blowup.")

    except Exception as err:
        st.error(f"Error computing time response: {err}")

with tab3:
    st.subheader("📚 Mathematical Modeling & Theoretical Foundations")
    
    st.markdown("""
    ### 1. Laplace Transform in IT Control Systems
    Differential equations describe how variables like **server CPU load**, **active user queue length**, or **audio circuit voltages** change over continuous time $t$. 
    By applying the unilateral Laplace Transform:
    """)
    st.latex(r"\mathcal{L}\{f(t)\} = F(s) = \int_{0}^{\infty} f(t) e^{-st} \, dt, \quad s = \sigma + j\omega")
    st.markdown("""
    All derivatives transform into algebraic multipliers ($df/dt \rightarrow sF(s) - f(0)$). Under zero initial conditions, the governing differential equations turn into an algebraic rational transfer function:
    """)
    st.latex(r"H(s) = \frac{Y(s)}{X(s)} = \frac{N(s)}{D(s)} = K \frac{\prod_{j=1}^m (s - z_j)}{\prod_{i=1}^n (s - p_i)}")
    
    st.markdown("""
    ---
    ### 2. Practical IT Case Study: Cloud Server Auto-Scaling Feedback Loop
    Consider a Kubernetes / AWS Auto-Scaler maintaining target CPU utilization $L_{\text{target}}$ against dynamic user traffic $R(t)$.
    
    - Let $L(t)$ be the actual server load.
    - Provisioning and booting virtual machine containers introduces an operational lag modeled by time constant $\tau$.
    - The Proportional-Integral (PI) auto-scaling controller adjusts container capacity based on error $e(t) = R(t) - L(t)$:
    """)
    st.latex(r"\tau \frac{d^2 L(t)}{dt^2} + \frac{dL(t)}{dt} = K_p \frac{de(t)}{dt} + K_i e(t)")
    st.markdown("""
    Taking the Laplace transform on both sides yields the closed-loop transfer function:
    """)
    st.latex(r"H(s) = \frac{L(s)}{R(s)} = \frac{K_p s + K_i}{\tau s^2 + (1 + K_p)s + K_i}")
    st.markdown("""
    - **Stable Parameter Selection:** When $(1 + K_p) > 0$ and $K_i > 0$, both poles have negative real parts ($\text{Re}(p) < 0$). Load smoothly converges to demand without thrashing.
    - **Thrashing / Instability:** If container startup delay or negative feedback delays introduce sign inversion (e.g. overcompensation where effective damping becomes negative), roots cross into the Right-Half Plane ($\text{Re}(p) > 0$). This causes the auto-scaler to repeatedly over-provision thousands of instances followed by immediate mass de-provisioning—a catastrophic phenomenon known as **cloud thrashing**.
    
    ---
    ### 3. Pole Locations vs. Physical System Behavior
    - **$\text{Re}(p) < 0$ (Left-Half Plane):** Exponential decay factor $e^{\sigma t}$ with $\sigma < 0 \rightarrow$ transient oscillations die out. **System is BIBO Stable.**
    - **$\text{Re}(p) > 0$ (Right-Half Plane):** Exponential factor $e^{\sigma t}$ with $\sigma > 0 \rightarrow$ amplitude grows infinitely. **System is Unstable.**
    - **$\text{Re}(p) = 0, \text{Im}(p) \neq 0$ (On $j\omega$ axis):** Pure sinusoidal oscillation $e^{j\omega t} = \cos(\omega t) + j\sin(\omega t)$. Neither grows nor decays. **System is Marginally Stable.**
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #94A3B8; font-size: 0.9rem;'>
    Department of Information Technology | Vidyavardhini's College of Engineering & Technology<br>
  Prepared By : Aayush Bhilare, Atharva Bhosale, Devyani Chaudhari, Ashish Chauhan, Pratikshya Das
</div>
""", unsafe_allow_html=True)