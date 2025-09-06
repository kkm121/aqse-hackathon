# streamlit_app.py
import streamlit as st
import numpy as np
from controller import run_aqse, pauli_list_4q, coeffs_4q

st.set_page_config(page_title="AQSE Hackathon", layout="wide")
st.title("AQSE — Adaptive Quantum Sampling Engine (Simulator-only Hackathon)")

with st.sidebar:
    st.header("Run settings")
    preset = st.selectbox("Preset", ["Fast Demo", "Balanced", "High Accuracy"])
    if preset == "Fast Demo":
        iters = st.slider("Iterations", 8, 80, 40, 4)
        shots_per_eval = st.selectbox("Baseline shots per eval", [128,256,512], index=2)
        trust_threshold = st.slider("Surrogate trust threshold", 0.01, 0.2, 0.07, 0.01)
        bootstrap_points = st.slider("Bootstrap measured points", 2, 10, 6)
        surrogate_models = st.selectbox("Surrogate ensemble size", [4,6,8], index=1)
    elif preset == "Balanced":
        iters = st.slider("Iterations", 20, 120, 60, 4)
        shots_per_eval = st.selectbox("Baseline shots per eval", [256,512,1024], index=1)
        trust_threshold = st.slider("Surrogate trust threshold", 0.01, 0.2, 0.06, 0.01)
        bootstrap_points = st.slider("Bootstrap measured points", 4, 12, 8)
        surrogate_models = st.selectbox("Surrogate ensemble size", [6,8,10], index=0)
    else:
        iters = st.slider("Iterations", 40, 240, 120, 4)
        shots_per_eval = st.selectbox("Baseline shots per eval", [512,1024,2048], index=0)
        trust_threshold = st.slider("Surrogate trust threshold", 0.005, 0.15, 0.05, 0.005)
        bootstrap_points = st.slider("Bootstrap measured points", 6, 20, 12)
        surrogate_models = st.selectbox("Surrogate ensemble size", [8,10,12], index=0)

    gate_error = st.slider("2Q gate error (simulated)", 0.0, 0.05, 0.02, 0.005)
    pruning_thresh = st.slider("Pruning fidelity threshold", 0.0, 0.2, 0.04, 0.01)
    run_button = st.button("Run AQSE")

pauli_list = pauli_list_4q
coeffs = coeffs_4q

if run_button:
    with st.spinner("Running AQSE (simulator-only)…"):
        df, summary = run_aqse(pauli_list, coeffs, n_qubits=4,
                               iters=iters, shots_per_eval=shots_per_eval,
                               trust_threshold=trust_threshold, bootstrap_points=bootstrap_points,
                               surrogate_models=surrogate_models, rf_estimators=50,
                               gate_error_two_q=gate_error, pruning_threshold=pruning_thresh, seed=42)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Final energy", f"{summary['final_energy']:.6f} Hartree")
    col2.metric("Baseline shots", f"{summary['total_shots_baseline']}")
    col3.metric("AQSE shots", f"{summary['total_shots_aqse']}")
    col4.metric("Shots saved", f"{summary['shots_saved_pct']:.1f}%")

    st.markdown("---")
    st.subheader("Energy vs Iteration")
    st.line_chart(df.set_index('iter')['used_energy'])
    st.subheader("Cumulative shots")
    st.line_chart(df.set_index('iter')['cumulative_shots'])
    st.subheader("Decision trace (sample)")
    st.dataframe(df.head(80))
    st.download_button("Download iteration CSV", df.to_csv(index=False).encode('utf-8'), "aqse_results.csv")
else:
    st.info("Configure settings and click Run AQSE.")
