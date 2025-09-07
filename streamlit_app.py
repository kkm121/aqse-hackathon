# streamlit_app.py
import os
import streamlit as st
import numpy as np
from controller import run_aqse, pauli_list_4q, coeffs_4q

st.set_page_config(page_title="AQSE Hackathon", layout="wide")
st.title("AQSE — Adaptive Quantum Sampling Engine (Simulator-only Hackathon)")

with st.sidebar:
    st.header("Run settings")
    use_real_backend = st.checkbox("Use real IBM Quantum backend (enter token below)", value=False)
    ibmq_token = ""
    if use_real_backend:
        ibmq_token = st.text_input("IBM Quantum API token (QISKIT_IBM_TOKEN)", type="password")
        if ibmq_token:
            os.environ['QISKIT_IBM_TOKEN'] = ibmq_token
            os.environ['AQSE_USE_QISKIT'] = '1'
        else:
            st.warning("Enter API token to enable real backend.")
    else:
        # Ensure simulator mode
        os.environ['AQSE_USE_QISKIT'] = '0'

    preset = st.selectbox("Preset", ["Fast Demo", "Balanced", "High Accuracy"])
    if preset == "Fast Demo":
        iters = st.slider("Iterations", 8, 80, 40, 4)
        shots_per_eval = st.selectbox("Baseline shots per eval", [128,256,512], index=2)
        trust_threshold = st.slider("Surrogate trust threshold", 0.01, 0.2, 0.07, 0.01)
        bootstrap_points = st.slider("Bootstrap measured points", 2, 10, 6)
        surrogate_models = st.selectbox("Surrogate ensemble size", [6,8,10], index=0)
    elif preset == "Balanced":
        iters = st.slider("Iterations", 8, 160, 80, 4)
        shots_per_eval = st.selectbox("Baseline shots per eval", [128,256,512,1024], index=2)
        trust_threshold = st.slider("Surrogate trust threshold", 0.005, 0.2, 0.05, 0.005)
        bootstrap_points = st.slider("Bootstrap measured points", 4, 12, 8)
        surrogate_models = st.selectbox("Surrogate ensemble size", [8,10,12], index=0)
    else:
        iters = st.slider("Iterations", 8, 320, 160, 8)
        shots_per_eval = st.selectbox("Baseline shots per eval", [256,512,1024,2048], index=1)
        trust_threshold = st.slider("Surrogate trust threshold", 0.001, 0.1, 0.02, 0.001)
        bootstrap_points = st.slider("Bootstrap measured points", 6, 20, 10)
        surrogate_models = st.selectbox("Surrogate ensemble size", [10,12,16], index=0)

    gate_error = st.slider("2Q gate error (simulated)", 0.0, 0.05, 0.02, 0.005)
    pruning_thresh = st.slider("Pruning fidelity threshold", 0.0, 0.2, 0.04, 0.01)
    run_button = st.button("Run AQSE")

    # compare controls
    compare_seeds = st.slider('Compare: number of seeds', 1, 10, 3)
    compare_iters = st.slider('Compare: iterations', 8, 120, 40, 4)
    compare_shots = st.selectbox('Compare: shots per eval', [128,256,512,1024], index=2)
    compare_button = st.button('Run AQSE vs Baseline (multi-seed)')

pauli_list = pauli_list_4q
coeffs = coeffs_4q

if run_button:
    with st.spinner("Running AQSE…"):
        try:
            df, summary = run_aqse(pauli_list, coeffs, n_qubits=4,
                                   iters=iters, shots_per_eval=shots_per_eval,
                                   trust_threshold=trust_threshold, bootstrap_points=bootstrap_points,
                                   surrogate_models=surrogate_models, rf_estimators=50,
                                   gate_error_two_q=gate_error, pruning_threshold=pruning_thresh,
                                   entangler_active=True, seed=1)
            st.success("Run complete.")
            st.write("Summary:", summary)
            st.dataframe(df)
        except Exception as e:
            st.error(f"Run failed: {e}")

if compare_button:
    import pandas as pd
    from compare_and_report import run_mode
    rows = []
    with st.spinner("Running multi-seed comparison..."):
        for mode in ('aqse','baseline'):
            for s in range(compare_seeds):
                seed = 1000 + s
                try:
                    dfm, summary = run_mode(mode, seed, compare_iters, compare_shots, outdir='results')
                    rows.append({'mode': mode, 'seed': seed, **summary})
                except Exception as e:
                    rows.append({'mode': mode, 'seed': seed, 'error': str(e)})
    df_all = pd.DataFrame(rows)
    st.success('Comparison complete.')
    st.dataframe(df_all)
    try:
        st.subheader("Final Energy by Mode")
        st.image("results/final_energy_box.png")
    except Exception as e:
        st.error(f"Could not load final_energy_box.png: {e}")

if not run_button and not compare_button:
    st.info("Configure settings and click Run AQSE.")
