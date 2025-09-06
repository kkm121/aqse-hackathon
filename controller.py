# controller.py
import numpy as np
import pandas as pd
from optimizer import SPSA
from surrogate import EnsembleCalibratedSurrogate
from measurement_nosim import measure_paulis
from shot_allocation import allocate_shots_opt
from fidelity import estimate_fidelity_drop

def energy_from_paulis(pauli_vals, coeffs):
    E = 0.0
    for p, c in coeffs.items():
        E += c * pauli_vals[p]
    return float(E)

def compute_sigmaE(coeffs_array, sigma_pauli_array, samplingVar_array=None):
    if samplingVar_array is None:
        samplingVar_array = np.zeros_like(sigma_pauli_array)
    var_terms = (coeffs_array ** 2) * (sigma_pauli_array ** 2 + samplingVar_array)
    return float(np.sqrt(np.sum(var_terms)))

# Default 4-qubit toy molecule
# controller.py (snippet)

# Try to auto-load a generated pauli file (pauli_h2.py) if present.
try:
    from pauli_h2 import pauli_list as _pauli_list_gen, coeffs as _coeffs_gen
    maxlen = max(len(p) for p in _pauli_list_gen)
    # normalize keys to full-length strings (pad with 'I' if needed)
    pauli_list_4q = [p.ljust(maxlen, 'I') for p in _pauli_list_gen]
    coeffs_4q = {p.ljust(maxlen, 'I'): float(_coeffs_gen[p]) for p in _pauli_list_gen}
    print("Loaded generated Hamiltonian from pauli_h2.py")
except Exception:
    # Fallback to built-in toy Hamiltonian (ensure 4-char Pauli strings)
    pauli_list_4q = [
        'ZIII', 'IZII', 'IIZI', 'IIIZ',
        'ZZII', 'IIZZ', 'ZIZI', 'IZIZ',
        'XXII', 'IIXX', 'XXXX', 'XYIZ'
    ]
    coeffs_4q = {
        'ZIII': -0.700, 'IZII': 0.300, 'IIZI': 0.250, 'IIIZ': -0.400,
        'ZZII': 0.200, 'IIZZ': -0.150, 'ZIZI': 0.050, 'IZIZ': -0.060,
        'XXII': -0.350, 'IIXX': 0.280, 'XXXX': 0.100, 'XYIZ': 0.070
    }
    # Pad all Pauli strings to length 4
    pauli_list_4q = [p.ljust(4, 'I') for p in pauli_list_4q]
    coeffs_4q = {p.ljust(4, 'I'): v for p, v in coeffs_4q.items()}


def run_aqse(pauli_terms=None, coeffs_dict=None,
             n_qubits=4, iters=40, shots_per_eval=512, trust_threshold=0.07,
             bootstrap_points=8, surrogate_models=6, rf_estimators=50,
             gate_error_two_q=0.02, pruning_threshold=0.04, seed=42):
    rng = np.random.default_rng(seed)
    if pauli_terms is None or coeffs_dict is None:
        pauli_terms = pauli_list_4q
        coeffs_dict = coeffs_4q
        n_qubits = 4
    coeffs_array = np.array([coeffs_dict[p] for p in pauli_terms])
    theta0 = np.random.normal(0, 0.1, size=n_qubits)
    optim = SPSA(theta0.copy())
    surrogate = EnsembleCalibratedSurrogate(n_models=surrogate_models, rf_estimators=rf_estimators, random_state=seed)
    entangler_active = True

    # bootstrap
    X = []; Y = []
    for i in range(bootstrap_points):
        th = theta0 + rng.normal(0, 0.2, size=n_qubits) * (i+1)
        pvals = measure_paulis(th, pauli_terms, shots_per_term=shots_per_eval, entangler=entangler_active, seed=int(seed+i))
        X.append(th); Y.append([pvals[p] for p in pauli_terms])
    X = np.vstack(X); Y = np.vstack(Y)
    surrogate.fit(X, Y)

    rows = []
    cumulative_shots = bootstrap_points * shots_per_eval
    baseline_shots = iters * shots_per_eval
    prev_energy = None

    for it in range(1, iters+1):
        def eval_fn(th):
            mean, std = surrogate.predict_mean_std(th.reshape(1, -1))
            sigmaE = compute_sigmaE(coeffs_array, std.flatten())
            if float(sigmaE) < trust_threshold:
                return float((mean.flatten() * coeffs_array).sum())
            pvals = measure_paulis(th, pauli_terms, shots_per_term=shots_per_eval, entangler=entangler_active, seed=int(seed+it))
            return energy_from_paulis(pvals, coeffs_dict)

        theta, _, _ = optim.step(eval_fn)

        mean_vec, std_vec = surrogate.predict_mean_std(theta.reshape(1, -1))
        mean_vec = mean_vec.flatten(); std_vec = std_vec.flatten()
        samplingVar_default = (1.0 - mean_vec**2) / max(1, shots_per_eval)
        sigmaE = compute_sigmaE(coeffs_array, std_vec, samplingVar_default)

        use_surrogate = sigmaE < trust_threshold
        measured_flag = False; shots_used = 0

        if use_surrogate:
            E_used = float((mean_vec * coeffs_array).sum())
        else:
            alloc = allocate_shots_opt(shots_per_eval, coeffs_array, std_vec, min_shot=8)
            pvals = {}
            for p, s in zip(pauli_terms, alloc):
                res = measure_paulis(theta, [p], shots_per_term=int(s), entangler=entangler_active, seed=int(seed+it))
                pvals[p] = res[p]
            E_used = energy_from_paulis(pvals, coeffs_dict)
            measured_flag = True
            shots_used = int(alloc.sum())
            cumulative_shots += shots_used
            X_new = np.vstack([surrogate.X, theta.reshape(1, -1)]) if surrogate.is_trained else theta.reshape(1, -1)
            Y_new = np.vstack([surrogate.Y, np.array([[pvals[p] for p in pauli_terms]])]) if surrogate.is_trained else np.array([[pvals[p] for p in pauli_terms]])
            surrogate.fit(X_new, Y_new)

        n_two_q = 1 if entangler_active else 0
        fidelity_drop = estimate_fidelity_drop(n_two_q, gate_error=gate_error_two_q)
        expected_improvement = 0.0 if prev_energy is None else max(0.0, prev_energy - E_used)
        prune = False
        if fidelity_drop > pruning_threshold and expected_improvement < fidelity_drop:
            entangler_active = False
            prune = True

        prev_energy = E_used

        rows.append({
            "iter": it,
            "theta": theta.tolist(),
            "used_energy": float(E_used),
            "surrogate_sigmaE": float(sigmaE),
            "used_surrogate": bool(use_surrogate),
            "measured_flag": bool(measured_flag),
            "shots_this_iter": int(shots_used),
            "cumulative_shots": int(cumulative_shots),
            "fidelity_drop": float(fidelity_drop),
            "pruned_this_iter": bool(prune),
            "entangler_active": bool(entangler_active)
        })

    df = pd.DataFrame(rows)
    summary = {
        "final_energy": float(df["used_energy"].iloc[-1]),
        "total_shots_baseline": int(baseline_shots),
        "total_shots_aqse": int(cumulative_shots),
        "shots_saved_pct": 100.0 * (1 - cumulative_shots / baseline_shots)
    }
    return df, summary
