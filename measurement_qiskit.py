# measurement_qiskit.py
"""
Qiskit-powered measurement backend for running on a real IBM Quantum backend or a local simulator.
Configuration via environment variables:

- AQSE_USE_QISKIT=1                # to activate this backend in controller.py
- QISKIT_IBM_TOKEN=...             # your IBM Quantum Platform API token
- IBMQ_INSTANCE="hub/group/project"  # optional instance
- QISKIT_BACKEND="ibm_oslo"        # backend name; default uses least-busy device
- QISKIT_SHOTS=4096                # default shots when allocating

Exports: measure_paulis(theta, pauli_terms, shots_per_term, entangler, seed)
"""
import os, numpy as np

# Qiskit imports (may raise if not installed)
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

def _get_service_and_backend():
    token = os.getenv("QISKIT_IBM_TOKEN", None)
    if not token:
        raise RuntimeError("QISKIT_IBM_TOKEN not set; cannot use real backend.")
    instance = os.getenv("IBMQ_INSTANCE", None)
    service = QiskitRuntimeService(channel="ibm_quantum", token=token, instance=instance)
    backend_name = os.getenv("QISKIT_BACKEND", None)
    if backend_name:
        backend = service.backend(backend_name)
    else:
        # pick a device with least queue automatically (best-effort)
        candidates = [b for b in service.backends(simulator=False, operational=True)]
        if not candidates:
            # fallback to any available backend
            candidates = service.backends()
        candidates.sort(key=lambda b: getattr(b.status(), "pending_jobs", 0))
        backend = candidates[0]
    return service, backend

def _ansatz(theta, n_qubits=4, entangler=True):
    qc = QuantumCircuit(n_qubits)
    # Simple Ry parameterized ansatz
    for i, t in enumerate(theta):
        qc.ry(float(t), i % n_qubits)
    if entangler:
        for i in range(n_qubits-1):
            qc.cx(i, i+1)
    return qc

def _pauli_to_measure_circuit(p, n_qubits=4):
    # Prepare basis change for measuring Pauli string p in Z basis
    qc = QuantumCircuit(n_qubits, n_qubits)
    for i, ch in enumerate(p):
        if ch == 'X':
            qc.h(i)
        elif ch == 'Y':
            qc.sdg(i); qc.h(i)
        # 'Z' or 'I' -> no change
    qc.measure(range(n_qubits), range(n_qubits))
    return qc

def _expectation_from_counts(p, counts, shots):
    # Expectation of tensor product observable with outcomes +/-1
    exp = 0.0
    for bitstr, c in counts.items():
        val = 1.0
        bits = bitstr[::-1]  # assume little-endian mapping
        for i, ch in enumerate(p):
            if ch == 'I':
                continue
            bit = int(bits[i])
            v = -1.0 if bit == 1 else 1.0
            val *= v
        exp += val * (c / shots)
    return float(exp)

def measure_paulis(theta, pauli_terms, shots_per_term=1024, entangler=True, seed=None):
    n_qubits = len(pauli_terms[0])
    prep = _ansatz(theta, n_qubits=n_qubits, entangler=entangler)

    service, backend = _get_service_and_backend()
    sampler = Sampler(backend=backend)

    results = {}
    for p in pauli_terms:
        meas = _pauli_to_measure_circuit(p, n_qubits=n_qubits)
        qc = prep.compose(meas, front=True)
        shots = int(os.getenv("QISKIT_SHOTS", str(shots_per_term)))
        job = sampler.run([qc], shots=shots)
        res = job.result()
        # There may be different result data formats; try to extract counts robustly
        data = res[0].data
        if hasattr(data, "meas"):
            dist = data.meas.get_counts()
        elif hasattr(data, "get_counts"):
            dist = data.get_counts()
        else:
            dist = {}
        results[p] = _expectation_from_counts(p, dist, shots)
    return results
