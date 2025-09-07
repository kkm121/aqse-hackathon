# measurement_qiskit.py
"""
Qiskit-powered measurement backend for running on a real IBM Quantum backend or a local simulator.
Robust against different qiskit-ibm-runtime return shapes (backend names vs objects)
and resilient to different runtime result shapes for counts.

Exports: measure_paulis(theta, pauli_terms, shots_per_term, entangler, seed, return_backend_info=False)
"""
import os
import numpy as np
import warnings

from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService

# Try common Sampler names; fallback to None and handle later.
try:
    from qiskit_ibm_runtime import SamplerV2 as Sampler
except Exception:
    try:
        from qiskit_ibm_runtime import Sampler as Sampler
    except Exception:
        Sampler = None

# Global variable to store last backend info for inspection
LAST_QISKIT_BACKEND_INFO = {"name": None, "is_simulator": None}


def _resolve_backend_entry(service, entry):
    """
    Given an entry from service.backends() (which may be a backend object or a string),
    return a backend object. Raises RuntimeError if resolution fails.
    """
    # If it's already an object with callable .name(), assume it's a backend object
    try:
        if hasattr(entry, "name") and callable(getattr(entry, "name")):
            return entry
    except Exception:
        pass

    # If it's a string, try to resolve via service.backend(name)
    if isinstance(entry, str):
        try:
            return service.backend(entry)
        except Exception as e:
            # Try to match among service.backends() items
            try:
                for cand in service.backends():
                    # cand may be string or object
                    if isinstance(cand, str):
                        if cand == entry:
                            return service.backend(cand)
                    else:
                        try:
                            if cand.name() == entry:
                                return cand
                        except Exception:
                            # If cand.name() fails, skip
                            continue
            except Exception:
                pass
            raise RuntimeError(f"Could not resolve backend name '{entry}': {e}")

    # If it's something else try a last-ditch inspection
    raise RuntimeError(f"Unrecognized backend entry type: {type(entry)}")


def _get_service_and_backend():
    token = os.getenv("QISKIT_IBM_TOKEN", None)
    if not token:
        raise RuntimeError("QISKIT_IBM_TOKEN not set; cannot use real backend.")

    # Always use channel="ibm_cloud" for modern Qiskit Runtime
    service = QiskitRuntimeService(channel="ibm_cloud", token=token)

    # If user set QISKIT_BACKEND use it; otherwise try to find a real device.
    backend_name_env = os.getenv("QISKIT_BACKEND", None)
    backend_obj = None

    if backend_name_env:
        try:
            # try direct resolution (service.backend handles names)
            backend_obj = _resolve_backend_entry(service, backend_name_env)
        except Exception as e:
            # log and continue to auto-select
            print(f"[measurement_qiskit] Requested backend '{backend_name_env}' not resolved: {e}. Will try auto-selection.")

    if backend_obj is None:
        # Attempt to auto-select a real device (non-simulator, operational) if available.
        try:
            raw_list = list(service.backends(simulator=False, operational=True))
            # Resolve into backend objects robustly
            candidates = []
            for ent in raw_list:
                try:
                    candidates.append(_resolve_backend_entry(service, ent))
                except Exception:
                    continue
            if not candidates:
                # try any backends
                raw_all = list(service.backends())
                for ent in raw_all:
                    try:
                        candidates.append(_resolve_backend_entry(service, ent))
                    except Exception:
                        continue

            if not candidates:
                raise RuntimeError("No backends returned from QiskitRuntimeService().")

            # sort by pending jobs if possible
            try:
                candidates.sort(key=lambda b: getattr(b.status(), "pending_jobs", 0))
            except Exception:
                pass

            backend_obj = candidates[0]
        except Exception as e:
            # If everything fails, raise a friendly error
            raise RuntimeError(f"Auto-selection of backend failed: {e}")

    # Confirm we have a backend object and get its properties
    try:
        cfg = backend_obj.configuration()
        is_sim = bool(getattr(cfg, "simulator", False))
    except Exception:
        is_sim = False

    try:
        backend_name = backend_obj.name()
    except Exception:
        # last-resort: if name() fails, try attribute or str()
        backend_name = getattr(backend_obj, "name", None) or str(backend_obj)

    print(f"[measurement_qiskit] Using backend: {backend_name}  simulator={is_sim}")

    # Store last backend info globally
    global LAST_QISKIT_BACKEND_INFO
    LAST_QISKIT_BACKEND_INFO = {"name": backend_name, "is_simulator": is_sim}
    return service, backend_obj


# ----------------- helper functions for measurement -----------------
def _ansatz(theta, n_qubits=4, entangler=True):
    qc = QuantumCircuit(n_qubits)
    for i, t in enumerate(theta):
        qc.ry(float(t), i % n_qubits)
    if entangler and n_qubits >= 2:
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
    return qc


def _pauli_to_measure_circuit(p, n_qubits=4):
    qc = QuantumCircuit(n_qubits, n_qubits)
    for i, ch in enumerate(p):
        if ch == 'X':
            qc.h(i)
        elif ch == 'Y':
            qc.sdg(i); qc.h(i)
        # Z/I -> nothing
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def _expectation_from_counts(p, counts, shots):
    if not counts or shots <= 0:
        return 0.0
    exp = 0.0
    for bitstr, c in counts.items():
        # normalize bitstring into a str
        try:
            bstr = str(bitstr)
        except Exception:
            bstr = repr(bitstr)
        bits = bstr[::-1]  # make little-endian interpretation consistent with sim code
        val = 1.0
        for i, ch in enumerate(p):
            if ch == 'I':
                continue
            if i >= len(bits):
                bit = 0
            else:
                try:
                    bit = int(bits[i])
                except Exception:
                    bit = 0
            v = -1.0 if bit == 1 else 1.0
            val *= v
        exp += val * (c / shots)
    return float(exp)


def _extract_counts_from_result(result_entry):
    """
    Extract counts dict from various runtime result shapes.
    Returns {} if no counts found.
    """
    try:
        # direct get_counts
        if hasattr(result_entry, "get_counts") and callable(result_entry.get_counts):
            return result_entry.get_counts()
    except Exception:
        pass

    try:
        data = getattr(result_entry, "data", None)
        if data is not None:
            # many runtime shapes put measurement counts under data.meas or data.get_counts()
            meas = getattr(data, "meas", None)
            if meas is not None:
                if hasattr(meas, "get_counts") and callable(meas.get_counts):
                    return meas.get_counts()
                if isinstance(meas, dict) and "counts" in meas:
                    return meas["counts"]
            if hasattr(data, "get_counts") and callable(data.get_counts):
                return data.get_counts()
            if isinstance(data, dict):
                for key in ("counts", "meas", "value_counts"):
                    if key in data and isinstance(data[key], dict):
                        return data[key]
    except Exception:
        pass

    # fallback: if entry itself is dict-like
    try:
        if isinstance(result_entry, dict):
            for key in ("counts", "value_counts", "measurement_counts"):
                if key in result_entry and isinstance(result_entry[key], dict):
                    return result_entry[key]
    except Exception:
        pass

    return {}


def measure_paulis(theta, pauli_terms, shots_per_term=1024, entangler=True, seed=None, return_backend_info=False):
    """
    Measure Pauli strings using IBM Qiskit Runtime sampler (or device).
    Returns dict {pauli_str: expectation_estimate}
    """
    if Sampler is None:
        raise ImportError("Qiskit Runtime Sampler not available in this environment (Sampler import failed).")

    theta = np.asarray(theta, dtype=float)
    n_qubits = len(pauli_terms[0])
    prep = _ansatz(theta, n_qubits=n_qubits, entangler=entangler)

    service, backend = _get_service_and_backend()

    # Try Sampler(session=service), fallback to Sampler(service), fallback to Sampler()
    sampler = None
    try:
        sampler = Sampler(session=service)
    except TypeError:
        try:
            sampler = Sampler(service)
        except TypeError:
            sampler = Sampler()

    results = {}
    for p in pauli_terms:
        meas = _pauli_to_measure_circuit(p, n_qubits=n_qubits)
        qc = prep.compose(meas, front=True)
        shots = int(os.getenv("QISKIT_SHOTS", str(shots_per_term)))
        try:
            # Try passing backend as mode if required
            job = sampler.run([qc], shots=shots, mode=backend)
        except TypeError:
            # Fallback: run without mode
            job = sampler.run([qc], shots=shots)
        res = job.result()

        counts = {}
        try:
            # common: result is list-like with entries that contain counts
            if hasattr(res, "__len__") and len(res) > 0:
                counts = _extract_counts_from_result(res[0])
            else:
                counts = _extract_counts_from_result(res)
        except Exception:
            counts = {}

        # fallback: try top-level get_counts on res if present
        if not counts:
            try:
                if hasattr(res, "get_counts") and callable(res.get_counts):
                    counts = res.get_counts()
            except Exception:
                counts = {}

        if not counts:
            warnings.warn("[measurement_qiskit] Could not extract counts from runtime result. "
                          "Counts dict is empty — check runtime version / result shape. "
                          "Returning expectation = 0.0 for this Pauli.", UserWarning)

        results[p] = _expectation_from_counts(p, counts, shots)
        print(f"[DEBUG] Pauli: {p}, Counts: {counts}, Expectation: {results[p]}")

    if return_backend_info:
        backend_info = dict(LAST_QISKIT_BACKEND_INFO)
        return results, backend_info
    return results
