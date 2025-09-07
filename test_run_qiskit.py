# test_run_qiskit.py  -- robust: tries ibm_cloud then ibm_quantum_platform
import os
import sys
from qiskit import QuantumCircuit

from qiskit_ibm_runtime import QiskitRuntimeService

def create_service_try_channels(token, instance):
    # Try the modern IBM Cloud channel first (most common for IAM tokens)
    for ch in ("ibm_cloud", "ibm_quantum_platform"):
        try:
            print(f"Attempting QiskitRuntimeService(channel='{ch}', instance={instance}) ...")
            svc = QiskitRuntimeService(channel=ch, token=token, instance=instance)
            print(f"Connected using channel='{ch}'")
            return svc
        except Exception as e:
            print(f"Channel '{ch}' failed: {e}")
    # Last resort: try creating without explicit channel (some SDK setups)
    try:
        print("Attempting QiskitRuntimeService(...) without explicit channel ...")
        svc = QiskitRuntimeService(token=token, instance=instance)
        print("Connected without explicit channel.")
        return svc
    except Exception as e:
        print("All attempts failed. Final exception:", e)
        raise

def main():
    token = os.getenv("QISKIT_IBM_TOKEN")
    instance = os.getenv("IBMQ_INSTANCE", None)
    backend_name = os.getenv("QISKIT_BACKEND", None)

    if not token:
        raise RuntimeError("QISKIT_IBM_TOKEN not set in environment. Set it and re-run.")

    print("Using instance:", instance)

    # Create the runtime service (tries channels)
    service = create_service_try_channels(token, instance)

    # Show available backends
    try:
        backends = service.backends()
        print("Available backends in instance (first 20):")
        for b in backends[:20]:
            try:
                cfg = b.configuration()
                st = b.status()
                print(f" - {b.name():25s}  simulator={getattr(cfg,'simulator',False)}  operational={getattr(st,'operational','unknown')}  pending_jobs={getattr(st,'pending_jobs','unknown')}")
            except Exception:
                print(f" - {b.name():25s}  (no detailed info)")
    except Exception as e:
        print("Could not list backends:", e)

    # Choose backend
    if backend_name:
        try:
            backend = service.backend(backend_name)
            print("Using forced backend (env):", backend_name)
        except Exception as e:
            print(f"Requested backend '{backend_name}' not available: {e}")
            print("Falling back to automatic selection.")
            backend_name = None

    if backend_name is None:
        # auto-select first operational real device, else first backend
        try:
            candidates = [b for b in service.backends() if not getattr(b.configuration(), "simulator", False)]
            candidates = [b for b in candidates if getattr(b.status(), "operational", True)]
            if not candidates:
                candidates = list(service.backends())
            candidates.sort(key=lambda b: getattr(b.status(), "pending_jobs", 0))
            backend = candidates[0]
        except Exception as e:
            print("Auto-selection failed, trying first backend directly:", e)
            backend = service.backends()[0]

    print("Selected backend:", backend.name(), " simulator=", getattr(backend.configuration(), "simulator", None))

    # Run a trivial single-qubit circuit via Sampler to verify
    try:
        from qiskit_ibm_runtime import Sampler
    except Exception as e:
        print("Could not import Sampler. Check qiskit-ibm-runtime version:", e)
        raise

    sampler = Sampler(backend=backend)
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    print("Submitting a trivial H-measure circuit (256 shots)... (this can take time on real hardware)")
    job = sampler.run([qc], shots=256)
    res = job.result()
    print("Raw result repr():")
    print(repr(res)[:1000])  # print a truncated repr for readability

    # Try to extract counts if present
    try:
        if hasattr(res, "__len__") and len(res) > 0:
            first = res[0]
            if hasattr(first, "get_counts"):
                print("Counts:", first.get_counts())
            else:
                data = getattr(first, "data", None)
                print("First entry type:", type(first))
                print("First entry data repr (truncated):", repr(data)[:800])
        else:
            print("Result object has no length or is empty; inspect repr above.")
    except Exception as e:
        print("Exception while extracting counts:", e)

if __name__ == "__main__":
    main()
