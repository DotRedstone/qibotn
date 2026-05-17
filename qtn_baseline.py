import argparse
import importlib.metadata as md
import inspect
import json
import os
import sys
import time

# CPU-only: hide GPUs before importing qibo.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import qibo
from qibo import Circuit, gates
from qibo.models import QFT


def versions():
    pkgs = ["qibo", "qibojit", "qibotn", "quimb", "cotengra", "numpy"]
    out = {}
    for pkg in pkgs:
        try:
            out[pkg] = md.version(pkg)
        except Exception:
            out[pkg] = None
    return out


def gate_count(circuit):
    return len(circuit.queue)


def build_qft_circuit(nqubits, ngates=None, seed=None):
    """Official-style QFT workload."""
    if ngates is not None:
        raise ValueError("QFT workload does not use --ngates.")
    return QFT(nqubits)


def build_random_local_circuit(nqubits, ngates, seed):
    """Build a deterministic local deep circuit for large-gate benchmarks."""
    if ngates is None:
        ngates = 1000

    rng = np.random.default_rng(seed)
    circuit = Circuit(nqubits)
    twoq_count = 0

    while gate_count(circuit) < ngates:
        index = gate_count(circuit)

        if nqubits > 1 and index % 4 == 3:
            offset = twoq_count % 2
            q = (2 * (twoq_count // 2) + offset) % (nqubits - 1)
            circuit.add(gates.CZ(q, q + 1))
            twoq_count += 1
            continue

        q = index % nqubits
        theta = float(rng.uniform(-np.pi, np.pi))
        rotation = index % 3
        if rotation == 0:
            circuit.add(gates.RX(q, theta=theta))
        elif rotation == 1:
            circuit.add(gates.RY(q, theta=theta))
        else:
            circuit.add(gates.RZ(q, theta=theta))

    return circuit


def build_circuit(workload, nqubits, ngates=None, seed=1234):
    builders = {
        "qft": build_qft_circuit,
        "random-local": build_random_local_circuit,
    }

    try:
        return builders[workload](nqubits=nqubits, ngates=ngates, seed=seed)
    except KeyError as exc:
        raise ValueError(f"Unsupported workload: {workload}") from exc


def build_observable_spec(observable, nqubits):
    if observable == "z0":
        return {
            "name": observable,
            "operators_list": ["z"],
            "sites_list": [(0,)],
            "coeffs_list": [1.0],
        }

    if observable == "zz-chain":
        return {
            "name": observable,
            "operators_list": ["zz"] * (nqubits - 1),
            "sites_list": [(i, i + 1) for i in range(nqubits - 1)],
            "coeffs_list": [1.0] * (nqubits - 1),
        }

    raise ValueError(f"Unsupported observable: {observable}")


def set_threads(threads):
    if threads is None:
        return

    # Better to export these before running Python, but setting here is still useful.
    for name in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        os.environ[name] = str(threads)

    try:
        qibo.set_threads(threads)
    except Exception:
        pass


def set_qibotn_cpu_backend(mode, max_bond, optimizer, svd_cutoff):
    """
    CPU-only QiboTN backend.

    Current source version supports:
      platform='quimb'
    Do not use:
      platform='cutensornet'  # GPU / cuQuantum
      platform='qutensornet'  # old name, unsupported by current source version
    """
    qibo.set_backend(
        backend="qibotn",
        platform="quimb",
        quimb_backend="numpy",
        contraction_optimizer=optimizer,
    )
    qibo.set_device("/CPU:0")

    backend_obj = qibo.get_backend()
    configure_meta = {
        "called": False,
        "reason": None,
        "signature": None,
        "config": None,
        "error": None,
    }

    # Configure ansatz explicitly. The quimb backend expects lowercase "mps";
    # None selects the generic tensor-network Circuit class.
    if mode in ["tn", "mps"]:
        if not hasattr(backend_obj, "configure_tn_simulation"):
            configure_meta["reason"] = "backend has no configure_tn_simulation"
        else:
            sig = inspect.signature(backend_obj.configure_tn_simulation)
            accepted = set(sig.parameters)
            config = {}

            if "ansatz" in accepted:
                config["ansatz"] = "mps" if mode == "mps" else None

            if mode == "mps" and max_bond is not None:
                if "max_bond_dimension" in accepted:
                    config["max_bond_dimension"] = max_bond
                elif "max_bond" in accepted:
                    config["max_bond"] = max_bond

            if mode == "mps" and svd_cutoff is not None:
                if "svd_cutoff" in accepted:
                    config["svd_cutoff"] = svd_cutoff

            try:
                backend_obj.configure_tn_simulation(**config)
                configure_meta.update(
                    {
                        "called": True,
                        "signature": str(sig),
                        "config": config,
                    }
                )
            except Exception as exc:
                configure_meta.update(
                    {
                        "called": False,
                        "signature": str(sig),
                        "config": config,
                        "error": repr(exc),
                    }
                )
                raise

    return {
        "backend": "qibotn",
        "platform": "quimb",
        "device": "/CPU:0",
        "quimb_backend": "numpy",
        "contraction_optimizer": optimizer,
        "mode": mode,
        "max_bond": max_bond,
        "svd_cutoff": svd_cutoff,
        "configure_tn_simulation": configure_meta,
    }


def set_qibojit_cpu_backend():
    """CPU reference backend."""
    qibo.set_backend("qibojit", platform="numba")
    qibo.set_device("/CPU:0")

    return {
        "backend": "qibojit",
        "platform": "numba",
        "device": "/CPU:0",
    }


def extract_state(result):
    """
    Accept different return types:
      1. numpy array
      2. QiboTN TensorNetworkResult with .statevector
      3. Qibo result with .state()
    """
    if isinstance(result, np.ndarray):
        return result

    if hasattr(result, "statevector") and result.statevector is not None:
        return result.statevector

    if hasattr(result, "state"):
        state = result.state()
        if state is not None:
            return state

    raise RuntimeError(
        "Cannot extract statevector. "
        "For qibotn+quimb use execute_circuit(..., return_array=True)."
    )


def canonicalize_state(state, nqubits):
    """
    Convert backend-specific state shape to standard 1D statevector.

    qibotn may return shape (2**n, 1).
    qibojit usually returns shape (2**n,).
    """
    arr = np.asarray(state)
    original_shape = tuple(arr.shape)

    arr = np.squeeze(arr)
    arr = arr.reshape(-1)

    expected_size = 2**nqubits
    if arr.size != expected_size:
        raise RuntimeError(
            f"Unexpected state size: got {arr.size}, expected {expected_size}. "
            f"Original shape was {original_shape}."
        )

    return arr, original_shape


def to_real_scalar(value):
    arr = np.asarray(value)
    return float(np.real(arr.reshape(-1)[0]))


def z_eigenvalue(index, nqubits, qubit):
    bit = (index >> (nqubits - 1 - qubit)) & 1
    return 1.0 if bit == 0 else -1.0


def expectation_from_statevector(state, observable, nqubits):
    """Compute simple Z-basis observables directly from a dense statevector."""
    state = np.asarray(state).reshape(-1)
    probabilities = np.abs(state) ** 2

    if observable == "z0":
        return sum(
            prob * z_eigenvalue(index, nqubits, 0)
            for index, prob in enumerate(probabilities)
        )

    if observable == "zz-chain":
        expectation = 0.0
        for index, prob in enumerate(probabilities):
            eigenvalue = 0.0
            for qubit in range(nqubits - 1):
                eigenvalue += z_eigenvalue(index, nqubits, qubit) * z_eigenvalue(
                    index, nqubits, qubit + 1
                )
            expectation += prob * eigenvalue
        return expectation

    raise ValueError(f"Unsupported observable: {observable}")


def reference_expectation(circuit, observable, nqubits):
    state = circuit().state(numpy=True)
    return expectation_from_statevector(state, observable, nqubits)


def quimb_operator_from_pauli_string(opstr):
    import quimb as qu

    operator = qu.pauli(opstr[0].lower())
    for char in opstr[1:]:
        operator = operator & qu.pauli(char.lower())
    return operator


def qibotn_expectation_local(backend_obj, quimb_circuit, spec):
    expectation_value = 0.0
    for opstr, sites, coeff in zip(
        spec["operators_list"],
        spec["sites_list"],
        spec["coeffs_list"],
    ):
        operator = quimb_operator_from_pauli_string(opstr)
        term_value = quimb_circuit.local_expectation(
            operator,
            where=sites,
            backend=backend_obj.backend,
            optimize=backend_obj.contractions_optimizer,
            simplify_sequence="R",
        )
        expectation_value = expectation_value + coeff * term_value

    return backend_obj.real(expectation_value)


def qibotn_expectation_mps_batched(backend_obj, quimb_circuit, spec):
    if not hasattr(quimb_circuit, "psi"):
        raise RuntimeError("Quimb circuit does not expose an MPS psi object.")

    psi = quimb_circuit.psi
    if not hasattr(psi, "compute_local_expectation"):
        raise RuntimeError("Quimb MPS object has no compute_local_expectation method.")

    terms = {}
    for opstr, sites, coeff in zip(
        spec["operators_list"],
        spec["sites_list"],
        spec["coeffs_list"],
    ):
        where = tuple(sites)
        operator = coeff * quimb_operator_from_pauli_string(opstr)
        if where in terms:
            terms[where] = terms[where] + operator
        else:
            terms[where] = operator

    sig = inspect.signature(psi.compute_local_expectation)
    accepted = set(sig.parameters)
    accepts_extra_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in sig.parameters.values()
    )

    kwargs = {}
    if "normalized" in accepted:
        kwargs["normalized"] = True
    if "return_all" in accepted:
        kwargs["return_all"] = False
    if "method" in accepted:
        kwargs["method"] = "canonical"
    elif "mode" in accepted:
        kwargs["mode"] = "canonical"
    if "backend" in accepted or accepts_extra_kwargs:
        kwargs["backend"] = backend_obj.backend
    if "optimize" in accepted or accepts_extra_kwargs:
        kwargs["optimize"] = backend_obj.contractions_optimizer

    expectation_value = psi.compute_local_expectation(terms, **kwargs)
    return backend_obj.real(expectation_value)


def qibotn_convert_to_quimb(
    backend_obj,
    circuit,
    circuit_type,
    circuit_kwargs,
    profile_conversion,
    fuse_single_qubit,
    absorb_1q_into_2q,
    two_qubit_apply,
):
    fuse_single_qubit = fuse_single_qubit or absorb_1q_into_2q

    if not profile_conversion and not fuse_single_qubit:
        t0 = time.perf_counter()
        quimb_circuit = backend_obj._qibo_circuit_to_quimb(
            circuit,
            quimb_circuit_type=circuit_type,
            **circuit_kwargs,
        )
        t1 = time.perf_counter()
        return quimb_circuit, {"stage_qibo_to_quimb_sec": t1 - t0}

    from qibo.gates.abstract import ParametrizedGate
    from qibotn.backends.quimb import GATE_MAP

    def apply_direct(quimb_circuit, gate_id, params, qubits, is_parametrized=False):
        if is_parametrized:
            quimb_circuit.apply_gate(
                gate_id, *params, *qubits, parametrized=is_parametrized
            )
        else:
            quimb_circuit.apply_gate(gate_id, *params, *qubits)

    def flush_pending_fast(quimb_circuit, pending_1q, qubit):
        matrix = pending_1q[qubit]
        if matrix is None:
            return False
        quimb_circuit.apply_gate(matrix, qubit)
        pending_1q[qubit] = None
        return True

    def absorb_pending_into_two_qubit_gate(gate, pending_1q, qubits):
        q0, q1 = qubits
        left0 = pending_1q[q0]
        left1 = pending_1q[q1]
        absorbed = int(left0 is not None) + int(left1 is not None)
        if absorbed == 0:
            return None, 0

        base = np.asarray(gate.matrix()).reshape(4, 4)
        identity = np.eye(2, dtype=base.dtype)
        if left0 is None:
            left0 = identity
        if left1 is None:
            left1 = identity
        combined = base @ np.kron(left0, left1)
        pending_1q[q0] = None
        pending_1q[q1] = None
        return combined, absorbed

    def two_qubit_gate_id(gate, quimb_gate_name):
        if two_qubit_apply == "matrix":
            return np.asarray(gate.matrix()).reshape(4, 4)
        return quimb_gate_name

    if fuse_single_qubit and not profile_conversion:
        t_total0 = time.perf_counter()
        quimb_circuit = circuit_type(circuit.nqubits, **circuit_kwargs)
        pending_1q = [None] * circuit.nqubits
        gate_count_1q = 0
        gate_count_2q = 0
        gate_count_other = 0
        gate_count_measure = 0
        gate_count_applied_1q = 0
        gate_count_applied_2q = 0
        gate_count_applied_other = 0
        gate_count_absorbed_1q = 0

        for gate in circuit.queue:
            gate_name = getattr(gate, "name", None)
            quimb_gate_name = GATE_MAP.get(gate_name, None)
            if quimb_gate_name == "measure":
                gate_count_measure += 1
                continue
            if quimb_gate_name is None:
                raise ValueError(f"Gate {gate_name} not supported in Quimb backend.")

            params = getattr(gate, "parameters", ())
            qubits = getattr(gate, "qubits", ())
            n_active_qubits = len(qubits)

            if n_active_qubits == 1:
                matrix = np.asarray(gate.matrix())
                qubit = qubits[0]
                if pending_1q[qubit] is None:
                    pending_1q[qubit] = matrix
                else:
                    pending_1q[qubit] = matrix @ pending_1q[qubit]
                gate_count_1q += 1
                continue

            if absorb_1q_into_2q and n_active_qubits == 2:
                combined_gate, absorbed = absorb_pending_into_two_qubit_gate(
                    gate=gate,
                    pending_1q=pending_1q,
                    qubits=qubits,
                )
                if combined_gate is not None:
                    quimb_circuit.apply_gate(combined_gate, *qubits)
                    gate_count_2q += 1
                    gate_count_applied_2q += 1
                    gate_count_absorbed_1q += absorbed
                    continue

            for qubit in qubits:
                if flush_pending_fast(quimb_circuit, pending_1q, qubit):
                    gate_count_applied_1q += 1

            is_parametrized = isinstance(gate, ParametrizedGate) and getattr(
                gate, "trainable", True
            )
            apply_direct(
                quimb_circuit=quimb_circuit,
                gate_id=(
                    two_qubit_gate_id(gate, quimb_gate_name)
                    if n_active_qubits == 2
                    else quimb_gate_name
                ),
                params=params,
                qubits=qubits,
                is_parametrized=is_parametrized,
            )

            if n_active_qubits == 2:
                gate_count_2q += 1
                gate_count_applied_2q += 1
            else:
                gate_count_other += 1
                gate_count_applied_other += 1

        for qubit in range(circuit.nqubits):
            if flush_pending_fast(quimb_circuit, pending_1q, qubit):
                gate_count_applied_1q += 1

        t_total1 = time.perf_counter()
        return quimb_circuit, {
            "stage_qibo_to_quimb_sec": t_total1 - t_total0,
            "two_qubit_apply": two_qubit_apply,
            "gate_count_1q": gate_count_1q,
            "gate_count_2q": gate_count_2q,
            "gate_count_other": gate_count_other,
            "gate_count_measure": gate_count_measure,
            "gate_count_applied_1q": gate_count_applied_1q,
            "gate_count_applied_2q": gate_count_applied_2q,
            "gate_count_applied_other": gate_count_applied_other,
            "gate_count_absorbed_1q": gate_count_absorbed_1q,
        }

    def add_apply_time(stats, dt, n_active_qubits):
        stats["apply_sec"] += dt
        if n_active_qubits == 1:
            stats["apply_1q_sec"] += dt
            stats["gate_count_applied_1q"] += 1
        elif n_active_qubits == 2:
            stats["apply_2q_sec"] += dt
            stats["gate_count_applied_2q"] += 1
        else:
            stats["apply_other_sec"] += dt
            stats["gate_count_applied_other"] += 1

    def flush_pending_one_qubit(quimb_circuit, pending_1q, qubit, stats):
        matrix = pending_1q[qubit]
        if matrix is None:
            return
        t_apply0 = time.perf_counter()
        quimb_circuit.apply_gate(matrix, qubit)
        t_apply1 = time.perf_counter()
        add_apply_time(stats, t_apply1 - t_apply0, 1)
        pending_1q[qubit] = None

    t_total0 = time.perf_counter()
    t_init0 = time.perf_counter()
    quimb_circuit = circuit_type(circuit.nqubits, **circuit_kwargs)
    t_init1 = time.perf_counter()

    metadata_sec = 0.0
    fusion_matrix_sec = 0.0
    absorb_matrix_sec = 0.0
    stats = {
        "apply_sec": 0.0,
        "apply_1q_sec": 0.0,
        "apply_2q_sec": 0.0,
        "apply_other_sec": 0.0,
        "gate_count_applied_1q": 0,
        "gate_count_applied_2q": 0,
        "gate_count_applied_other": 0,
        "gate_count_absorbed_1q": 0,
    }
    gate_count_1q = 0
    gate_count_2q = 0
    gate_count_other = 0
    gate_count_measure = 0
    pending_1q = [None] * circuit.nqubits

    t_loop0 = time.perf_counter()
    for gate in circuit.queue:
        t_meta0 = time.perf_counter()
        gate_name = getattr(gate, "name", None)
        quimb_gate_name = GATE_MAP.get(gate_name, None)
        if quimb_gate_name == "measure":
            gate_count_measure += 1
            metadata_sec += time.perf_counter() - t_meta0
            continue
        if quimb_gate_name is None:
            raise ValueError(f"Gate {gate_name} not supported in Quimb backend.")

        params = getattr(gate, "parameters", ())
        qubits = getattr(gate, "qubits", ())
        n_active_qubits = len(qubits)
        is_parametrized = isinstance(gate, ParametrizedGate) and getattr(
            gate, "trainable", True
        )
        metadata_sec += time.perf_counter() - t_meta0

        if fuse_single_qubit and n_active_qubits == 1:
            t_fuse0 = time.perf_counter()
            matrix = np.asarray(gate.matrix())
            qubit = qubits[0]
            if pending_1q[qubit] is None:
                pending_1q[qubit] = matrix
            else:
                pending_1q[qubit] = matrix @ pending_1q[qubit]
            fusion_matrix_sec += time.perf_counter() - t_fuse0
            gate_count_1q += 1
            continue

        if fuse_single_qubit:
            if absorb_1q_into_2q and n_active_qubits == 2:
                t_absorb0 = time.perf_counter()
                combined_gate, absorbed = absorb_pending_into_two_qubit_gate(
                    gate=gate,
                    pending_1q=pending_1q,
                    qubits=qubits,
                )
                absorb_matrix_sec += time.perf_counter() - t_absorb0
                if combined_gate is not None:
                    t_apply0 = time.perf_counter()
                    quimb_circuit.apply_gate(combined_gate, *qubits)
                    t_apply1 = time.perf_counter()
                    add_apply_time(stats, t_apply1 - t_apply0, 2)
                    stats["gate_count_absorbed_1q"] += absorbed
                    gate_count_2q += 1
                    continue

            for qubit in qubits:
                flush_pending_one_qubit(quimb_circuit, pending_1q, qubit, stats)

        t_apply0 = time.perf_counter()
        apply_direct(
            quimb_circuit=quimb_circuit,
            gate_id=(
                two_qubit_gate_id(gate, quimb_gate_name)
                if n_active_qubits == 2
                else quimb_gate_name
            ),
            params=params,
            qubits=qubits,
            is_parametrized=is_parametrized,
        )
        t_apply1 = time.perf_counter()

        dt_apply = t_apply1 - t_apply0
        add_apply_time(stats, dt_apply, n_active_qubits)
        if n_active_qubits == 1:
            gate_count_1q += 1
        elif n_active_qubits == 2:
            gate_count_2q += 1
        else:
            gate_count_other += 1

    if fuse_single_qubit:
        for qubit in range(circuit.nqubits):
            flush_pending_one_qubit(quimb_circuit, pending_1q, qubit, stats)

    t_loop1 = time.perf_counter()
    t_total1 = time.perf_counter()

    return quimb_circuit, {
        "stage_qibo_to_quimb_sec": t_total1 - t_total0,
        "stage_quimb_circuit_init_sec": t_init1 - t_init0,
        "stage_gate_loop_sec": t_loop1 - t_loop0,
        "stage_gate_metadata_sec": metadata_sec,
        "stage_gate_apply_sec": stats["apply_sec"],
        "stage_gate_apply_1q_sec": stats["apply_1q_sec"],
        "stage_gate_apply_2q_sec": stats["apply_2q_sec"],
        "stage_gate_apply_other_sec": stats["apply_other_sec"],
        "stage_gate_fusion_matrix_sec": fusion_matrix_sec,
        "stage_gate_absorb_matrix_sec": absorb_matrix_sec,
        "gate_count_1q": gate_count_1q,
        "gate_count_2q": gate_count_2q,
        "gate_count_other": gate_count_other,
        "gate_count_measure": gate_count_measure,
        "gate_count_applied_1q": stats["gate_count_applied_1q"],
        "gate_count_applied_2q": stats["gate_count_applied_2q"],
        "gate_count_applied_other": stats["gate_count_applied_other"],
        "gate_count_absorbed_1q": stats["gate_count_absorbed_1q"],
        "two_qubit_apply": two_qubit_apply,
    }


def qibotn_expectation(
    backend_obj,
    circuit,
    observable,
    mode,
    max_bond,
    svd_cutoff,
    expectation_engine,
    profile_conversion,
    fuse_single_qubit,
    absorb_1q_into_2q,
    two_qubit_apply,
):
    import quimb.tensor as qtn

    if expectation_engine == "mps-batched" and mode != "mps":
        raise RuntimeError("--expectation-engine mps-batched requires --mode mps.")

    spec = build_observable_spec(observable, circuit.nqubits)
    circuit_type = qtn.CircuitMPS if mode == "mps" else qtn.Circuit
    circuit_kwargs = {}

    if mode == "mps":
        gate_opts = {}
        if max_bond is not None:
            gate_opts["max_bond"] = max_bond
        if svd_cutoff is not None:
            gate_opts["cutoff"] = svd_cutoff
        circuit_kwargs["gate_opts"] = gate_opts

    # This private conversion hook avoids materializing a dense statevector for
    # expectation runs. Keep this localized so official workload adaptation has
    # a single place to revisit if the backend API changes.
    quimb_circuit, conversion_meta = qibotn_convert_to_quimb(
        backend_obj=backend_obj,
        circuit=circuit,
        circuit_type=circuit_type,
        circuit_kwargs=circuit_kwargs,
        profile_conversion=profile_conversion,
        fuse_single_qubit=fuse_single_qubit,
        absorb_1q_into_2q=absorb_1q_into_2q,
        two_qubit_apply=two_qubit_apply,
    )

    fallback_reason = None
    if expectation_engine in {"mps-batched", "auto"} and mode == "mps":
        t_engine0 = time.perf_counter()
        try:
            value = qibotn_expectation_mps_batched(
                backend_obj=backend_obj,
                quimb_circuit=quimb_circuit,
                spec=spec,
            )
            t_engine1 = time.perf_counter()
            return value, {
                "expectation_engine": "mps-batched",
                "expectation_terms": len(spec["operators_list"]),
                "expectation_engine_fallback": None,
                "stage_expectation_contract_sec": t_engine1 - t_engine0,
                "stage_mps_batched_attempt_sec": t_engine1 - t_engine0,
                **conversion_meta,
            }
        except Exception as exc:
            t_engine1 = time.perf_counter()
            if expectation_engine == "mps-batched":
                raise
            fallback_reason = repr(exc)
            batched_attempt_sec = t_engine1 - t_engine0
    else:
        batched_attempt_sec = None

    t_engine0 = time.perf_counter()
    value = qibotn_expectation_local(
        backend_obj=backend_obj,
        quimb_circuit=quimb_circuit,
        spec=spec,
    )
    t_engine1 = time.perf_counter()
    return value, {
        "expectation_engine": "local",
        "expectation_terms": len(spec["operators_list"]),
        "expectation_engine_fallback": fallback_reason,
        "stage_expectation_contract_sec": t_engine1 - t_engine0,
        "stage_mps_batched_attempt_sec": batched_attempt_sec,
        **conversion_meta,
    }


def run_baseline(
    mode,
    output,
    workload,
    nqubits,
    ngates,
    seed,
    observable,
    max_bond,
    threads,
    optimizer,
    svd_cutoff,
    max_statevector_qubits,
    expectation_engine,
    profile_conversion,
    fuse_single_qubit,
    absorb_1q_into_2q,
    two_qubit_apply,
):
    total_t0 = time.perf_counter()
    stage_times = {}

    stage_t0 = time.perf_counter()
    set_threads(threads)
    stage_times["stage_set_threads_sec"] = time.perf_counter() - stage_t0

    stage_t0 = time.perf_counter()
    if mode in ["tn", "mps"]:
        backend_meta = set_qibotn_cpu_backend(
            mode=mode,
            max_bond=max_bond,
            optimizer=optimizer,
            svd_cutoff=svd_cutoff,
        )
    elif mode == "ref":
        backend_meta = set_qibojit_cpu_backend()
    else:
        raise ValueError("mode must be one of: tn, mps, ref")
    stage_times["stage_set_backend_sec"] = time.perf_counter() - stage_t0

    stage_t0 = time.perf_counter()
    circuit = build_circuit(
        workload=workload,
        nqubits=nqubits,
        ngates=ngates,
        seed=seed,
    )
    stage_times["stage_build_circuit_sec"] = time.perf_counter() - stage_t0

    t0 = time.perf_counter()

    if output == "statevector" and nqubits > max_statevector_qubits:
        raise RuntimeError(
            f"Refusing to materialize a dense statevector for nqubits={nqubits}. "
            f"Increase --max-statevector-qubits above {max_statevector_qubits} "
            "if this is intentional."
        )

    if output == "statevector" and mode in ["tn", "mps"]:
        backend_obj = qibo.get_backend()

        # Important for current qibotn+quimb:
        # return_array=True is required to obtain a dense statevector.
        result = backend_obj.execute_circuit(
            circuit,
            return_array=True,
        )
        raw_state = extract_state(result)
        value, original_shape = canonicalize_state(raw_state, nqubits)
        value_meta = {
            "state_original_shape": list(original_shape),
            "state_shape": list(value.shape),
            "state_dtype": str(value.dtype),
            "state_norm": float(np.linalg.norm(value)),
        }
    elif output == "statevector":
        result = circuit()
        raw_state = extract_state(result)
        value, original_shape = canonicalize_state(raw_state, nqubits)
        value_meta = {
            "state_original_shape": list(original_shape),
            "state_shape": list(value.shape),
            "state_dtype": str(value.dtype),
            "state_norm": float(np.linalg.norm(value)),
        }
    elif output == "expectation" and mode in ["tn", "mps"]:
        backend_obj = qibo.get_backend()
        value, expectation_meta = qibotn_expectation(
            backend_obj=backend_obj,
            circuit=circuit,
            observable=observable,
            mode=mode,
            max_bond=max_bond,
            svd_cutoff=svd_cutoff,
            expectation_engine=expectation_engine,
            profile_conversion=profile_conversion,
            fuse_single_qubit=fuse_single_qubit,
            absorb_1q_into_2q=absorb_1q_into_2q,
            two_qubit_apply=two_qubit_apply,
        )
        value_meta = {
            "observable": observable,
            "expectation_value": to_real_scalar(value),
        }
        value_meta.update(expectation_meta)
    elif output == "expectation":
        value = reference_expectation(circuit, observable, nqubits)
        value_meta = {
            "observable": observable,
            "expectation_value": to_real_scalar(value),
            "expectation_engine": "dense-reference",
            "expectation_terms": len(build_observable_spec(observable, nqubits)["operators_list"]),
            "expectation_engine_fallback": None,
        }
    else:
        raise ValueError("output must be one of: statevector, expectation")

    t1 = time.perf_counter()
    stage_times["stage_execute_sec"] = t1 - t0
    stage_times["stage_total_sec"] = time.perf_counter() - total_t0

    meta = {
        "mode": mode,
        "output": output,
        "workload": workload,
        "nqubits": nqubits,
        "requested_ngates": ngates,
        "actual_gate_count": gate_count(circuit),
        "seed": seed,
        "elapsed_sec": t1 - t0,
        "backend": backend_meta,
        "threads_arg": threads,
    }
    meta.update(stage_times)
    meta.update(value_meta)

    return value, meta


def add_arguments(parser):
    parser.add_argument("--mode", choices=["tn", "mps", "ref"], default="tn")
    parser.add_argument("--output", choices=["statevector", "expectation"], default="statevector")
    parser.add_argument("--workload", choices=["qft", "random-local"], default="qft")
    parser.add_argument("--nqubits", type=int, default=8)
    parser.add_argument("--ngates", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--observable", choices=["z0", "zz-chain"], default="z0")
    parser.add_argument("--max-bond", type=int, default=None)
    parser.add_argument("--svd-cutoff", type=float, default=1e-10)
    parser.add_argument(
        "--expectation-engine",
        choices=["local", "mps-batched", "auto"],
        default="local",
        help=(
            "Expectation contraction path. 'local' preserves the original "
            "per-term local_expectation loop; 'mps-batched' uses quimb's MPS "
            "batched local expectation helper; 'auto' tries mps-batched for "
            "MPS and falls back to local."
        ),
    )
    parser.add_argument(
        "--profile-conversion",
        action="store_true",
        help=(
            "Use an instrumented Qibo-to-quimb conversion loop that records "
            "gate metadata and apply_gate timing. Default conversion path is "
            "unchanged when this flag is not set."
        ),
    )
    parser.add_argument(
        "--fuse-single-qubit",
        action="store_true",
        help=(
            "Fuse pending single-qubit gates per qubit before Qibo-to-quimb "
            "MPS conversion. This is an algebraically equivalent circuit "
            "preprocessing path and is disabled by default."
        ),
    )
    parser.add_argument(
        "--absorb-1q-into-2q",
        action="store_true",
        help=(
            "Absorb pending single-qubit matrices into the next two-qubit "
            "gate matrix. Implies --fuse-single-qubit."
        ),
    )
    parser.add_argument(
        "--two-qubit-apply",
        choices=["name", "matrix"],
        default="name",
        help=(
            "How to feed ordinary two-qubit gates to quimb during the custom "
            "conversion path. 'name' preserves the original quimb gate name; "
            "'matrix' passes gate.matrix() directly."
        ),
    )
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--optimizer", type=str, default="auto-hq")
    parser.add_argument("--check-ref", action="store_true")
    parser.add_argument("--check-ref-max-qubits", type=int, default=20)
    parser.add_argument("--max-statevector-qubits", type=int, default=30)
    parser.add_argument("--save-state", type=str, default=None)
    parser.add_argument(
        "--batch-json",
        type=str,
        default=None,
        help=(
            "Run multiple cases from a JSONL file. Each line is an object whose "
            "fields override the normal single-case arguments. Use '-' for stdin."
        ),
    )


def ensure_cpu_only():
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        raise RuntimeError(
            "This baseline is CPU-only, but CUDA_VISIBLE_DEVICES is not empty. "
            "Run: export CUDA_VISIBLE_DEVICES=\"\""
        )


def run_case(args):
    value, meta = run_baseline(
        mode=args.mode,
        output=args.output,
        workload=args.workload,
        nqubits=args.nqubits,
        ngates=args.ngates,
        seed=args.seed,
        observable=args.observable,
        max_bond=args.max_bond,
        threads=args.threads,
        optimizer=args.optimizer,
        svd_cutoff=args.svd_cutoff,
        max_statevector_qubits=args.max_statevector_qubits,
        expectation_engine=args.expectation_engine,
        profile_conversion=args.profile_conversion,
        fuse_single_qubit=args.fuse_single_qubit,
        absorb_1q_into_2q=args.absorb_1q_into_2q,
        two_qubit_apply=args.two_qubit_apply,
    )

    meta["versions"] = versions()

    if args.check_ref and args.mode != "ref":
        if args.nqubits <= args.check_ref_max_qubits:
            ref_value, ref_meta = run_baseline(
                mode="ref",
                output=args.output,
                workload=args.workload,
                nqubits=args.nqubits,
                ngates=args.ngates,
                seed=args.seed,
                observable=args.observable,
                max_bond=args.max_bond,
                threads=args.threads,
                optimizer=args.optimizer,
                svd_cutoff=args.svd_cutoff,
                max_statevector_qubits=args.max_statevector_qubits,
                expectation_engine=args.expectation_engine,
                profile_conversion=False,
                fuse_single_qubit=False,
                absorb_1q_into_2q=False,
                two_qubit_apply=args.two_qubit_apply,
            )

            if args.output == "statevector":
                max_abs_error = float(np.max(np.abs(value - ref_value)))
                allclose = bool(np.allclose(value, ref_value, atol=1e-8, rtol=1e-8))
                reference = {
                    "backend": "qibojit-numba",
                    "elapsed_sec": ref_meta["elapsed_sec"],
                    "state_original_shape": ref_meta["state_original_shape"],
                    "max_abs_error": max_abs_error,
                    "allclose_1e-8": allclose,
                }
            else:
                abs_error = abs(to_real_scalar(value) - to_real_scalar(ref_value))
                reference = {
                    "backend": "qibojit-numba",
                    "elapsed_sec": ref_meta["elapsed_sec"],
                    "expected_expectation_value": ref_meta["expectation_value"],
                    "abs_error": float(abs_error),
                    "isclose_1e-8": bool(abs_error <= 1e-8),
                }

            meta["reference"] = reference
        else:
            meta["reference"] = {
                "skipped": True,
                "reason": (
                    f"nqubits={args.nqubits} exceeds "
                    f"--check-ref-max-qubits={args.check_ref_max_qubits}"
                ),
            }

    if args.save_state is not None:
        if args.output != "statevector":
            raise RuntimeError("--save-state is only valid with --output statevector")
        np.save(args.save_state, value)
        meta["saved_state"] = args.save_state

    return meta


def read_batch_cases(path):
    if path == "-":
        lines = sys.stdin
    else:
        lines = open(path, "r", encoding="utf-8")

    try:
        for line_number, line in enumerate(lines, start=1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            try:
                case = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on batch line {line_number}: {exc}") from exc
            if not isinstance(case, dict):
                raise ValueError(f"Batch line {line_number} must be a JSON object.")
            yield line_number, case
    finally:
        if path != "-":
            lines.close()


def case_value(case, name, default):
    dashed = name.replace("_", "-")
    return case.get(name, case.get(dashed, default))


def optional_int(value):
    if value is None or value == "":
        return None
    return int(value)


def optional_float(value):
    if value is None or value == "":
        return None
    return float(value)


def bool_value(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def batch_case_args(base_args, line_number, case):
    args = argparse.Namespace(**vars(base_args))
    args.batch_json = None
    args.mode = str(case_value(case, "mode", args.mode))
    args.output = str(case_value(case, "output", args.output))
    args.workload = str(case_value(case, "workload", args.workload))
    args.nqubits = int(case_value(case, "nqubits", args.nqubits))
    args.ngates = optional_int(case_value(case, "ngates", args.ngates))
    args.seed = int(case_value(case, "seed", args.seed))
    args.observable = str(case_value(case, "observable", args.observable))
    args.max_bond = optional_int(case_value(case, "max_bond", args.max_bond))
    args.svd_cutoff = optional_float(case_value(case, "svd_cutoff", args.svd_cutoff))
    args.expectation_engine = str(
        case_value(case, "expectation_engine", args.expectation_engine)
    )
    args.profile_conversion = bool_value(
        case_value(case, "profile_conversion", args.profile_conversion)
    )
    args.fuse_single_qubit = bool_value(
        case_value(case, "fuse_single_qubit", args.fuse_single_qubit)
    )
    args.absorb_1q_into_2q = bool_value(
        case_value(case, "absorb_1q_into_2q", args.absorb_1q_into_2q)
    )
    args.two_qubit_apply = str(case_value(case, "two_qubit_apply", args.two_qubit_apply))
    args.threads = optional_int(case_value(case, "threads", args.threads))
    args.optimizer = str(case_value(case, "optimizer", args.optimizer))
    args.check_ref = bool_value(case_value(case, "check_ref", args.check_ref))
    args.check_ref_max_qubits = int(
        case_value(case, "check_ref_max_qubits", args.check_ref_max_qubits)
    )
    args.max_statevector_qubits = int(
        case_value(case, "max_statevector_qubits", args.max_statevector_qubits)
    )
    args.save_state = case_value(case, "save_state", args.save_state)
    args.case_name = case_value(case, "case_name", case_value(case, "name", None))
    args.batch_line = line_number
    return args


def validate_args(args):
    if args.mode not in {"tn", "mps", "ref"}:
        raise ValueError("mode must be one of: tn, mps, ref")
    if args.output not in {"statevector", "expectation"}:
        raise ValueError("output must be one of: statevector, expectation")
    if args.workload not in {"qft", "random-local"}:
        raise ValueError("workload must be one of: qft, random-local")
    if args.observable not in {"z0", "zz-chain"}:
        raise ValueError("observable must be one of: z0, zz-chain")
    if args.expectation_engine not in {"local", "mps-batched", "auto"}:
        raise ValueError("expectation_engine must be one of: local, mps-batched, auto")
    if args.expectation_engine == "mps-batched" and args.mode != "mps":
        raise ValueError("expectation_engine=mps-batched requires mode=mps")
    if args.two_qubit_apply not in {"name", "matrix"}:
        raise ValueError("two_qubit_apply must be one of: name, matrix")


def run_batch(base_args):
    for index, (line_number, case) in enumerate(
        read_batch_cases(base_args.batch_json),
        start=1,
    ):
        args = batch_case_args(base_args, line_number, case)
        validate_args(args)
        meta = run_case(args)
        meta["batch"] = {
            "index": index,
            "line": line_number,
            "case_name": args.case_name,
        }
        print(json.dumps(meta, separators=(",", ":")), flush=True)


def main():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    ensure_cpu_only()
    if args.batch_json is not None:
        run_batch(args)
    else:
        validate_args(args)
        print(json.dumps(run_case(args), indent=2))


if __name__ == "__main__":
    main()
