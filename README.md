# Qibotn

The tensor network translation module for Qibo to support large-scale simulation of quantum circuits and acceleration.

## ASC26 CPU-only Experiment Notes

This repository also contains CPU-only baseline tooling for the ASC26 QiboTN
task. The benchmark path is intentionally conservative:

- QiboTN/quimb remains the simulation backend.
- Workload and circuit definitions are not modified by the C++ tools.
- GPU, cuQuantum, cuTensorNet, MPI, and NCCL paths are not used for the CPU-only
  baseline.
- Large 30-qubit runs should use expectation output rather than dense
  statevector output.

The C++ files in `tools/` only replace low-risk experiment plumbing:

- `tools/qtn_runner.cpp` runs predefined benchmark suites, sets CPU-only
  environment variables, captures `/usr/bin/time -v`, and writes summaries.
- `tools/qtn_summarize.cpp` aggregates existing `*.stdout.json` and
  `*.stderr.log` files into CSV.

They do not implement an independent quantum simulator and do not bypass
QiboTN for the actual tensor-network calculation.

Build on Linux:

```sh
chmod +x tools/build_cpp_tools.sh
tools/build_cpp_tools.sh
```

Example:

```sh
tools/qtn_runner --suite final --repeats 3 --optimizer auto-hq \
  --log-dir logs_cpp_final_auto_hq

tools/qtn_summarize logs_cpp_final_auto_hq > logs_cpp_final_auto_hq/summary_cpp.csv
```

### Reversible Optimization Layers

The experiment code is organized so each optimization layer can be disabled
independently if the competition interpretation becomes stricter.

| Layer | How to enable | What it changes | Rollback |
| --- | --- | --- | --- |
| L0 single-case baseline | `python qtn_baseline.py ...` | Original one-process-per-case path | Default path |
| L1 batch baseline | `python qtn_baseline.py --batch-json cases.jsonl` | Reuses one Python import/backend process for many cases | Do not use `--batch-json` |
| L2 C++ runner | `tools/qtn_runner ...` | Replaces shell loops and summary plumbing | Use `sweep_qft_baseline.py` |
| L2 batch runner | `tools/qtn_runner --batch ...` | Lets C++ runner call one Python batch process | Drop `--batch` |
| L3a MPS batched expectation | `--expectation-engine mps-batched` | Uses quimb's MPS batched local-expectation helper for multi-term observables | Use `--expectation-engine local` |
| L3b single-qubit fusion | `--fuse-single-qubit` | Algebraically fuses pending 1-qubit gates before MPS application | Drop `--fuse-single-qubit` |
| L3c absorb 1q into 2q | `--absorb-1q-into-2q` | Absorbs pending 1-qubit matrices into the following 2-qubit matrix | Drop `--absorb-1q-into-2q` |
| L3d 2q apply input mode | `--two-qubit-apply matrix` | Tests direct 2-qubit matrix application instead of quimb gate-name lookup | Use `--two-qubit-apply name` |
| L3 native CPU helper | not implemented | Would optimize CPU tensor/MPS kernels | Keep disabled unless validated |

The default expectation engine is `local`, which preserves the original
per-term `local_expectation` loop. The optional `mps-batched` engine still uses
QiboTN/quimb and keeps the same observable definition, but asks quimb to compute
all local MPS terms together. Use `auto` to try the batched path and fall back to
`local` if the installed quimb API rejects it.

Batch examples:

```sh
cat > cases.jsonl <<'EOF'
{"mode":"mps","output":"expectation","workload":"random-local","nqubits":30,"ngates":10000,"observable":"z0","max_bond":64,"svd_cutoff":"1e-8","threads":8,"optimizer":"auto-hq"}
{"mode":"mps","output":"expectation","workload":"random-local","nqubits":30,"ngates":10000,"observable":"zz-chain","max_bond":64,"svd_cutoff":"1e-8","threads":8,"optimizer":"auto-hq"}
EOF

python qtn_baseline.py --batch-json cases.jsonl > batch.stdout.jsonl

tools/qtn_runner --suite final --batch --repeats 3 \
  --optimizer auto-hq \
  --log-dir logs_cpp_batch_final_auto_hq

tools/qtn_runner --suite final --batch --repeats 3 \
  --optimizer auto-hq \
  --expectation-engine mps-batched \
  --log-dir logs_l3a_batch_final_mps_batched
```

Do not use `--batch` for the thread sweep, because each thread count should be
set before Python starts:

```sh
tools/qtn_runner --suite threads --optimizer auto-hq \
  --log-dir logs_cpp_threads_auto_hq
```

Hotspot timing fields are emitted in every `summary.csv`:

- `stage_set_threads_sec`
- `stage_set_backend_sec`
- `stage_build_circuit_sec`
- `stage_execute_sec`
- `stage_qibo_to_quimb_sec`
- `stage_expectation_contract_sec`
- `stage_total_sec`

For MPS expectation workloads, `stage_qibo_to_quimb_sec` includes the expensive
Qibo circuit to quimb/MPS construction path, including applying gates to the
MPS representation. This is the first place to inspect before changing lower
level kernels.

If `stage_gate_apply_1q_sec` is large, test the reversible single-qubit fusion
path:

```sh
tools/qtn_runner --suite correctness --batch \
  --optimizer auto-hq \
  --profile-conversion \
  --fuse-single-qubit \
  --log-dir logs_l3b_correctness_fuse1q

tools/qtn_runner --suite final --batch --repeats 3 \
  --optimizer auto-hq \
  --profile-conversion \
  --fuse-single-qubit \
  --log-dir logs_l3b_final_fuse1q

tools/qtn_runner --suite correctness --batch \
  --optimizer auto-hq \
  --fuse-single-qubit \
  --absorb-1q-into-2q \
  --log-dir logs_l3c_correctness_absorb1q2q

tools/qtn_runner --suite final --batch --repeats 3 \
  --optimizer auto-hq \
  --fuse-single-qubit \
  --absorb-1q-into-2q \
  --log-dir logs_l3c_final_absorb1q2q

tools/qtn_runner --suite final --batch --repeats 3 \
  --optimizer auto-hq \
  --fuse-single-qubit \
  --absorb-1q-into-2q \
  --two-qubit-apply matrix \
  --log-dir logs_l3d_final_2q_matrix
```

## Supported Computation

Tensor Network Types:

- Tensornet (TN)
- Matrix Product States (MPS)

Tensor Network contractions to:

- dense vectors
- expecation values of given Pauli string

The supported HPC configurations are:

- single-node CPU
- single-node GPU or GPUs
- multi-node multi-GPU with Message Passing Interface (MPI)
- multi-node multi-GPU with NVIDIA Collective Communications Library (NCCL)

Currently, the supported tensor network libraries are:

- [cuQuantum](https://github.com/NVIDIA/cuQuantum), an NVIDIA SDK of optimized libraries and tools for accelerating quantum computing workflows.
- [quimb](https://quimb.readthedocs.io/en/latest/), an easy but fast python library for ‘quantum information many-body’ calculations, focusing primarily on tensor networks.

## Installation

To get started:

```sh
pip install qibotn
```

to install the tools and dependencies. A few extras are provided, check `pyproject.toml` in
case you need them.

<!-- TODO: describe extras, after Poetry adoption and its groups -->

## Contribute

To contribute, please install using poetry:

```sh
git clone https://github.com/qiboteam/qibotn.git
cd qibotn
poetry install
```

## Sample Codes

### Single-Node Example

The code below shows an example of how to activate the Cuquantum TensorNetwork backend of Qibo.

```py
import numpy as np
from qibo import Circuit, gates
import qibo

# Below shows how to set the computation_settings
# Note that for MPS_enabled and expectation_enabled parameters the accepted inputs are boolean or a dictionary with the format shown below.
# If computation_settings is not specified, the default setting is used in which all booleans will be False.
# This will trigger the dense vector computation of the tensornet.

computation_settings = {
    "MPI_enabled": False,
    "MPS_enabled": {
        "qr_method": False,
        "svd_method": {
            "partition": "UV",
            "abs_cutoff": 1e-12,
        },
    },
    "NCCL_enabled": False,
    "expectation_enabled": False,
}


qibo.set_backend(
    backend="qibotn", platform="cutensornet", runcard=computation_settings
)  # cuQuantum
# qibo.set_backend(backend="qibotn", platform="qutensornet", runcard=computation_settings) #quimb


# Construct the circuit
c = Circuit(2)
# Add some gates
c.add(gates.H(0))
c.add(gates.H(1))

# Execute the circuit and obtain the final state
result = c()

print(result.state())
```

Other examples of setting the computation_settings

```py
# Expectation computation with specific Pauli String pattern
computation_settings = {
    "MPI_enabled": False,
    "MPS_enabled": False,
    "NCCL_enabled": False,
    "expectation_enabled": {
        "pauli_string_pattern": "IXZ",
    },
}

# Dense vector computation using multi node through MPI
computation_settings = {
    "MPI_enabled": True,
    "MPS_enabled": False,
    "NCCL_enabled": False,
    "expectation_enabled": False,
}
```

### Multi-Node Example

Multi-node is enabled by setting either the MPI or NCCL enabled flag to True in the computation settings. Below shows the script to launch on 2 nodes with 2 GPUs each. $node_list contains the IP of the nodes assigned.

```sh
mpirun -n 4 -hostfile $node_list python test.py
```
