# Legacy experiment snapshots (frozen)

Historical monoliths, preserved **verbatim** (except local interpreter paths
in docstrings) for provenance — they are the exact code that produced the
Phase A / Phase B results reported in `docs/`:

| File | Phase |
|---|---|
| `sallora.py` | Phase A — SALLoRA vs LoRA (negative result) |
| `gatelora.py` | Phase B — early routing / residual gating (v1) |
| `gatelora_v2.py` | Variant search — GateConcat emerges (v2) |

These files are **not maintained** and duplicate code on purpose: editing a
frozen snapshot would break the link between the reported numbers and the
code that produced them. Current, maintained code is the `gateconcat`
package plus `gatelora_v3.py` / `gatelora_v4.py` at the repository root.
