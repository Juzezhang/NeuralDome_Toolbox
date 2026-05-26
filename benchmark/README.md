# HODome monocular HOI benchmark — evaluation code

This folder holds the **metric-computation code** used in the HODome single-view human–object
interaction benchmark. The full protocol, results, and fairness audit are written up in
[`../docs/benchmark_monocular_hoi.md`](../docs/benchmark_monocular_hoi.md).

## What's here (`eval/`)

| file | what it computes |
|---|---|
| `metrics.py` | MPJPE, PA-MPJPE (Procrustes), object Chamfer, V2V, p.V2V; `to_camera` (world→view26) |
| `behave_chamfer.py` | **BEHAVE-protocol Human-CD / Object-CD (cm)** — the `chamfer_distance` + `rigid_transform_3D` copied **verbatim** from CONTHO/CHORE, applied with one uniform judge protocol |
| `surface.py` | area-weighted surface sampling (N=10000, faithful to CHORE/CONTHO) |
| `faces.py` | resolves human faces by topology (SMPL 6890 / SMPL-X 10475) and object faces from the result file or GT template |
| `run_eval.py` | per-seq driver: transforms GT to the view-26 camera frame, then scores a method's predictions |

## How metrics are computed (summary)

1. **Coordinate frame.** HODome GT (SMPL-X mesh + posed object) is in the world (ground-aligned)
   frame; predictions are in each method's own camera. Before any metric, GT is transformed into the
   **view-26 camera frame** via the ground calibration extrinsic: `X_cam = X_world @ R^T + T`.
2. **Human.** MPJPE (root-aligned) and **PA-MPJPE** (Procrustes) over the 22 SMPL body joints.
3. **Object / scene.** Symmetric Chamfer on **10000 area-weighted surface samples**; V2V / p.V2V on
   the combined human+object cloud.
4. **BEHAVE Human-/Object-CD (cm).** Scaled-rigid align (Umeyama) estimated from the 22 corresponded
   body joints — the only cross-topology correspondence, since methods output SMPL / SMPL-H / SMPL-X
   / point clouds — then the methods' **verbatim** bidirectional Chamfer on surface samples.

Run: `python eval/run_eval.py --method {METHOD}` (expects per-seq prediction npz in the common
format documented in the writeup). `--sanity` uses GT-as-pred and returns ~0 on every metric.
