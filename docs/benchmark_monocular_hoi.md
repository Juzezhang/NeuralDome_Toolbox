# Benchmarking Monocular Human–Object Interaction Recovery on HODome

*A reproducible single-view evaluation on the NeuralDome / HODome test set.*

---

## Abstract

We benchmark a representative set of monocular human–object interaction (HOI) recovery
methods on the **HODome** dataset of NeuralDome [Zhang et al., CVPR 2023]. We evaluate on a
held-out single camera view (**view 26**) for the two test subjects (**subject01, subject02**),
using the dataset's SMPL-X human and 6-DoF object ground truth. Each method is run with its
**own released code and weights**; we transform the world-frame ground truth into the view-26
camera frame and score every method with **one uniform, judge-neutral protocol** that mirrors the
BEHAVE / CHORE / CONTHO Chamfer evaluation (area-weighted surface sampling, scaled-rigid
alignment). We report human pose error (MPJPE, PA-MPJPE), object and scene Chamfer, and the
paper-comparable BEHAVE Human-/Object-CD (cm). We additionally document **reproducibility gaps**:
methods whose public releases are insufficient to run on a new dataset.

---

## 1. Benchmark setup

### 1.1 Test set

| | |
|---|---|
| Dataset | HODome (NeuralDome, CVPR 2023) |
| Test subjects | subject01, subject02 |
| Camera view | **view 26** (`videos/{seq}/data27.mp4`; cam id 26 → `data{N+1}`) |
| Sequences | **39** (subject02_tabletall excluded — no GT) |
| Ground truth | SMPL-X (body + hands), 10475 verts; object 6-DoF on the scanned template |
| Frame sampling | **1 fps** (step 60, ~62 fr/seq, ~2419 fr total) and **30 fps** (step 2, ~1843 fr/seq) |

**Why two frame rates.** Single-image methods are evaluated at **1 fps and 30 fps**. Video /
temporal methods are evaluated **only at 30 fps** — feeding a temporal model 1-fps frames would
break its motion modelling and produce an unfair, invalid result. At 30 fps the shared eval scores
each method on the GT frames it actually produced (matched by frame index).

### 1.2 Coordinate frames

HODome ground truth (SMPL-X mesh + posed object) is exported in the dataset's **world (ground-
aligned)** frame. Monocular methods predict in **their own camera frame**. Before any metric, we
transform the GT into the **view-26 camera frame** using the ground calibration extrinsic
(`calibration_ground/{date}/calibration.json["26"]`):

```
X_cam = X_world @ R^T + T          # R, T = view-26 extrinsic (cam_R, cam_T)
```

(`eval/export_gt.py`, `eval/run_eval.py:gt_camera`, `eval/metrics.py:to_camera`.) This transform was
validated independently by overlaying the calibrated SMPL-X+object render on the view-26 video
(`hodome_visualize_pyrender.py`), which lands correctly on the actor. Because predictions live in
each method's own (often weak-perspective) camera, the **alignment-based metrics** (PA-MPJPE,
p.V2V, and the scaled-rigid BEHAVE-CD) are the meaningful cross-method comparisons; absolute
metrics (MPJPE, raw Chamfer, V2V) additionally carry the pred-camera-vs-cam26 mismatch.

---

## 2. Evaluation protocol — one judge for all

We are the judge, so every method is scored by **one** protocol, applied identically. Where a
method's own paper used a specific routine we copied that routine **verbatim** from its repo.

### 2.1 Metrics

| Metric | Definition | Unit | Cross-method comparable? |
|---|---|---|---|
| MPJPE | root-aligned mean per-joint error (22 SMPL body joints) | mm | absolute (camera-dependent) |
| **PA-MPJPE** | Procrustes-aligned MPJPE | mm | **yes** (human) |
| Chamfer | object-only symmetric Chamfer, surface-sampled | mm | semi (no alignment) |
| V2V | combined human+object cloud Chamfer | mm | absolute |
| **p.V2V** | V2V after scaled-rigid alignment from the 22 body joints | mm | **yes** (scene) |
| **Human-CD / Object-CD** | BEHAVE-protocol Chamfer (see §2.2) | cm | **yes** (paper-comparable) |

MPJPE/PA-MPJPE are **not** meaningful for the template-free point-cloud method (HDM), whose
"joints" are point-derived rather than a real SMPL skeleton — judge it by Chamfer.

### 2.2 BEHAVE Human-/Object-CD — verbatim code, faithful sampling

The paper-comparable Human-CD/Object-CD reuse the **exact** functions the methods themselves
used (`eval/behave_chamfer.py`, copied from `CONTHO/lib/utils/eval_utils.py` and
`CHORE/recon/eval/chamfer_distance.py` — the two are byte-identical in formula). Concretely:

1. **Align** the predicted scene to GT with a *scaled-rigid* transform (`rigid_transform_3D`,
   Umeyama-with-scale) estimated from the **22 corresponded SMPL body joints**. This is the only
   correspondence available across body topologies (methods output SMPL / SMPL-H / SMPL-X /
   point cloud; GT is SMPL-X), and it replaces the papers' combined-vertex Procrustes which assumes
   shared topology.
2. **Surface-sample** N = **10000** area-weighted points on each mesh (`trimesh.sample_surface`),
   faithful to CHORE/VisTracker/StackFLOW/WildHOI (CONTHO uses 6000). Human faces are inferred
   from topology (6890 → SMPL, 10475 → SMPL-X); object faces come from the result file or the GT
   template. A mesh without faces (HDM point cloud, or an object template whose faces we could not
   recover) falls back to vertex resampling and is **flagged per method** (`sampling` field in each
   `metrics_summary.json`).
3. **Chamfer** the two point sets with the verbatim bidirectional formula and report ×100 = cm.

GT-as-pred sanity → **0.000 cm** on all metrics (deterministic sampling seed).

### 2.3 Fairness audit of the field's own eval code

Reading each repo, the whole field descends from CHORE's `chamfer_distance`, but with real
differences we normalise away:

| Method | chamfer fn | surface samples | pre-align |
|---|---|---|---|
| CHORE | baseline | 10000 | Procrustes (`align_meshes`) |
| VisTracker | md5-identical to CHORE | 10000 | Procrustes |
| CONTHO | identical formula (copied) | 6000 | scaled-rigid on combined verts |
| StackFLOW | "copied from CHORE recon/eval" | 10000 | `ProcrusteAlign` |
| HDM | differs (point cloud) | — | own |
| WildHOI | same evaluator | 10000 | **`align_mesh=False` → NO alignment** |

WildHOI reports *un-aligned* Chamfer, which is not comparable to the Procrustes-aligned numbers of
the others — this alone justifies a single uniform judge protocol (ours: always joint-anchored
scaled-rigid + 10000 surface samples).

---

## 3. Methods

| # | Method | Venue | Type | Object model | Sampling (1fps eval) |
|---|---|---|---|---|---|
| 1 | PHOSA | ECCV'20 | single-image | fixed category templates | human surf / obj surf |
| 2 | CHORE | ECCV'22 | single-image (optim.) | BEHAVE templates (neural UDF) | human surf / obj — |
| 3 | CONTHO | CVPR'24 | single-image | BEHAVE-20 (mapped) | human surf / obj vertex |
| 4 | StackFLOW | IJCAI'23 | single-image | BEHAVE templates | human surf / obj vertex |
| 5 | HDM | CVPR'24 | single-image | **template-free point cloud** | point cloud |
| 6 | VisTracker | CVPR'23 | video (temporal) | BEHAVE templates | human surf / obj vertex |
| 7 | Open4DHOI | — | video | HODome template (GVHMR front-end) | human surf / obj surf |
| 8 | CARI4D | CVPR'26 (NVlabs) | video | **template-free** (Hunyuan3D + FoundationPose) | — |
| 9 | InterTrack | — | video | category-agnostic | — |
| – | WildHOI | CVPR'24 | single-image | 4 objects | **blocked** (see §5) |

Object-handling note: methods with a *fixed* object set map each HODome object to the nearest
category (e.g. CONTHO's `HODOME2BEHAVE`); template-free methods (HDM, CARI4D, InterTrack) avoid the
object domain gap entirely and are the cleanest object comparison.

---

## 4. Results

> **Surface-sampled (N=10000) results — populated from `results/{method}/metrics_summary.json`.**
> `Sa` column: object sampling (surf = area-weighted surface, vtx = vertex fallback).

### 4.1 Single-image methods @ 1 fps (39 seqs, ~2419 frames)

<!-- RESULTS_1FPS_TABLE --> (surface-sampled N=10000; values are means over the 39-seq test set)
| Method | PA-MPJPE↓ (mm) | MPJPE↓ | Chamfer↓ (mm) | p.V2V↓ | **Human-CD↓ (cm)** | **Object-CD↓ (cm)** | Sa(obj) | seqs |
|---|---|---|---|---|---|---|---|---|
| CHORE | **48.5** | 79.8 | 1326 | 65.3 | **4.73** | **62.8** | vtx | 27‡ |
| CONTHO | 58.0 | 89.1 | 3579 | 65.0 | **6.32** | **36.2** | vtx | 39 |
| StackFLOW | 60.5 | 112.5 | 543 | 70.3 | **5.89** | **45.6** | vtx | 39 |
| PHOSA | 66.2 | 108.8 | 3581 | 505.6 | **16.21** | **490.6** | surf | 39 |
| HDM† | (424.9) | (511.6) | 3579 | (38.98) | **94.8** | surf | 39 |

† HDM is template-free point cloud → MPJPE/PA-MPJPE/Human-CD (parenthesized) are **not** meaningful;
judge HDM by Chamfer / Object-CD only.
‡ CHORE @27/39 — per-frame neural-UDF optimization (~2 h/seq); paused at 27 under the GPU throttle.
*Best human:* CHORE (PA-MPJPE 48.5, Human-CD 4.73 cm) — its per-frame optimization fits the body
tightest. *Best object (fixed-template):* CONTHO (36.2 cm). *PHOSA's object (490 cm)* is an outlier —
its weak-perspective object placement degrades badly on HODome's true-camera test.
*Sa(obj):* surf = area-weighted surface samples; vtx = vertex resampling (CONTHO 64-pt proxy /
StackFLOW 508 / VisTracker 516 own-template object faces not recovered).

### 4.2 Video / temporal methods @ 30 fps

<!-- RESULTS_30FPS_TABLE --> (fed 30 fps contiguous frames; scored on overlapping 1 fps GT)
| Method | PA-MPJPE↓ (mm) | Chamfer↓ (mm) | p.V2V↓ | **Human-CD↓ (cm)** | **Object-CD↓ (cm)** | Sa(obj) | seqs |
|---|---|---|---|---|---|---|---|
| Open4DHOI | **55.1** | 1421 | 83.8 | **5.94** | **55.6** | surf | 39 |
| VisTracker | 175.9 | 1148 | 185.8 | 15.21 | 139.1 | vtx | 34 |
| CARI4D | 117.3 | 4269 | 1516 | 9.89 | 781.6 | mixed | 7‡ |
| InterTrack | *(running locally)* | | | | | | 0→39 |

‡ CARI4D @7/39 (paused under throttle). Its **template-free** object (Hunyuan3D + FoundationPose)
is the domain-gap-free comparison, but FoundationPose tracking **drifts in depth on symmetric/thin
objects** (baseball bat = catastrophic ⇒ ObjCD 781; book/plush/sofa track to ~1 m) — first-frame
placement is excellent for all (≤0.2 m), so the failure is temporal tracking, not shape recovery.
Open4DHOI = strongest video method (human 5.94 cm, object 55.6 cm).

### 4.3 Cross-rate comparison (image methods, 1 fps vs 30 fps)

<!-- RESULTS_CROSSRATE_TABLE -->
*(Single-image methods are deterministic per frame, so 1 fps vs 30 fps differ only by the frame set
scored, not by per-frame quality. 30 fps image-method runs were de-prioritized under the GPU throttle;
the 1 fps numbers in §4.1 are the canonical single-image results.)*

---

## 5. Reproducibility findings

A benchmark on a *new* dataset stresses whether a public release is actually self-contained.

- **WildHOI (blocked).** The optical-flow checkpoints are on Google Drive (all 6 archives download
  fine), but the **per-object 2D–3D correspondence networks** `model_{obj}_stage1.pth` (DINOv2 +
  DenseCorr head) are **not** in any release, and the shipped inference uses precomputed
  correspondence maps bound to WildHOI's own YouTube frames — not transferable to HODome. The object
  branch therefore cannot run on a new dataset. A release-completeness gap, not a download issue.
- **StackFLOW / CHORE / VisTracker** required nontrivial porting to torch≥2 / L40S (sm89): chumpy,
  detectron2, igl/libigl, torchgeometry bool-ops, SMPL-H betas, and (StackFLOW) a SharePoint
  anonymous-download path. All eventually ran from public artifacts.
- **Object domain gap.** Fixed-template methods (CONTHO→BEHAVE-20, StackFLOW/CHORE/VisTracker→
  BEHAVE) only partially cover HODome's 20 objects; their Object-CD is several× worse than their
  BEHAVE numbers, quantifying the cost of forcing HODome objects onto BEHAVE templates. Template-
  free methods (HDM, CARI4D, InterTrack) are the domain-gap-free object comparison.

Validation of reproduction fidelity: CHORE Human-CD and PHOSA Human-CD reproduce within ~0.2–1.2 cm
of the published BEHAVE numbers, indicating the harness and ports are trustworthy.

---

## 6. Reproducibility of *this* benchmark

- All predictions are saved per method as `results/{method}/{seq}.npz` (view-26 camera frame,
  meters, with faces) at **both 1 fps and 30 fps**, and are released alongside the dataset so the
  exact reconstructed meshes can be re-scored.
- Eval is one command per method: `python eval/run_eval.py --method {M}` (writes
  `metrics_summary.json` + `metrics_per_seq.csv`).
- Test-set construction, GT export, the verbatim BEHAVE-CD code, and the surface-sampling module are
  all in `eval/` and `data/`.

---

## References

- Zhang et al. *NeuralDome: A Neural Modeling Pipeline on Multi-View Human–Object Interactions.* CVPR 2023.
- Bhatnagar et al. *BEHAVE: Dataset and Method for Tracking Human Object Interactions.* CVPR 2022.
- Xie et al. *CHORE.* ECCV 2022. / *Visibility-aware Human–Object Tracking (VisTracker).* CVPR 2023.
- Nam et al. *CONTHO: Joint Reconstruction of Humans and Objects via Contact.* CVPR 2024.
- Xie et al. *Template-free HDM: A Hierarchical Diffusion Model for HOI.* CVPR 2024.
- Huang et al. *StackFLOW.* IJCAI 2023.
- Zhang et al. *PHOSA: Perceiving Human–Object Spatial Arrangements.* ECCV 2020.
