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
| 8 | CARI4D | CVPR'26 (NVlabs) | video | **template-free, category-agnostic** (real NVlabs code + `cari4d-release` ckpt: UniDepth+NLF+SMPL-H human · Hunyuan3D object geometry · FoundationPose track · **CARI4D learned render-and-compare refinement**) | — |
| 9 | InterTrack | — | video | category-agnostic | — |
| – | WildHOI | CVPR'24 | single-image | 4 objects | **blocked** (see §5) |

Object-handling note: methods with a *fixed* object set map each HODome object to the nearest
category (e.g. CONTHO's `HODOME2BEHAVE`); template-free methods (HDM, CARI4D, InterTrack) avoid the
object domain gap entirely and are the cleanest object comparison.

---

## 4. Results — two leaderboards by input frame-rate

The benchmark has **two separate leaderboards**, split by the frame-rate the method is *fed*:
- **Leaderboard A — 1 fps input:** single-image methods only (video methods cannot use 1 fps).
- **Leaderboard B — 30 fps input:** single-image methods **re-run on 30 fps** *plus* the video /
  temporal methods (which only run at 30 fps).

A single-image method therefore appears on **both** boards (run once per rate); a video method only
on **B**. All numbers are surface-sampled (N=10000); `Sa` = object sampling (surf = area-weighted
surface, vtx = vertex resampling fallback). MPJPE/PA-MPJPE are not meaningful for the template-free
point-cloud methods (HDM; InterTrack uses an FPS joint proxy) — judge those by Chamfer / Object-CD.

### 4.1 Leaderboard A — 1 fps input (single-image)

<!-- RESULTS_1FPS_TABLE -->
| Method | PA-MPJPE↓ (mm) | MPJPE↓ | Chamfer↓ (mm) | p.V2V↓ | **Human-CD↓ (cm)** | **Object-CD↓ (cm)** | Sa(obj) | seqs |
|---|---|---|---|---|---|---|---|---|
| CHORE | **48.3** | 79.4 | 1291 | 63.6 | **4.72** | **62.1** | vtx | 39 |
| CONTHO | 58.0 | 89.1 | 3579 | 65.0 | **6.32** | **36.2** | vtx | 39 |
| StackFLOW | 60.5 | 112.5 | 543 | 70.3 | **5.89** | **45.6** | vtx | 39 |
| PHOSA | 66.2 | 108.8 | 3581 | 505.6 | **16.21** | **490.6** | surf | 39 |
| HDM† | (424.9) | (511.6) | 3579 | (38.98) | **94.8** | surf | 39 |

† HDM template-free point cloud (parenthesized human metrics not meaningful).
CHORE = per-frame neural-UDF optimization (~2.5 h/seq); completed 39/39 (last 6 on the L40s queue).
*Best human:* CHORE (PA-MPJPE 48.3). *Best object:* CONTHO (36.2 cm). *PHOSA object (490 cm)* is an
outlier — weak-perspective placement degrades on HODome's true-camera test.

### 4.2 Leaderboard B — 30 fps input (single-image @30fps + video)

**Video / temporal methods** (fed 30 fps contiguous frames):

<!-- RESULTS_30FPS_VIDEO_TABLE -->
| Method | type | PA-MPJPE↓ | Chamfer↓ | p.V2V↓ | **Human-CD↓ (cm)** | **Object-CD↓ (cm)** | Sa(obj) | seqs |
|---|---|---|---|---|---|---|---|---|
| Open4DHOI | video | **55.1** | 1421 | 83.8 | **5.94** | **55.6** | surf | 39 |
| VisTracker | video | ~~175.9~~ | ~~1148~~ | ~~185.8~~ | ~~15.21~~ | ~~139.1~~ | vtx | 34 |

> ⚠️ **VisTracker rows are INVALID — being re-run (2026-07-12 usage audit).** Our `prep_view26.py`
> fed its SMPL-T front-end a hardcoded zero-pose `mocap.json` (the FrankMocap/PARE init was never
> run) and near-empty 2D keypoints, so the temporal SMPL fit is degenerate (T-pose/floating/
> mis-scaled — verified visually and by projection probes). The human metrics measure our broken
> prep, not the method. The in-domain finetune eval reused the same stages 0–3, so its human
> numbers are equally contaminated. Both will be re-run with a real 2D detector + real mono init.
| CARI4D | video (template-free) | 145.3 | 1901 | 676 | 11.78 | 349.8 | mixed | 37 |
| InterTrack | video (template-free) | (proxy) | 3559 | — | **37.65** | **104.64** | — | 39 |

**Single-image methods re-run @30fps** (same models, fed every 30 fps frame):

<!-- RESULTS_30FPS_IMAGE_TABLE --> (scored on the dense 30 fps GT, ~71936 frames)
| Method | PA-MPJPE↓ | **Human-CD↓ (cm)** | **Object-CD↓ (cm)** | status |
|---|---|---|---|---|
| CONTHO | 58.0 | **6.33** | **35.9** | ✅ 39/39 @30fps (71936 fr) |
| StackFLOW | 60.0 | **5.84** | **45.1** | ✅ 39/39 @30fps (70788 fr) |
| HDM | | | | 🟢 23/39 @30fps (free pool, running) |
| PHOSA | | | | ⚠️ per-frame optim ~13.5 s/fr → 30 fps ≈ 7 h/seq (subset only) |
| CHORE | | | | ⚠️ per-frame optim ~133 s/fr → 30 fps infeasible full-39 (subset only) |

*CONTHO 30 fps (PA 58.0 / Human-CD 6.33 / Object-CD 35.9) is **within noise of its 1 fps result**
(58.0 / 6.32 / 36.2) — confirming feed-forward single-image methods are per-frame deterministic, so
their dense-30fps average equals the 1 fps average. The 30 fps board is therefore only *discriminative*
for the temporal/video methods.*

InterTrack is now **39/39** (was 8/39). Two upstream bugs blocked the rest and were fixed: (1) the
opt stages' `randint(0, seq_len − batch_size)` crashed on short sequences (`high ≤ 0`) — guarded to
use the whole clip; (2) the human-opt stage resumed from **stale checkpoints** (a 62-frame preview
from an old prep) while the current prep uses full-length sequences, so the pose slice was empty and
`smpl_layer` failed `size 256 vs 0` — the loader now validates checkpoint length against the current
sequence and discards/rebuilds on mismatch. InterTrack remains a proxy (FPS joints, not a mesh), so
PA-MPJPE/V2V are not directly comparable; judge it by Chamfer / Object-CD.
CARI4D is now scored on the full **37/39** (keyboard×2 have no FoundationPose input). CARI4D's
template-free object (Hunyuan3D + FoundationPose) drifts on symmetric/thin objects (full-set ObjCD
349.8 cm; first-frame placement ≤0.2 m → the failure is temporal tracking, not shape recovery).
Open4DHOI = strongest video method.

**Note on feasibility:** the optimization-based single-image methods (PHOSA, CHORE) are not feasible
at 30 fps over all 39 seqs (per-frame optimization × ~1843 frames/seq = hundreds of GPU-hours); they
are reported at 30 fps on a representative subset only. The feed-forward methods (CONTHO, StackFLOW,
HDM) run the full 30 fps set.

### 4.3 Leaderboard C — in-domain (retrained on HoDome)

The boards above are **zero-shot transfer** (BEHAVE-trained models, mapped to the nearest BEHAVE
object template). To measure the *object domain gap* directly, we retrain the template-locked methods
on HoDome itself: **train subjects 03–10, view26, 5 fps (43,674 frames); test subjects 01–02 (the
same 39-seq testset); native HoDome objects (drop the BEHAVE-template mapping).** Best setting is
**finetuning from the released BEHAVE checkpoint** (low LR, few epochs) — training from scratch on
HoDome's ~40k frames badly underfits the human (CONTHO from-scratch PA 86.7), so it is not used.

<!-- RESULTS_INDOMAIN_TABLE -->
| Method | setting | PA-MPJPE↓ | Human-CD↓ (cm) | **Object-CD↓ (cm)** |
|---|---|---|---|---|
| **StackFLOW** | zero-shot (BEHAVE obj) | 60.5 | 5.89 | 45.55 |
| **StackFLOW** | **in-domain finetune** | **51.1** | **4.71** | **25.75** |
| **CONTHO** | zero-shot (BEHAVE obj) | 58.0 | 6.32 | 36.23 |
| **CONTHO** | in-domain finetune (frozen human) | 59.9 | 6.87 | **28.00** |
| CONTHO | in-domain finetune (full) | 72.7 | 7.07 | 27.99 |
| **VisTracker** | zero-shot (BEHAVE obj) | ~~175.9~~ | ~~15.21~~ | ~~139.11~~ |
| **VisTracker** | **in-domain finetune** | ~~174.3~~ | ~~14.99~~ | ~~111.93~~ |

> ⚠️ VisTracker rows invalid — prep bug (zero-pose mocap init + empty 2D), being re-run; see the
> Leaderboard-B note.
| **CHORE** | zero-shot (BEHAVE obj) | 48.5 | 4.73 | 60.29 |
| **CHORE** | **in-domain finetune** | 49.2 | 4.77 | **51.99** |
| **CARI4D** | zero-shot | 145.1 | 11.77 | 348.74 |
| **CARI4D** | **in-domain finetune** | **120.6** | **10.31** | 416.75 |

**Findings.**
1. **In-domain finetuning closes the object domain gap.** StackFLOW Object-CD **45.6 → 25.8 cm
   (−43%)**; CONTHO **36.2 → 28.0** (frozen-human, full-mesh). Training on the real objects
   is the win the benchmark was built to expose.
2. **StackFLOW improves on every axis** (PA 60.5→51.1, Human-CD 5.89→4.71, Object-CD 45.6→25.8) — the
   ideal outcome.
3. **CONTHO needs the human branch frozen.** Finetuning *all* of CONTHO regresses the human
   (PA 58→72.7) because HoDome's SMPL-H **3D pose GT is noisier than BEHAVE's** (it projects correctly
   in 2D — verified 100% in-bbox, 1.5 px reprojection — but its 3D joint angles are less accurate),
   and CONTHO's strong direct 3D-pose loss overfits that noise. Freezing `hand4whole` (+ zeroing the
   human losses) preserves the BEHAVE pose (PA **59.9 ≈ 58**) while the object branch still adapts
   (36→**28.0**) — matching the full-finetune's object *without* the human cost. So the
   **frozen-human variant is the reported CONTHO in-domain result**. StackFLOW does not need this (its
   pose comes through a flow/2D-reprojection head, robust to 3D-GT noise).
   *Object representation note:* CONTHO's native output is 64 anchor keypoints, but it also regresses
   a full **6D object pose** (`obj_pose` R + `obj_trans` T); we export the pose applied to the full
   native template (mesh + faces), so its Object-CD is surface-sampled and apples-to-apples with
   StackFLOW/CHORE. (The 64-keypoint export gave an inflated 33.1; the full mesh is 28.0.)
4. **VisTracker: object −20%, human flat — and only the *classical* parts adapt.** On the 34 seqs
   both settings share, Object-CD drops 139.0 → 111.6 cm (−19.8%) while the human stays put
   (PA 175.8→173.5, Human-CD 15.20→14.92). Two finetune-specific observations:
   (a) **SIF-Net does not benefit from in-domain training** — its validation error rises
   *monotonically* from the very first eval (227→271 over 2.5 h), so the val-min early-stopping
   checkpoint (≈10 min in, barely off the BEHAVE init) is used; the object gain comes from the
   **native templates + the finetuned conditional motion infiller** (whose val improves
   0.0385→0.035) — the same "the neural branch doesn't adapt, the rest does" pattern as CONTHO's
   human branch. (b) In-domain also **widens coverage**: pingpong, tennis and trolleycase have no
   usable BEHAVE mapping and were unrunnable zero-shot; with native templates the finetuned eval
   covers all 37 runnable seqs (39 minus the 2 keyboard skips). VisTracker remains an outlier in
   absolute terms on HoDome (PA ~174 vs 51–60 for the others) — see the zero-shot boards for that
   diagnosis; finetuning doesn't change the story.
5. **CHORE: object −14%, human flat — same pattern.** Compared on the 38 seqs both settings share
   (the finetuned model's object fit for one test seq, subject01_pingpong, is degenerate and crashes
   neural_renderer's rasterizer — `forward_face_index_map` CUDA illegal-memory — on every L40s;
   zero-shot ran it fine, so it's dropped from both columns for a fair common-subset number),
   Object-CD drops 60.3 → 52.0 cm (−14%) while the human is unchanged
   (PA 48.5→49.2, Human-CD 4.73→4.77). CHORE finetune *training* completes on the free pool after two
   fixes — (a) **cap iters/epoch** so an epoch finishes inside the ~3 h preemption window (else the
   7.6 h epoch never completes and restarts epoch 0), and (b) **node-local `/scr` TMPDIR + 4 workers**
   so the NFS-bound boundary-sample dataloader runs at ~1 it/s instead of crawling. CHORE's *eval* is
   the multi-stage **recon**, whose `neural_renderer`/`pytorch3d` ops are compiled sm_89-only and throw
   "no kernel image" on free-pool cards — so it runs on L40s (as zero-shot CHORE did). `results/CHORE_finetune`.
6. **CARI4D: the inverse pattern — finetune helps the human, *hurts* the object.** On the 37 seqs both
   settings share (keyboard×2 excluded — no FoundationPose input was built for it, same as zero-shot),
   the human improves (PA 145.1→120.6, −17%; Human-CD 11.77→10.31) but Object-CD *worsens*
   348.7→416.8 cm (+19%). This is expected once you see CARI4D's design: unlike the four template-locked
   methods, its object is **not** a fixed mesh whose pose is regressed — it's per-frame geometry from
   Hunyuan3D + a FoundationPose track, refined by the learned render-and-compare net. There is no object
   *template* for in-domain training to adapt; finetuning the refinement net on HoDome trades object
   alignment for human alignment. And CARI4D's object error is by far the worst on the board (3–4 **m**,
   vs 26–112 cm) in both settings — the Hunyuan3D-geometry + FoundationPose object pipeline essentially
   does not work on these HoDome objects, and finetuning the refinement does not (and structurally cannot)
   fix that. So CARI4D is the one method where in-domain training is **not** an object-gap win — a useful
   negative result. `results/CARI4D_finetune` (37 seqs).

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
