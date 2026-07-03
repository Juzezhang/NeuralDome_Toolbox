# Egocentric HOI Rendering from the 76-View HoDome Dome

**Goal.** Given the HoDome capture — 76 calibrated 4K dome cameras, per-frame SMPL-X
human pose, and per-frame rigid object pose — synthesize the **egocentric
(head-mounted) view** of the human-object interaction: a virtual camera placed at the
subject's eyes, looking down/outward at their own hands and the object they manipulate.

All code lives in [`scripts/egonvs/`](../scripts/egonvs). Everything is validated on
`subject04_book` and `subject04_pink`. The whole effort is a *reconstruction* from the
real capture — no generative hallucination is used in the final deliverables (see
[§7](#7-alternatives-considered) for why a generative prior was built and then dropped).

---

## 1. Problem framing

The ego camera sits **inside** the dome at the subject's head, ~20–30 cm from the
hands/object. Every dome camera, by contrast, sits on the dome surface metres away
looking inward. So the ego view is an **extreme near-range novel viewpoint**, and —
critically — it sees surfaces (palm interiors, object underside, chest-from-above)
that **no dome camera ever observes**. This "blind zone" is the fundamental
difficulty; it is a data-coverage limit, not a method bug.

Two questions had to be answered up front:

1. *Do recent methods give layer-wise human+object geometry + rendering?* — Yes; this
   is exactly the user's own **NeuralDome (CVPR 2023)** layer-wise neural pipeline. But
   that neural-rendering code is **not** in this repo (only mesh-overlay viz), and no
   3DGS/NeRF tooling existed anywhere on the system → photorealistic appearance is
   **built from scratch**.
2. *Can we render the ego view from 76 views + known head pose?* — Yes; demonstrated in
   two phases below.

---

## 2. Inputs and key facts (verified)

| Input | Path | Note |
|---|---|---|
| 76-view video | `videos/{seq}/data{1..76}.mp4` | 4K (3840×2160), ~60 fps |
| Calibration | `calibration_ground/{date}/calibration.json` | K(4K), RT(4×4 OpenCV), **distCoeff=0** (pinhole) |
| Human pose | `smplx/{seq}.npz` (SMPL-X, via `./HODome` symlink) | global_orient/body/hands/betas/transl |
| Object mesh | `scaned_object/{obj}/{obj}_face1000.obj` | decimated; some have UV texture, many don't |
| Object pose | `object/{seq}.npz` | per-frame `object_R` (3×3), `object_T` |
| Layer masks | `mask_refine/{seq}/{human,object}/{view}/{f:06d}.png` | 720p per-view segmentation |

**Decisive verified facts** (each saved a wrong assumption):

- **Calibration removes COLMAP.** distCoeff=0 + exact K/R/T in the SMPL-X world frame →
  feed cameras straight to gsplat; init Gaussians from SMPL-X + posed-object verts.
- **Calibration is accurate** (two independent checks): held-out dome-view renders are
  pixel-aligned with the real image; learnable per-view camera refinement converges to
  only ~0.04°/0.5 mm. ⇒ **Ego blur is the blind zone, not the cameras.**
- **Frame alignment is per-sequence and must be re-verified.** `subject04_book` → offset
  0; `subject04_pink` → +4 (within IoU noise of 0, treated as 0). Tool:
  `check_alignment.py` (silhouette-IoU offset search; run in the `hodome` env).
- **Resolution.** K is 4K; masks are 720p. Either scale K by 1/3 (720p path) or crop
  from the native 4K video (hi-res path).

---

## 3. Two-phase approach

### Phase A — geometry ego render (fast, exact, no appearance)
Place a virtual camera at the SMPL-X head and rasterize the SMPL-X human + posed object
**mesh** with the existing PyTorch3D `Pyt3DWrapper`. Flat-shaded; validates the ego
camera math and yields ego silhouettes/depth. Env: `hodome`.

- `ego_camera.py` — head camera from SMPL-X joints (face→−Z, right→+X, Y-up, verified
  empirically). `--aim {gaze,object,hands}` aims at the interaction so it stays centered.
- `render_ego_geometry.py` — driver → `output/ego/{seq}/ego.mp4`.

### Phase B — photorealistic ego render (3DGS from the 76 RGB views)
Reconstruct appearance with **3D Gaussian Splatting**, supervised by the 76 dome views,
masked to the foreground by `mask_refine`. Env: `hodome_gs` (torch cu124 + gsplat 1.5.3;
needs `CUDA_HOME=/usr/local/cuda-12.8` on PATH for the JIT CUDA build).

Two model families (see [§5](#5-results)):

- **Single-frame static 3DGS** (`gs_scene.py`) — one frame, 76 views, full densification
  → highest per-frame quality ("hero" frame). Cannot make a video by itself.
- **Animatable compositional avatar** (`gs_avatar.py`) — the NeuralDome layer-wise idea
  in 3DGS: **human Gaussians bound to SMPL-X faces** (ride each face by barycentric
  coords + a learned local offset; deform with the *known* per-frame posed verts — no LBS
  reconstruction) **+ rigid object Gaussians** driven by the known object R/T. One trained
  model is **pose-driven**, so it renders *any* frame's ego camera → a full, temporally
  consistent **ego video**.

The "v2 recipe" that won (applies to both): **native-4K human+object crops** (`--hires`,
supervise where the interaction actually is — fixes the ~4× under-sampling at the ego camera) +
**green-spill suppression** (`--despill`, cap green at max(R,B) — removes the green
fringe from the chroma-key dome) + compactness **regularization** (`--scale_reg
--offset_reg --max_scale`, kills foggy oversized/anisotropic Gaussians) + denser,
smaller-initialized Gaussians. For the final video, `--antialias` reduces edge shimmer
and the renderer preserves source timing by default (`source_fps / render_step`).

---

## 4. Pipeline files

| File | Role |
|---|---|
| `ego_camera.py` | head/eye → OpenCV (K,R,T); `look_at` for object-aimed gaze |
| `render_ego_geometry.py` | Phase A mesh ego video |
| `cameras.py` | shared 76-cam loader (date lookup, K-scale, RT→R,T) |
| `extract_frames.py` | subsampled 720p frame dump (for the 720p path) |
| `dataset.py` | (frame,view) → RGB + masks + camera; alignment self-test |
| `gs_scene.py` | single-frame static 3DGS (densification, `--hires/--despill/--opt_cam/--gs2d/--hand_weight`) |
| `gs_avatar.py` | animatable compositional avatar (train + pose-driven render) |
| `gs_render_ego.py` | render ego (+ sanity dome view) from a single-frame ckpt |
| `ego_refine.py` | **optional, unused** generative prior (SDXL img2img) |
| `check_alignment.py` | per-sequence frame-offset verification |

Reused from the repo: `scripts/utils/pose_source.py` (SMPL-X verts/joints),
`viz/pyt3d_wrapper.py` (Phase A renderer), `scripts/video_extraction.py`.

---

## 5. Results

Foreground PSNR on 3 held-out dome views (never seen in training):

| Setting | Gaussians | Holdout PSNR | Note |
|---|---|---|---|
| Single-frame 720p (first try) | 324k | **29.2 dB** | proves calibration; ego view blurry |
| Single-frame 4K + despill (**v2, best single frame**) | 293k | 27.4 dB | sharp hand/book; the "hero" |
| Single-frame + 2DGS | 781k | 24.8 dB | **worse** — smeary |
| Single-frame + hand-weight | 503k | 26.9 dB | no gain, more streaks |
| Avatar video, 720p (first try) | 137k | — | foggy/streaky |
| Avatar video, v2 recipe (**final video**) | 308k | 21.1–21.6 dB | clean, temporally consistent |

(Avatar PSNR < single-frame because one shared model covers 60 frames with no per-frame
densification; and 4K-crop PSNR isn't directly comparable to 720p — more high-freq detail
to fit.)

**Deliverables produced:** Phase-A geometry ego video; single-frame hero ego for
`subject04_book` and `subject04_pink`; full ego videos (`ego_avatar.mp4`) for both, each
with a `dome9.mp4` held-out-view reference. The "pink" object is a grey bucket with a
vivid pink rim — sharper than the book (larger object, better-observed).

---

## 6. The core finding

The ego-view softness is **not** capacity, not 3DGS tuning, and not calibration. It is
the **unobserved blind zone**: surfaces visible only to the head-down camera are seen by
no dome camera, so adding Gaussians/weight/2D-surfels there only produces streaks
(nothing to supervise against). The faithful reconstruction ceiling is the v2 single
frame. Exceeding it requires *inventing* the unseen surfaces — i.e. a generative prior.

---

## 7. Alternatives considered

### Pose/appearance source
- **Reuse NeuralDome's trained neural rep** — not on disk; rejected.
- **Mesh-only ego (Phase A)** — kept as the fast geometry/validation path, not as the
  photorealistic deliverable.
- **Texture-bake the meshes** — most objects lack UV maps and SMPL-X has no per-subject
  texture; would still need multi-view appearance. 3DGS subsumes this. Rejected.

### Reconstruction representation
- **Per-frame static 3DGS** — best single-frame quality; kept as the hero/baseline. Can't
  make a video (would be 600 independent flickering models).
- **4D / deformable 3DGS** — would re-learn motion we already know exactly; wasteful and
  less robust. Rejected in favor of the known-pose compositional avatar.
- **Compositional avatar (chosen for video)** — exploits known SMPL-X pose + known object
  pose + per-layer masks; pose-driven → dense smooth video. This is the deliverable.

### Tried to sharpen the ego, and the verdict
- **4K person crops** ✓ — biggest faithful win (fixes resolution under-sampling).
- **Green-spill suppression** ✓ — removes green fringe; clean matte.
- **Gaussian compactness regularization** ✓ — removes fog/streaks.
- **Camera-pose refinement** (`--opt_cam`) — learned ~0.04°/0.5 mm ⇒ confirms cameras are
  already accurate; no quality gain. Useful as a *diagnostic*, not a fix.
- **2D Gaussian Splatting** (`--gs2d`) ✗ — worse (24.8 dB, smeary); flat surfels' normal
  reg is ill-posed in the blind zone.
- **Hand-region loss weighting** (`--hand_weight`) ✗ — no clear gain, more streaks on the
  object's unobserved far side.
- **Generative prior** (`ego_refine.py`, SDXL img2img) — *worked* visually (sharpens +
  plausibly fills the blind zone), but it **fabricates** unobserved detail. The user
  prefers faithful reconstruction, so it is **built but not used** in the final outputs.

### Remaining faithful lever (not yet built)
Binding-aware densification for the avatar (let face-bound Gaussians split/clone) to
approach the single-frame density → sharper video without any hallucination.

---

## 8. How to reproduce

Prereq once: `ln -sfn /simurgh2/users/juze/datasets/hodome HODome` at the repo root
(the SMPL-X loader uses the relative `./HODome/smplx/{seq}.npz`).

```bash
# env for 3DGS (gsplat JIT-compiles on first import)
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/simurgh2/users/juze/anaconda3/envs/hodome_gs/bin:$CUDA_HOME/bin:$PATH
PY=/simurgh2/users/juze/anaconda3/envs/hodome_gs/bin/python

# 0) verify per-sequence frame alignment (run in the `hodome` env; needs pytorch3d)
conda run -n hodome python scripts/egonvs/check_alignment.py --seq SEQ --view 17 --search=-20,40,4

# A) geometry ego video (env: hodome)
conda run -n hodome python scripts/egonvs/render_ego_geometry.py --seq SEQ --aim object

# B1) single-frame hero (best per-frame quality)
$PY scripts/egonvs/gs_scene.py --seq SEQ --frame 1500 --hires --despill --antialias \
    --scale_reg 0.5 --max_scale 0.008 --iters 20000 \
    --out output/ego_gs/SEQ_v2
$PY scripts/egonvs/gs_render_ego.py --seq SEQ --frame 1500 \
    --ckpt output/ego_gs/SEQ_v2/gauss_001500.pt --height 1080 --width 1080 --sanity_view 9 --antialias

# B2) full ego VIDEO (animatable avatar, v2 recipe)
$PY scripts/egonvs/gs_avatar.py --mode train --seq SEQ --hires --despill --antialias \
    --gpf 14 --n_obj 15000 --scale_reg 1.0 --offset_reg 0.1 --max_scale 0.008 \
    --max_frames 60 --iters 22000 --out output/ego_avatar/SEQ_v2seq
$PY scripts/egonvs/gs_avatar.py --mode render --seq SEQ \
    --ckpt output/ego_avatar/SEQ_v2seq/avatar.pt \
    --render_start 300 --render_end 2700 --render_step 4 --height 1080 --width 1080 --antialias
```

Each training writes side-by-side `holdout_v{N}.png` (`rendered | real`) — the calibration
sanity check — and the avatar render writes `ego_avatar.mp4` + `dome9.mp4`.
