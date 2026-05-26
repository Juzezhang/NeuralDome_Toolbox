#!/usr/bin/env bash
# Body models for the HoDome SMPL-X / MHR visualization pipeline are LICENSE-RESTRICTED and are
# NOT redistributed in this repo. Register + download them yourself from the official sites, then
# place them under models/model_files/. This script creates the target dirs, prints exactly what
# goes where, and verifies the install.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MF="$ROOT/models/model_files"
mkdir -p "$MF/smplx"

cat <<EOF
============================ HoDome body-model setup ============================
The RECOMMENDED pose source is SMPL-X (or MHR). For the modern visualization
pipeline (scripts/hodome_visualize_pyrender.py, --source smplx) you only need SMPL-X.

1) SMPL-X  (REQUIRED for --source smplx)
   - Register & download at:  https://smpl-x.is.tue.mpg.de   (SMPL-X v1.1)
   - Place the NEUTRAL model here:
         $MF/smplx/SMPLX_NEUTRAL.npz
   - It is loaded via smplx.create(model_type="smplx", gender="neutral",
     num_betas=10, num_expression_coeffs=10, use_pca=False, flat_hand_mean=True).

2) MHR  (--source mhr)
   - No body model needed here: vertices are pre-computed by
     scripts/forward_mhr_to_npz.py in the MHR env (see README "Installation").

3) (LEGACY ONLY) SMPL / MANO for the old SMPL-H scripts
   - SMPL:  https://smpl.is.tue.mpg.de    MANO: https://mano.is.tue.mpg.de
   - NOT needed for the recommended SMPL-X / MHR pipeline.

⚠ The older SMPL-H pose data (legacy 'smplh/') is lower quality and is NOT recommended.
  Prefer --source smplx (recommended) or --source mhr.
================================================================================
EOF

PY="$(command -v python || command -v python3)"
if [ -n "${PY:-}" ]; then
  "$PY" -c "import importlib.util as u; print('  smplx package:', 'OK' if u.find_spec('smplx') else 'MISSING (pip install smplx)')"
fi
if [ -f "$MF/smplx/SMPLX_NEUTRAL.npz" ]; then
  echo "  SMPLX_NEUTRAL.npz: present ✓"
else
  echo "  SMPLX_NEUTRAL.npz: MISSING → download per step 1 above"
fi
