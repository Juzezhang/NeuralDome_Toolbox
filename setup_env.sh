#!/usr/bin/env bash
# One-shot environment setup for the HoDome SMPL-X / MHR visualization pipeline.
#
#   bash setup_env.sh [ENV_NAME] [MHR_ASSETS_DIR]
#       ENV_NAME         conda env to create        (default: hodome)
#       MHR_ASSETS_DIR   where to unzip MHR assets   (default: $HOME/mhr_assets)
#
# Why conda-forge for torch/pytorch3d/pymomentum: their compiled extensions must share one
# ABI. Installing torch from the `pytorch` channel and adding `pymomentum` later silently
# upgrades torch and breaks PyTorch3D's `_C` extension. So we pin them all to conda-forge.
set -euo pipefail

ENV_NAME="${1:-hodome}"
MHR_ASSETS_DIR="${2:-$HOME/mhr_assets}"
MHR_ASSETS_URL="https://github.com/facebookresearch/MHR/releases/download/v1.0.0/assets.zip"

echo ">>> [1/4] conda env '$ENV_NAME' (python 3.12, conda-forge torch+pytorch3d+pymomentum)"
# 3.12 because pymomentum wheels are 3.12+ on conda-forge and mhr needs >=3.11.
conda create -n "$ENV_NAME" --override-channels -c conda-forge -y \
    python=3.12 pytorch pytorch3d pymomentum \
    trimesh py-opencv tqdm yacs pyyaml matplotlib scikit-learn \
    hatchling hatch-vcs editables pip

# Resolve the env's python without needing `conda activate` inside a script.
PY="$(conda run -n "$ENV_NAME" which python)"

echo ">>> [2/4] pip: smplx + mhr + pyrender (not on conda-forge / pip is the tested path)"
"$PY" -m pip install --quiet smplx mhr pyrender

echo ">>> [3/4] MHR model assets → $MHR_ASSETS_DIR/assets  (~200 MB, one-time)"
if [ -f "$MHR_ASSETS_DIR/assets/lod1.fbx" ]; then
    echo "    already present, skipping download"
else
    mkdir -p "$MHR_ASSETS_DIR"
    ( cd "$MHR_ASSETS_DIR" && curl -fL -O "$MHR_ASSETS_URL" && unzip -oq assets.zip && rm -f assets.zip )
fi

echo ">>> [4/4] verify"
MHR_ASSETS_DIR="$MHR_ASSETS_DIR/assets" "$PY" - <<'PYEOF'
import os, torch, pytorch3d, smplx, trimesh, cv2, pyrender
import pymomentum.geometry  # noqa
from pytorch3d.renderer import RasterizationSettings  # noqa
print(f"  torch={torch.__version__}  cuda={torch.cuda.is_available()}  cuda_ver={torch.version.cuda}")
print(f"  pytorch3d={pytorch3d.__version__}  cv2={cv2.__version__}  trimesh={trimesh.__version__}")
from pathlib import Path
a = Path(os.environ["MHR_ASSETS_DIR"])
ok = (a / "lod1.fbx").exists()
print(f"  MHR assets at {a}: {'OK' if ok else 'MISSING lod1.fbx'}")
print("  ALL OK")
PYEOF

cat <<EOF

Done. Use the env with:  conda activate $ENV_NAME
For MHR rendering, set:   export MHR_ASSETS_DIR=$MHR_ASSETS_DIR/assets
(or pass --mhr_assets_dir to scripts/forward_mhr_to_npz.py)

Developing on MHR itself? replace the pip install with an editable checkout:
  pip install --no-build-isolation -e /path/to/your/MHR    # ships its own assets/, no download needed
EOF
