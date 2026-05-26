#!/usr/bin/env python
"""One-click HoDome dataset download from Google Drive (+ optional tar extraction).

The dataset lives in a public (anyone-with-link) Google Drive folder. This script fetches it with
`gdown` and, by default, extracts the per-subject / per-object tars in place so you end up with the
layout documented in the README.

Usage:
    pip install -U gdown
    python scripts/download_hodome.py --out ./HODome                 # everything
    python scripts/download_hodome.py --out ./HODome \
        --modalities smplx object mhr scaned_object calibration_ground   # a subset
    python scripts/download_hodome.py --out ./HODome --no-extract         # download only

Notes:
  * Large folders occasionally hit Google Drive's per-file quota — just re-run; gdown skips files
    already downloaded.
  * For very large / flaky transfers, `rclone copy` against the same folder is a robust alternative
    (see the README).
"""
import argparse
import os
import subprocess
import sys
import tarfile

# Public HoDome Drive folder (anyone-with-link).
HODOME_FOLDER_ID = "19td-JTT38bduUAw384znXs_9dyS95q5e"
HODOME_FOLDER_URL = f"https://drive.google.com/drive/folders/{HODOME_FOLDER_ID}"

# Modalities that ship as tars and need extraction in place.
TAR_MODALITIES = ["videos", "mask_refine", "mhr", "scaned_object"]


def have_gdown():
    try:
        import gdown  # noqa: F401
        return True
    except ImportError:
        return False


def gdown_folder(url, out_dir, modalities=None):
    """Download the Drive folder (or named sub-folders) into out_dir via the gdown CLI."""
    os.makedirs(out_dir, exist_ok=True)
    cmd = [sys.executable, "-m", "gdown", "--folder", url, "-O", out_dir, "--remaining-ok"]
    print("[download]", " ".join(cmd))
    subprocess.check_call(cmd)


def extract_tars(out_dir, modalities):
    for m in modalities:
        d = os.path.join(out_dir, m)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.endswith((".tar", ".tar.gz", ".tgz")):
                path = os.path.join(d, fn)
                print(f"[extract] {path}")
                with tarfile.open(path) as tf:
                    tf.extractall(d)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="./HODome", help="destination dataset root")
    ap.add_argument("--modalities", nargs="*", default=None,
                    help="optional subset of top-level dirs to keep after download "
                         "(e.g. smplx object mhr scaned_object calibration_ground)")
    ap.add_argument("--no-extract", dest="extract", action="store_false",
                    help="download only, do not untar")
    args = ap.parse_args()

    if not have_gdown():
        sys.exit("gdown not found. Install it first:  pip install -U gdown")

    gdown_folder(HODOME_FOLDER_URL, args.out, args.modalities)

    if args.extract:
        extract_tars(args.out, TAR_MODALITIES)
        print("\n[note] images are not shipped (too large) — extract them from the videos with:")
        print("       python ./scripts/video2image.py")
    print(f"\nHoDome ready under: {args.out}")


if __name__ == "__main__":
    main()
