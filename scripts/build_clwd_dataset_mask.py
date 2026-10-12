"""Build the CLWD support mask given to SORD: the dataset's own watermark mask at the benchmark resolution.

    python scripts/build_clwd_dataset_mask.py --clwd-root /path/to/CLWD

For every image of the CLWD benchmark (bench_b2r.npz, field `ids` = CLWD test-split image ids) the
256x256 binary mask test/Mask/<id>.png is resized to 64x64 (bilinear) and thresholded at 0.05, i.e. every
pixel the logo touches.  The benchmark's scoring region M_true is this mask dilated by one pixel; SORD
receives the undilated mask as its support (configs/sord/clwd.yaml) and is scored on M_true.
Writes experiments/extbench/results/b2r/mask_raw_dataset.npz (array M, shape (100, 64, 64)) under DATA_ROOT.
CLWD: https://github.com/vinthony/deep-blind-watermark-removal (dataset by Liu et al., WDNet).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from checkpoint_common import paths

BENCH = "experiments/extbench/results/b2r/bench_b2r.npz"
OUT = "experiments/extbench/results/b2r/mask_raw_dataset.npz"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clwd-root", type=Path, default=paths.DATA_ROOT / "data/external_validation/clwd/full/CLWD",
                    help="CLWD root with test/Mask/<id>.png")
    ap.add_argument("--threshold", type=float, default=0.05)
    a = ap.parse_args()

    ids = np.load(paths.data(BENCH))["ids"]
    masks = []
    for i in ids:
        m = Image.open(a.clwd_root / "test" / "Mask" / f"{int(i)}.png").convert("L").resize((64, 64), Image.BILINEAR)
        masks.append(np.asarray(m, np.float32) / 255 > a.threshold)
    out = paths.data(OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, M=np.stack(masks))
    print(f"wrote {out}: {len(masks)} masks, mean coverage {np.mean(masks):.3f}")


if __name__ == "__main__":
    main()
