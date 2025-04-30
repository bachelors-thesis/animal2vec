#!/usr/bin/env python3 -u
"""
make_manifest_anuraset.py
=========================
Create **Fairseq‐style** manifest files (TSV) from *AnuraSet* which keeps all
annotation in a single `metadata.csv` instead of per‑file HDF5 labels.

Assumptions
-----------
* `metadata.csv` lives in the directory passed as *root*.
* WAV files are in `<root>/wav/` and the filename is given in the
  `sample_name` column.  If your audio lives elsewhere, override with
  `--audio-dir`.
* Columns after `subset` are one‑hot species indicators; their order
  (left → right) defines the integer **class id** used in the label list.

Usage example
-------------
```bash
python make_manifest_anuraset.py data/anuraset \
       --dest manifests_full \
       --valid-percent 0.05   # 5 % validation, 95 % train
```
The script writes:
```
manifests_full/
  ├─ train_0.tsv   # k‑fold train splits (k = --n-split)
  ├─ valid_0.tsv
  ├─ train_1.tsv
  ├─ …
  └─ pretrain.tsv  # all unlabeled + labeled audio for unsupervised pre‑train
```

The manifest format is exactly what Fairseq/Animal2Vec expects:
* **line 0** – absolute path of the audio *root* directory.
* **other lines** – `relative_path<TAB>n_frames`.
"""

from __future__ import annotations
import os
import argparse
import soundfile as sf
from typing import List, Dict

import pandas as pd
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("root", help="Dataset root that contains metadata.csv and wav/")
    p.add_argument("--audio-dir", default="wav", help="Sub‑directory (inside root) that holds WAV files")
    p.add_argument("--dest", default="manifest_out", help="Output folder for the manifest TSV files")
    p.add_argument("--valid-percent", type=float, default=0.2, help="Portion of labelled data in validation set")
    p.add_argument("--n-split", type=int, default=5, help="Number of cross‑validation folds")
    p.add_argument("--seed", type=int, default=1612, help="Random seed for the stratified splitter")
    p.add_argument("--limit-fraction", type=float, default=None,
                    help="Use only this fraction of the total dataset "
                         "for *train+valid* manifest generation.")

    return p

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def relative_path(abs_path: str, root: str) -> str:
    """Return *abs_path* relative to *root* (without leading ./)."""
    return os.path.relpath(abs_path, root)


def get_frame_count(wav_path: str) -> int:
    """Return number of PCM frames in *wav_path* (uses soundfile metadata only)."""
    return sf.info(wav_path).frames


# --------------------------------------------------------------------------------------
# Main logic
# --------------------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    root = os.path.abspath(args.root)
    audio_root = os.path.join(root, args.audio_dir)
    meta_path = os.path.join(root, "metadata.csv")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"metadata.csv not found at {meta_path}")

    df = pd.read_csv(meta_path)

    # ---- new code ----
    if args.limit_fraction and 0 < args.limit_fraction < 1.0:
        df = df.sample(frac=args.limit_fraction,
                       random_state=args.seed).reset_index(drop=True)
    # ------------------


    # Identify label columns (everything after the column named "subset")
    try:
        subset_idx = list(df.columns).index("subset")
    except ValueError as e:
        raise ValueError("Column 'subset' not found in metadata.csv") from e

    label_cols: List[str] = df.columns[subset_idx + 1 :].tolist()
    n_classes = len(label_cols)
    print(f"Discovered {n_classes} classes → IDs 0 … {n_classes-1}")

    # ----------------------------------------------------------------------------
    # Build lists required for stratified splitter
    # ----------------------------------------------------------------------------
    wav_paths: List[str] = []      # absolute paths of *labelled* wavs
    targets: List[np.ndarray] = [] # multi‑hot vectors (shape = n_classes)

    missing_audio: List[str] = []  # rows where the WAV file is absent on disk

    for _, row in df.iterrows():
        file_sub_dir = str(row["site"]).strip()
        filename = str(row["fname"]).strip()
        filename_appendix = f"_{row['min_t']}_{row['max_t']}"
        abs_path = os.path.join(audio_root, file_sub_dir, filename + filename_appendix + ".wav")

        if not os.path.isfile(abs_path):
            missing_audio.append(abs_path)
            continue

        label_vector = row[label_cols].to_numpy(dtype=int)
       
        if label_vector.sum() == 0:
            # unlabeled clip → goes only into pre‑train set
            continue
        wav_paths.append(abs_path)
        targets.append(label_vector)

    if missing_audio:
        print(f"Warning: {len(missing_audio)} audio files listed in metadata.csv were not found on disk.")

    if len(wav_paths) == 0:
        raise RuntimeError("No labeled audio files with existing WAV found.")

    targets_arr = np.vstack(targets)

    # ----------------------------------------------------------------------------
    # Prepare output directory & utilities
    # ----------------------------------------------------------------------------
    os.makedirs(args.dest, exist_ok=True)

    def write_manifest(pathlist: List[str], tsv_path: str) -> None:
        with open(tsv_path, "w") as f:
            print(audio_root, file=f)  # header line
            for p in pathlist:
                line = f"{relative_path(p, audio_root)}\t{get_frame_count(p)}"
                print(line, file=f)

    # ----------------------------------------------------------------------------
    # pretrain.tsv = all audio (labelled + unlabelled)
    # ----------------------------------------------------------------------------
    print("Writing pretrain.tsv (all audio files)…")
    all_audio = [os.path.join(audio_root, fn) for fn in os.listdir(audio_root) if fn.lower().endswith(".wav")]
    write_manifest(all_audio, os.path.join(args.dest, "pretrain.tsv"))

    # ----------------------------------------------------------------------------
    # K‑fold train / valid splits (labelled only)
    # ----------------------------------------------------------------------------
    splitter = MultilabelStratifiedShuffleSplit(n_splits=args.n_split,
                                               test_size=args.valid_percent,
                                               random_state=args.seed)

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(wav_paths, targets_arr)):
        print(f"Fold {fold}: {len(train_idx)} train, {len(valid_idx)} valid clips")
        train_paths = [wav_paths[i] for i in train_idx]
        valid_paths = [wav_paths[i] for i in valid_idx]
        write_manifest(train_paths, os.path.join(args.dest, f"train_{fold}.tsv"))
        write_manifest(valid_paths, os.path.join(args.dest, f"valid_{fold}.tsv"))

    print("Manifest generation complete →", args.dest)


# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = get_parser()
    main(parser.parse_args())