#!/usr/bin/env python3 -u
"""
Create Fairseq‑style manifest files (TSV) for the AnuraSet dataset
which uses a single `metadata.csv` file instead of per‑file HDF5 labels.

* The CSV **must** contain a column with the relative path to each WAV file and
  a column holding one or more labels per file (semicolon‑separated for
  multi‑label clips).
* The script keeps the original functionality of the Max‑Planck script:
    - k‑fold multilabel‑stratified splits (`--n-split`)
    - few‑shot sub‑splits (`--few-shot`)
    - leave‑p‑out strategy (`--leave-p-out`)

Example usage
-------------
python make_manifest_anuraset.py /datasets/anuraset \
       --dest manifests/anuraset \
       --metadata-file metadata.csv \
       --relpath-col rel_path \
       --label-col labels \
       --valid-percent 0.05
"""

import argparse
import os
import random
import re
import soundfile as sf
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Fairseq/Animal2Vec manifests from AnuraSet metadata.csv")

    # Core dataset paths
    parser.add_argument("root", metavar="DIR", help="Root directory that contains the audio files and metadata.csv")
    parser.add_argument("--metadata-file", default="metadata.csv", type=str,
                        help="CSV file inside <root> describing the dataset (default: metadata.csv)")

    # CSV layout options
    parser.add_argument("--relpath-col", default="rel_path", type=str,
                        help="CSV column that stores the *relative* WAV path (default: rel_path)")
    parser.add_argument("--label-col", default="labels", type=str,
                        help="CSV column that stores labels (semicolon‑separated for multi‑label) (default: labels)")

    # Manifest options
    parser.add_argument("--dest", default=".", type=str, metavar="DIR", help="Output directory for .tsv manifests")
    parser.add_argument("--ext", default="wav", type=str, metavar="EXT", help="File extension to look for (default: wav)")
    parser.add_argument("--valid-percent", default=0.2, type=float,
                        help="Percentage of labelled data to use as validation set (0–1)")
    parser.add_argument("--n-split", default=5, type=int,
                        help="Number of k‑fold splits for cross‑validation (default: 5)")

    # Extra strategies
    parser.add_argument("--few-shot", action="store_true",
                        help="Create five nested few‑shot train subsets (1 %, 10 %, 25 %, 50 %, 75 %) for each split")
    parser.add_argument("--leave-p-out", action="store_true",
                        help="Generate an additional leave‑p‑out split where ≈20 % of unique recordings are held out for eval only")

    # Misc
    parser.add_argument("--seed", default=1612, type=int, help="Random seed (default: 1612)")
    parser.add_argument("--path-must-contain", default=None, type=str,
                        help="If set, only rows whose path contains this substring will be considered")
    return parser

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _parse_label_string(label_str: str) -> List[str]:
    """Split a semicolon‑separated label string into a list. Empty string → []."""
    label_str = str(label_str).strip()
    return [] if label_str == "" else [lbl.strip() for lbl in label_str.split(";")]


def _build_class_mapping(all_label_lists: List[List[str]]) -> Dict[str, int]:
    """Assign an integer id to every unique class name found in the dataset."""
    classes = sorted({lbl for sub in all_label_lists for lbl in sub})
    return {cls_name: idx for idx, cls_name in enumerate(classes)}


def _frames_for(path: Path) -> int:
    """Return number of audio frames for a file using *soundfile* metadata (no decoding)."""
    return sf.info(path).frames

# --------------------------------------------------------------------------------------
# Main logic
# --------------------------------------------------------------------------------------

def main(args: argparse.Namespace):
    random.seed(args.seed)
    np.random.seed(args.seed)

    # --------------------------------------------------
    # 0. I/O setup
    # --------------------------------------------------
    root = Path(args.root).expanduser().resolve()
    args.dest = Path(args.dest).expanduser().resolve()
    args.dest.mkdir(parents=True, exist_ok=True)

    print(f"Reading metadata from {root / args.metadata_file}")
    df = pd.read_csv(root / args.metadata_file)

    # Basic sanity checks on columns
    for col in (args.relpath_col, args.label_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {args.metadata_file}")

    # --------------------------------------------------
    # 1. Collect file paths and labels
    # --------------------------------------------------
    records_with_labels: List[str] = []   # absolute paths with ≥1 label
    label_lists:        List[List[str]] = []
    records_wo_labels:  List[str] = []   # absolute paths with zero/empty label

    ext_pattern = re.compile(rf".*\.{args.ext}$", re.IGNORECASE)

    for _, row in df.iterrows():
        rel_path = str(row[args.relpath_col])
        if args.path_must_contain and args.path_must_contain not in rel_path:
            continue
        if not ext_pattern.match(rel_path):
            continue

        abs_path = str(root / rel_path)
        lbls = _parse_label_string(row[args.label_col])
        if len(lbls) == 0:
            records_wo_labels.append(abs_path)
        else:
            records_with_labels.append(abs_path)
            label_lists.append(lbls)

    if len(records_with_labels) == 0:
        raise RuntimeError("No labelled audio files were found – aborting.")

    # --------------------------------------------------
    # 2. Convert class names → integer IDs for stratification
    # --------------------------------------------------
    class_to_id = _build_class_mapping(label_lists)
    print("Discovered classes:")
    for cls, idx in class_to_id.items():
        print(f"  {idx:>2d} : {cls}")

    # Build binary indicator arrays per file for stratified splitting
    Y = np.zeros((len(label_lists), len(class_to_id)), dtype=int)
    for i, lbls in enumerate(label_lists):
        Y[i, [class_to_id[lbl] for lbl in lbls]] = 1

    # --------------------------------------------------
    # 3. Leave‑p‑out split (optional)
    # --------------------------------------------------
    if args.leave_p_out:
        unique_parents = sorted({Path(p).stem for p in records_with_labels})
        p = round(0.2 * len(unique_parents))
        held_out = set(random.sample(unique_parents, p))
        test_mask = np.array([Path(p).stem in held_out for p in records_with_labels])
        train_mask = ~test_mask
        _write_leave_p_out(root, args.dest, records_with_labels, records_wo_labels,
                           train_mask, test_mask)
        print(f"[leave‑p‑out]  Wrote train_lof.tsv, valid_lof.tsv, pretrain_lof.tsv")

    # --------------------------------------------------
    # 4. k‑fold stratified splits
    # --------------------------------------------------
    splitter = MultilabelStratifiedShuffleSplit(n_splits=args.n_split,
                                               test_size=args.valid_percent,
                                               random_state=args.seed)

    for split_idx, (train_idx, valid_idx) in enumerate(splitter.split(records_with_labels, Y)):
        _write_split(root, args.dest, split_idx,
                     records_with_labels, records_wo_labels,
                     train_idx, valid_idx,
                     few_shot=args.few_shot,
                     seed=args.seed)
        print(f"[split {split_idx}]  train_{split_idx}.tsv & valid_{split_idx}.tsv written")

    # --------------------------------------------------
    # 5. Pretrain manifest – all *unlabelled* + all *labelled* files
    # --------------------------------------------------
    pretrain_path = args.dest / "pretrain.tsv"
    with pretrain_path.open("w") as f:
        print(root, file=f)  # header
        for p in (*records_with_labels, *records_wo_labels):
            print(f"{os.path.relpath(p, root)}\t{_frames_for(p)}", file=f)
    print("[pretrain]      pretrain.tsv written")

# --------------------------------------------------------------------------------------
# Writers
# --------------------------------------------------------------------------------------

def _write_split(root: Path, dest: Path, split_idx: int,
                 labelled_paths: List[str], unlabelled_paths: List[str],
                 train_idx: np.ndarray, valid_idx: np.ndarray,
                 few_shot: bool, seed: int):
    """Write train/valid TSVs (+ optional few‑shot subsets) for one split."""

    train_path = dest / f"train_{split_idx}.tsv"
    valid_path = dest / f"valid_{split_idx}.tsv"

    with train_path.open("w") as ft, valid_path.open("w") as fv:
        print(root, file=ft)
        print(root, file=fv)
        for idx in train_idx:
            p = labelled_paths[idx]
            print(f"{os.path.relpath(p, root)}\t{_frames_for(p)}", file=ft)
        for idx in valid_idx:
            p = labelled_paths[idx]
            print(f"{os.path.relpath(p, root)}\t{_frames_for(p)}", file=fv)

    # ----- few‑shot sub‑sets -----------------------------------------------------
    if few_shot:
        shot_fracs = [0.01, 0.10, 0.25, 0.50, 0.75]
        rng = np.random.default_rng(seed)
        for level, frac in enumerate(shot_fracs):
            k = max(1, int(frac * len(train_idx)))
            subset_idx = rng.choice(train_idx, size=k, replace=False)
            shot_path = dest / f"train_{split_idx}_few_{level}.tsv"
            with shot_path.open("w") as fs:
                print(root, file=fs)
                for idx in subset_idx:
                    p = labelled_paths[idx]
                    print(f"{os.path.relpath(p, root)}\t{_frames_for(p)}", file=fs)

    # Also append *labelled* train/valid files to pretrain.tsv via caller

def _write_leave_p_out(root: Path, dest: Path,
                       labelled_paths: List[str], unlabelled_paths: List[str],
                       train_mask: np.ndarray, test_mask: np.ndarray):
    """Write train_lof.tsv / valid_lof.tsv / pretrain_lof.tsv for leave‑p‑out."""
    train_lof = dest / "train_lof.tsv"
    valid_lof = dest / "valid_lof.tsv"
    pretrain_lof = dest / "pretrain_lof.tsv"

    with train_lof.open("w") as ft, valid_lof.open("w") as fv, pretrain_lof.open("w") as fp:
        # headers
        for f in (ft, fv, fp):
            print(root, file=f)
        # train labelled
        for idx, keep in enumerate(train_mask):
            if keep:
                p = labelled_paths[idx]
                line = f"{os.path.relpath(p, root)}\t{_frames_for(p)}"
                print(line, file=ft)
                print(line, file=fp)
        # valid labelled
        for idx, keep in enumerate(test_mask):
            if keep:
                p = labelled_paths[idx]
                line = f"{os.path.relpath(p, root)}\t{_frames_for(p)}"
                print(line, file=fv)
        # unlabelled always to pretrain
        for p in unlabelled_paths:
            print(f"{os.path.relpath(p, root)}\t{_frames_for(p)}", file=fp)

# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = get_parser()
    main(parser.parse_args())
