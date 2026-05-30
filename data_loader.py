"""
data_loader.py
==============
Single source of truth for loading data across all model notebooks.

Place this file at the project root (next to clean_pan.ipynb).

All CSVs live in data/processed/:
    master.csv        — full PAN dataset (unsplit)
    pan_train.csv     — PAN train split (70%)
    pan_val.csv       — PAN val split (15%)
    pan_test.csv      — PAN test split (15%)
    quora_train.csv   — Quora train split
    quora_test.csv    — Quora test split

Usage from any notebook in models/:
-----------------------------------
    import sys
    sys.path.append('..')          # so notebooks in models/ can import this
    from data_loader import load_pan, load_quora, get_pair_dataset

    train, val, test = load_pan()
    pairs = get_pair_dataset(train, dataset='pan')
"""
from __future__ import annotations
from pathlib import Path
from typing import Literal

import pandas as pd


# Path resolution — works whether running from project root or models/
def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / 'data' / 'processed').exists():
            return parent
    return Path.cwd()


PROJECT_ROOT  = _find_project_root()
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'


# ─────────────────────────────────────────────────────────────────────────
# PAN loaders
# ─────────────────────────────────────────────────────────────────────────
def load_pan_master() -> pd.DataFrame:
    """Load the full unsplit PAN master.csv."""
    return pd.read_csv(PROCESSED_DIR / 'master.csv')


def load_pan() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, val, test) PAN DataFrames."""
    train = pd.read_csv(PROCESSED_DIR / 'pan_train.csv')
    val   = pd.read_csv(PROCESSED_DIR / 'pan_val.csv')
    test  = pd.read_csv(PROCESSED_DIR / 'pan_test.csv')
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────
# Quora loaders
# ─────────────────────────────────────────────────────────────────────────
def load_quora() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, test) Quora DataFrames.
    Schema: q1_clean, q2_clean, is_duplicate
    """
    train = pd.read_csv(PROCESSED_DIR / 'quora_train.csv')
    test  = pd.read_csv(PROCESSED_DIR / 'quora_test.csv')
    return train, test


# ─────────────────────────────────────────────────────────────────────────
# Unified pair extractor — used by all 3 model notebooks
# ─────────────────────────────────────────────────────────────────────────
def get_pair_dataset(
    df: pd.DataFrame,
    dataset: Literal['pan', 'quora'],
    *,
    drop_no_source: bool = True,
    drop_intrinsic: bool = False,
) -> pd.DataFrame:
    """
    Convert any input DataFrame into a unified format:

        text_a  | text_b  | label

    Parameters
    ----------
    df : input DataFrame (PAN or Quora schema)
    dataset : 'pan' or 'quora'
    drop_no_source : (PAN only) drop rows where source_text is missing/'NONE'
                     This excludes intrinsic + translation cases that have no source.
    drop_intrinsic : (PAN only) drop intrinsic-corpus rows entirely
    """
    if dataset == 'pan':
        out = df.copy()
        if drop_intrinsic:
            out = out[out['corpus_type'] != 'intrinsic']
        if drop_no_source:
            out = out[(out['source_text'].notna()) & (out['source_text'] != 'NONE')]
        out = out.rename(columns={
            'suspicious_text': 'text_a',
            'source_text'    : 'text_b',
            'is_plagiarism'  : 'label',
        })
        return out[['text_a', 'text_b', 'label']].reset_index(drop=True)

    elif dataset == 'quora':
        out = df.rename(columns={
            'q1_clean'    : 'text_a',
            'q2_clean'    : 'text_b',
            'is_duplicate': 'label',
        })
        return out[['text_a', 'text_b', 'label']].reset_index(drop=True)

    raise ValueError(f"Unknown dataset: {dataset}. Use 'pan' or 'quora'.")


def summarize_split(df: pd.DataFrame, name: str = 'split') -> None:
    """Print a one-line summary of any pair dataset."""
    n = len(df)
    label_col = 'label' if 'label' in df.columns else 'is_plagiarism'
    pos = int(df[label_col].sum()) if label_col in df.columns else None
    print(f'  {name:8s} | rows={n:>7,d}  pos={pos:>6,d}  neg={n - (pos or 0):>6,d}')


if __name__ == '__main__':
    print(f'Project root: {PROJECT_ROOT}')
    print(f'Looking in  : {PROCESSED_DIR}')
    print()
    print('Available CSVs:')
    for f in sorted(PROCESSED_DIR.glob('*.csv')):
        size = f.stat().st_size / 1_048_576
        print(f'  {f.name:25s}  ({size:>6.1f} MB)')
