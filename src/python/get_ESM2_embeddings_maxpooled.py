#!/usr/bin/env python3
"""
Generate MAX-POOLED ESM-2 layer-18 embeddings for a subset of secretome sequences.

Model: esm2_t33_650M_UR50D (CPU)
Layer: 18
Pooling: max across residues (matches F1 + centroid analyses)
"""

import argparse
import os
import re
from pathlib import Path
import datetime as dt
import gc

import torch
import pandas as pd
import esm

MAX_LEN = 2000  # skip ultra-long sequences


# ── Logging helper ──────────────────────────────────────────────────────────

def log(msg: str):
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


# ── FASTA utilities ─────────────────────────────────────────────────────────

def fasta_iter_ids(path: Path):
    """Yield (token, header, sequence) from FASTA."""
    with path.open() as fh:
        hdr = None
        seq = []
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if hdr is not None:
                    token = hdr.split()[0]
                    yield token, hdr, "".join(seq)
                hdr = line[1:].strip()
                seq = []
            else:
                seq.append(line.strip())
        if hdr is not None:
            token = hdr.split()[0]
            yield token, hdr, "".join(seq)


def token_to_entry(id_token: str) -> str:
    """Extract UniProt accession from FASTA token."""
    parts = id_token.split("|")
    if len(parts) >= 3:
        return parts[1]
    return id_token


# ── ESM-2 model loading ─────────────────────────────────────────────────────

def load_esm2_model(model_name: str, repr_layer: int):
    log(f"[ESM] Loading model: {model_name} (CPU)")
    model, alphabet = getattr(esm.pretrained, model_name)()
    model.eval()
    model.to("cpu")

    if repr_layer < 1 or repr_layer > model.num_layers:
        raise ValueError(
            f"repr_layer {repr_layer} out of range "
            f"(1–{model.num_layers})"
        )

    log(f"[ESM] Model loaded. Using repr_layer={repr_layer}")
    return model, alphabet


# ── Embedding computation (MAX pooling) ────────────────────────────────────

def compute_embeddings(
    sequences,
    out_dir: Path,
    model,
    alphabet,
    batch_size: int = 4,
    repr_layer: int = 18,
    skip_existing: bool = True,
):
    """
    Compute MAX-POOLED ESM-2 embeddings and save one .pt per Entry.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    batch_converter = alphabet.get_batch_converter()

    total = len(sequences)
    log(f"[ESM] Computing embeddings for {total} sequences "
        f"(batch size {batch_size}, pooling=max)")

    num_written = 0
    skipped_long = []

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = sequences[start:end]

        effective_batch = []
        for entry, seq in batch:
            out_file = out_dir / f"{entry}.pt"

            if skip_existing and out_file.exists():
                continue

            if len(seq) > MAX_LEN:
                skipped_long.append((entry, len(seq)))
                continue

            effective_batch.append((entry, seq))

        if not effective_batch:
            log(f"[ESM] Skipped batch {start}-{end}")
            continue

        data = [(entry, seq) for entry, seq in effective_batch]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to("cpu")

        with torch.no_grad():
            results = model(
                batch_tokens,
                repr_layers=[repr_layer],
                return_contacts=False,
            )

        token_reps = results["representations"][repr_layer].to("cpu")

        for i, (entry, seq) in enumerate(effective_batch):
            seq_len = len(seq)

            # 🔑 MAX POOLING ACROSS RESIDUES
            rep = token_reps[i, 1 : seq_len + 1].max(0).values

            out_file = out_dir / f"{entry}.pt"
            torch.save(
                {
                    "entry": entry,
                    "embedding": rep,          # (1280,)
                    "sequence": seq,
                    "pooling": "max",
                    "model": "esm2_t33_650M_UR50D",
                    "repr_layer": repr_layer,
                },
                out_file,
            )
            num_written += 1

        del results, token_reps, batch_tokens, batch_labels, batch_strs
        gc.collect()

        log(f"[ESM] Processed {end}/{total}")

    log(f"[ESM] Finished. Wrote {num_written} embeddings.")
    return num_written, skipped_long


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate MAX-pooled ESM-2 layer-18 embeddings for secretome."
    )
    parser.add_argument("--fasta", required=True,
                        help="Input UniProt FASTA")
    parser.add_argument("--csv", required=True,
                        help="CSV with 'Entry' column (UniProt accessions)")
    parser.add_argument("--out-dir", required=True,
                        help="Output directory for .pt embeddings")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--repr-layer", type=int, default=18)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.num_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.num_threads)
    torch.set_num_threads(args.num_threads)
    torch.set_num_interop_threads(args.num_threads)

    fasta_path = Path(args.fasta)
    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)

    if not fasta_path.exists():
        raise SystemExit(f"FASTA not found: {fasta_path}")
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    # Load allowed entries (accessions)
    df = pd.read_csv(csv_path)
    if "Entry" not in df.columns:
        raise SystemExit("CSV must contain an 'Entry' column")

    target_entries = set(df["Entry"].astype(str))
    log(f"[DATA] Target entries: {len(target_entries)}")

    # Collect sequences from FASTA
    selected = []
    found = set()

    log(f"[DATA] Scanning FASTA: {fasta_path}")
    for token, hdr, seq in fasta_iter_ids(fasta_path):
        entry = token_to_entry(token)
        if entry in target_entries:
            selected.append((entry, seq))
            found.add(entry)

    log(f"[DATA] Sequences matched: {len(found)}")

    if not selected:
        raise SystemExit("No matching sequences found in FASTA")

    # Load model
    model, alphabet = load_esm2_model(
        "esm2_t33_650M_UR50D",
        repr_layer=args.repr_layer,
    )

    # Compute embeddings
    num_embeddings, skipped_long = compute_embeddings(
        sequences=selected,
        out_dir=out_dir,
        model=model,
        alphabet=alphabet,
        batch_size=args.batch_size,
        repr_layer=args.repr_layer,
        skip_existing=not args.no_skip_existing,
    )

    # Save skipped sequences
    if skipped_long:
        skip_df = pd.DataFrame(skipped_long, columns=["Entry", "Length"])
        skip_df.to_csv("skipped_long_sequences.csv", index=False)
        log("[WARN] Some sequences skipped due to length")

    log("========== SUMMARY ==========")
    log(f"Embeddings written: {num_embeddings}")
    log(f"Output directory: {out_dir}")
    log("================================")


if __name__ == "__main__":
    main()
