#!/usr/bin/env python3
"""
Generate max-pooled ESM-2 layer-18 embeddings for ALL sequences in a FASTA file.

Model: esm2_t33_650M_UR50D (CPU)
Layer: 18

Per-residue embeddings are computed internally and max-pooled across residues
to produce a fixed-length (1280,) vector per sequence.
"""

import argparse
import os
from pathlib import Path
import datetime as dt
import gc

import torch
import esm

MAX_LEN = 10000  # skip ultra-long sequences


# ── Logging ────────────────────────────────────────────────────────────────

def log(msg: str):
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


# ── FASTA reader ───────────────────────────────────────────────────────────

def fasta_iter_ids(path: Path):
    """Yield (entry_id, sequence) from a FASTA file."""
    with path.open() as fh:
        header = None
        seq = []
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    entry = header.split()[0]
                    yield entry, "".join(seq)
                header = line[1:].strip()
                seq = []
            else:
                seq.append(line)
        if header is not None:
            entry = header.split()[0]
            yield entry, "".join(seq)


# ── ESM logic ──────────────────────────────────────────────────────────────

def load_esm2_model(model_name: str, repr_layer: int):
    log(f"[ESM] Loading model: {model_name} (CPU)")
    model, alphabet = getattr(esm.pretrained, model_name)()
    model.eval()
    model.to("cpu")

    if repr_layer < 1 or repr_layer > model.num_layers:
        raise ValueError(
            f"repr_layer must be between 1 and {model.num_layers}, got {repr_layer}"
        )

    log(f"[ESM] Model loaded (layers={model.num_layers}, repr_layer={repr_layer})")
    return model, alphabet


def compute_embeddings(
    sequences,
    out_dir: Path,
    model,
    alphabet,
    batch_size: int = 4,
    repr_layer: int = 18,
    skip_existing: bool = True,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    batch_converter = alphabet.get_batch_converter()

    total = len(sequences)
    log(f"[ESM] Computing embeddings for {total} sequences")

    num_written = 0
    skipped_long = []

    for start in range(0, total, batch_size):
        batch = sequences[start : start + batch_size]
        effective = []

        for entry, seq in batch:
            # filesystem-safe filename
            safe_entry = entry.replace(":", "_")
            out_file = out_dir / f"{safe_entry}.pt"

            if skip_existing and out_file.exists():
                continue
            if len(seq) > MAX_LEN:
                skipped_long.append((entry, len(seq)))
                continue

            effective.append((entry, seq))

        if not effective:
            continue

        data = [(e, s) for e, s in effective]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to("cpu")

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[repr_layer])

        reps = results["representations"][repr_layer].to("cpu")

        for i, (entry, seq) in enumerate(effective):
            # per-residue embeddings: (L, F)
            residue_reps = reps[i, 1 : len(seq) + 1]

            # max-pool across residues → (F,)
            pooled = residue_reps.max(dim=0).values

            safe_entry = entry.replace(":", "_")
            torch.save(
                {
                    "entry": entry,
                    "sequence": seq,
                    "embedding": pooled,   # (1280,)
                    "pooling": "max",
                    "model": "esm2_t33_650M_UR50D",
                    "repr_layer": repr_layer,
                },
                out_dir / f"{safe_entry}.pt",
            )

            num_written += 1

        del results, reps, batch_tokens, batch_labels, batch_strs
        gc.collect()

        log(f"[ESM] Processed {min(start + batch_size, total)}/{total}")

    log(f"[ESM] Finished. Wrote {num_written} embeddings.")
    return num_written, skipped_long


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--repr-layer", type=int, default=18)
    parser.add_argument("--num-threads", type=int, default=4)
    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.num_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.num_threads)
    torch.set_num_threads(args.num_threads)
    torch.set_num_interop_threads(args.num_threads)

    fasta = Path(args.fasta)
    out_dir = Path(args.out_dir)

    log(f"[DATA] Reading FASTA: {fasta}")
    sequences = list(fasta_iter_ids(fasta))
    log(f"[DATA] Sequences found: {len(sequences)}")

    model, alphabet = load_esm2_model(
        model_name="esm2_t33_650M_UR50D",
        repr_layer=args.repr_layer,
    )

    compute_embeddings(
        sequences=sequences,
        out_dir=out_dir,
        model=model,
        alphabet=alphabet,
        batch_size=args.batch_size,
        repr_layer=args.repr_layer,
    )


if __name__ == "__main__":
    main()
