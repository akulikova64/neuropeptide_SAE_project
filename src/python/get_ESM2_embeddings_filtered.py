#!/usr/bin/env python3
"""
Generate ESM-2 layer-18 embeddings for a subset of sequences from a UniProt FASTA,
restricted to entries listed in a CSV file.

Model: esm2_t33_650M_UR50D (CPU)
Layer: 18
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

MAX_LEN = 2000  # max sequence length (aa) to keep; longer ones will be skipped

# ── Logging helper ──────────────────────────────────────────────────────────

def log(msg: str):
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


# ── FASTA utilities ─────────────────────────────────────────────────────────

def fasta_iter_ids(path: Path):
    """
    Iterate over a FASTA file, yielding (token, header_without_>, sequence_string).
    """
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
    parts = id_token.split("|")
    if len(parts) >= 3:
        return parts[1]
    return id_token


def base_acc(acc: str) -> str:
    return re.sub(r"-\d+$", "", acc)


# ── ESM-2 embedding logic ───────────────────────────────────────────────────

def load_esm2_model(model_name: str, repr_layer: int):
    log(f"[ESM] Loading model: {model_name} (CPU)")
    model, alphabet = getattr(esm.pretrained, model_name)()
    model.eval()
    model.to("cpu")

    if repr_layer < 1 or repr_layer > model.num_layers:
        raise ValueError(
            f"repr_layer {repr_layer} is out of range for this model "
            f"(must be between 1 and {model.num_layers}, got {repr_layer})."
        )

    log(f"[ESM] Model loaded. Using repr_layer={repr_layer} / num_layers={model.num_layers}")
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
    """
    Compute ESM-2 embeddings and save one .pt per Entry.

    sequences: list of (entry, sequence_str)

    Returns:
        num_written: int
        skipped_long: list of (entry, length)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_converter = alphabet.get_batch_converter()
    total = len(sequences)
    log(f"[ESM] Computing embeddings for {total} sequences "
        f"(batch size {batch_size}, repr_layer {repr_layer}, device=cpu)")

    num_written = 0
    processed_so_far = 0  # purely for progress reporting
    skipped_long: list[tuple[str, int]] = []  # to record what we skip

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = sequences[start:end]

        # Filter out entries whose files already exist OR are too long
        effective_batch = []
        for entry, seq in batch:
            out_file = out_dir / f"{entry}.pt"

            # Skip if embedding already exists
            if skip_existing and out_file.exists():
                continue

            # Skip if sequence is too long
            if len(seq) > MAX_LEN:
                log(f"[ESM] SKIPPING ultra-long sequence {entry} "
                    f"(len={len(seq)} > {MAX_LEN})")
                skipped_long.append((entry, len(seq)))
                continue

            effective_batch.append((entry, seq))

        if not effective_batch:
            processed_so_far = end
            log(f"[ESM] Skipped batch {start}-{end} (all already done or too long). "
                f"Processed {processed_so_far}/{total} sequences")
            continue

        # Log batch sizes so you can see when big ones come in
        lengths = [len(seq) for _, seq in effective_batch]
        ids = [entry for entry, _ in effective_batch]
        log(f"[ESM] Next batch entries: {ids}, sizes (aa): {lengths}")

        try:
            data = [(entry, seq) for entry, seq in effective_batch]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to("cpu")

            with torch.no_grad():
                results = model(
                    batch_tokens,
                    repr_layers=[repr_layer],
                    return_contacts=False,
                )
            token_representations = results["representations"][repr_layer].to("cpu")

            for i, (entry, seq) in enumerate(effective_batch):
                seq_len = len(seq)
                rep = token_representations[i, 1 : seq_len + 1].mean(0)

                out_file = out_dir / f"{entry}.pt"
                torch.save(
                    {
                        "entry": entry,
                        "embedding": rep,
                        "sequence": seq,
                    },
                    out_file,
                )
                num_written += 1

        except MemoryError:
            log("[ERROR] MemoryError during batch – try reducing --batch-size or lowering MAX_LEN.")
            raise

        # clean up between batches to free RAM
        del results, token_representations, batch_tokens, batch_labels, batch_strs
        gc.collect()

        processed_so_far = end
        log(f"[ESM] Processed {processed_so_far}/{total} sequences")

    log(f"[ESM] Finished writing {num_written} embeddings in this run.")
    return num_written, skipped_long


# ── Main script ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate ESM-2 layer-18 embeddings (650M model, CPU) "
                    "for secretome sequences filtered by secretome_no_immonoglobulins.csv."
    )
    parser.add_argument(
        "--fasta",
        default="../../data/uniprotkb_taxonomy_id_9606_AND_keyword_2025_08_20.fasta",
        help="Input UniProt FASTA file.",
    )
    parser.add_argument(
        "--csv",
        default="../../data/secretome_filter_out_enzymes_5/secretome_no_immonoglobulins.csv",
        help="CSV file with an 'Entry' column listing sequences to embed.",
    )
    parser.add_argument(
        "--out-dir",
        default="/Volumes/T7 Shield/ESM2_secretome_embeddings",
        help="Directory to save ESM-2 embeddings (.pt files).",
    )
    parser.add_argument(
        "--model",
        default="esm2_t33_650M_UR50D",  # keep big model for consistency
        help="ESM-2 model name from esm.pretrained.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,  # safer than 8 for 650M on CPU
        help="Batch size for ESM-2 embedding computation.",
    )
    parser.add_argument(
        "--repr-layer",
        type=int,
        default=18,
        help="ESM-2 layer index to use for embeddings (1-based). Default: 18.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="Number of CPU threads to use (Torch/BLAS). Default: 4.",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="If set, do NOT skip entries whose .pt files already exist.",
    )
    args = parser.parse_args()

    # threading settings
    os.environ["OMP_NUM_THREADS"] = str(args.num_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.num_threads)
    torch.set_num_threads(args.num_threads)
    torch.set_num_interop_threads(args.num_threads)
    log(f"[THREADS] Using {args.num_threads} CPU threads")

    fasta_path = Path(args.fasta)
    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)

    if not fasta_path.exists():
        raise SystemExit(f"❌ FASTA not found: {fasta_path}")
    if not csv_path.exists():
        raise SystemExit(f"❌ CSV not found: {csv_path}")

    # 1) Entries from CSV
    df = pd.read_csv(csv_path)
    if "Entry" not in df.columns:
        raise SystemExit("❌ CSV must contain an 'Entry' column.")
    target_entries = set(df["Entry"].astype(str))
    num_entries_csv = len(target_entries)
    log(f"[DATA] Entries in filtered CSV: {num_entries_csv}")

    # 2) Collect sequences from FASTA
    selected = []
    found_entries = set()
    total_fasta_seqs = 0

    log(f"[DATA] Scanning FASTA: {fasta_path}")
    for token, hdr, seq in fasta_iter_ids(fasta_path):
        total_fasta_seqs += 1
        entry = token_to_entry(token)
        if entry in target_entries:
            selected.append((entry, seq))
            found_entries.add(entry)

    log(f"[DATA] Total sequences in FASTA: {total_fasta_seqs}")
    log(f"[DATA] Sequences matching CSV entries: {len(found_entries)}")

    if len(found_entries) == 0:
        raise SystemExit("❌ No sequences found for CSV entries in the FASTA.")

    # 3) Load model
    model, alphabet = load_esm2_model(args.model, repr_layer=args.repr_layer)

    # 4) Compute embeddings
    num_embeddings, skipped_long = compute_embeddings(
        sequences=selected,
        out_dir=out_dir,
        model=model,
        alphabet=alphabet,
        batch_size=args.batch_size,
        repr_layer=args.repr_layer,
        skip_existing=not args.no_skip_existing,
    )

    # Save skipped ultra-long sequences, if any
    if skipped_long:
        skip_file = "../../data/skipped_secretome_long_sequences.csv"
        df_skip = pd.DataFrame(skipped_long, columns=["Entry", "Length"])
        df_skip.to_csv(skip_file, index=False)
        log(f"[ESM] Wrote list of skipped ultra-long sequences to: {skip_file}")
    else:
        log("[ESM] No ultra-long sequences were skipped.")

    # 5) Summary
    log("========== SUMMARY ==========")
    log(f"Number of embeddings generated in this run: {num_embeddings}")
    log(f"Number of sequences in original FASTA: {total_fasta_seqs}")
    log(f"Number of entries in filtered CSV: {num_entries_csv}")
    log(f"Embeddings directory: {out_dir}")
    log("================================")


if __name__ == "__main__":
    main()
