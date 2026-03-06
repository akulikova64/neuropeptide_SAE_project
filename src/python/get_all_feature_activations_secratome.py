#!/usr/bin/env python3
import os
import re
import torch
import pandas as pd

feaures_list = [3355, 645, 5544, 595, 10051, 4151, 2100, 4219, 7479, 3836, 4896] #4813

for feature_idx in feaures_list:
    print("Working on feature:", feature_idx)

    # ─── USER PARAMETERS ────────────────────────────────────────────────────
    embeddings_dir = "/Volumes/T7 Shield/layer_18_secratome_embeddings"
    # change to the feature you want
    output_csv     = "../../data/neuropep_features/feature_{}_activations_ALL.csv".format(feature_idx)
    flags_csv      = "../../data/secratome_neuropeptide_flags.csv"   # Entry, neuropeptide (TRUE/FALSE)

    # ─── HELPERS ───────────────────────────────────────────────────────────
    ACC_RE = re.compile(r"^[A-NR-Z0-9]{6,10}$")  # basic UniProt accession shape

    def accession_from_entry_string(s: str) -> str:
        """
        Normalize an Entry-like string to a UniProt accession.
        - If it contains pipes (sp|ACC|ID or tr|ACC|ID), return the middle field.
        - Else if it looks like a UniProt accession already, return as-is.
        - Else, strip anything after first whitespace, then after ':' or '_' (fallback).
        """
        s = str(s).strip()
        if '|' in s:
            parts = s.split('|')
            if len(parts) >= 2:
                return parts[1]
        # if already looks like accession
        tok = s.split()[0]
        if ACC_RE.match(tok):
            return tok
        # fallback trims
        tok = tok.split(':', 1)[0]
        tok = tok.split('_', 1)[0]
        return tok

    def accession_from_filename(fn: str) -> str:
        """
        From filenames like:
        sp|P0DI86|OXLA_BOTAL_original_SAE.pt  → P0DI86
        tr|A0A023FBW4|E1142_AMBCJ_original_SAE.pt → A0A023FBW4
        A0A8C6VKS0_original_SAE.pt → A0A8C6VKS0
        """
        base = os.path.splitext(fn)[0]
        if base.endswith("_original_SAE"):
            base = base[:-len("_original_SAE")]
        # If pipe-format, use the middle field
        if '|' in base:
            parts = base.split('|')
            if len(parts) >= 2:
                return parts[1]
        # No pipes: often already the accession at start
        return accession_from_entry_string(base)

    # ─── LOAD NEUROPEPTIDE ACCESSIONS ──────────────────────────────────────
    flags = pd.read_csv(flags_csv)
    if not {"Entry", "neuropeptide"} <= set(flags.columns):
        raise ValueError("Flags CSV must contain columns: 'Entry' and 'neuropeptide'.")

    # Normalize boolean
    flags["neuropeptide"] = flags["neuropeptide"].astype(str).str.upper().map({"TRUE": True, "FALSE": False})
    # Keep only TRUE and normalize entries to accessions
    np_accessions = set(
        flags.loc[flags["neuropeptide"] == True, "Entry"].astype(str).map(accession_from_entry_string)
    )

    print(f"[info] neuropeptide TRUE accessions loaded: {len(np_accessions):,}")

    # ─── COLLECT ACTIVATIONS (NEUROPEPTIDES ONLY) ──────────────────────────
    records = []
    missing = []
    skipped_non_np = 0

    for fn in os.listdir(embeddings_dir):
        if not fn.endswith(".pt") or fn.startswith("._"):
            continue

        acc = accession_from_filename(fn)
        if acc not in np_accessions:
            skipped_non_np += 1
            continue

        fpath = os.path.join(embeddings_dir, fn)
        try:
            tensor = torch.load(fpath, map_location="cpu")  # (seq_len, num_features)
            if tensor.ndim != 2:
                raise ValueError(f"bad shape {tuple(tensor.shape)}")
        except Exception:
            missing.append(acc)
            continue

        seq_len, num_features = tensor.shape
        if feature_idx < 0 or feature_idx >= num_features:
            raise ValueError(f"feature_idx {feature_idx} out of range (0–{num_features-1}) in {acc}")

        # extract this feature's activation at every position
        feat_vec = tensor[:, feature_idx].float().tolist()
        for i, val in enumerate(feat_vec, 1):  # 1-based positions
            records.append({"Entry": acc, "position": i, "activation": float(val)})

    if missing:
        print(f"⚠️  Failed to load or bad shape for {len(missing)} accessions (examples: {missing[:5]})")
    print(f"[info] skipped non-neuropeptides: {skipped_non_np:,}")

    # ─── SAVE TO CSV ───────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_out = pd.DataFrame.from_records(records, columns=["Entry", "position", "activation"])
    df_out.to_csv(output_csv, index=False)
    print(f"✅ Saved {len(records):,} rows to {output_csv}")
