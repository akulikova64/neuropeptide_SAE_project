#!/usr/bin/env python3
"""
DeepTMHMM filtering via BioLib (run API, resume-safe, auto-splits on failure).

Inputs:
  - CSV of accessions: ../../data/secretome_filter_out_nocdhit/secretome_filtered_nonenzymes_nocsdhit.csv  (col: Entry)
  - Master FASTA:      ../../data/uniprotkb_taxonomy_id_9606_AND_keyword_2025_08_20.fasta

Outputs (under ../../data/secretome_filter_out_nocdhit/deeptmhmm_biolib):
  - chunk folders with DeepTMHMM outputs
  - secretome_nontransmembrane_entries.csv  (single column Entry)
"""

from __future__ import annotations
import os, sys, json, time
from pathlib import Path
from typing import List, Tuple, Dict

# ---------- Paths ----------
BASE             = Path(__file__).resolve().parent
INPUT_CSV        = BASE / "../../data/secretome_filter_out_nocdhit/secretome_filtered_nonenzymes_nocsdhit.csv"
MASTER_FASTA     = BASE / "../../data/uniprotkb_taxonomy_id_9606_AND_keyword_2025_08_20.fasta"
OUT_ROOT         = BASE / "../../data/secretome_filter_out_nocdhit/deeptmhmm_biolib"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
FINAL_NON_TM_CSV = OUT_ROOT / "secretome_nontransmembrane_entries.csv"

# ---------- Controls ----------
CHUNK_SIZE   = int(os.environ.get("CHUNK_SIZE", "120"))   # starting batch size
MIN_CHUNK    = int(os.environ.get("MIN_CHUNK",  "15"))    # smallest allowed on split
ONLY_FIRST_N = os.environ.get("ONLY_FIRST_N")
ONLY_FIRST_N = int(ONLY_FIRST_N) if (ONLY_FIRST_N or "").isdigit() else None

# ---------- BioLib (pybiolib client) ----------
try:
    import biolib  # must be the client that has .load and returns a Result from .run()
    if not hasattr(biolib, "load"):
        raise ImportError("This 'biolib' is not the pybiolib client. Install with: python -m pip install pybiolib")
except Exception as e:
    sys.exit(f"❌ BioLib client not available in this Python. Install and re-run:\n"
             f"   python -m pip install pybiolib\nError: {e}")

# ---------- FASTA helpers ----------
def token_to_entry(id_token: str) -> str:
    parts = id_token.split("|")
    if len(parts) >= 3:
        return parts[1]
    return id_token

def parse_fasta(path: Path):
    """Yield (header_without_>, sequence)."""
    hdr, seq = None, []
    with path.open() as fh:
        for line in fh:
            if line.startswith(">"):
                if hdr is not None:
                    yield hdr, "".join(seq)
                hdr, seq = line[1:].strip(), []
            else:
                seq.append(line.strip())
        if hdr is not None:
            yield hdr, "".join(seq)

def write_fasta(records: List[Tuple[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as out:
        for hdr, seq in records:
            out.write(f">{hdr}\n")
            for i in range(0, len(seq), 80):
                out.write(seq[i:i+80] + "\n")

# ---------- DeepTMHMM run + parsing ----------
def run_deeptmhmm_blocking(fa_path: Path, workdir: Path) -> bool:
    """
    Run DeepTMHMM via BioLib .run(), stream logs, and save outputs into workdir.
    Returns True if outputs saved & topology file(s) found, else False.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    app = biolib.load("DTU/DeepTMHMM")
    print(f"     • Running DeepTMHMM on {fa_path.name} (blocking, logs below) …")

    try:
        # This call blocks until the job finishes remotely and returns a Result
        result = app.run(fasta=str(fa_path), biolib_stream_logs=True)
    except Exception as e:
        print(f"     ✖ DeepTMHMM run() raised: {e}")
        return False

    # Save all output files into workdir
    try:
        result.save_files(str(workdir))
    except TypeError:
        # old client versions didn't accept kwargs; this matches their signature
        result.save_files(str(workdir))
    except Exception as e:
        print(f"     ✖ result.save_files() failed: {e}")
        return False

    # Check we actually got a topology-like file
    topo = []
    for pat in ("**/*topolog*.tsv", "**/*topolog*.txt", "**/*topolog*.csv", "**/*topolo*"):
        topo.extend(workdir.glob(pat))
    if not topo:
        print("     ⚠️ No topology files present after save; will treat as failure.")
        return False

    (workdir / "_done.ok").write_text("ok\n")
    return True

def parse_topologies_from_dir(chunk_dir: Path) -> Dict[str, str]:
    """
    Return {accession: topology_string} from any '*topolog*' files in the chunk folder.
    Very permissive parsing: first column token, last column topology.
    """
    topo_map: Dict[str, str] = {}
    files = []
    for pat in ("**/*topolog*.tsv", "**/*topolog*.txt", "**/*topolog*.csv", "**/*topolo*"):
        files.extend(sorted(chunk_dir.glob(pat)))
    for fp in files:
        try:
            text = fp.read_text()
        except Exception:
            continue
        for line in text.splitlines():
            if not line.strip():
                continue
            # TSV first, then CSV fallback
            parts = [p.strip() for p in line.split("\t")]
            if len(parts) == 1:
                parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                seq_id = parts[0].split()[0]
                topo   = parts[-1]
                acc    = token_to_entry(seq_id)
                topo_map[acc] = topo
    return topo_map

# ---------- Adaptive splitting ----------
def process_chunk(records: List[Tuple[str, str]], label: str) -> Dict[str, str]:
    """
    Attempt DeepTMHMM for this set of records; on failure, split into halves (down to MIN_CHUNK).
    Returns merged topology map for this label.
    """
    chunk_dir = OUT_ROOT / label
    fa_path   = chunk_dir / f"{label}.fasta"

    # Fast path: already finished earlier
    if (chunk_dir / "_done.ok").exists():
        topo = parse_topologies_from_dir(chunk_dir)
        if topo:
            print(f"[{label}] already done; parsed {len(topo)} topologies")
            return topo

    if not fa_path.exists():
        write_fasta(records, fa_path)
        size_kb = fa_path.stat().st_size / 1024.0
        print(f"[{label}] {len(records)} seq → {fa_path} ({size_kb:.1f} KB)")

    print(f"   • Submitting DeepTMHMM for {label} …")
    ok = run_deeptmhmm_blocking(fa_path, chunk_dir)
    if ok:
        topo = parse_topologies_from_dir(chunk_dir)
        print(f"   ✔ {label}: parsed {len(topo)} topologies")
        return topo

    # If failure and can still split, recurse
    if len(records) > MIN_CHUNK:
        mid = len(records) // 2
        left, right = records[:mid], records[mid:]
        print(f"   ↳ Splitting {label} into {label}a ({len(left)}) and {label}b ({len(right)})")
        topo_a = process_chunk(left,  label + "a")
        topo_b = process_chunk(right, label + "b")
        # mark parent as "split"
        (chunk_dir / "_done.ok").write_text("split\n")
        merged = {**topo_a, **topo_b}
        (chunk_dir / f"{label}_merged_topologies.json").write_text(json.dumps(merged, indent=2))
        return merged

    print(f"   ✖ {label}: failed and cannot split further (<= MIN_CHUNK={MIN_CHUNK}). Skipping.")
    return {}

# ---------- Main ----------
def main():
    import pandas as pd

    # 1) Load accession list
    if not INPUT_CSV.exists():
        sys.exit(f"❌ Input CSV not found: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    if "Entry" not in df.columns:
        sys.exit("❌ Input CSV must contain an 'Entry' column.")
    accs = df["Entry"].astype(str).tolist()
    if ONLY_FIRST_N:
        accs = accs[:ONLY_FIRST_N]
    print(f"• Starting with {len(accs)} accessions from {INPUT_CSV}")

    # 2) Build records from master FASTA
    if not MASTER_FASTA.exists():
        sys.exit(f"❌ Master FASTA not found: {MASTER_FASTA}")
    print("• Indexing master FASTA …")
    desired = set(accs)
    records: List[Tuple[str, str]] = []
    for hdr, seq in parse_fasta(MASTER_FASTA):
        acc = token_to_entry(hdr.split()[0])
        if acc in desired:
            records.append((hdr, seq))
    if not records:
        sys.exit("❌ None of the requested accessions were found in the FASTA.")

    total = len(records)
    print(f"• Will submit {total} sequences to DeepTMHMM "
          f"(sequential, resume-safe; initial chunk size={CHUNK_SIZE}, min={MIN_CHUNK})")

    # 3) Initial chunking
    labels_records: List[Tuple[str, List[Tuple[str, str]]]] = []
    for i in range(0, total, CHUNK_SIZE):
        label = f"chunk_{i//CHUNK_SIZE + 1:03d}"
        labels_records.append((label, records[i:i+CHUNK_SIZE]))

    # 4) Process chunks (adaptive split on failure)
    all_topo: Dict[str, str] = {}
    for label, recs in labels_records:
        topo = process_chunk(recs, label)
        all_topo.update(topo)

    if not all_topo:
        print("⚠️ No topologies parsed. Producing pass-through CSV (no TM removed).")
        pd.DataFrame({"Entry": sorted(desired)}).to_csv(FINAL_NON_TM_CSV, index=False)
        print(f"• Output CSV: {FINAL_NON_TM_CSV}")
        return

    # 5) Select non-TM (no 'M' in topology string)
    tm_accs = {acc for acc, topo in all_topo.items() if "M" in str(topo)}
    non_tm  = sorted(desired - tm_accs)

    pd.DataFrame({"Entry": non_tm}).to_csv(FINAL_NON_TM_CSV, index=False)

    print("\n✅ DeepTMHMM filtering complete.")
    print(f"   • Input accessions:     {len(desired)}")
    print(f"   • Parsed topologies:    {len(all_topo)}")
    print(f"   • Predicted TM (has M): {len(tm_accs)}")
    print(f"   • Remaining (non-TM):   {len(non_tm)}")
    print(f"   • Output CSV:           {FINAL_NON_TM_CSV}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\n✋ Interrupted.")
    except Exception as e:
        sys.exit(f"❌ Error: {e}")
