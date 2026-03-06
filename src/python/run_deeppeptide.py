#!/usr/bin/env python3
import argparse, json, os, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Dict

# ---------- FASTA utils ----------
def read_fasta(path: Path) -> List[Tuple[str, str]]:
    recs, h, buf = [], None, []
    with path.open() as fh:
        for line in fh:
            if line.startswith(">"):
                if h is not None:
                    recs.append((h, "".join(buf).replace(" ", "")))
                h, buf = line[1:].strip(), []
            else:
                buf.append(line.strip())
    if h is not None:
        recs.append((h, "".join(buf).replace(" ", "")))
    return recs

def write_fasta(records: List[Tuple[str,str]], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for h, s in records:
            f.write(f">{h}\n")
            for i in range(0, len(s), 80):
                f.write(s[i:i+80] + "\n")

def bin_by_length(records: List[Tuple[str,str]], max_aa: int, max_seqs: int) -> List[List[Tuple[str,str]]]:
    recs = sorted(records, key=lambda r: len(r[1]), reverse=True)
    bins, aa_sums = [], []
    for r in recs:
        placed = False
        for i, b in enumerate(bins):
            if aa_sums[i] + len(r[1]) <= max_aa and len(b) < max_seqs:
                b.append(r); aa_sums[i] += len(r[1]); placed = True; break
        if not placed:
            bins.append([r]); aa_sums.append(len(r[1]))
    return bins

# ---------- predictor CLI detection ----------
def run_help(cmd: List[str]) -> str:
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, text=True)
        return p.stdout or ""
    except Exception:
        return ""

def detect_entry_and_flags(python_exe: str) -> Tuple[List[str], str, str]:
    """
    Try common entrypoints, parse --help to find input/output flags.
    Returns: (base_cmd, input_flag, output_flag)
    """
    candidates = [
        [python_exe, "-m", "predictor", "--help"],
        [python_exe, "predictor/main.py", "--help"],
        [python_exe, "predictor/app.py", "--help"],
    ]
    for c in candidates:
        help_txt = run_help(c)
        if not help_txt:
            continue
        # guess input flag
        in_flag = next((f for f in ["--fasta","--input","--in","-i"] if f in help_txt), "--fasta")
        # guess output flag (directory or file)
        out_flag = next((f for f in ["--outdir","--output_dir","--output","--out","-o"] if f in help_txt), "--outdir")
        # return the same command but without --help
        base_cmd = c[:-1]
        return base_cmd, in_flag, out_flag
    # fallback (still works for many CLIs)
    return [python_exe, "predictor/main.py"], "--fasta", "--outdir"

# ---------- run a chunk ----------
def run_one_chunk(python_exe: str, chunk_fa: Path, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    base_cmd, in_flag, out_flag = detect_entry_and_flags(python_exe)
    cmd = base_cmd + [in_flag, str(chunk_fa), out_flag, str(out_dir)]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with (out_dir / "cmd.txt").open("w") as f: f.write(" ".join(cmd))

    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    (out_dir / "stdout.txt").write_text(p.stdout or "")
    (out_dir / "stderr.txt").write_text(p.stderr or "")
    (out_dir / "exit_code.txt").write_text(str(p.returncode))
    return p.returncode

# ---------- merging ----------
def merge_tabular(chunks_root: Path, combined_root: Path, exts=(".csv",".tsv")):
    combined_root.mkdir(parents=True, exist_ok=True)
    filemap: Dict[str, List[Path]] = {}
    for cdir in sorted(chunks_root.glob("chunk_*")):
        for ext in exts:
            for f in cdir.glob(f"*{ext}"):
                filemap.setdefault(f.name, []).append(f)
    for fname, paths in filemap.items():
        if not paths: continue
        sep = "," if fname.endswith(".csv") else "\t"
        out = combined_root / fname
        wrote_header = False
        with out.open("w", encoding="utf-8") as outfh:
            for p in sorted(paths):
                with p.open("r", encoding="utf-8") as infh:
                    for i, line in enumerate(infh):
                        if i == 0:
                            if not wrote_header:
                                outfh.write(line); wrote_header = True
                            continue
                        outfh.write(line)

def merge_json(chunks_root: Path, combined_root: Path):
    # If predictor writes JSON files, concatenate arrays into one big array; else write NDJSON.
    files = sorted(chunks_root.glob("chunk_*/*.json"))
    if not files: return
    big_list, nd_lines = [], []
    for p in files:
        try:
            obj = json.loads(p.read_text())
            if isinstance(obj, list):
                big_list.extend(obj)
            else:
                nd_lines.append(json.dumps(obj))
        except Exception:
            nd_lines.append(p.read_text().strip())
    if big_list:
        (combined_root / "deeppeptide_combined.json").write_text(json.dumps(big_list, ensure_ascii=False))
    if nd_lines:
        (combined_root / "deeppeptide_combined.ndjson").write_text("\n".join(nd_lines))

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fasta", help="input FASTA")
    ap.add_argument("outdir", help="output directory")
    ap.add_argument("--max-aa", type=int, default=60_000, help="max total residues per chunk")
    ap.add_argument("--max-seqs", type=int, default=40, help="max sequences per chunk")
    ap.add_argument("--workers", type=int, default=2, help="parallel local processes")
    ap.add_argument("--python", default=sys.executable, help="python interpreter to run predictor")
    args = ap.parse_args()

    fasta = Path(args.fasta).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    chunks_dir = outdir / "chunks"
    combined_dir = outdir / "combined"
    tmp_dir = outdir / "tmp_len_chunks"

    if not fasta.exists():
        print(f"ERROR: FASTA not found: {fasta}", file=sys.stderr)
        sys.exit(2)

    outdir.mkdir(parents=True, exist_ok=True)
    print("Reading FASTA …")
    recs = read_fasta(fasta)
    total_aa = sum(len(s) for _, s in recs)
    print(f"  sequences: {len(recs)} | total aa: {total_aa:,}")

    print(f"Binning into chunks (≤{args.max_aa:,} aa, ≤{args.max_seqs} seqs)…")
    bins = bin_by_length(recs, args.max_aa, args.max_seqs)
    print(f"  created {len(bins)} chunks")

    # materialize FASTAs
    chunk_paths: List[Path] = []
    for i, b in enumerate(bins, start=1):
        cp = tmp_dir / f"chunk_{i:04d}.fasta"
        write_fasta(b, cp)
        chunk_paths.append(cp)

    failures = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        fut2path = {}
        for cp in chunk_paths:
            cdir = chunks_dir / cp.stem
            if (cdir / "exit_code.txt").exists():
                print(f"Skipping {cp.name} (already done)")
                continue
            fut = ex.submit(run_one_chunk, args.python, cp, cdir)
            fut2path[fut] = cp.name

        for fut in as_completed(fut2path):
            name = fut2path[fut]
            try:
                code = fut.result()
                print(f"  finished {name}: exit_code={code}")
                if code != 0:
                    failures.append((name, code))
            except Exception as e:
                print(f"  error in {name}: {e}")
                failures.append((name, -1))

    print("Merging outputs …")
    combined_dir.mkdir(parents=True, exist_ok=True)
    merge_tabular(chunks_dir, combined_dir)
    merge_json(chunks_dir, combined_dir)

    if failures:
        print("\nSome chunks failed:")
        for n, c in failures:
            print(f"  - {n}: exit_code={c}")
        print("Re-run with smaller --max-aa / --max-seqs to resume the failed ones.")

if __name__ == "__main__":
    main()
