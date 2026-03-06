#!/usr/bin/env python3
# --- optional: raise open-files limit on macOS to help mmseqs ---
try:
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = 65536 if hard >= 65536 else hard
    if soft < target:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
except Exception:
    pass
# ----------------------------------------------------------------

import argparse, os, sys, re, io, random, shutil, subprocess, time
from pathlib import Path
from typing import Iterable, Tuple, Dict, Optional
from urllib.parse import urlparse, parse_qs

try:
    import requests
except ImportError:
    requests = None
    import urllib.request as urllib_request

# ------------------------------ Config defaults ------------------------------
UNIPROT_SEARCH_BASE = "https://rest.uniprot.org/uniprotkb/search"

DEF_MAX_DOWNLOAD = 500_000
DEF_IDENTITY = 0.40      # sequence identity threshold (0.0–1.0)
DEF_COVERAGE = 0.80      # coverage threshold (0.0–1.0)
DEF_COV_MODE = 1         # mmseqs: 0=alignment, 1=target, 2=query
DEF_FINAL_SAMPLE = 32_493
RNG_SEED = 13

FASTA_ID_RE = re.compile(r'^>(\S+).*')  # capture first token after '>'


# ------------------------------ Helpers --------------------------------------
def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def iter_fasta_records(handle: Iterable[str]) -> Iterable[Tuple[str, str, str]]:
    """
    Robust FASTA iterator: skips malformed headers (e.g., lone '>'),
    tolerates stray blank lines, and ignores leading BOMs.
    Yields (header_line, seq_id, seq_string).
    """
    header, seq_lines = None, []

    def safe_yield(hdr: str, seq_lines_):
        if hdr.startswith("\ufeff"):
            hdr = hdr.lstrip("\ufeff")
        m = FASTA_ID_RE.match(hdr.strip())
        if not m:
            return None  # skip malformed header
        seq = ''.join(seq_lines_).replace(' ', '').replace('\r', '')
        if not seq:
            return None  # skip empty sequence
        seq_id = m.group(1)
        return (hdr, seq_id, seq)

    for raw in handle:
        if isinstance(raw, bytes):
            raw = raw.decode('utf-8', errors='ignore')
        line = raw.rstrip('\n')
        if not line:
            continue
        if line.startswith('>') or line.startswith('\ufeff>'):
            if header is not None:
                rec = safe_yield(header, seq_lines)
                if rec is not None:
                    yield rec
            header, seq_lines = line, []
        else:
            if header is None:
                continue
            seq_lines.append(line)

    if header is not None:
        rec = safe_yield(header, seq_lines)
        if rec is not None:
            yield rec


def write_fasta(records: Iterable[Tuple[str,str,str]], out_path: Path) -> int:
    out_path = out_path.resolve()
    n = 0
    with out_path.open('w') as fh:
        for header, _, seq in records:
            print(header, file=fh)
            for i in range(0, len(seq), 60):
                print(seq[i:i+60], file=fh)
            n += 1
    return n


def fasta_ids(fasta_path: Path) -> set:
    fasta_path = fasta_path.resolve()
    ids = set()
    with fasta_path.open() as fh:
        for line in fh:
            if line.startswith('>'):
                m = FASTA_ID_RE.match(line.strip())
                ids.add(m.group(1) if m else line.strip()[1:].split()[0])
    return ids


def run_mmseqs(cmd: list, workdir: Path):
    workdir = workdir.resolve()
    log("MMseqs2: " + ' '.join(cmd) + f"  [cwd={workdir}]")
    res = subprocess.run(cmd, cwd=str(workdir), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        sys.stderr.write("\n=== mmseqs STDOUT ===\n" + (res.stdout or "") + "\n")
        sys.stderr.write("=== mmseqs STDERR ===\n" + (res.stderr or "") + "\n")
        raise RuntimeError(f"MMseqs2 command failed (exit {res.returncode}). See STDERR above.")
    return res


def build_mmseqs_db(fasta: Path, db_out: Path, tmp: Path):
    run_mmseqs(["mmseqs", "createdb", str(fasta.resolve()), str(db_out.resolve())], tmp)


def mmseqs_search(query_db: Path, target_db: Path, aln_out: Path, tmp: Path,
                  identity=DEF_IDENTITY, coverage=DEF_COVERAGE, cov_mode=DEF_COV_MODE, threads: int = 0):
    """
    Run sensitive mmseqs search (NP queries -> secreted targets),
    then convert to TSV and filter by identity & coverage in Python.
    """
    query_db = query_db.resolve()
    target_db = target_db.resolve()
    aln_out = aln_out.resolve()
    tmp = tmp.resolve()
    tmp_search = (tmp / "tmp_search").resolve()

    # 1) search (alignment DB)
    run_mmseqs([
        "mmseqs", "search",
        str(query_db), str(target_db), str(aln_out), str(tmp_search),
        "--threads", str(threads if threads > 0 else os.cpu_count() or 1),
        "-s", "7.5"
    ], tmp)

    # 2) convert entire alignment DB to TSV (no pre-filtering here)
    tsv = aln_out.with_suffix(".tsv")
    run_mmseqs([
        "mmseqs", "convertalis",
        str(query_db), str(target_db), str(aln_out), str(tsv),
        "--format-output",
        # qId,tId,pident,alnlen,mismatch,gapopen,qStart,qEnd,tStart,tEnd,qLen,tLen
        "query,target,pident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,qlen,tlen"
    ], tmp)

    # 3) Filter by identity & coverage in Python
    keep = []
    with tsv.open() as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 12:
                continue
            qid, tid = parts[0], parts[1]
            try:
                pident = float(parts[2])
                qstart, qend = int(parts[6]), int(parts[7])
                tstart, tend = int(parts[8]), int(parts[9])
                qlen,  tlen  = int(parts[10]), int(parts[11])
            except ValueError:
                continue
            if pident < identity * 100.0:  # convert fraction to %
                continue
            alen = abs(qend - qstart) + 1
            cov_t = alen / max(1, tlen)
            cov_q = alen / max(1, qlen)
            cov = cov_t if cov_mode == 1 else (cov_q if cov_mode == 2 else (alen / max(1, min(qlen, tlen))))
            if cov >= coverage:
                keep.append((qid, tid, pident, cov))

    tsv_filt = tsv.with_suffix(".covfilt.tsv")
    with tsv_filt.open("w") as out:
        for qid, tid, pident, cov in keep:
            out.write(f"{qid}\t{tid}\t{pident:.3f}\t{cov:.3f}\n")
    log(f"MMseqs2: kept {len(keep):,} NP↔secreted hits after identity≥{identity}, coverage≥{coverage} (cov_mode={cov_mode})")
    return tsv_filt

def drop_ids_from_fasta(source_fasta: Path, ids_to_drop: set, out_fasta: Path) -> int:
    source_fasta = source_fasta.resolve()
    out_fasta = out_fasta.resolve()
    dropped = 0
    with source_fasta.open() as fh, out_fasta.open('w') as out:
        write_seq = False
        current_id = None
        for line in fh:
            if line.startswith('>'):
                m = FASTA_ID_RE.match(line.strip())
                current_id = m.group(1) if m else line.strip()[1:].split()[0]
                if current_id in ids_to_drop:
                    write_seq = False
                    dropped += 1
                else:
                    write_seq = True
                    out.write(line)
            else:
                if write_seq:
                    out.write(line)
    kept = sum(1 for _ in fasta_ids(out_fasta))
    log(f"Filtered FASTA: dropped {dropped:,}, kept {kept:,}")
    return kept


def sample_fasta(source_fasta: Path, n_samples: int, out_fasta: Path, seed=RNG_SEED) -> int:
    source_fasta = source_fasta.resolve()
    out_fasta = out_fasta.resolve()
    log("Indexing FASTA for sampling …")
    records = []
    with source_fasta.open() as fh:
        pos = fh.tell()
        header = None
        while True:
            line = fh.readline()
            if not line:
                if header is not None:
                    records.append((header_pos, seq_start, fh.tell()))
                break
            if line.startswith('>'):
                if header is not None:
                    records.append((header_pos, seq_start, pos))
                header = line
                header_pos = pos
                seq_start = fh.tell()
            pos = fh.tell()

    total = len(records)
    if total == 0:
        log("No records to sample.")
        return 0
    if n_samples > total:
        log(f"Requested {n_samples:,} but only {total:,} available. Will take all.")
        n_samples = total

    random.seed(RNG_SEED if seed is None else seed)
    idxs = set(random.sample(range(total), n_samples))

    written = 0
    with source_fasta.open() as fh, out_fasta.open('w') as out:
        for i, (hpos, sstart, endpos) in enumerate(records):
            if i in idxs:
                fh.seek(hpos)
                header = fh.readline()
                out.write(header)
                fh.seek(sstart)
                while fh.tell() < endpos:
                    out.write(fh.readline())
                written += 1
    log(f"Sampled {written:,} sequences → {out_fasta}")
    return written


# ------------------------- UniProt paginated download -------------------------
def _next_link(headers: Dict[str, str]) -> Optional[str]:
    link = headers.get("Link") or headers.get("link")
    if not link:
        return None
    for part in [p.strip() for p in link.split(",")]:
        if 'rel="next"' in part:
            s = part.find("<")
            e = part.find(">")
            if s != -1 and e != -1 and e > s:
                return part[s+1:e]
    return None

def paged_uniprot_fasta(query: str, limit: int, out_fasta: Path,
                        page_size: int = 500, reviewed: bool = False,
                        timeout: int = 90, max_retries: int = 6, backoff: float = 1.6):
    """
    Robust, paginated download using UniProt /search with cursor paging.
    Streams lines via requests.iter_lines() to avoid 'I/O on closed file' errors.
    """
    out_fasta = out_fasta.resolve()
    out_fasta.parent.mkdir(parents=True, exist_ok=True)

    params = {
        "query": query + (" AND reviewed:true" if reviewed else ""),
        "format": "fasta",
        "size": str(page_size),
    }

    n_total = 0
    url = UNIPROT_SEARCH_BASE
    log(f"Downloading up to {limit:,} sequences via paged API (size={page_size}, reviewed={reviewed}) …")
    with out_fasta.open("w") as out:
        while n_total < limit and url:
            tries = 0
            while True:
                try:
                    if requests is None:
                        # Fallback: single-shot without paging robustness
                        from urllib.request import urlopen, Request
                        req_url = url if "cursor=" in url else (url + "?" + "&".join(f"{k}={v}" for k,v in params.items()))
                        req = Request(req_url, headers={"User-Agent":"ak-negset/1.0"})
                        with urllib_request.urlopen(req, timeout=timeout) as resp:
                            # Simple read-all fallback
                            content = resp.read().decode("utf-8", errors="ignore").splitlines()
                            wrote = 0
                            for header, seq_id, seq in iter_fasta_records(content):
                                print(header, file=out)
                                for i in range(0, len(seq), 60):
                                    print(seq[i:i+60], file=out)
                                n_total += 1; wrote += 1
                                if n_total >= limit:
                                    break
                        next_url = None
                    else:
                        r = requests.get(
                            url,
                            params=params if "cursor" not in url else None,
                            stream=True,
                            timeout=timeout,
                            headers={"User-Agent":"ak-negset/1.0"}
                        )
                        r.raise_for_status()

                        # SAFER: iterate decoded lines
                        wrote = 0
                        line_iter = r.iter_lines(decode_unicode=True, chunk_size=8192)
                        # feed directly to our robust FASTA iterator
                        for header, seq_id, seq in iter_fasta_records(line_iter):
                            print(header, file=out)
                            for i in range(0, len(seq), 60):
                                print(seq[i:i+60], file=out)
                            n_total += 1; wrote += 1
                            if n_total >= limit:
                                break

                        # parse Link header for cursor
                        next_url = (r.headers.get("Link") or r.headers.get("link") or "")
                        if 'rel="next"' in next_url:
                            # extract <...> with rel=next
                            nxt = None
                            for part in [p.strip() for p in next_url.split(",")]:
                                if 'rel="next"' in part:
                                    s = part.find("<"); e = part.find(">")
                                    if s != -1 and e != -1 and e > s: nxt = part[s+1:e]
                            next_url = nxt
                        else:
                            next_url = None

                    # no more data?
                    if wrote == 0 and not next_url:
                        url = None
                        break

                    url = next_url
                    params = None  # after first successful call cursor is embedded in URL
                    break  # page succeeded
                except Exception as e:
                    tries += 1
                    if tries > max_retries:
                        raise
                    sleep_s = backoff ** tries
                    log(f"Page fetch error ({e}); retry {tries}/{max_retries} in {sleep_s:.1f}s …")
                    time.sleep(sleep_s)

            if n_total and (n_total % 5000 == 0 or n_total >= limit):
                log(f"…downloaded {n_total:,} so far")

    log(f"Downloaded {n_total:,} sequences → {out_fasta}")

# ------------------------------ Main pipeline --------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Curate secreted non-neuropeptide negatives from UniProt and remove sequences similar to your neuropeptides (via MMseqs2)."
    )
    ap.add_argument("--neuropep-fasta", default="/Volumes/T7 Shield/neuropep_training_SAE.fasta",
                    help="Path to your neuropeptide FASTA (positives).")
    ap.add_argument("--outdir", default="negatives_build",
                    help="Output directory.")
    ap.add_argument("--download", action="store_true",
                    help="Download from UniProt (KW-0964) using paged API.")
    ap.add_argument("--preDownloaded-fasta", default="",
                    help="If provided, use this FASTA as the secreted candidate pool (skip download).")
    ap.add_argument("--max-download", type=int, default=DEF_MAX_DOWNLOAD,
                    help="Max sequences to pull from UniProt.")
    ap.add_argument("--page-size", type=int, default=500,
                    help="UniProt page size per request (max ~500).")
    ap.add_argument("--reviewed", action="store_true",
                    help="Limit to reviewed:true (Swiss-Prot).")
    ap.add_argument("--timeout", type=int, default=60,
                    help="HTTP timeout (seconds) per page.")

    ap.add_argument("--identity", type=float, default=DEF_IDENTITY,
                    help="MMseqs identity threshold (e.g., 0.4).")
    ap.add_argument("--coverage", type=float, default=DEF_COVERAGE,
                    help="MMseqs coverage threshold (e.g., 0.8).")
    ap.add_argument("--cov-mode", type=int, default=DEF_COV_MODE,
                    help="Coverage mode: 1=target (default), 2=query, 0=alignment.")
    ap.add_argument("--final-n", type=int, default=DEF_FINAL_SAMPLE,
                    help="Final number of negatives to sample.")
    ap.add_argument("--threads", type=int, default=0, help="Threads for MMseqs2.")
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve(); outdir.mkdir(parents=True, exist_ok=True)
    tmpdir = (outdir / "mmseqs_tmp").resolve()
    if tmpdir.exists():
        shutil.rmtree(tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)

    neuropep_fasta = Path(args.neuropep_fasta).resolve()
    if not neuropep_fasta.exists():
        sys.exit(f"Neuropeptide FASTA not found: {neuropep_fasta}")

    # Step 1: get secreted sequences (KW-0964)
    if args.preDownloaded_fasta:
        secreted_raw = Path(args.preDownloaded_fasta).resolve()
        log(f"Using pre-downloaded FASTA: {secreted_raw}")
    else:
        secreted_raw = (outdir / "secreted_kw0964_raw.fasta").resolve()
        if args.download:
            paged_uniprot_fasta(
                query="keyword:KW-0964",
                limit=args.max_download,
                out_fasta=secreted_raw,
                page_size=args.page_size,
                reviewed=args.reviewed,
                timeout=args.timeout
            )
        else:
            sys.exit("Provide --download or --preDownloaded-fasta to supply the secreted candidate pool.")

    # Optional: light scrub to pre-exclude obvious neuropeptide/prohormone names
    log("Light keyword scrub (drop headers containing neuropeptide-ish terms)…")
    keywords = re.compile(r"(neuropeptide|prohormone|proneuropeptide|NPY|POMC|neurotensin|tachykinin)", re.I)
    scrubbed = (outdir / "secreted_kw0964_scrubbed.fasta").resolve()
    with secreted_raw.open() as fh, scrubbed.open('w') as out:
        write = False
        for line in fh:
            if line.startswith('>'):
                hdr = line.strip()
                write = not keywords.search(hdr)
                if write:
                    out.write(line if line.endswith("\n") else line + "\n")
            else:
                if write:
                    out.write(line)
    log(f"Scrubbed FASTA saved → {scrubbed}")

    # Step 2: remove anything too similar to neuropeptides using MMseqs2
    secreted_db = (outdir / "secreted_DB").resolve()
    neuro_db    = (outdir / "neuropep_DB").resolve()
    aln_out     = (outdir / "np_vs_secreted_aln").resolve()

    log("Building MMseqs2 DBs …")
    build_mmseqs_db(scrubbed, secreted_db, tmpdir)
    build_mmseqs_db(neuropep_fasta, neuro_db, tmpdir)

    # Search (NP queries → secreted targets), then filter by identity/coverage
    tsv_covfilt = mmseqs_search(
        query_db=neuro_db, target_db=secreted_db, aln_out=aln_out, tmp=tmpdir,
        identity=args.identity, coverage=args.coverage, cov_mode=args.cov_mode, threads=args.threads
    )

    # Collect secreted IDs that were "too close" to neuropeptides
    too_close_ids = set()
    with tsv_covfilt.open() as fh:
        for line in fh:
            tid = line.split("\t")[1]
            too_close_ids.add(tid)

    # Also ensure exact neuropeptides are dropped if present
    np_ids = fasta_ids(neuropep_fasta)
    drop_ids = too_close_ids | np_ids
    log(f"IDs to drop (similar to NP or exact NP): {len(drop_ids):,}")

    filtered_fasta = (outdir / "secreted_filtered_not_similar_to_NP.fasta").resolve()
    _ = drop_ids_from_fasta(scrubbed, drop_ids, filtered_fasta)

    # Step 3: sample desired count (or all if fewer available)
    final_out = (outdir / f"secreted_non_neuropeptides_{args.final_n}.fasta").resolve()
    sample_fasta(filtered_fasta, args.final_n, final_out, seed=RNG_SEED)

    log("Done.")
    log(f"Final negatives FASTA → {final_out}")


if __name__ == "__main__":
    main()
