#!/usr/bin/env python3
"""
Download UniProt FASTA for: secreted proteins (KW-0964) EXCLUDING neuropeptides (KW-0527).

- Uses UniProt REST 'stream' endpoint with server-side gzip.
- Writes a single .fasta.gz file (safe for very large datasets).
- Prints an estimate of total matching records before download.

Example:
    python download_uniprot_secreted_minus_neuropeptides.py \
        --out "/Volumes/T7 Shield/uniprot_secreted_minus_neuropeptides.fasta.gz"
"""

import argparse
import gzip
import io
import os
import sys
import time
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


UNIPROT_STREAM_URL = "https://rest.uniprot.org/uniprotkb/stream"
UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"

# Query: secreted (KW-0964) AND NOT neuropeptide (KW-0527)
DEFAULT_QUERY = "(keyword:KW-0964) NOT (keyword:KW-0527)"


def make_session(total_retries: int = 5, backoff_factor: float = 0.5, timeout: int = 120) -> requests.Session:
    """Create a requests Session with retry/backoff."""
    retry = Retry(
        total=total_retries,
        read=total_retries,
        connect=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=10)
    s = requests.Session()
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    # Helpful, but not strictly necessary:
    s.headers.update({
        "User-Agent": "uniprot-bulk-download/1.0 (+https://example.org/contact)",
        "Accept": "*/*",
    })
    # Stash timeout on the session for convenience
    s.request_timeout = timeout
    return s


def get_estimated_count(session: requests.Session, query: str) -> Optional[int]:
    """Ask UniProt for an estimated total using a zero-size search."""
    try:
        # size=0 returns only headers with x-total-results
        r = session.get(
            UNIPROT_SEARCH_URL,
            params={"query": query, "size": 0},
            timeout=session.request_timeout,
        )
        r.raise_for_status()
        total = r.headers.get("x-total-results") or r.headers.get("X-Total-Results")
        return int(total) if total is not None else None
    except Exception:
        return None


def download_fasta_gz(session: requests.Session, query: str, out_path: str) -> None:
    """Stream server-gzipped FASTA and write to a .gz file on disk."""
    params = {
        "query": query,
        "format": "fasta",
        "compressed": "true",  # server-side gzip
    }

    # Stream the HTTP response to disk verbatim (already gzipped)
    with session.get(UNIPROT_STREAM_URL, params=params, stream=True, timeout=session.request_timeout) as r:
        r.raise_for_status()

        # Create parent dir if needed
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        bytes_written = 0
        t0 = time.time()
        with open(out_path, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1 MiB chunks
                if chunk:
                    fh.write(chunk)
                    bytes_written += len(chunk)

        dt = time.time() - t0
        mb = bytes_written / (1024 * 1024)
        print(f"Download complete → {out_path}")
        print(f"Wrote: {mb:,.1f} MiB in {dt:,.1f} s ({mb / max(dt,1e-6):,.1f} MiB/s)")

    # Optional quick sanity: peek first line without decompressing the entire file.
    # (We can stream-decompress just the beginning.)
    try:
        with open(out_path, "rb") as fh:
            with gzip.GzipFile(fileobj=fh, mode="rb") as gz:
                head = gz.readline().decode("utf-8", errors="replace").strip()
        if not head.startswith(">"):
            print("Warning: first line in FASTA does not start with '>' — file may be empty or unexpected format.", file=sys.stderr)
        else:
            print(f"First FASTA header: {head[:100]}{'...' if len(head) > 100 else ''}")
    except Exception as e:
        print(f"Note: could not peek into gzip file to validate header ({e}).", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description="Download UniProt secreted proteins (KW-0964) minus neuropeptides (KW-0527) as FASTA (.gz).")
    ap.add_argument("--out", required=True, help="Output .fasta.gz path (e.g., /Volumes/T7 Shield/secreted_minus_neuropeptides.fasta.gz)")
    ap.add_argument("--query", default=DEFAULT_QUERY, help=f'Custom UniProt query (default: {DEFAULT_QUERY!r})')
    ap.add_argument("--retries", type=int, default=5, help="HTTP retries (default: 5)")
    ap.add_argument("--timeout", type=int, default=120, help="Per-request timeout in seconds (default: 120)")
    args = ap.parse_args()

    if not args.out.endswith(".gz"):
        print("Note: adding .gz extension to output path for compressed FASTA.")
        args.out += ".gz"

    session = make_session(total_retries=args.retries, timeout=args.timeout)

    # Get a quick estimate
    est = get_estimated_count(session, args.query)
    if est is not None:
        print(f"Estimated matching records: {est:,}")
    else:
        print("Estimated matching records: (unavailable)")

    print("Starting download from UniProt (server-side gzip, FASTA format)…")
    download_fasta_gz(session, args.query, args.out)

    print("Done.")


if __name__ == "__main__":
    main()
