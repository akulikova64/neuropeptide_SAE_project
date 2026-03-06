#!/usr/bin/env python3

import time
import csv
import requests
from bs4 import BeautifulSoup
from Bio import SeqIO

# ---------------- CONFIG ----------------

FRL_URL = "http://kurata14.bio.kyutech.ac.jp/NeuroPred-FRL/prediction.php"   
FASTA_IN = "../../data/final_human_secretome_entries_1201.fasta"
OUT_CSV  = "../../data/neuropredFRL_predictions_1201.csv"

BATCH_SIZE = 5
SLEEP_SECONDS = 10   # be polite — avoid hammering server

# HTML form field names (inspect page source!)
SEQ_FIELD_NAME = "sequence"     # 🔴 REPLACE
SUBMIT_FIELD   = "submit"       # 🔴 REPLACE

# ---------------- HELPERS ----------------

def batched(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def parse_results(html_text):
    """
    Parse NeuroPred-FRL HTML output.
    You MUST adjust this to match the page structure.
    """
    soup = BeautifulSoup(html_text, "html.parser")

    results = []

    # 🔴 EXAMPLE: adjust selector to match real output
    table = soup.find("table")
    if not table:
        return results

    rows = table.find_all("tr")[1:]  # skip header
    for r in rows:
        cols = [c.get_text(strip=True) for c in r.find_all("td")]
        if len(cols) >= 2:
            entry = cols[0]
            score = cols[1]
            results.append((entry, score))

    return results

# ---------------- MAIN ----------------

def main():
    sequences = []
    for rec in SeqIO.parse(FASTA_IN, "fasta"):
        sequences.append((rec.id.split()[0], str(rec.seq)))

    print(f"[i] Loaded {len(sequences)} sequences")

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Entry", "NeuroPredFRL_score"])

        processed = 0

        for batch in batched(sequences, BATCH_SIZE):
            print(f"[i] Sending batch {processed+1}–{processed+len(batch)}")

            # Build FASTA-like submission text
            seq_text = "\n".join(
                f">{entry}\n{seq}" for entry, seq in batch
            )

            payload = {
                SEQ_FIELD_NAME: seq_text,
                SUBMIT_FIELD: "Submit"
            }

            response = requests.post(FRL_URL, data=payload, timeout=120)
            response.raise_for_status()

            batch_results = parse_results(response.text)

            for entry, score in batch_results:
                writer.writerow([entry, score])

            processed += len(batch)
            print(f"[i] Processed {processed}/{len(sequences)}")

            time.sleep(SLEEP_SECONDS)

    print("== DONE ===============================")
    print(f"Output written to: {OUT_CSV}")
    print("======================================")

if __name__ == "__main__":
    main()
