import time
import requests
import pandas as pd
from Bio import SeqIO
from Bio.Blast import NCBIXML
from io import StringIO

# ---------------- CONFIG ----------------

#INPUT_FASTA = "../../data/novoall_high_prob_high_sinalP.fasta"
#SIGNALP_CSV = "../../data/novo_277_signalP_scores.csv"
#OUTPUT_FASTA = "../../data/novoall_high_prob_high_sinalP_blastp_top_hits.fasta"

INPUT_FASTA = "../../data/novoall_50_60_high_signalP.fasta"
SIGNALP_CSV = "../../data/novo_signalP_scores_50_60.csv"
OUTPUT_FASTA = "../../data/novoall_50_60_high_sinalP_blastp_top_hits.fasta"

EBI_SUBMIT_URL = "https://www.ebi.ac.uk/Tools/services/rest/ncbiblast/run"
EBI_STATUS_URL = "https://www.ebi.ac.uk/Tools/services/rest/ncbiblast/status/"
EBI_RESULT_URL = "https://www.ebi.ac.uk/Tools/services/rest/ncbiblast/result/"

EMAIL = "anastasiya.kulikova@biochem.utah.edu"
DATABASE = "uniprotkb_swissprot"

POLL_TIME = 15  # seconds

def normalize_entry(entry: str) -> str:
    """
    Normalize Entry IDs so FASTA headers and SignalP CSV match.
    """
    return (
        entry
        .replace("+chr", "_chr")
        .replace("-chr", "_chr")
        .replace(":", "_")
    )

# ---------------- LOAD SIGNALP ----------------

signalp_df = pd.read_csv(SIGNALP_CSV)

signalp_scores = {
    normalize_entry(entry): prob
    for entry, prob in zip(
        signalp_df["Entry"].astype(str),
        signalp_df["signal_prob"].astype(float),
    )
}

# ---------------- LOAD FASTA ----------------

records = list(SeqIO.parse(INPUT_FASTA, "fasta"))

print(f"Submitting {len(records)} sequences to EBI BLASTP")

# =========================================================
# ESSENTIAL FIX: submit ONE sequence per EBI job
# =========================================================

with open(OUTPUT_FASTA, "w") as out:

    for record in records:
        seq = str(record.seq).replace("*", "")
        if len(seq) < 20:
            continue

        fasta = f">{record.id}\n{seq}\n"

        print(f"Submitting {record.id}")

        submit_resp = requests.post(
            EBI_SUBMIT_URL,
            data={
                "email": EMAIL,
                "program": "blastp",
                "stype": "protein",
                "database": DATABASE,
                "sequence": fasta,
            }
        )

        if submit_resp.status_code != 200:
            print("EBI error response:")
            print(submit_resp.text)
            submit_resp.raise_for_status()

        job_id = submit_resp.text.strip()
        print(f"  Job ID: {job_id}")

        # ---------------- POLL ----------------

        while True:
            time.sleep(POLL_TIME)
            status = requests.get(EBI_STATUS_URL + job_id).text.strip()
            print(f"  Status: {status}")

            if status == "FINISHED":
                break
            if status in {"ERROR", "FAILURE"}:
                raise RuntimeError(f"EBI BLAST job failed for {record.id}")

        # ---------------- RETRIEVE ----------------

        result_xml = requests.get(f"{EBI_RESULT_URL}{job_id}/out",params={"format": 5}).text
        blast_record = next(NCBIXML.parse(StringIO(result_xml)))

        if not blast_record.alignments:
            continue

        aln = blast_record.alignments[0]
        hsp = aln.hsps[0]

        pct_id = round(hsp.identities / hsp.align_length * 100, 2)
        hit_name = aln.hit_def.replace(" ", "_")
        hit_seq = hsp.sbjct

        norm_id = normalize_entry(record.id)

        signalp = signalp_scores.get(norm_id)
        if signalp is not None:
            signalp = round(signalp, 6)
        else:
            signalp = "NA"

        out.write(f">{record.id}_SignalP:{signalp}\n")
        out.write(f"{seq}\n")
        out.write(
            f">hit_percent_identity:{pct_id}_to_{record.id}_hitname:{hit_name}\n"
        )
        out.write(f"{hit_seq}\n\n")

print("Done. Output written to:")
print(OUTPUT_FASTA)
