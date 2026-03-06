import re

# -----------------------
# File paths
# -----------------------
signalp_files = [
    "../../data/fig_6_zebrafish_secretome/signalP_output_zebrafish.txt",
]

input_fasta = "../../data/fig_6_zebrafish_secretome/secretome_no_TMRs.fasta"

output_fasta = "../../data/fig_6_zebrafish_secretome/secretome_no_TMRs_signalP_filtered.fasta"


# -----------------------
# Helper: extract UniProt accession
# -----------------------
def extract_accession(text):
    """
    Extract UniProt accession like A0A0C5B5G6 from:
    - sp|A0A0C5B5G6|MOTSC_HUMAN
    - sp_A0A0C5B5G6_MOTSC_HUMAN
    """
    match = re.search(r'[A-Z0-9]{6,10}', text)
    return match.group(0) if match else None

# -----------------------
# Parse SignalP outputs
# -----------------------
keep_accessions = set()

for file in signalp_files:
    with open(file) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue

            parts = line.rstrip("\n").split("\t")

            full_id = parts[0]
            prediction = parts[1]
            sp_score = float(parts[3])  # SP(Sec/SPI)

            accession = extract_accession(full_id)

            if accession and sp_score >= 0.4:
                keep_accessions.add(accession)

# -----------------------
# Parse FASTA and filter
# -----------------------
kept_records = []
total_records = 0

with open(input_fasta) as f:
    header = None
    seq_lines = []

    for line in f:
        if line.startswith(">"):
            if header:
                total_records += 1
                accession = extract_accession(header)
                if accession in keep_accessions:
                    kept_records.append((header, "".join(seq_lines)))

            header = line.strip()
            seq_lines = []
        else:
            seq_lines.append(line.strip())

    # last record
    if header:
        total_records += 1
        accession = extract_accession(header)
        if accession in keep_accessions:
            kept_records.append((header, "".join(seq_lines)))

# -----------------------
# Write filtered FASTA
# -----------------------
with open(output_fasta, "w") as out:
    for header, seq in kept_records:
        out.write(f"{header}\n")
        for i in range(0, len(seq), 60):
            out.write(seq[i:i+60] + "\n")

# -----------------------
# Report counts
# -----------------------
print(f"Total sequences before filtering: {total_records}")
print(f"Total sequences after filtering:  {len(kept_records)}")
