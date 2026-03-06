from interplm.sae.inference import load_sae_from_hf
from interplm.esm.embed import embed_single_sequence
from pathlib import Path
from Bio import SeqIO
import torch

# ── Fixed input FASTA files ────────────────────────────────────────────────────
positive_fasta = Path("../../data/concept_groups/sequences/group_1/group_1_positive.fasta")
negative_fasta = Path("../../data/concept_groups/sequences/group_1/group_1_negative.fasta")

fasta_dict = {
    "positive": positive_fasta,
    "negative": negative_fasta,
}

# ── Root of the external drive ─────────────────────────────────────────────────
t7_root = Path("/Volumes") / "T7 Shield"

# ── Which layers to embed (inclusive) ──────────────────────────────────────────
# InterPLM publishes a separate, sparsely-trained SAE for only a handful of ESM-2 layers—
# for the 650-million-parameter model the valid layers are:
layers = [1, 9, 18, 24, 30, 33] 

# ── Main loop ──────────────────────────────────────────────────────────────────
for layer in layers:
    print(f"▶︎ Processing ESM-2 layer {layer} and generating SAE embeddings")

    # Load the SAE **once** per layer
    sae = load_sae_from_hf(plm_model="esm2-650m", plm_layer=layer)

    # Iterate over the two datasets (positive / negative)
    for dataset_type, fasta_path in fasta_dict.items():

        # /Volumes/T7 Shield/group_1_embeddings/layer_<LAYER>/<dataset_type>/
        out_dir = (
            t7_root
            / "group_1_embeddings"
            / f"layer_{layer}"
            / dataset_type
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        # Parse each record in the FASTA file
        for record in SeqIO.parse(fasta_path, "fasta"):
            seq_id   = record.id
            sequence = str(record.seq)

            # 1) ESM embedding
            embedding = embed_single_sequence(
                sequence   = sequence,
                model_name = "esm2_t33_650M_UR50D",
                layer      = layer,
            )

            # 2) SAE encoding
            features = sae.encode(embedding)     # (L, 10200) tensor

            # 3) Save
            out_file = out_dir / f"{seq_id}_{dataset_type}_original_SAE.pt"
            torch.save(features, out_file)

            print(f"  ✓ Saved {out_file}")

print("✅ All layers finished.")
