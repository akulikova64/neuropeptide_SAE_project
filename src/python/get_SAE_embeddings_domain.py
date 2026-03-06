from interplm.sae.inference import load_sae_from_hf
from interplm.esm.embed import embed_single_sequence
from Bio import SeqIO
from pathlib import Path
import torch
import os
import sys

# getting embeddings for cleavage site prediction


'''
t7_root = Path("/Volumes") / "T7 Shield"

# /Volumes/T7 Shield/ESM_embeddings/SAE_unnorm_cleavage/
output_folder = (
    t7_root
    / "ESM_embeddings"
    / "SAE_unnorm_cleavage"
)
output_folder.mkdir(parents=True, exist_ok=True)
'''

# ============== Input and Output Data Paths =============
#output_folder = "../../data/concept_groups/embeddings/group_2/training"
#output_folder = "../../data/concept_groups/embeddings/group_2/"
#output_folder = "/Volumes/T7 Shield/secratome_analysis/secratome_SAE_embeddings/"

#output_folder = "/Volumes/T7 Shield/layer_18_embeddings/"
#output_folder = "/Volumes/T7 Shield/layer_18_secratome_embeddings/"
#output_folder = "/Volumes/T7 Shield/layer_18_negative_embeddings/"
#output_folder = "/Volumes/T7 Shield/layer_18_novo_smORF_embeddings/"
#output_folder = "/Volumes/T7 Shield/layer_18_de_Souza_embeddings/"
#output_folder = "/Volumes/T7 Shield/layer_18_Whited_embeddings/"
#output_folder = "/Volumes/T7 Shield/layer_18_150_novo_embeddings/"
#output_folder = "/Volumes/T7 Shield/layer_33_highest_scoring_novo_embeddings/"
#output_folder = "/Volumes/T7 Shield/layer_33_whited_embeddings"
#output_folder = "/Volumes/T7 Shield/layer_33_de_Souza_embeddings"
#output_folder = "/Volumes/T7 Shield/layer_33_highest_scoring_secretome_embeddings"

#output_folder = "/Volumes/T7 Shield/layer_33_80_90_score_novo_embeddings"
#output_folder = "/Volumes/T7 Shield/layer_33_70_80_score_novo_embeddings"
#output_folder = "/Volumes/T7 Shield/layer_33_60_70_score_novo_embeddings"
#output_folder = "/Volumes/T7 Shield/layer_33_50_60_score_novo_embeddings"

#secretomes_output
#output_folder = "/Volumes/T7 Shield/layer_18_secretome_mouse"
#output_folder = "/Volumes/T7 Shield/layer_18_secretome_c_elegans"
#output_folder = "/Volumes/T7 Shield/layer_18_secretome_drosophila"
output_folder = "/Volumes/T7 Shield/layer_18_secretome_zebrafish"


#input_path = "../../data/novo_examples.fasta"
#input_path = "../../data/uniprotkb_taxonomy_id_9606_AND_keyword_2025_08_20.fasta"

#input_path = "/Volumes/T7 Shield/neuropep_training_SAE.fasta"
#input_path = "../../data/uniprotkb_taxonomy_id_9606_AND_keyword_2025_08_20.fasta"
#input_path = "../../data/negatives_build_simple/secreted_non_neuropeptides_32493.fasta"
#input_path = "../../data/negatives_build_matched/secreted_non_neuropeptides_32492_matched.fasta"
#input_path = "../../data/combined_l18_no_TMRs.fasta"
#input_path = "../../data/negative_dataset_matching_dist.fasta"
#input_path = "../../data/novo_smORFpipe_full_data.fasta"
#input_path = "../../data/Whited_2024_Entire_Gold_Standard.fasta"
#input_path = "../../data/peptides_162_pept.fasta"
#input_path = "../../data/novo_smORF_highest_scoring_data.fasta"
#input_path = "../../data/novo_alldata_l18_no_TMRs_highprob.fasta"
#input_path = "../../data/novo_alldata_filter_out_enzymes/neuropeptide_filter/novo_flagged_neuropeptides.fasta"
#input_path = "../../data/de_Souza_l18_highprob.fasta"
#input_path = "../../data/top_secretome_hits.fasta"

#input_path = "../../data/novo_alldata_l18_no_TMRs_50_to_60.fasta"
#input_path = "../../data/novo_alldata_l18_no_TMRs_60_to_70.fasta"
#input_path = "../../data/novo_alldata_l18_no_TMRs_70_to_80.fasta"
#input_path = "../../data/novo_alldata_l18_no_TMRs_80_to_90.fasta"

#secretomes
#input_path = "../../data/fig_6_mouse_secretome/secretome_final_cdhit_95.fasta"
#input_path = "../../data/fig_6_c_elegans_secretome/secretome_final_cdhit_95_no_long.fasta"
input_path = "../../data/fig_6_zebrafish_secretome/secretome_final_cdhit_95_no_A0AB32TF33.fasta"

# ── Root of the external drive ─────────────────────────────────────────────────

# === Load SAE only once ===
#sae = load_sae_from_hf(plm_model="esm2-650m", plm_layer=33)
sae = load_sae_from_hf(plm_model="esm2-650m", plm_layer=18)

# === Parce input fasta - get embeddings ===

for record in SeqIO.parse(input_path, "fasta"):
        
    seq_id = record.id
    sequence = str(record.seq)
    filename = f"{seq_id}_original_SAE.pt"
    output_path = os.path.join(output_folder, filename)

    if os.path.exists(output_path):         # <-- skip-existing guard
        print(f"⏭️  Skipping existing {output_path}")
        continue

    # === Generate ESM Embeddings ===
    embedding = embed_single_sequence(
        sequence=sequence,
        model_name="esm2_t33_650M_UR50D",
        layer=18 #33 
    )

    # === Encode with SAE ===
    features = sae.encode(embedding)  # shape: (sequence_length, 10200)

    # === Save features ===
    filename = f"{seq_id}_original_SAE.pt"
    output_path = os.path.join(output_folder, filename)
    
    torch.save(features, output_path)

    print(f"✅ Embedding saved to {output_path}")