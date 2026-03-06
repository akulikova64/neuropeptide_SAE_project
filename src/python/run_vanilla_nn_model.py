#!/usr/bin/env python3
import os
import pandas as pd
import torch
import numpy as np
from Bio import SeqIO
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# ─── Model factory ─────────────────────────────────────────────────────────

def make_tiny_net(num_features):
    # 1 hidden layer of 8 units →  num_features*8 + 8*1  params
    return nn.Sequential(
        nn.Linear(num_features, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
        nn.Sigmoid()
    )

'''
# "best_mine" model
def make_model(num_features):
    h1 = num_features
    h2 = max(1, num_features // 5)   # second layer is one-fifth the width
    return nn.Sequential(
        nn.Linear(num_features, h1),
        nn.ReLU(),
        nn.Linear(h1, h2),
        nn.ReLU(),
        nn.Linear(h2, 1),
        nn.Sigmoid()
    )
'''

'''
def make_model(num_features):
    hidden = num_features   # a small constant
    return nn.Sequential(
        nn.Linear(num_features, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
        nn.Sigmoid()
    )
'''
'''
def make_model(num_features):
    h1 = max(5, num_features // 5)
    h2 = max(5, num_features // 10)   # second layer is one-fifth the width
    return nn.Sequential(
        nn.Linear(num_features, h1),
        nn.ReLU(),
        nn.Linear(h1, h2),
        nn.ReLU(),
        nn.Linear(h2, 1),
        nn.Sigmoid()
    )
'''

# ─── Hyperparameters ───────────────────────────────────────────────────────
homology_training_list = [40, 50, 60, 70, 80, 90] 
feature_steps          = range(10, 501, 10)
val_fraction           = 0.2
random_seed            = 42

# parameters for "mine_model"
'''
n_epochs               = 10
batch_size             = 32
learning_rate          = 1e-5
'''
learning_rate = 1e-4        # small enough for stable convergence
batch_size    = 32          # fits comfortably in memory
n_epochs      = 15          # enough to learn but not overfit
weight_decay  = 1e-4        # L2 regularization
dropout_rate  = 0.2         # optional, if you see overfitting
device                 = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Where to save your models & plots ─────────────────────────────────────
models_dir = "/Volumes/T7 Shield/nn_models/"
os.makedirs(models_dir, exist_ok=True)

for homology_training in homology_training_list:
    # ─── Paths ─────────────────────────────────────────────────────────────
    embeddings_dir    = "../../data/concept_groups/embeddings/group_2/training"
    test_emb_dir      = "../../data/concept_groups/embeddings/group_2/test"
    feature_rank_path = "../../data/linear_regression_data/feature_ranking_hybrid.csv"
    pos_path          = "../../data/linear_regression_data/group_2_positive.csv"
    neg_path          = "../../data/linear_regression_data/group_2_negative.csv"
    positive_test     = "../../data/concept_groups/sequences/group_2/training/group_2_test_positives.csv"
    negative_test     = "../../data/concept_groups/sequences/group_2/training/group_2_test_negatives.csv"
    repr_fasta        = f"../../data/classifier_training_data_mmseqs{homology_training}/final_rep_train_val_nr{homology_training}.fasta"
    output_csv        = f"../../data/vanilla_nn_results/nn_results_{homology_training}_hom.csv"

    hom_models_dir = os.path.join(models_dir, str(homology_training))
    os.makedirs(hom_models_dir, exist_ok=True)

    # ─── Load feature ranking ────────────────────────────────────────────────
    feat_df = pd.read_csv(feature_rank_path).sort_values("rank")
    feature_indices = feat_df["feature"].tolist()

    # ─── Load & label train data ─────────────────────────────────────────────
    pos_df = pd.read_csv(pos_path).assign(label=1)
    neg_df = pd.read_csv(neg_path).assign(label=0)
    labels_df = pd.concat([pos_df, neg_df], ignore_index=True)

    # ─── Filter to clustered entries ─────────────────────────────────────────
    allowed = {rec.id for rec in SeqIO.parse(repr_fasta, "fasta")}
    labels_df = labels_df[labels_df["Entry"].isin(allowed)].reset_index(drop=True)

    # ─── Build full train arrays ──────────────────────────────────────────────
    X_list, y_list, groups = [], [], []
    for _, row in labels_df.iterrows():
        entry = row["Entry"]
        idx   = int(row["residue_number"]) - 1
        path  = os.path.join(embeddings_dir, f"{entry}_original_SAE.pt")
        if not os.path.isfile(path):
            continue
        emb = torch.load(path, map_location="cpu").numpy()
        X_list.append(emb[idx, :])
        y_list.append(row["label"])
        groups.append(entry)
    X_all  = np.vstack(X_list)
    y_all  = np.array(y_list)
    groups = np.array(groups)

    # ─── Fixed group‐aware split for validation ──────────────────────────────
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=random_seed)
    train_idx, val_idx = next(splitter.split(X_all, y_all, groups=groups))
    X_train_full, y_train_full = X_all[train_idx], y_all[train_idx]
    X_val, y_val               = X_all[val_idx], y_all[val_idx]

    # ─── Build test set ───────────────────────────────────────────────────────
    pos_test_df = pd.read_csv(positive_test).assign(label=1)
    neg_test_df = pd.read_csv(negative_test).assign(label=0)
    test_df = pd.concat([pos_test_df, neg_test_df], ignore_index=True)

    X_test_list, y_test_list = [], []
    for _, row in test_df.iterrows():
        entry = row["Entry"]
        idx   = int(row["residue_num"]) - 1
        path  = os.path.join(test_emb_dir, f"{entry}_original_SAE.pt")
        if not os.path.isfile(path):
            continue
        emb = torch.load(path, map_location="cpu").numpy()
        X_test_list.append(emb[idx, :])
        y_test_list.append(row["label"])
    X_test = np.vstack(X_test_list)
    y_test = np.array(y_test_list)

    # ─── Loop over feature counts ─────────────────────────────────────────────
    results = []
    for num_feat in feature_steps:
        top_idxs = feature_indices[:num_feat]

        # prepare tensors
        X_tr = torch.from_numpy(X_train_full[:, top_idxs]).float().to(device)
        y_tr = torch.from_numpy(y_train_full).float().view(-1,1).to(device)
        X_v  = torch.from_numpy(X_val[:, top_idxs]).float().to(device)
        y_v  = torch.from_numpy(y_val).float().view(-1,1).to(device)
        X_te = torch.from_numpy(X_test[:, top_idxs]).float().to(device)
        y_te = torch.from_numpy(y_test).float().view(-1,1).to(device)

        train_ds = torch.utils.data.TensorDataset(X_tr, y_tr)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        #model     = make_model(num_feat).to(device)
        model     = make_tiny_net(num_feat).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        criterion = nn.BCELoss()

        # recorders
        epoch_losses = []
        val_accuracies = []

        # train loop
        model.train()
        for epoch in range(n_epochs):
            running_loss = 0.0
            for xb, yb in train_loader:
                pred = model(xb)
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            avg_loss = running_loss / len(train_loader)
            epoch_losses.append(avg_loss)

            # validation at end of epoch
            model.eval()
            with torch.no_grad():
                val_pred = (model(X_v) > 0.5).cpu().numpy()
                val_acc  = accuracy_score(y_val, val_pred) * 100
            val_accuracies.append(val_acc)
            model.train()

        # save model
        model_path = os.path.join(hom_models_dir, f"nn_hom{homology_training}_{num_feat}.pt")
        torch.save(model.state_dict(), model_path)

        # final test accuracy
        model.eval()
        with torch.no_grad():
            test_pred = (model(X_te) > 0.5).cpu().numpy()
            test_acc  = accuracy_score(y_test, test_pred) * 100

        results.append({
            "num_features":        num_feat,
            "validation_accuracy": f"{np.mean(val_accuracies):.2f}",
            "test_accuracy":       f"{test_acc:.2f}"
        })

        # ─── Plot loss & validation curve ─────────────────────────────────
        epochs = np.arange(1, n_epochs+1)
        plt.figure(figsize=(6,4))
        plt.plot(epochs, epoch_losses, '-o', label='Training Loss')
        plt.plot(epochs, val_accuracies, '-s', label='Validation Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Loss / Accuracy (%)')
        plt.title(f'H{homology_training} F{num_feat}')
        plt.legend()
        plot_path = os.path.join(hom_models_dir, f"curve_hom_{homology_training}_features_{num_feat}.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    # save results CSV
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"Saved NN results and models to {hom_models_dir}")
