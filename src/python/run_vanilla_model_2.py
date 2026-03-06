#!/usr/bin/env python3
import os
import pandas as pd
import torch
import numpy as np
from Bio import SeqIO
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ─── Model factory ─────────────────────────────────────────────────────────
def make_model(num_features, dropout_rate):
    h1 = min(max(5, num_features // 5), 50)
    return nn.Sequential(
        nn.Linear(num_features, h1),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(h1, 1),
        nn.Sigmoid()
    )

# ─── Hyperparameters ───────────────────────────────────────────────────────
homology_training_list = [40, 50, 60, 70, 80, 90]
feature_steps          = range(10, 501, 10)
val_fraction           = 0.1
random_seed            = 42

learning_rate = 1e-4
batch_size    = 32
n_epochs      = 20
weight_decay  = 1e-4
dropout_rate  = 0.3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Directories ───────────────────────────────────────────────────────────
models_dir = "/Volumes/T7 Shield/nn_models/"
os.makedirs(models_dir, exist_ok=True)

# ─── Reproducibility ───────────────────────────────────────────────────────
import random
random.seed(random_seed)
np.random.seed(random_seed)

for homology_training in homology_training_list:
    data_root = "/Volumes/T7 Shield/toxin_dataset/toxins_and_neuro"
    embeddings_dir = os.path.join(data_root, "toxin_neuro_SAE_embeddings")
    test_emb_dir   = embeddings_dir

    feature_rank_path = os.path.join(
        data_root, "input_data", "training_data",
        f"homology_group{homology_training}",
        f"feature_ranking_hybrid_hom{homology_training}.csv"
    )

    pos_path = os.path.join(
        data_root, "input_data", "training_data",
        f"homology_group{homology_training}",
        "group_2_positive_toxin_neuro_train.csv"
    )
    neg_path = os.path.join(
        data_root, "input_data", "training_data",
        f"homology_group{homology_training}",
        "group_2_negative_toxin_neuro_train.csv"
    )

    positive_test = os.path.join(
        data_root, "input_data", "test_data",
        f"homology_group{homology_training}",
        "group_2_positive_toxin_neuro_test.csv"
    )
    negative_test = os.path.join(
        data_root, "input_data", "test_data",
        f"homology_group{homology_training}",
        "group_2_negative_toxin_neuro_test.csv"
    )

    repr_fasta = os.path.join(
        data_root, "input_data", "training_data",
        f"homology_group{homology_training}",
        f"final_rep_train_val_nr{homology_training}.fasta"
    )

    output_csv = os.path.join(
        data_root, "classifier_results", "nn_csv_results",
        f"nn_results_{homology_training}_hom.csv"
    )

    models_dir = os.path.join(data_root, "classifier_results", "neural_net_models")
    hom_models_dir = os.path.join(models_dir, f"hom{homology_training}")
    os.makedirs(hom_models_dir, exist_ok=True)

    feat_df = pd.read_csv(feature_rank_path).sort_values("rank")
    feature_indices = feat_df["feature"].tolist()

    pos_df = pd.read_csv(pos_path).assign(label=1)
    neg_df = pd.read_csv(neg_path).assign(label=0)
    labels_df = pd.concat([pos_df, neg_df], ignore_index=True)

    X_list, y_list, groups = [], [], []
    for _, row in labels_df.iterrows():
        entry = row["Entry"]
        idx   = int(row["residue_number"]) - 1
        path  = os.path.join(embeddings_dir, f"{entry}_original_SAE.pt")
        if not os.path.isfile(path): continue
        emb = torch.load(path, map_location="cpu").numpy()
        X_list.append(emb[idx, :])
        y_list.append(row["label"])
        groups.append(entry)

    X_all = np.vstack(X_list)
    y_all = np.array(y_list)
    groups = np.array(groups)

    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=random_seed)
    train_idx, val_idx = next(splitter.split(X_all, y_all, groups=groups))
    X_train_full, y_train_full = X_all[train_idx], y_all[train_idx]
    X_val, y_val               = X_all[val_idx],   y_all[val_idx]

    pos_test_df = pd.read_csv(positive_test).assign(label=1)
    neg_test_df = pd.read_csv(negative_test).assign(label=0)
    test_df     = pd.concat([pos_test_df, neg_test_df], ignore_index=True)

    X_test_list, y_test_list = [], []
    for _, row in test_df.iterrows():
        entry = row["Entry"]
        idx   = int(row["residue_number"]) - 1
        path  = os.path.join(test_emb_dir, f"{entry}_original_SAE.pt")
        if not os.path.isfile(path): continue
        emb = torch.load(path, map_location="cpu").numpy()
        X_test_list.append(emb[idx, :])
        y_test_list.append(row["label"])

    X_test = np.vstack(X_test_list)
    y_test = np.array(y_test_list)

    results = []
    for num_feat in feature_steps:
        top_idxs = feature_indices[:num_feat]

        X_tr = torch.from_numpy(X_train_full[:, top_idxs]).float().to(device)
        y_tr = torch.from_numpy(y_train_full).float().view(-1,1).to(device)
        X_v  = torch.from_numpy(X_val[:, top_idxs]).float().to(device)
        y_v  = torch.from_numpy(y_val).float().view(-1,1).to(device)
        X_te = torch.from_numpy(X_test[:, top_idxs]).float().to(device)
        y_te = torch.from_numpy(y_test).float().view(-1,1).to(device)

        train_ds = torch.utils.data.TensorDataset(X_tr, y_tr)
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(random_seed)
        )

        model = make_model(num_feat, dropout_rate).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.BCELoss()

        epoch_losses = []
        val_accuracies = []

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

            model.eval()
            with torch.no_grad():
                val_pred = (model(X_v) > 0.5).cpu().numpy()
                val_acc  = accuracy_score(y_val, val_pred) * 100
            val_accuracies.append(val_acc)
            model.train()

        model_path = os.path.join(hom_models_dir, f"nn_hom{homology_training}_{num_feat}.pt")
        torch.save(model.state_dict(), model_path)

        model.eval()
        with torch.no_grad():
            test_pred = (model(X_te) > 0.5).cpu().numpy()
            test_acc  = accuracy_score(y_test, test_pred) * 100
            test_bal_acc = balanced_accuracy_score(y_test, test_pred) * 100

        results.append({
            "num_features": num_feat,
            "validation_accuracy": f"{np.mean(val_accuracies):.2f}",
            "test_accuracy": f"{test_acc:.2f}",
            "balanced_test_accuracy": f"{test_bal_acc:.2f}"
        })

        print(f"  • H{homology_training} | F{num_feat} → Val: {np.mean(val_accuracies):.2f}% | Test: {test_acc:.2f}% | Balanced Test: {test_bal_acc:.2f}%")

        epochs = np.arange(1, n_epochs+1)
        plt.figure(figsize=(6,4), dpi=300)
        plt.plot(epochs, epoch_losses, '-o', label='Training Loss')
        plt.plot(epochs, val_accuracies, '-s', label='Validation Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Loss / Accuracy (%)')
        plt.title(f'H{homology_training} F{num_feat}')
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(hom_models_dir, f"curve_hom{homology_training}_features_{num_feat}.png")
        plt.savefig(plot_path)
        plt.close()

    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"Saved NN results and models to {hom_models_dir}")
