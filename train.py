"""Training script for DualGraphSHM."""

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from data_loader import load_dataset
from graph_utils import get_adjacency
from model import DualGraphSHM

# ---------------------------------------------------------------------------
# Dataset configurations
# ---------------------------------------------------------------------------
DATASET_CONFIG = {
    'lumo': {
        'num_sensors': 18,
        'num_classes': 7,
        'class_names': [
            'Healthy',
            'DAM3-1bolt', 'DAM3-all', 'DAM4-1bolt',
            'DAM4-all', 'DAM6-1bolt', 'DAM6-all',
        ],
    },
    'qugs': {
        'num_sensors': 30,
        'num_classes': 5,
        'class_names': [
            'Healthy', 'Damage-1', 'Damage-2', 'Damage-3', 'Damage-4',
        ],
    },
}


def parse_args():
    p = argparse.ArgumentParser(description='Train DualGraphSHM')
    p.add_argument('--dataset', default='lumo', choices=['lumo', 'qugs'])
    p.add_argument('--data_dir', type=str, required=True,
                   help='Path to directory containing CSV files')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--history', type=int, default=200,
                   help='Window length (time steps per sample)')
    p.add_argument('--max_rows', type=int, default=500000,
                   help='Max rows to read per CSV file')
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--mode', default='dual',
                   choices=['dual', 'horizontal', 'vertical'],
                   help='Graph branch mode')
    p.add_argument('--save_path', type=str, default='best_model.pt',
                   help='Path to save the best checkpoint')
    return p.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    set_seed(args.seed)

    cfg = DATASET_CONFIG[args.dataset]
    num_sensors = cfg['num_sensors']
    num_classes = cfg['num_classes']
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print(f"Dataset: {args.dataset} | Sensors: {num_sensors} | "
          f"Classes: {num_classes} | Mode: {args.mode}")
    print(f"Device: {device} | Seed: {args.seed}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    X, labels = load_dataset(args.data_dir, args.history, num_sensors,
                             args.max_rows)
    print(f"Loaded {X.shape[0]} samples, shape: {X.shape}")

    X = torch.from_numpy(X).permute(0, 2, 1)  # (N, sensors, history)

    # 80 / 10 / 10 stratified split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, labels, test_size=0.2, random_state=args.seed, stratify=labels)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=args.seed, stratify=y_temp)

    y_train = torch.from_numpy(y_train)
    y_val = torch.from_numpy(y_val)
    y_test = torch.from_numpy(y_test)

    train_loader = data_utils.DataLoader(
        data_utils.TensorDataset(X_train, y_train),
        batch_size=args.batch_size, shuffle=True)
    val_loader = data_utils.DataLoader(
        data_utils.TensorDataset(X_val, y_val),
        batch_size=args.batch_size, shuffle=False)
    test_loader = data_utils.DataLoader(
        data_utils.TensorDataset(X_test, y_test),
        batch_size=args.batch_size, shuffle=False)

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    adj_norm, adj_self, _ = get_adjacency(args.dataset)
    model = DualGraphSHM(
        num_sensors=num_sensors,
        num_classes=num_classes,
        adj_norm=adj_norm,
        adj_self=adj_self,
        graph_mode=args.mode,
    ).double().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    best_val_acc = 0.0
    best_epoch = 0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        # --- train ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for signals, labs in train_loader:
            signals = signals.double().to(device)
            labs = labs.long().to(device)

            optimizer.zero_grad()
            out = model(signals)
            loss = criterion(out, labs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * labs.size(0)
            train_correct += (out.argmax(1) == labs).sum().item()
            train_total += labs.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # --- validate ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for signals, labs in val_loader:
                signals = signals.double().to(device)
                labs = labs.long().to(device)
                out = model(signals)
                loss = criterion(out, labs)
                val_loss += loss.item() * labs.size(0)
                val_correct += (out.argmax(1) == labs).sum().item()
                val_total += labs.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'args': vars(args),
            }, args.save_path)

    elapsed = time.time() - start
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"Best val accuracy: {best_val_acc*100:.2f}% (epoch {best_epoch})")

    # ------------------------------------------------------------------
    # Test evaluation (load best checkpoint)
    # ------------------------------------------------------------------
    ckpt = torch.load(args.save_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for signals, labs in test_loader:
            signals = signals.double().to(device)
            labs = labs.long().to(device)
            out = model(signals)
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(labs.cpu().numpy())

    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(classification_report(
        all_labels, all_preds, target_names=cfg['class_names'], digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == '__main__':
    main()
