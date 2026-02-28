"""Evaluate a trained DualGraphSHM checkpoint on the test set."""

import argparse

import numpy as np
import torch
import torch.utils.data as data_utils
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from data_loader import load_dataset
from graph_utils import get_adjacency
from model import DualGraphSHM

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
    p = argparse.ArgumentParser(description='Evaluate DualGraphSHM')
    p.add_argument('--checkpoint', type=str, required=True,
                   help='Path to saved checkpoint (.pt)')
    p.add_argument('--dataset', default='lumo', choices=['lumo', 'qugs'])
    p.add_argument('--data_dir', type=str, required=True,
                   help='Path to directory containing CSV files')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--history', type=int, default=200)
    p.add_argument('--max_rows', type=int, default=500000)
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--mode', default='dual',
                   choices=['dual', 'horizontal', 'vertical'])
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = DATASET_CONFIG[args.dataset]
    num_sensors = cfg['num_sensors']
    num_classes = cfg['num_classes']
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load data and reproduce the same split
    X, labels = load_dataset(args.data_dir, args.history, num_sensors,
                             args.max_rows)
    X = torch.from_numpy(X).permute(0, 2, 1)

    _, X_temp, _, y_temp = train_test_split(
        X, labels, test_size=0.2, random_state=args.seed, stratify=labels)
    _, X_test, _, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=args.seed, stratify=y_temp)

    y_test = torch.from_numpy(y_test)
    test_loader = data_utils.DataLoader(
        data_utils.TensorDataset(X_test, y_test),
        batch_size=args.batch_size, shuffle=False)

    # Build model and load checkpoint
    adj_norm, adj_self, _ = get_adjacency(args.dataset)
    model = DualGraphSHM(
        num_sensors=num_sensors,
        num_classes=num_classes,
        adj_norm=adj_norm,
        adj_self=adj_self,
        graph_mode=args.mode,
    ).double().to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} "
          f"(val acc: {ckpt['val_acc']*100:.2f}%)")

    # Evaluate
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for signals, labs in test_loader:
            signals = signals.double().to(device)
            labs = labs.long().to(device)
            out = model(signals)
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(labs.cpu().numpy())

    test_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"\nTest Accuracy: {test_acc*100:.2f}%\n")
    print(classification_report(
        all_labels, all_preds, target_names=cfg['class_names'], digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == '__main__':
    main()
