# DualGraphSHM

Official implementation of **"Robust and efficient dual-graph neural networks for structural damage detection and localization"**, published in *Engineering Structures*, Volume 343, 2025, Article 121265.

## Architecture

```
Raw sensor signals (B, num_sensors, history)
        |
   1D-CNN backbone (4 layers)
        |
        v
   (B, num_sensors, 200)
       / \
      /   \
  H-GC    V-GC
  branch  branch
     |      |
     v      v
  Adaptive  LGFM
  Aggreg.   (temporal
  (spatial)  gating)
     |      |
     +--SE--+  (cross-branch squeeze-and-excitation attention)
        |
   Classifier
        |
   damage class
```

- **Horizontal branch (H-GC)**: Multi-scale Chebyshev GCN with learned + spatial adjacency fusion
- **Vertical branch (V-GC)**: LSTM-gated temporal GCN across time segments
- **SE Fusion**: Dual squeeze-and-excitation cross-attention between branches

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: PyTorch, PyTorch Geometric, NumPy, pandas, scikit-learn.

## Data Preparation

Place your dataset CSV files in the `data/` directory. Each CSV should contain:
- First `num_sensors` columns: accelerometer readings
- Last column: class label (integer)
- First row: header (skipped during loading)

**LUMO dataset** (18 sensors, 7 classes): columns `R1, M1, R2, M2, ..., R9, M9, label`

**QUGS dataset** (30 sensors, 5 classes): similar format with 30 sensor columns.

## Usage

### Training

```bash
# LUMO dataset (18 sensors, 7 damage classes)
python train.py --dataset lumo --data_dir data/ --epochs 100 --device cuda:0

# QUGS dataset (30 sensors, 5 classes)
python train.py --dataset qugs --data_dir data/ --epochs 100 --device cuda:0

# Ablation: horizontal branch only
python train.py --dataset lumo --data_dir data/ --mode horizontal

# Ablation: vertical branch only
python train.py --dataset lumo --data_dir data/ --mode vertical
```

### Evaluation

```bash
python evaluate.py --checkpoint best_model.pt --dataset lumo --data_dir data/
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `lumo` | Dataset name (`lumo` or `qugs`) |
| `--data_dir` | (required) | Path to CSV data directory |
| `--batch_size` | `64` | Training batch size |
| `--lr` | `0.0001` | Learning rate |
| `--epochs` | `100` | Number of training epochs |
| `--history` | `200` | Window length (time steps per sample) |
| `--mode` | `dual` | Branch mode: `dual`, `horizontal`, `vertical` |
| `--seed` | `42` | Random seed for reproducibility |
| `--device` | `cuda:0` | CUDA device |
| `--save_path` | `best_model.pt` | Checkpoint save path |

## File Structure

| File | Description |
|------|-------------|
| `train.py` | Main training script with argparse |
| `evaluate.py` | Load checkpoint and evaluate on test set |
| `model.py` | `DualGraphSHM` model (main class) |
| `cnn.py` | 1D-CNN backbone |
| `mgcn.py` | Multi-scale GCN (horizontal branch) |
| `lgfm.py` | LSTM-gated temporal GCN (vertical branch) |
| `adaptive_aggregation.py` | Frobenius-norm adaptive adjacency fusion |
| `signal_gcn.py` | GCN wrapper for spatial convolution |
| `layers.py` | `GraphConvolution` layer |
| `se_module.py` | Squeeze-and-Excitation attention |
| `frobenius.py` | Frobenius norm penalty module |
| `graph_utils.py` | Adjacency matrix construction |
| `data_loader.py` | CSV data loading and windowing |

## Citation

```bibtex
@article{dualgraphshm2025,
  title     = {Robust and efficient dual-graph neural networks for structural
               damage detection and localization},
  journal   = {Engineering Structures},
  volume    = {343},
  pages     = {121265},
  year      = {2025},
  doi       = {10.1016/j.engstruct.2025.121265}
}
```

## License

This project is licensed under the [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) license.
