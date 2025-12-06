# Kiran vs Malhar — CNN From Scratch (PyTorch)

This project trains a small but strong CNN **from scratch** (no pretrained backbone) to classify two classes: **Diksha** and **poornima**.

## Expected data layout (before splitting)
```
dataset/
├── Diksha/
│   ├── img1.jpg
│   ├── ...
└── poornima/
    ├── imgA.jpg
    ├── ...
```
Any mix of `.jpg/.jpeg/.png` is fine.

## Quick start

### 1) Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Split your dataset into train/val/test
```bash
python prepare_data.py --source_dir dataset --output_dir data --val_ratio 0.15 --test_ratio 0.15
```

This creates:
```
data/
├── train/
│   ├── Diksha/..., Poornima/...
├── val/
│   ├── Diksha/..., poornima/...
└── test/
    ├── Diksha/..., Poornima/...
```

### 3) Train
```bash
python train.py --data_dir data --epochs 30 --batch_size 32 --img_size 224 --lr 1e-3
```
The best model checkpoint (by validation accuracy) is saved to `checkpoints/best.pt`.
Full training logs + final confusion matrix are also saved in `runs/`.

You can enable mixed precision on GPU with `--amp`.

### 4) Evaluate (test set)
```bash
python evaluate.py --data_dir data --checkpoint checkpoints/best.pt --img_size 224 --batch_size 64
```

### 5) Inference on a single image or folder
```bash
# Single image
python predict.py --checkpoint checkpoints/best.pt --image /path/to/file.jpg --img_size 224

# Folder with images
python predict.py --checkpoint checkpoints/best.pt --folder /path/to/folder --img_size 224
```

## Notes

- **No pretrained model** is used. The network `SmallConvNet` is implemented from scratch (`model.py`).
- Standard augmentations are applied for training. You can tweak them in `data.py`.
- Early stopping + best-checkpoint saving are implemented.
- Reproducible seeds by default.
- Works on CPU or GPU. Use `--device cuda` if you have a CUDA GPU.
