import argparse, os, time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from data import make_loaders
from model import SmallConvNet
from utils import set_seed, save_confusion_matrix, count_params
from sklearn.metrics import classification_report

def train_one_epoch(model, loader, device, criterion, optimizer, scaler=None):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        if scaler:
            with torch.cuda.amp.autocast():
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in tqdm(loader, desc="Eval", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc, all_labels, all_preds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true", help="Use mixed precision on GPU")
    ap.add_argument("--patience", type=int, default=7, help="Early stopping patience (epochs)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_dir", default="checkpoints")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    device = torch.device(args.device)
    train_loader, val_loader, test_loader, classes = make_loaders(args.data_dir, args.img_size, args.batch_size)
    num_classes = len(classes)
    model = SmallConvNet(num_classes=num_classes).to(device)
    print(f"Model params: {count_params(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type=="cuda")

    best_acc, best_path = 0.0, os.path.join(args.save_dir, "best.pt")
    epochs_no_improve = 0

    for epoch in range(1, args.epochs+1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, device, criterion, optimizer, scaler)
        va_loss, va_acc, y_true, y_pred = evaluate(model, val_loader, device, criterion)
        scheduler.step()

        print(f"Train  loss={tr_loss:.4f} acc={tr_acc:.4f}")
        print(f"Val    loss={va_loss:.4f} acc={va_acc:.4f}")
        with open(os.path.join("runs", "val_report.txt"), "a") as f:
            f.write(f"Epoch {epoch}: val_acc={va_acc:.4f} val_loss={va_loss:.4f}\n")

        # Save best
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({"model": model.state_dict(), "classes": classes, "img_size": args.img_size}, best_path)
            print(f"Saved new best checkpoint: {best_path} (val_acc={best_acc:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= args.patience:
            print("Early stopping triggered.")
            break

    # Final evaluation on test
    print("\nEvaluating on TEST set with best checkpoint...")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    te_loss, te_acc, y_true, y_pred = evaluate(model, test_loader, device, criterion)
    print(f"TEST: loss={te_loss:.4f} acc={te_acc:.4f}")
    # Reports
    import json
    from utils import save_confusion_matrix
    save_confusion_matrix(y_true, y_pred, classes, os.path.join("runs", "confusion_matrix.png"))
    with open(os.path.join("runs", "classification_report.txt"), "w") as f:
        f.write(str(classification_report(y_true, y_pred, target_names=classes)))
    print("Saved confusion_matrix.png and classification_report.txt in runs/.")

if __name__ == "__main__":
    main()
