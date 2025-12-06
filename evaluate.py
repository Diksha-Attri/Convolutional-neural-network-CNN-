import argparse, os, torch, torch.nn as nn
from data import make_loaders
from model import SmallConvNet
from utils import set_seed, save_confusion_matrix
from sklearn.metrics import classification_report

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    train_loader, val_loader, test_loader, classes = make_loaders(args.data_dir, args.img_size, args.batch_size)
    model = SmallConvNet(num_classes=len(classes)).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    criterion = nn.CrossEntropyLoss()
    _, _, y_true, y_pred = None, None, None, None

    # Evaluate on test
    from tqdm import tqdm
    running_loss, correct, total = 0.0, 0, 0
    y_true, y_pred = [], []
    for imgs, labels in tqdm(test_loader, desc="Test", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
    loss = running_loss / total
    acc = correct / total
    print(f"TEST: loss={loss:.4f} acc={acc:.4f}")
    os.makedirs("runs", exist_ok=True)
    save_confusion_matrix(y_true, y_pred, classes, os.path.join("runs", "confusion_matrix_eval.png"))
    with open(os.path.join("runs", "classification_report_eval.txt"), "w") as f:
        f.write(str(classification_report(y_true, y_pred, target_names=classes)))
    print("Saved confusion_matrix_eval.png and classification_report_eval.txt in runs/.")

if __name__ == "__main__":
    main()
