import argparse, os, torch
from PIL import Image
from torchvision import transforms
from model import SmallConvNet

def load_ckpt(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = SmallConvNet(num_classes=len(ckpt["classes"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    img_size = ckpt.get("img_size", 224)
    classes = ckpt["classes"]
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
    ])
    return model, classes, tfm

@torch.no_grad()
def predict_image(path, model, device, tfm, classes):
    img = Image.open(path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    logits = model(x)
    prob = torch.softmax(logits, dim=1)[0].cpu().tolist()
    pred_idx = logits.argmax(dim=1).item()
    return classes[pred_idx], prob

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", default=None, help="Path to a single image")
    ap.add_argument("--folder", default=None, help="Path to a folder of images")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--img_size", type=int, default=None, help="Override image size (optional)")
    args = ap.parse_args()

    device = torch.device(args.device)
    model, classes, tfm = load_ckpt(args.checkpoint, device)

    if args.image:
        label, prob = predict_image(args.image, model, device, tfm, classes)
        print(f"{os.path.basename(args.image)} -> {label}  (probs={prob})")

    if args.folder:
        for fname in os.listdir(args.folder):
            if fname.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp")):
                path = os.path.join(args.folder, fname)
                label, prob = predict_image(path, model, device, tfm, classes)
                print(f"{fname} -> {label}  (probs={prob})")

if __name__ == "__main__":
    main()
