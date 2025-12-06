import argparse, os
from data import stratified_split

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_dir", required=True, help="Path to original dataset with subfolders per class")
    ap.add_argument("--output_dir", default="data", help="Where to write train/val/test")
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    classes = stratified_split(args.source_dir, args.output_dir, args.val_ratio, args.test_ratio, args.seed)
    print("Classes:", classes)
    print(f"Split done. Wrote to: {args.output_dir}")

if __name__ == "__main__":
    main()
