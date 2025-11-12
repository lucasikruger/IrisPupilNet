import argparse, os, math
from pathlib import Path
from typing import Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import __init__ as ds_init  # to ensure registry import side-effects
from models   import __init__ as md_init  # same
from datasets import __init__ as _
from models import MODEL_REGISTRY
from datasets import DATASET_REGISTRY
from utils.metrics import mean_iou_ignore_bg

try:
    import albumentations as A  # just to ensure requirement is installed
except Exception:
    pass

def build_model(model_name: str, in_channels: int, n_classes: int, base: int):
    ModelCtor = MODEL_REGISTRY[model_name]
    return ModelCtor(in_channels=in_channels, n_classes=n_classes, base=base)

def train_one_epoch(model, dl, optimizer, loss_fn, device):
    model.train()
    total = 0.0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
    return total / len(dl.dataset)

@torch.no_grad()
def evaluate(model, dl, loss_fn, device, num_classes: int):
    model.eval()
    tot_loss, tot_iou, n = 0.0, 0.0, 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y).item()
        iou = mean_iou_ignore_bg(logits, y, num_classes=num_classes)
        bs = x.size(0)
        tot_loss += loss * bs
        tot_iou  += iou  * bs
        n += bs
    return tot_loss / n, tot_iou / n


def build_dataloaders(dataset_name: str,
                      data_root: str,
                      csv_path: str,
                      img_size: int,
                      batch_size: int,
                      workers: int,
                      default_format: str):
    DatasetCls = DATASET_REGISTRY[dataset_name]
    if dataset_name == "csv_seg":
        train_ds = DatasetCls(dataset_base_dir=data_root, csv_path=csv_path, split="train",
                              img_size=img_size, default_format=default_format)
        val_ds   = DatasetCls(dataset_base_dir=data_root, csv_path=csv_path, split="val",
                              img_size=img_size, default_format=default_format)
    else:
        raise ValueError("Use dataset=csv_seg for the CSV-driven loader.")

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return train_dl, val_dl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="csv_seg", choices=list(DATASET_REGISTRY.keys()))
    ap.add_argument("--data-root", required=True, help="Base dir for relative paths in CSV.")
    ap.add_argument("--csv", required=True, help="CSV with rel_image_path, rel_mask_path, split[, dataset_format].")
    ap.add_argument("--default-format", default="mobius_3c", help="Fallback dataset_format if CSV row lacks it.")

    ap.add_argument("--model", default="unet_se_small", choices=list(MODEL_REGISTRY.keys()))
    ap.add_argument("--img-size", type=int, default=160)
    ap.add_argument("--in-channels", type=int, default=3)
    ap.add_argument("--num-classes", type=int, default=3, help="0=bg, 1=iris, 2=pupil")
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", type=Path, default=Path("runs/mobius_unet_se"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)

    train_dl, val_dl = build_dataloaders(args.dataset, args.data_root, args.csv,
                                         args.img_size, args.batch_size, args.workers,
                                         default_format=args.default_format)
    model = build_model(args.model, args.in_channels, args.num_classes, args.base).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    best_iou = -1.0
    for ep in range(1, args.epochs+1):
        tr_loss = train_one_epoch(model, train_dl, opt, loss_fn, device)
        val_loss, val_iou = evaluate(model, val_dl, loss_fn, device, num_classes=args.num_classes)
        print(f"epoch {ep:02d} | train {tr_loss:.4f} | val {val_loss:.4f} | IoU(iris+pupil) {val_iou:.3f}")

        if val_iou > best_iou:
            best_iou = val_iou
            ckpt = args.out / "best.pt"
            torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt)
            print(f"  ↳ saved {ckpt}")

if __name__ == "__main__":
    main()
