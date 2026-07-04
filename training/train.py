"""
Обучение CRNN с CTC-loss на синтетическом датасете.

Использование:
    pip install torch pillow
    python generate_dataset.py   # сначала сгенерировать данные
    python train.py
"""
import os

import torch
from torch.nn import CTCLoss
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from model import CRNN

CHARSET = (
    " !\"#$%&'()*+,-./0123456789:;<=>?@"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
    "abcdefghijklmnopqrstuvwxyz{|}~"
)
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(CHARSET)}  # 0 зарезервирован под CTC-blank


class OCRDataset(Dataset):
    def __init__(self, root: str, img_height: int = 32, img_width: int = 128):
        self.root = root
        self.img_height = img_height
        self.img_width = img_width
        with open(os.path.join(root, "labels.txt"), encoding="utf-8") as f:
            self.samples = [line.strip().split("\t") for line in f if line.strip()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, text = self.samples[idx]
        img = Image.open(os.path.join(self.root, fname)).convert("L")
        img = img.resize((self.img_width, self.img_height))
        tensor = torch.tensor(list(img.getdata()), dtype=torch.float32)
        tensor = tensor.view(1, self.img_height, self.img_width)
        tensor = (tensor / 255.0 - 0.5) / 0.5
        target = torch.tensor([CHAR_TO_IDX[c] for c in text if c in CHAR_TO_IDX], dtype=torch.long)
        return tensor, target, len(target)


def collate(batch):
    imgs, targets, lengths = zip(*batch)
    imgs = torch.stack(imgs)
    targets_cat = torch.cat(targets)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return imgs, targets_cat, lengths


def train(dataset_dir: str = "dataset", epochs: int = 20, batch_size: int = 64, lr: float = 1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"устройство: {device}")

    ds = OCRDataset(dataset_dir)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=2)

    model = CRNN(num_classes=len(CHARSET) + 1).to(device)
    criterion = CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for imgs, targets, lengths in dl:
            imgs, targets, lengths = imgs.to(device), targets.to(device), lengths.to(device)

            logits = model(imgs)                             # (T, B, C)
            log_probs = torch.log_softmax(logits, dim=2)
            input_lengths = torch.full(
                (imgs.size(0),), logits.size(0), dtype=torch.long, device=device
            )

            loss = criterion(log_probs, targets, input_lengths, lengths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dl)
        print(f"эпоха {epoch + 1}/{epochs}  loss={avg_loss:.4f}")
        torch.save(model.state_dict(), "crnn_ocr.pt")

    print("Обучение завершено, веса сохранены в crnn_ocr.pt")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train(
        dataset_dir=args.dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
