"""
Обучение CRNN с CTC-loss на синтетическом датасете.

Использование:
    pip install torch pillow
    python generate_dataset.py   # сначала сгенерировать данные
    python train.py
"""
import os

import numpy as np
import torch
from torch.nn import CTCLoss
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from model import CRNN

from charset import get_charset, get_char_to_idx


class OCRDataset(Dataset):
    def __init__(self, root: str, char_to_idx: dict, img_height: int = 32, max_width: int = 256):
        self.root = root
        self.img_height = img_height
        self.max_width = max_width
        self.char_to_idx = char_to_idx
        with open(os.path.join(root, "labels.txt"), encoding="utf-8") as f:
            self.samples = [line.strip().split("\t") for line in f if line.strip()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, text = self.samples[idx]
        img = Image.open(os.path.join(self.root, fname)).convert("L")
        w, h = img.size
        # ВАЖНО: сохраняем пропорции, как это делает Rust-инференс в
        # preprocess.rs. Раньше здесь был img.resize((128, 32)) — жёсткое
        # сжатие, которое искажало геометрию текста и не совпадало с тем,
        # что модель видит в проде.
        new_w = max(int(round(w * self.img_height / h)), 8)
        new_w = min(new_w, self.max_width)
        img = img.resize((new_w, self.img_height))
        arr = np.asarray(img, dtype=np.float32)  # (H, W)
        tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)
        tensor = (tensor / 255.0 - 0.5) / 0.5
        target = torch.tensor(
            [self.char_to_idx[c] for c in text if c in self.char_to_idx], dtype=torch.long
        )
        return tensor, target, len(target), new_w


def collate(batch):
    imgs, targets, lengths, widths = zip(*batch)
    max_w = max(widths)
    padded = []
    for img in imgs:
        c, h, w = img.shape
        if w < max_w:
            # Паддинг значением 1.0 (= белый фон после нормализации (x/255-0.5)/0.5)
            pad = torch.full((c, h, max_w - w), 1.0)
            img = torch.cat([img, pad], dim=2)
        padded.append(img)
    imgs = torch.stack(padded)
    targets_cat = torch.cat(targets)
    lengths = torch.tensor(lengths, dtype=torch.long)
    widths = torch.tensor(widths, dtype=torch.long)
    return imgs, targets_cat, lengths, widths


def train(
    dataset_dir: str = "dataset",
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    lang: str = "en",
    max_width: int = 256,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"устройство: {device}")
    print(f"язык: {lang}")

    charset = get_charset(lang)
    char_to_idx = get_char_to_idx(lang)
    checkpoint_path = f"crnn_ocr_{lang}.pt"

    ds = OCRDataset(dataset_dir, char_to_idx=char_to_idx, max_width=max_width)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=2)

    model = CRNN(num_classes=len(charset) + 1).to(device)
    criterion = CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # CNN уменьшает ширину в 4 раза (два MaxPool2d(2,2); третий пулинг
    # трогает только высоту). Нужно для корректных input_lengths при паддинге.
    width_downsample = 4

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for imgs, targets, lengths, widths in dl:
            imgs, targets, lengths = imgs.to(device), targets.to(device), lengths.to(device)

            logits = model(imgs)                             # (T, B, C)
            log_probs = torch.log_softmax(logits, dim=2)
            input_lengths = torch.clamp(widths // width_downsample, min=1, max=logits.size(0))
            input_lengths = input_lengths.to(device)

            loss = criterion(log_probs, targets, input_lengths, lengths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dl)
        print(f"эпоха {epoch + 1}/{epochs}  loss={avg_loss:.4f}")
        torch.save(model.state_dict(), checkpoint_path)

    print(f"Обучение завершено, веса сохранены в {checkpoint_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lang", choices=["en", "ru"], default="en", help="Язык модели")
    parser.add_argument("--max-width", type=int, default=256, help="Максимальная ширина изображения (px)")
    args = parser.parse_args()

    train(
        dataset_dir=args.dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lang=args.lang,
        max_width=args.max_width,
    )
