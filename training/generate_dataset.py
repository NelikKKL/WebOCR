"""
Генерирует синтетические изображения текстовых строк для обучения OCR.
Кладите .ttf/.otf шрифты в training/fonts/ — чем их больше и чем они
разнообразнее (serif, sans, monospace, рукописные), тем лучше модель
обобщается на реальные шрифты.

Использование:
    pip install pillow
    python generate_dataset.py
"""
import os
import random
import string
from PIL import Image, ImageDraw, ImageFont, ImageFilter

CHARSET = (
    " !\"#$%&'()*+,-./0123456789:;<=>?@"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
    "abcdefghijklmnopqrstuvwxyz{|}~"
)

WORDS_POOL = list(string.ascii_letters) + [
    "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
    "Hello", "World", "Rust", "WASM", "OCR", "2026", "Invoice", "Total",
]


def random_text(min_len: int = 3, max_len: int = 14) -> str:
    if random.random() < 0.5:
        n = random.randint(min_len, max_len)
        return "".join(random.choice(CHARSET.strip()) for _ in range(n))
    n_words = random.randint(1, 3)
    return " ".join(random.choice(WORDS_POOL) for _ in range(n_words))[:max_len]


def render_sample(text: str, font_path: str, img_height: int = 32) -> Image.Image:
    font_size = random.randint(16, 34)
    font = ImageFont.truetype(font_path, font_size)

    dummy = Image.new("L", (10, 10), 255)
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font)
    w = max(bbox[2] - bbox[0] + 12, 10)
    h = max(bbox[3] - bbox[1] + 12, 10)

    bg = random.randint(200, 255)
    fg = random.randint(0, 60)
    img = Image.new("L", (w, h), color=bg)
    draw = ImageDraw.Draw(img)
    draw.text((6, 6), text, font=font, fill=fg)

    # Лёгкие аугментации, чтобы модель была устойчива к качеству реальных фото/сканов
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))
    if random.random() < 0.2:
        img = img.rotate(random.uniform(-3, 3), fillcolor=bg, expand=True)

    new_w = max(int(w * img_height / h), 8)
    img = img.resize((new_w, img_height))
    return img


def build_dataset(fonts_dir: str, out_dir: str, n_samples: int = 50_000) -> None:
    fonts = [
        os.path.join(fonts_dir, f)
        for f in os.listdir(fonts_dir)
        if f.lower().endswith((".ttf", ".otf"))
    ]
    if not fonts:
        raise RuntimeError(f"Не найдено шрифтов в {fonts_dir}. Добавьте .ttf/.otf файлы.")

    os.makedirs(out_dir, exist_ok=True)
    labels = []
    for i in range(n_samples):
        text = random_text()
        font_path = random.choice(fonts)
        try:
            img = render_sample(text, font_path)
        except Exception:
            continue  # некоторые шрифты могут не поддерживать отдельные символы
        fname = f"{i:07d}.png"
        img.save(os.path.join(out_dir, fname))
        labels.append(f"{fname}\t{text}")

        if (i + 1) % 5000 == 0:
            print(f"сгенерировано {i + 1}/{n_samples}")

    with open(os.path.join(out_dir, "labels.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(labels))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fonts-dir", default="fonts")
    parser.add_argument("--out-dir", default="dataset")
    parser.add_argument("--n-samples", type=int, default=50_000)
    args = parser.parse_args()

    build_dataset(fonts_dir=args.fonts_dir, out_dir=args.out_dir, n_samples=args.n_samples)
