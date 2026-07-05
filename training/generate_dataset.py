"""
Генерирует синтетические изображения текстовых строк для обучения OCR.
Кладите .ttf/.otf шрифты в training/fonts/ — чем их больше и чем они
разнообразнее (serif, sans, monospace, рукописные), тем лучше модель
обобщается на реальные шрифты.

Использование:
    pip install pillow
    python generate_dataset.py
"""
import io
import os
import random
import string

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from charset import get_charset

CYRILLIC_LETTERS = list(
    "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя"
)

WORDS_POOL_EN = list(string.ascii_letters) + [
    "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
    "Hello", "World", "Rust", "WASM", "OCR", "2026", "Invoice", "Total",
]

WORDS_POOL_RU = CYRILLIC_LETTERS + [
    "привет", "мир", "быстрая", "коричневая", "лиса", "прыгает", "через",
    "ленивую", "собаку", "текст", "документ", "счёт", "итого", "накладная",
    "2026", "Россия", "Москва", "программа",
]

WORDS_POOLS = {"en": WORDS_POOL_EN, "ru": WORDS_POOL_RU}


def random_text(lang: str = "en", min_len: int = 3, max_len: int = 14) -> str:
    charset = get_charset(lang)
    words_pool = WORDS_POOLS[lang]
    if random.random() < 0.5:
        n = random.randint(min_len, max_len)
        return "".join(random.choice(charset.strip()) for _ in range(n))
    n_words = random.randint(1, 3)
    return " ".join(random.choice(words_pool) for _ in range(n_words))[:max_len]


def _random_bg_fg(dark_bg: bool) -> tuple[int, int]:
    """Возвращает (цвет фона, цвет текста). Раньше фон был всегда светлым —
    модель никогда не видела белый текст на тёмном фоне."""
    if dark_bg:
        bg = random.randint(0, 55)
        fg = random.randint(195, 255)
    else:
        bg = random.randint(200, 255)
        fg = random.randint(0, 60)
    return bg, fg


def _add_background_texture(img: Image.Image) -> Image.Image:
    """Добавляет шум или мягкий градиент на фон — реальные фото/сканы редко
    имеют идеально ровную заливку."""
    arr = np.asarray(img, dtype=np.float32)
    h, w = arr.shape
    if random.random() < 0.5:
        noise = np.random.normal(0, random.uniform(3, 18), (h, w))
        arr = arr + noise
    else:
        direction = random.choice(["h", "v"])
        ramp = np.linspace(-1, 1, w if direction == "h" else h) * random.uniform(10, 35)
        grad = np.tile(ramp, (h, 1)) if direction == "h" else np.tile(ramp.reshape(-1, 1), (1, w))
        arr = arr + grad
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def _jpeg_artifact(img: Image.Image) -> Image.Image:
    """Пережимает изображение в JPEG с низким качеством, чтобы приучить
    модель к артефактам реальных фото/скриншотов."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=random.randint(35, 80))
    buf.seek(0)
    return Image.open(buf).convert("L")


def render_sample(text: str, font_path: str, img_height: int = 32) -> Image.Image:
    # font_size снижен до 14, чтобы модель училась и на мелком тексте
    font_size = random.randint(14, 34)
    font = ImageFont.truetype(font_path, font_size)

    dummy = Image.new("L", (10, 10), 255)
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font)
    w = max(bbox[2] - bbox[0] + 12, 10)
    h = max(bbox[3] - bbox[1] + 12, 10)

    # ~45% сэмплов — тёмный фон/светлый текст (раньше не встречалось вообще)
    dark_bg = random.random() < 0.45
    bg, fg = _random_bg_fg(dark_bg)

    img = Image.new("L", (w, h), color=bg)
    draw = ImageDraw.Draw(img)
    draw.text((6, 6), text, font=font, fill=fg)

    # Текстура/шум фона вместо идеально ровной заливки
    if random.random() < 0.6:
        img = _add_background_texture(img)

    if random.random() < 0.4:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.6, 1.5))
    if random.random() < 0.4:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))

    # Лёгкие аугментации, чтобы модель была устойчива к качеству реальных фото/сканов
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))
    if random.random() < 0.2:
        img = img.rotate(random.uniform(-3, 3), fillcolor=bg, expand=True)
    if random.random() < 0.25:
        img = _jpeg_artifact(img)

    new_w = max(int(w * img_height / h), 8)
    img = img.resize((new_w, img_height))
    return img


def _font_supports_text(font_path: str, text: str) -> bool:
    """Грубая проверка, что шрифт умеет рисовать нужные символы (например,
    кириллицу) — иначе PIL молча подставит «квадратики»/пустые глифы."""
    try:
        font = ImageFont.truetype(font_path, 20)
        for ch in set(text):
            if ch == " ":
                continue
            if font.getmask(ch).getbbox() is None:
                return False
        return True
    except Exception:
        return False


def build_dataset(fonts_dir: str, out_dir: str, n_samples: int = 50_000, lang: str = "en") -> None:
    all_fonts = [
        os.path.join(fonts_dir, f)
        for f in os.listdir(fonts_dir)
        if f.lower().endswith((".ttf", ".otf"))
    ]
    if not all_fonts:
        raise RuntimeError(f"Не найдено шрифтов в {fonts_dir}. Добавьте .ttf/.otf файлы.")

    if lang == "ru":
        probe = "".join(CYRILLIC_LETTERS[:10])
        fonts = [f for f in all_fonts if _font_supports_text(f, probe)]
        if not fonts:
            raise RuntimeError(
                "Ни один шрифт в "
                f"{fonts_dir} не поддерживает кириллицу. Установите шрифты с "
                "поддержкой русского языка (например, DejaVu Sans, Noto Sans)."
            )
    else:
        fonts = all_fonts

    os.makedirs(out_dir, exist_ok=True)
    labels = []
    for i in range(n_samples):
        text = random_text(lang=lang)
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
    parser.add_argument("--lang", choices=["en", "ru"], default="en", help="Язык датасета")
    args = parser.parse_args()

    build_dataset(
        fonts_dir=args.fonts_dir,
        out_dir=args.out_dir,
        n_samples=args.n_samples,
        lang=args.lang,
    )
