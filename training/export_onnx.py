"""
Экспортирует обученную PyTorch-модель в ONNX для инференса через tract в Rust/WASM.

Использование:
    pip install torch onnx
    python export_onnx.py
"""
import torch

from model import CRNN
from charset import get_charset


def export(weights_path: str | None = None, out_path: str | None = None, lang: str = "en"):
    weights_path = weights_path or f"crnn_ocr_{lang}.pt"
    out_path = out_path or f"../rust/assets/model_{lang}.onnx"

    charset = get_charset(lang)
    model = CRNN(num_classes=len(charset) + 1)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    # Динамическая ширина по оси W, т.к. строки текста бывают разной длины
    dummy_input = torch.randn(1, 1, 32, 128)

    torch.onnx.export(
        model,
        dummy_input,
        out_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {3: "width"}, "output": {0: "time"}},
        opset_version=12,
        dynamo=False,
    )
    print(f"Модель ({lang}) экспортирована в {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-path", default=None)
    parser.add_argument("--out-path", default=None)
    parser.add_argument("--lang", choices=["en", "ru"], default="en", help="Язык модели")
    args = parser.parse_args()

    export(weights_path=args.weights_path, out_path=args.out_path, lang=args.lang)
