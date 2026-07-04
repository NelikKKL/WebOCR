"""
Экспортирует обученную PyTorch-модель в ONNX для инференса через tract в Rust/WASM.

Использование:
    pip install torch onnx
    python export_onnx.py
"""
import torch

from model import CRNN
from train import CHARSET

def export(weights_path: str = "crnn_ocr.pt", out_path: str = "../rust/assets/model.onnx"):
    model = CRNN(num_classes=len(CHARSET) + 1)
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
    )
    print(f"Модель экспортирована в {out_path}")


if __name__ == "__main__":
    export()
