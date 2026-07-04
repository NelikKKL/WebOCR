"""
CRNN: CNN-backbone + BiLSTM + линейный классификатор, обучается с CTC-loss.
Классическая и проверенная архитектура для распознавания строк текста.
Компактная (десятки МБ до квантования), поэтому хорошо подходит под
ограничение <100 МБ и инференс в браузере.
"""
import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, num_classes: int, img_height: int = 32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),   # H/2
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),  # H/4
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d((2, 1)),  # H/8
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        feat_h = img_height // 8
        self.rnn = nn.LSTM(
            input_size=256 * feat_h,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, H, W)
        feat = self.cnn(x)                                   # (B, C, H', W')
        b, c, h, w = feat.shape
        feat = feat.permute(0, 3, 1, 2).reshape(b, w, c * h)  # (B, W, C*H) — по временной оси W
        out, _ = self.rnn(feat)                               # (B, W, 256)
        out = self.fc(out)                                    # (B, W, num_classes)
        return out.permute(1, 0, 2)                           # (T, B, num_classes) — формат для CTCLoss
