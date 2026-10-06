"""Modul attention untuk varian ablasi YOLO11n (CBAM, SE, ECA, Coordinate Attention).

Checkpoint ablasi (`best.pt`) disimpan dari notebook training, tempat keempat class ini didefinisikan di `__main__`.
`torch.load` mencari class itu lewat nama `__main__.<Class>`, sehingga di luar notebook pemuatan gagal dengan
"Can't get attribute 'CBAM' on <module '__main__'>". File ini mendefinisikan ulang class yang sama (nama class, nama
atribut, dan operasi forward) lalu mendaftarkannya ke `__main__` sebelum bobot dimuat. Bobot tidak diubah.

Kesesuaian dengan checkpoint (diperiksa dari isi pickle, lihat test_attention.py):
  CBAM      3 sisipan sebelum Detect (64/128/256 kanal); channel (avg+max MLP, rasio 16) lalu spatial (conv 7x7)
  SE        3 sisipan; avg-pool -> Linear(C, C/16) -> ReLU -> Linear(C/16, C) -> Sigmoid
  ECA       3 sisipan; Conv1d lintas kanal, kernel 3 (64 kanal) atau 5 (128/256 kanal)
  CoordAtt  3 sisipan; pooling per sumbu H dan W, conv1+BN+Hardswish, conv_h/conv_w, reduksi kanal minimum 8

Model pavedroad dilatih dari notebook lain: satu sisipan di akhir backbone (256 kanal), CBAM bernama `CBAMAttention`
dan CoordAtt memakai `_HSwish` buatan sendiri. Keduanya didukung lewat `CBAMAttention` dan `_HSwish` di bawah.

Saat checkpoint dimuat, `__init__` class tidak dipanggil: atribut diisi dari pickle. Yang menentukan hasil hanya `forward`
dan nama atribut, jadi `__init__` di sini hanya untuk membangun modul baru (mis. pada pengujian).
"""
import math
import sys
from typing import Dict, Type

import torch
import torch.nn as nn


class _ChannelAttentionCBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))


class _SpatialAttentionCBAM(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Woo et al., 2018): perhatian kanal lalu perhatian spasial."""

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = _ChannelAttentionCBAM(channels, reduction)
        self.spatial_attention = _SpatialAttentionCBAM(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attention(x)
        return x * self.spatial_attention(x)


class CBAMAttention(CBAM):
    """Nama yang dipakai notebook pavedroad untuk CBAM yang sama."""


class _HSwish(nn.Module):
    """Hard-swish buatan notebook pavedroad: x * ReLU6(x + 3) / 6."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.nn.functional.relu6(x + 3.0) / 6.0


class SEAttention(nn.Module):
    """Squeeze-and-Excitation (Hu et al., 2018)."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(1, channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channels // reduction), channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        return x * self.fc(y).view(b, c, 1, 1).expand_as(x)


class ECAAttention(nn.Module):
    """Efficient Channel Attention (Wang et al., 2020): Conv1d lintas kanal, ukuran kernel adaptif terhadap jumlah kanal."""

    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)  # B,C,1,1
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)  # B,C,1,1
        return x * self.sigmoid(y).expand_as(x)


class CoordAtt(nn.Module):
    """Coordinate Attention (Hou et al., 2021): menyandikan posisi lewat pooling terpisah pada sumbu tinggi dan lebar."""

    def __init__(self, channels: int, reduction: int = 32):
        super().__init__()
        mip = max(8, channels // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mip, eps=1e-3, momentum=0.03)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        _, _, h, w = x.size()
        x_h = self.pool_h(x)  # B,C,H,1
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # B,C,W,1
        y = self.act(self.bn1(self.conv1(torch.cat([x_h, x_w], dim=2))))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        a_h = self.sigmoid(self.conv_h(x_h))
        a_w = self.sigmoid(self.conv_w(x_w.permute(0, 1, 3, 2)))
        return identity * a_w * a_h


ATTENTION_CLASSES: Dict[str, Type[nn.Module]] = {
    "CBAM": CBAM,
    "CBAMAttention": CBAMAttention,
    "_HSwish": _HSwish,
    "SEAttention": SEAttention,
    "ECAAttention": ECAAttention,
    "CoordAtt": CoordAtt,
    # kelas pembantu CBAM ikut dipickle dengan namanya sendiri
    "_ChannelAttentionCBAM": _ChannelAttentionCBAM,
    "_SpatialAttentionCBAM": _SpatialAttentionCBAM,
}


def register_attention_modules() -> None:
    """Menaruh class di `__main__` agar `torch.load` menemukannya. Aman dipanggil berulang; tidak menimpa nama yang sudah ada."""
    main = sys.modules["__main__"]
    for name, cls in ATTENTION_CLASSES.items():
        if not hasattr(main, name):
            setattr(main, name, cls)
