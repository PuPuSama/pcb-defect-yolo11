from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn


def _ultralytics_conv():
    try:
        from ultralytics.nn.modules import Conv
    except ImportError:
        from ultralytics.nn.modules.conv import Conv
    return Conv


class SimAM(nn.Module):
    """Parameter-free SimAM attention."""

    def __init__(self, e_lambda: float = 1e-4) -> None:
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = x.shape[-2] * x.shape[-1]
        if spatial <= 1:
            return x
        mean = x.mean(dim=(-2, -1), keepdim=True)
        deviation = (x - mean).pow(2)
        variance = deviation.sum(dim=(-2, -1), keepdim=True) / (spatial - 1)
        weights = deviation / (4.0 * (variance + self.e_lambda)) + 0.5
        return x * torch.sigmoid(weights)


class SimAMWithSlicing(nn.Module):
    """Apply SimAM independently on interleaved local slices."""

    def __init__(self, blocks: int = 2, e_lambda: float = 1e-4) -> None:
        super().__init__()
        if blocks < 1:
            raise ValueError("blocks must be >= 1")
        self.blocks = blocks
        self.attention = SimAM(e_lambda=e_lambda)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.blocks == 1 or min(x.shape[-2:]) < self.blocks:
            return self.attention(x)

        out = torch.empty_like(x)
        for row in range(self.blocks):
            for col in range(self.blocks):
                out[..., row :: self.blocks, col :: self.blocks] = self.attention(
                    x[..., row :: self.blocks, col :: self.blocks]
                )
        return out


class ConvSWS(nn.Module):
    """SWS attention wrapper for an existing Ultralytics Conv block."""

    def __init__(self, conv_block: nn.Module, blocks: int = 2, e_lambda: float = 1e-4) -> None:
        super().__init__()
        self.sws = SimAMWithSlicing(blocks=blocks, e_lambda=e_lambda)
        self.block = conv_block

    @property
    def conv(self) -> nn.Module | None:
        return getattr(self.block, "conv", None)

    @property
    def bn(self) -> nn.Module | None:
        return getattr(self.block, "bn", None)

    @property
    def act(self) -> nn.Module | None:
        return getattr(self.block, "act", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.sws(x))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.block, "forward_fuse"):
            return self.block.forward_fuse(self.sws(x))
        return self.forward(x)


class PartialConv2d(nn.Module):
    """PConv: spatial convolution on only a subset of channels."""

    def __init__(self, channels: int, n_div: int = 4) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        self.channels = channels
        self.dim_conv = max(1, channels // max(1, n_div))
        self.dim_untouched = channels - self.dim_conv
        self.partial_conv = nn.Conv2d(self.dim_conv, self.dim_conv, 3, 1, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim_untouched == 0:
            return self.partial_conv(x)
        x_conv, x_keep = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        return torch.cat((self.partial_conv(x_conv), x_keep), dim=1)


class FasterNetBlock(nn.Module):
    """Small FasterNet-style block used inside FBC2f."""

    def __init__(self, channels: int, shortcut: bool = True, n_div: int = 4) -> None:
        super().__init__()
        Conv = _ultralytics_conv()
        self.spatial = PartialConv2d(channels, n_div=n_div)
        self.mix = Conv(channels, channels, 1, 1)
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.mix(self.spatial(x))
        return x + y if self.shortcut else y


class FBC2f(nn.Module):
    """FasterNet block cross-stage partial fusion module."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
        pconv_n_div: int = 4,
    ) -> None:
        super().__init__()
        del g
        Conv = _ultralytics_conv()
        self.c = max(1, int(c2 * e))
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(FasterNetBlock(self.c, shortcut=shortcut, n_div=pconv_n_div) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(block(y[-1]) for block in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(block(y[-1]) for block in self.m)
        return self.cv2(torch.cat(y, 1))


@dataclass
class ReplacementStats:
    conv_sws: list[str] = field(default_factory=list)
    fbc2f: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    copied_tensors: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "conv_sws": self.conv_sws,
            "fbc2f": self.fbc2f,
            "skipped": self.skipped,
            "copied_tensors": self.copied_tensors,
        }


def _is_stride2_conv(module: nn.Module) -> bool:
    if isinstance(module, ConvSWS):
        return False
    if module.__class__.__name__ != "Conv" or not hasattr(module, "conv"):
        return False
    stride = getattr(module.conv, "stride", None)
    if isinstance(stride, tuple):
        return stride == (2, 2)
    return stride == 2


def _is_csp_like(module: nn.Module, target_names: set[str]) -> bool:
    if isinstance(module, FBC2f):
        return False
    if module.__class__.__name__ not in target_names:
        return False
    return all(hasattr(module, attr) for attr in ("cv1", "cv2", "m"))


def _infer_fbc2f_args(module: nn.Module) -> tuple[int, int, int, float]:
    c1 = int(module.cv1.conv.in_channels)
    c2 = int(module.cv2.conv.out_channels)
    n = len(module.m) if hasattr(module.m, "__len__") else 1
    hidden = int(getattr(module, "c", max(1, module.cv1.conv.out_channels // 2)))
    e = hidden / c2 if c2 else 0.5
    return c1, c2, max(1, n), max(0.125, min(1.0, e))


def _load_matching_state(dst: nn.Module, src: nn.Module) -> int:
    src_state = src.state_dict()
    dst_state = dst.state_dict()
    matched = {key: value for key, value in src_state.items() if key in dst_state and dst_state[key].shape == value.shape}
    if matched:
        dst_state.update(matched)
        dst.load_state_dict(dst_state)
    return len(matched)


def _replace_children(
    parent: nn.Module,
    stats: ReplacementStats,
    prefix: str,
    replace_sws: bool,
    replace_fbc2f: bool,
    sws_blocks: int,
    simam_lambda: float,
    pconv_n_div: int,
    target_csp_names: set[str],
) -> None:
    for name, child in list(parent.named_children()):
        path = f"{prefix}.{name}" if prefix else name

        if replace_sws and _is_stride2_conv(child):
            setattr(parent, name, ConvSWS(child, blocks=sws_blocks, e_lambda=simam_lambda))
            stats.conv_sws.append(path)
            continue

        if replace_fbc2f and _is_csp_like(child, target_csp_names):
            try:
                c1, c2, n, e = _infer_fbc2f_args(child)
                replacement = FBC2f(c1, c2, n=n, shortcut=True, e=e, pconv_n_div=pconv_n_div)
                stats.copied_tensors += _load_matching_state(replacement.cv1, child.cv1)
                stats.copied_tensors += _load_matching_state(replacement.cv2, child.cv2)
                setattr(parent, name, replacement)
                stats.fbc2f.append(path)
            except Exception as exc:  # pragma: no cover - defensive for Ultralytics version drift
                stats.skipped.append(f"{path}: {exc}")
            continue

        _replace_children(
            child,
            stats,
            path,
            replace_sws,
            replace_fbc2f,
            sws_blocks,
            simam_lambda,
            pconv_n_div,
            target_csp_names,
        )


def apply_pcb_yolo_mvp(
    yolo_or_model: Any,
    *,
    replace_sws: bool = True,
    replace_fbc2f: bool = True,
    sws_blocks: int = 2,
    simam_lambda: float = 1e-4,
    pconv_n_div: int = 4,
    target_csp_names: set[str] | None = None,
) -> ReplacementStats:
    """Inject PCB-YOLO MVP modules into an Ultralytics YOLO model."""

    model = getattr(yolo_or_model, "model", yolo_or_model)
    targets = target_csp_names or {"C2f", "C3k2"}
    stats = ReplacementStats()
    _replace_children(
        model,
        stats,
        "",
        replace_sws=replace_sws,
        replace_fbc2f=replace_fbc2f,
        sws_blocks=sws_blocks,
        simam_lambda=simam_lambda,
        pconv_n_div=pconv_n_div,
        target_csp_names=targets,
    )
    return stats
