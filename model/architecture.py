# model/architecture.py
# HOI-TCN: causal temporal convolutional network for hand-object interaction events.
#
# Input : (B, T, 105) per-frame features from modules/features.py (hands, object, camera motion)
# Output: event_logits (B, T, 3)  per-frame {0: none, 1: pickup, 2: putdown}
#         hold_logits  (B, T)     per-frame "a hand is holding a tracked object" (auxiliary)
#         act_logits   (B, T)     optional (act_head=True): "the hands are doing ANY action" (cut, stir, wash,
#                                 open, ... not only pick/put): the interaction trigger's score
#
# Why this design (vs. the old CLIP + 2-transformer clip classifier):
#   - Pick-up vs put-down is defined by hand/object motion over ~1-3 s, not scene appearance.
#     Using geometry features instead of 512-d CLIP stops the model learning "kitchen ⇒ take".
#   - Causal convolutions only look at the past → the same model runs live, frame by frame.
#   - Per-frame outputs → events are localised in time (needed for continuous footage).
#   - ~0.2 M parameters: trains on CPU/Colab in minutes; small datasets don't overfit as fast.
#
# Receptive field = 1 + 2 * (kernel - 1) * sum(dilations) frames
#                 = 1 + 2 * 2 * (1 + 2 + 4 + 8) = 61 frames ≈ 6 s at 10 fps.

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.features import FEATURE_DIM, GROUPS

NUM_CLASSES = 3
CLASS_NAMES = ["none", "pickup", "putdown"]


class CausalConv1d(nn.Module):
    def __init__(self, c_in, c_out, kernel, dilation):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, kernel, dilation=dilation)

    def forward(self, x):                       # x: (B, C, T)
        return self.conv(F.pad(x, (self.pad, 0)))


class ChannelNorm(nn.Module):
    """LayerNorm over channels at each time step. (GroupNorm/BatchNorm would mix time steps
    and leak future frames into a 'causal' model.)"""

    def __init__(self, channels):
        super().__init__()
        self.ln = nn.LayerNorm(channels)

    def forward(self, x):                       # x: (B, C, T)
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class TemporalBlock(nn.Module):
    def __init__(self, channels, kernel, dilation, dropout):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel, dilation)
        self.norm1 = ChannelNorm(channels)
        self.norm2 = ChannelNorm(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.drop(F.gelu(self.norm1(self.conv1(x))))
        y = self.drop(F.gelu(self.norm2(self.conv2(y))))
        return x + y


class HOITCN(nn.Module):
    def __init__(self, hidden: int = 96, kernel: int = 3, dilations=(1, 2, 4, 8),
                 dropout: float = 0.2, group_dropout: float = 0.1, act_head: bool = False):
        super().__init__()
        self.kernel = kernel
        self.dilations = tuple(dilations)
        self.group_dropout = group_dropout
        # One small encoder per input group, so each modality is projected on its own scale
        # (also makes the "remove one modality" ablation clean).
        g = {k: s.stop - s.start for k, s in GROUPS.items()}
        self.enc = nn.ModuleDict({
            "hands": nn.Sequential(nn.Linear(g["hands"], 64), nn.GELU()),
            "object": nn.Sequential(nn.Linear(g["object"], 32), nn.GELU()),
            "ego": nn.Sequential(nn.Linear(g["ego"], 8), nn.GELU()),
        })
        self.proj = nn.Conv1d(64 + 32 + 8, hidden, 1)
        self.blocks = nn.ModuleList(TemporalBlock(hidden, kernel, d, dropout) for d in dilations)
        self.event_head = nn.Conv1d(hidden, NUM_CLASSES, 1)
        self.hold_head = nn.Conv1d(hidden, 1, 1)
        self.act_head = nn.Conv1d(hidden, 1, 1) if act_head else None

    @property
    def receptive_field(self) -> int:
        return 1 + 2 * (self.kernel - 1) * sum(self.dilations)

    def forward(self, x: torch.Tensor, drop_groups: tuple[str, ...] = (), return_act: bool = False):
        """x: (B, T, FEATURE_DIM). drop_groups zeroes whole modalities (ablation).
        return_act: also return the any-action logits (None if the model has no act head)."""
        assert x.shape[-1] == FEATURE_DIM, f"expected {FEATURE_DIM} features, got {x.shape[-1]}"
        parts = []
        for name, sl in GROUPS.items():
            xi = x[..., sl]
            if name in drop_groups:
                xi = torch.zeros_like(xi)
            elif self.training and self.group_dropout > 0 and name != "hands":
                # randomly hide the object/ego stream so the model copes when YOLO misses
                keep = (torch.rand(x.shape[0], 1, 1, device=x.device) > self.group_dropout).float()
                xi = xi * keep
            parts.append(self.enc[name](xi))
        h = self.proj(torch.cat(parts, dim=-1).transpose(1, 2))        # (B, hidden, T)
        for blk in self.blocks:
            h = blk(h)
        ev, hold = self.event_head(h).transpose(1, 2), self.hold_head(h).squeeze(1)
        if return_act:
            return ev, hold, (self.act_head(h).squeeze(1) if self.act_head is not None else None)
        return ev, hold

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_model(path: str, device: str = "cpu") -> HOITCN:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = HOITCN(**ckpt["hparams"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.threshold = ckpt.get("threshold")      # tuned on validation videos by train.py
    model.gate_score = ckpt.get("gate_score", "event")   # which output the trigger uses (train_interaction.py)
    return model


class StreamingModel:
    """Runs HOI-TCN live: keeps the last receptive-field frames and returns the newest
    frame's probabilities. Identical to running on the full sequence because it's causal."""

    def __init__(self, model: HOITCN):
        self.model = model.eval()
        self.window = model.receptive_field
        self.buf: list = []

    def reset(self):
        self.buf = []

    @torch.no_grad()
    def step(self, feat) -> "torch.Tensor":
        self.buf.append(torch.as_tensor(feat, dtype=torch.float32))
        self.buf = self.buf[-self.window:]
        x = torch.stack(self.buf).unsqueeze(0)
        ev, _ = self.model(x)
        return torch.softmax(ev[0, -1], dim=-1)
