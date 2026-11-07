"""Geometry-conditioned translation/rotation queries over ViT patch features.

Geometry contains only observable image/camera information, never pose labels.
All attention uses ordinary PyTorch operations supported by CPU and Ascend NPU.
"""
import math

import torch
from torch import nn


def encode_bbox_geometry(bboxes, camera, device=None):
    """Encode pixel xyxy boxes as [(u-cx)/fx, (v-cy)/fy, w/fx, h/fy].

    Boxes must be the final boxes AFTER spatial augmentation, in original-image
    pixel coordinates. Zero/missing/degenerate boxes are rejected explicitly.
    """
    if camera is None or bboxes is None:
        raise ValueError("Geometry decoder requires bounding boxes and camera intrinsics")
    if camera.fx <= 0 or camera.fy <= 0:
        raise ValueError("Camera focal lengths must be positive")
    b = bboxes.to(device=device, dtype=torch.float32)
    if b.ndim != 2 or b.shape[-1] != 4:
        raise ValueError("Bounding boxes must have shape [B, 4] in pixel xyxy format")
    size = b[:, 2:] - b[:, :2]
    if not torch.isfinite(b).all() or not (size > 0).all():
        raise ValueError("Geometry decoder received non-finite or non-positive bounding boxes")
    center = (b[:, :2] + b[:, 2:]) * 0.5
    return torch.stack(((center[:, 0] - camera.cx) / camera.fx,
                        (center[:, 1] - camera.cy) / camera.fy,
                        size[:, 0] / camera.fx, size[:, 1] / camera.fy), dim=-1)


def forward_pose(model, images, bboxes=None, camera=None):
    """Keep the baseline call unchanged; condition the optional decoder only."""
    if getattr(model, "pose_decoder", "baseline") in ("geometry_dual_token", "geometry_dual_residual"):
        geometry = encode_bbox_geometry(bboxes, camera, images.device)
        return model(images, geometry=geometry)
    return model(images)


class CrossAttention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.query = nn.Linear(dim, dim)
        self.key_value = nn.Linear(dim, 2 * dim)
        self.output = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, memory):
        b, n, d = memory.shape
        q = self.query(query).reshape(b, 1, self.heads, self.head_dim).transpose(1, 2)
        k, v = self.key_value(memory).reshape(b, n, 2, self.heads, self.head_dim).unbind(2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        # FP32 scores/softmax avoid overflow under AMP; values stay in AMP dtype.
        score = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = self.dropout(score.softmax(dim=-1)).to(v.dtype)
        output = (weights @ v).transpose(1, 2).reshape(b, 1, d)
        return self.output(output)


class DualQueryBlock(nn.Module):
    def __init__(self, dim, heads, dropout, gated_interaction=True):
        super().__init__()
        self.query_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(2)])
        self.memory_norm = nn.LayerNorm(dim)
        self.attentions = nn.ModuleList([CrossAttention(dim, heads, dropout) for _ in range(2)])
        self.ffns = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4 * dim), nn.GELU(),
                          nn.Dropout(dropout), nn.Linear(4 * dim, dim), nn.Dropout(dropout))
            for _ in range(2)
        ])
        self.gated_interaction = gated_interaction
        if gated_interaction:
            self.gates = nn.ModuleList([nn.Linear(2 * dim, dim) for _ in range(2)])
            self.messages = nn.ModuleList([
                nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim)) for _ in range(2)
            ])
            for gate in self.gates:
                nn.init.constant_(gate.bias, -2.0)

    def forward(self, queries, memory):
        memory = self.memory_norm(memory)
        tokens = [queries[:, i:i + 1] + self.attentions[i](
            self.query_norms[i](queries[:, i:i + 1]), memory) for i in range(2)]
        if self.gated_interaction:
            # Both directions use the same pre-interaction tokens.
            pair = torch.cat(tokens, dim=-1)
            tokens = [tokens[i] + torch.sigmoid(self.gates[i](pair)) *
                      self.messages[i](tokens[1 - i]) for i in range(2)]
        return torch.cat([tokens[i] + self.ffns[i](tokens[i]) for i in range(2)], dim=1)


class GeometryDualTokenDecoder(nn.Module):
    def __init__(self, feature_dim, dim=256, heads=4, layers=2, dropout=0.0,
                 gated_interaction=True):
        super().__init__()
        if dim <= 0 or heads <= 0 or dim % heads or layers < 1:
            raise ValueError("Decoder dim must be positive/divisible by heads; layers >= 1")
        self.dim = dim
        self.memory_projection = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, dim))
        self.geometry_encoder = nn.Sequential(nn.Linear(4, dim), nn.GELU(), nn.Linear(dim, 2 * dim))
        self.task_tokens = nn.Parameter(torch.empty(1, 2, dim))
        nn.init.normal_(self.task_tokens, std=0.02)
        self.blocks = nn.ModuleList([
            DualQueryBlock(dim, heads, dropout, gated_interaction) for _ in range(layers)
        ])
        self.translation_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 3))
        self.rotation_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 6))
        # Start near valid depth and a non-degenerate 6D rotation, with nonzero
        # weights so all decoder/backbone paths receive gradients on step one.
        for head in (self.translation_head, self.rotation_head):
            nn.init.normal_(head[-1].weight, std=0.01)
            nn.init.zeros_(head[-1].bias)
        with torch.no_grad():
            self.translation_head[-1].bias[2] = 1.0
            self.rotation_head[-1].bias.copy_(torch.tensor([1., 0., 0., 0., 1., 0.]))

    def forward(self, patches, geometry):
        if geometry is None or geometry.shape != (patches.shape[0], 4):
            raise ValueError("Geometry decoder requires a [B, 4] geometry tensor")
        if not torch.isfinite(geometry).all():
            raise ValueError("Geometry tensor must be finite")
        memory = self.memory_projection(patches)
        condition = self.geometry_encoder(geometry.to(device=patches.device, dtype=patches.dtype))
        queries = self.task_tokens + condition.reshape(-1, 2, self.dim)
        for block in self.blocks:
            queries = block(queries, memory)
        return self.translation_head(queries[:, 0]), self.rotation_head(queries[:, 1])


def add_decoder_arguments(parser):
    group = parser.add_argument_group("Optional geometry-conditioned pose decoder")
    group.add_argument("--pose_decoder", choices=["baseline", "geometry_dual_token", "geometry_dual_residual"], default="baseline")
    group.add_argument("--decoder_dim", type=int, default=256)
    group.add_argument("--decoder_heads", type=int, default=4)
    group.add_argument("--decoder_layers", type=int, default=2)
    group.add_argument("--no_decoder_interaction", action="store_true",
                       help="Disable cross-task gates for an ablation")


def decoder_kwargs(args):
    return {"pose_decoder": args.pose_decoder, "decoder_dim": args.decoder_dim,
            "decoder_heads": args.decoder_heads, "decoder_layers": args.decoder_layers,
            "decoder_interaction": not args.no_decoder_interaction}
