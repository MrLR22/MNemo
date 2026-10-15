# model/architecture.py
# Multi-Modal Temporal Action Recognition Model for Egocentric Video
#
# Architecture:
#   1. Modal Encoders: project each modality (CLIP, YOLO, Hand, Flow) to shared dim
#   2. Cross-Modal Transformer: attention-based fusion across modalities per frame
#   3. Temporal Transformer: attention across T frames to capture action dynamics
#   4. Prediction Heads: verb classification + noun classification
#
# Designed for EPIC-KITCHENS-100 evaluation. Input is pre-extracted features,
# NOT raw video frames — feature extraction uses the existing MNemo pipeline.

import math
import torch
import torch.nn as nn


# ── 1. Modal Encoders ──────────────────────────────────────────────────────────
# Each modality has different dimensionality. These small networks project
# them all into the same embedding space so the transformer can attend over them.

class ModalEncoder(nn.Module):
    """Projects one modality's raw features to a shared embedding dimension."""

    def __init__(self, input_dim: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, input_dim) → (batch, seq_len, embed_dim)"""
        return self.net(x)


# ── 2. Cross-Modal Transformer ────────────────────────────────────────────────
# For a single frame, we have 4 modality tokens. This transformer lets them
# attend to each other — e.g., "hand is near a cup" (hand ↔ object attention)
# becomes a stronger signal for "picking up" than either modality alone.

class CrossModalFusion(nn.Module):
    """Self-attention across modality tokens for a single timestep."""

    def __init__(self, embed_dim: int, num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Learned modality-type embeddings (like token type embeddings in BERT)
        # 4 modalities: CLIP=0, YOLO=1, Hand=2, Flow=3
        self.modality_embed = nn.Embedding(4, embed_dim)

    def forward(self, modality_tokens: torch.Tensor) -> torch.Tensor:
        """
        modality_tokens: (batch, 4, embed_dim) — one token per modality
        Returns: (batch, embed_dim) — single fused representation
        """
        batch_size = modality_tokens.size(0)

        # Add modality type embeddings so the model knows which token is which
        type_ids = torch.arange(4, device=modality_tokens.device).unsqueeze(0).expand(batch_size, -1)
        modality_tokens = modality_tokens + self.modality_embed(type_ids)

        # Self-attention across the 4 modality tokens
        fused = self.transformer(modality_tokens)  # (batch, 4, embed_dim)

        # Mean pool across modalities to get one vector per frame
        return fused.mean(dim=1)  # (batch, embed_dim)


# ── 3. Temporal Transformer ───────────────────────────────────────────────────
# Actions unfold over time. "take" = reach → grasp → lift. A single frame can't
# distinguish "reaching towards a cup" from "putting down a cup." This transformer
# models the temporal sequence of T fused-frame features.

class TemporalTransformer(nn.Module):
    """Self-attention across T timesteps to capture action dynamics."""

    def __init__(self, embed_dim: int, num_heads: int = 4, num_layers: int = 2,
                 max_seq_len: int = 16, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Sinusoidal positional encoding for temporal ordering
        self.pos_encoding = self._build_sinusoidal_pe(max_seq_len, embed_dim)

    def forward(self, frame_features: torch.Tensor) -> torch.Tensor:
        """
        frame_features: (batch, T, embed_dim) — T fused frame features
        Returns: (batch, embed_dim) — single clip-level representation
        """
        T = frame_features.size(1)
        pe = self.pos_encoding[:T].unsqueeze(0).to(frame_features.device)
        x = frame_features + pe

        x = self.transformer(x)  # (batch, T, embed_dim)

        # Mean pool across time to get one vector per clip
        return x.mean(dim=1)  # (batch, embed_dim)

    @staticmethod
    def _build_sinusoidal_pe(max_len: int, dim: int) -> torch.Tensor:
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe


# ── 4. Full Model ─────────────────────────────────────────────────────────────

class EgoActionRecognizer(nn.Module):
    """
    Multi-Modal Temporal Action Recognition Model for Egocentric Video.

    Input: pre-extracted features from the MNemo pipeline
        - clip_features:  (batch, T, 512)  CLIP ViT-B/32 frame embeddings
        - obj_features:   (batch, T, 80)   YOLO object presence (COCO classes)
        - hand_features:  (batch, T, 42)   MediaPipe hand landmarks (normalized)
        - flow_features:  (batch, T, 8)    Optical flow (direction + magnitude)

    Output:
        - verb_logits:    (batch, num_verbs)  action verb prediction
        - noun_logits:    (batch, num_nouns)  object noun prediction

    Total trainable params: ~800K-1.2M (lightweight, trainable on CPU/Colab)
    """

    # Feature dimensions from the MNemo pipeline (these are fixed)
    CLIP_DIM = 512
    OBJ_DIM = 80     # one-hot COCO class presence
    HAND_DIM = 42    # 21 landmarks × 2 (x, y normalized)
    FLOW_DIM = 8     # vx, vy, mag, dir_encoded × 2 (mean + std over frame)

    def __init__(
        self,
        num_verbs: int = 20,
        num_nouns: int = 30,
        embed_dim: int = 128,
        cross_modal_heads: int = 4,
        cross_modal_layers: int = 2,
        temporal_heads: int = 4,
        temporal_layers: int = 2,
        max_clip_frames: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Modal encoders — project each modality to embed_dim
        self.clip_encoder = ModalEncoder(self.CLIP_DIM, embed_dim, dropout)
        self.obj_encoder = ModalEncoder(self.OBJ_DIM, embed_dim, dropout)
        self.hand_encoder = ModalEncoder(self.HAND_DIM, embed_dim, dropout)
        self.flow_encoder = ModalEncoder(self.FLOW_DIM, embed_dim, dropout)

        # Cross-modal fusion — attention across 4 modality tokens per frame
        self.cross_modal = CrossModalFusion(embed_dim, cross_modal_heads, cross_modal_layers, dropout)

        # Temporal modeling — attention across T frames in a clip
        self.temporal = TemporalTransformer(embed_dim, temporal_heads, temporal_layers, max_clip_frames, dropout)

        # Prediction heads
        self.verb_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_verbs),
        )
        self.noun_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_nouns),
        )

    def forward(
        self,
        clip_features: torch.Tensor,
        obj_features: torch.Tensor,
        hand_features: torch.Tensor,
        flow_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        All inputs: (batch, T, feature_dim) where T = number of sampled frames.
        Returns: (verb_logits, noun_logits) each (batch, num_classes).
        """
        batch, T, _ = clip_features.shape

        # Step 1: Encode each modality to shared dimension
        clip_enc = self.clip_encoder(clip_features)    # (B, T, D)
        obj_enc = self.obj_encoder(obj_features)       # (B, T, D)
        hand_enc = self.hand_encoder(hand_features)    # (B, T, D)
        flow_enc = self.flow_encoder(flow_features)    # (B, T, D)

        # Step 2: Cross-modal fusion per frame
        # Reshape to process all frames across the batch together
        # Stack modalities: (B*T, 4, D)
        modal_tokens = torch.stack([clip_enc, obj_enc, hand_enc, flow_enc], dim=2)  # (B, T, 4, D)
        modal_tokens = modal_tokens.view(batch * T, 4, self.embed_dim)

        fused_frames = self.cross_modal(modal_tokens)  # (B*T, D)
        fused_frames = fused_frames.view(batch, T, self.embed_dim)  # (B, T, D)

        # Step 3: Temporal modeling across frames
        clip_embedding = self.temporal(fused_frames)  # (B, D)

        # Step 4: Predict verb and noun
        verb_logits = self.verb_head(clip_embedding)  # (B, num_verbs)
        noun_logits = self.noun_head(clip_embedding)  # (B, num_nouns)

        return verb_logits, noun_logits

    def count_parameters(self) -> int:
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Quick sanity check ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = EgoActionRecognizer(num_verbs=20, num_nouns=30)
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"\nArchitecture:\n{model}")

    # Dummy forward pass — batch=4, T=8 frames per clip
    B, T = 4, 8
    clip_feat = torch.randn(B, T, 512)
    obj_feat = torch.randn(B, T, 80)
    hand_feat = torch.randn(B, T, 42)
    flow_feat = torch.randn(B, T, 8)

    verb_logits, noun_logits = model(clip_feat, obj_feat, hand_feat, flow_feat)
    print(f"\nVerb logits shape: {verb_logits.shape}")   # (4, 20)
    print(f"Noun logits shape: {noun_logits.shape}")     # (4, 30)
    print("Forward pass OK.")
