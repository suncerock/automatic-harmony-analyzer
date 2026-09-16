import math
from collections.abc import Mapping
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from src.data.utils import ALL_TETRADS_INTERVALS, ALL_TRIADS_INTERVALS, ChordLabelConfig


def coerce_label_config(
    config: Union[ChordLabelConfig, Mapping[str, object], None],
) -> ChordLabelConfig:
    if config is None:
        return ChordLabelConfig()
    if isinstance(config, ChordLabelConfig):
        return config
    if isinstance(config, Mapping):
        return ChordLabelConfig(**dict(config))
    raise TypeError(
        "label_config must be a ChordLabelConfig, a mapping, or None."
    )


HARTE_CLASS_FACTORS = {
    "triads": len(ALL_TRIADS_INTERVALS),
    "tetrads": len(ALL_TETRADS_INTERVALS),
}

class RBPInformedChordHead(nn.Module):
    """Root-Bass-Pitch-informed chord head used by the CRNN backbone.

    Architecture
    ------------
    1. RBP predictions are produced directly from the backbone embedding
       (N, T, C) via linear layers.
    2. They are concatenated back to the embedding and fed
       through a linear layer to obtain the chord predictions.
    
    Parameters
    ----------
    feature_dim : int
        Dimensionality of the backbone output embedding.
    """
    def __init__(self, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.root_head = nn.Linear(feature_dim, 12)
        self.bass_head = nn.Linear(feature_dim, 12)
        self.pitch_head = nn.Linear(feature_dim, 12)
        self.chord_head = nn.Linear(feature_dim + 12 + 12 + 12, num_classes)

    def forward(self, embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        embedding : (N, T, C)

        Returns
        -------
        chord_logits : (N, num_classes, T) - permuted to match ACRModel convention
        """
        root_logits = self.root_head(embedding)  # (N, T, 12)
        bass_logits = self.bass_head(embedding)  # (N, T, 12)
        pitch_logits = self.pitch_head(embedding)  # (N, T, 12)

        fused = torch.cat([embedding, root_logits, bass_logits, pitch_logits], dim=-1)  # (N, T, C+12+12+12)
        chord_logits = self.chord_head(fused).permute(0, 2, 1)  # (N, num_classes, T)
        root_logits = root_logits.permute(0, 2, 1)   # (N, 12, T)
        bass_logits = bass_logits.permute(0, 2, 1)   # (N, 12, T)

        return dict(chord=chord_logits, root=root_logits, bass=bass_logits, pitch=pitch_logits)

class PitchInformedRBPHead(nn.Module):
    """Pitch-informed Root-Bitmap-Bass head used by the Conformer backbone.

    Architecture
    ------------
    1. A linear ``pitch`` head produces 12-dim pitch-class logits directly from
       the backbone embedding  (N, T, 12).
    2. Root and bass predictions are each conditioned on both the embedding *and*
       the pitch logits via a small fusion MLP, so they can leverage the
       pitch-class information without being constrained to it.

    Parameters
    ----------
    feature_dim : int
        Dimensionality of the backbone output embedding.
    fusion_dropout : float
        Dropout probability inside the fusion MLP.
    """

    def __init__(self, feature_dim: int, fusion_dropout: float = 0.1) -> None:
        super().__init__()

        # Primary pitch-class prediction (12 binary logits)
        self.pitch = nn.Linear(feature_dim, 12)

        fused_dim = feature_dim + 12  # embedding ‖ pitch logits

        self.root_fusion = nn.Sequential(
            nn.Linear(fused_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
        )
        self.root = nn.Linear(feature_dim, 12)

        self.bass_fusion = nn.Sequential(
            nn.Linear(fused_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
        )
        self.bass = nn.Linear(feature_dim, 12)

    def forward(self, embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        embedding : (N, T, C)

        Returns
        -------
        pitch_logits : (N, T, 12)
        root_logits  : (N, 12, T)   - permuted to match ACRModel convention
        bass_logits  : (N, 12, T)   - permuted to match ACRModel convention
        """
        pitch_logits = self.pitch(embedding)                            # (N, T, 12)

        fused = torch.cat([embedding, pitch_logits], dim=-1)            # (N, T, C+12)
        root_logits = self.root(self.root_fusion(fused)).permute(0, 2, 1)  # (N, 12, T)
        bass_logits = self.bass(self.bass_fusion(fused)).permute(0, 2, 1)  # (N, 12, T)

        return dict(pitch=pitch_logits, root=root_logits, bass=bass_logits)


class ACRModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        loss_fn: Optional[nn.Module] = None,
        optim_cfg: Optional[Dict] = None,
        mpe_model: Optional[nn.Module] = None,
        mpe_ckpt_path: Optional[str] = None,
        backbone_ckpt_path: Optional[str] = None,
        model_input_key: str = "x",
        label_config: Union[ChordLabelConfig, Mapping[str, object], None] = None,
        pitch_informed_rbp: bool = False,
        rbp_informed_chord: bool = False,
        compile: bool = True,
    ):
        super().__init__()

        self.backbone = backbone
        self.loss_fn = loss_fn

        self.label_config = coerce_label_config(label_config)
        self.pitch_informed_rbp = pitch_informed_rbp
        self.rbp_informed_chord = rbp_informed_chord

        self.mpe_model = mpe_model
        self.model_input_key = model_input_key

        if self.mpe_model is not None:
            self._initialize_mpe_stage(mpe_ckpt_path)

        self._initialize_backbone_stage(backbone_ckpt_path)
        feature_dim = getattr(self.backbone, "output_dim")
        self._initialize_heads(feature_dim)

    def _initialize_heads(self, feature_dim: int) -> None:
        if self.pitch_informed_rbp:
            assert self.label_config.require_rbp
            self.heads = PitchInformedRBPHead(feature_dim)
        elif self.rbp_informed_chord:
            assert self.label_config.require_harte and self.label_config.require_rbp
            num_classes = HARTE_CLASS_FACTORS[self.label_config.harte_vocab] * 12
            self.heads = RBPInformedChordHead(feature_dim, num_classes=num_classes)
        else:
            raise ValueError("Model must enable either pitch_informed_rbp or rbp_informed_chord.")

        if getattr(self, "_pending_heads_sd", None):
            missing, unexpected = self.heads.load_state_dict(self._pending_heads_sd, strict=False)
            if missing or unexpected:
                print(f"Heads checkpoint load mismatch. Missing: {missing}, unexpected: {unexpected}.")
            self._pending_heads_sd = None

    def _initialize_mpe_stage(self, checkpoint_path: Optional[str]) -> None:
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            missing, unexpected = self.mpe_model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                msg = f"MPE checkpoint load mismatch. Missing keys: {missing}, unexpected keys: {unexpected}."
                print(msg)
        self.mpe_model.requires_grad_(False)

    def _initialize_backbone_stage(self, checkpoint_path: Optional[str]) -> None:
        self._pending_heads_sd: Optional[Dict] = None

        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location="cpu")

            backbone_sd = {
                k.removeprefix("backbone."): v
                for k, v in state_dict.items()
                if k.startswith("backbone.")
            }
            heads_sd = {
                k.removeprefix("heads."): v
                for k, v in state_dict.items()
                if k.startswith("heads.")
            }

            if backbone_sd:
                missing, unexpected = self.backbone.load_state_dict(backbone_sd, strict=False)
                if missing or unexpected:
                    print(f"Backbone checkpoint load mismatch. Missing: {missing}, unexpected: {unexpected}.")
            self._pending_heads_sd = heads_sd

    def _run_mpe_stage(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            mpe_output = self.mpe_model(inputs).detach()
        return torch.sigmoid(mpe_output)

    def _forward_heads(self, embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.heads(embedding)

    def _gather_targets(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        targets: Dict[str, torch.Tensor] = {}
        if "y_chord" in batch:
            targets["chord"] = batch["y_chord"].long()
        if "y_root" in batch:
            targets["root"] = batch["y_root"].long()
        if "y_bass" in batch:
            targets["bass"] = batch["y_bass"].long()
        if "y_pitch" in batch:
            targets["pitch"] = batch["y_pitch"].float()
        if "y_hchord_str" in batch:
            targets["hchord_str"] = batch["y_hchord_str"]
        return targets
