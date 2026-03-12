import torch
from torch import Tensor, nn
from curves import energy_decay_curve, echo_density_profile, mel_energy_decay_relief


def lp_error_fn(pred: Tensor, target: Tensor, power: float = 1.0,
                normalize: bool = False) -> Tensor:
    """L^p loss with optional normalization by the target norm."""
    loss = torch.mean(torch.abs(target - pred) ** power)
    if normalize:
        norm = torch.mean(torch.abs(target) ** power)
        return loss / norm
    return loss

def _to_batch(x: Tensor) -> Tensor:
    """Ensure tensor is 2-D (B, T). Adds a batch dimension if input is 1-D."""
    if x.ndim == 1:
        return x.unsqueeze(0)
    if x.ndim == 2:
        return x
    raise ValueError(f"Expected 1-D or 2-D input tensor, got shape {tuple(x.shape)}.")

def _apply_batched(fn, x: Tensor) -> Tensor:
    """
    Apply a 1-D curve function to each row of a (B, T) tensor.

    Returns a stacked tensor whose shape depends on the output of fn:
      - scalar per sample  → (B,)
      - 1-D array per sample → (B, L)
      - 2-D array per sample → (B, F, L)
    """
    return torch.stack([fn(x[i]) for i in range(x.shape[0])])

class EDCLoss(nn.Module):
    def __init__(self, power: float = 2.0, **kwargs) -> None:
        """Normalized MSE between linear-amplitude full-band EDC curves."""
        super().__init__(**kwargs)
        self.power = power

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            pred:   (T,) or (B, T)
            target: (T,) or (B, T)
        Returns:
            Scalar loss.
        """
        pred   = _to_batch(pred)    # (B, T)
        target = _to_batch(target)  # (B, T)
        edc_pred   = _apply_batched(energy_decay_curve, pred)    # (B, T)
        edc_target = _apply_batched(energy_decay_curve, target)  # (B, T)
        return lp_error_fn(edc_pred, edc_target, self.power, normalize=True)


class MelEDRLogLoss(nn.Module):
    def __init__(self, sr: int, power: float = 1.0, **kwargs) -> None:
        """Normalized MAE between log-amplitude mel-frequency EDRs."""
        super().__init__(**kwargs)
        self.sr    = sr
        self.power = power

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            pred:   (T,) or (B, T)
            target: (T,) or (B, T)
        Returns:
            Scalar loss.
        """
        pred   = _to_batch(pred)    # (B, T)
        target = _to_batch(target)  # (B, T)
        edr_fn = lambda x: mel_energy_decay_relief(x, self.sr, return_db=True)  # noqa: E731
        edr_pred   = _apply_batched(edr_fn, pred)    # (B, F, L)
        edr_target = _apply_batched(edr_fn, target)  # (B, F, L)
        return lp_error_fn(edr_pred, edr_target, self.power, normalize=True)


class EDPLoss(nn.Module):
    def __init__(self, sr: int, win_duration: float = 0.02, power: float = 2.0, **kwargs) -> None:
        """MSE between SoftEDP curves. Reference: https://doi.org/10.1186/s13636-024-00371-5"""
        super().__init__(**kwargs)
        self.sr           = sr
        self.win_duration = win_duration
        self.power        = power

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            pred:   (T,) or (B, T)
            target: (T,) or (B, T)
        Returns:
            Scalar loss.
        """
        pred   = _to_batch(pred)    # (B, T)
        target = _to_batch(target)  # (B, T)
        edp_fn = lambda x: echo_density_profile(x, sr=self.sr, win_duration=self.win_duration, differentiable=True) # noqa: E731
        profile_pred   = _apply_batched(edp_fn, pred)    # (B, L)
        profile_target = _apply_batched(edp_fn, target)  # (B, L)

        return lp_error_fn(profile_pred, profile_target, self.power, normalize=False)