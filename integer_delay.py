import torch
from torch import Tensor, nn


class IntegerDelayLines(nn.Module):
    """
    Non-differentiable Integer Delay Lines.
    Batch-aware vectorized implementation.
    """

    def __init__(self,
                 N: int,
                 buffer_len: int,
                 batch_size: int = 1,
                 device=None,
                 dtype=None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.factory_kwargs = {"device": device, "dtype": dtype}

        self.N = N
        self.L = buffer_len
        self.B = batch_size

        # Buffer shape: (B, N, L)
        self.buffer = torch.zeros(self.B, self.N, self.L, **self.factory_kwargs)

        # Base indices for circular indexing: (1, 1, L) → broadcast over B and N
        self.base = torch.arange(self.L, device=device).view(1, 1, self.L)

    def forward(self,
                inputs: Tensor,
                delays: Tensor,
                reflection_filters: Tensor = None) -> Tensor:
        """
        Args:
            inputs             : (B, N) or (B, N, 1)
            delays             : (N,) or (B, N)  — same delay for all batch items or per-item
            reflection_filters : (N, fir_order) or None

        Returns:
            outputs : (B, N, 1)
        """
        # Normalize input to (B, N)
        if inputs.ndim == 3:
            inputs = inputs.squeeze(-1)   # (B, N, 1) → (B, N)

        # Shift buffer left and write new samples
        # buffer: (B, N, L)
        self.buffer = torch.roll(self.buffer, -1, dims=-1)
        self.buffer[:, :, -1] = inputs    # (B, N)

        # Normalize delays to (B, N, 1) for broadcasting
        delays = delays.long() % self.L
        if delays.ndim == 1:
            # (N,) → (1, N, 1) → broadcast over B
            delays = delays.view(1, self.N, 1).expand(self.B, self.N, 1)
        elif delays.ndim == 2:
            # (B, N) → (B, N, 1)
            delays = delays.unsqueeze(-1)

        # Compute circular read indices: (B, N, L)
        shifted_idx = (self.base - delays) % self.L

        # Gather from buffer: (B, N, L)
        delayed_buffer = torch.gather(self.buffer, dim=2, index=shifted_idx)

        if reflection_filters is not None:
            # reflection_filters: (N, fir_order)
            n_bins = reflection_filters.shape[-1]
            # Extract last n_bins samples: (B, N, n_bins)
            windowed = delayed_buffer[:, :, self.L - n_bins: self.L]
            # Apply FIR filter: sum over last dim → (B, N)
            outputs = (reflection_filters.unsqueeze(0) * windowed).sum(-1)
        else:
            # Take last sample: (B, N)
            outputs = delayed_buffer[:, :, self.L - 1]

        return outputs.unsqueeze(-1)   # (B, N, 1)