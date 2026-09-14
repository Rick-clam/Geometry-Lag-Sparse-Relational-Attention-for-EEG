from __future__ import annotations

import torch


def masked_reconstruction_loss(prediction, target, masked, token_valid):
    effective = masked & token_valid
    weights = effective.unsqueeze(-1).float()
    count = weights.sum().clamp_min(1.0)
    time_loss = ((prediction.float() - target.float()).abs() * weights).sum()
    time_loss = time_loss / (count * target.shape[-1])
    pred_power = torch.log1p(torch.fft.rfft(prediction.float(), dim=-1, norm="ortho").abs().square())
    target_power = torch.log1p(torch.fft.rfft(target.float(), dim=-1, norm="ortho").abs().square())
    spectral_loss = ((pred_power - target_power).abs() * weights).sum()
    spectral_loss = spectral_loss / (count * pred_power.shape[-1])
    return time_loss + 0.1 * spectral_loss, time_loss, spectral_loss

