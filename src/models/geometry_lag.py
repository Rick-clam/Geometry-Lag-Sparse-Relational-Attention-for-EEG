from __future__ import annotations

import re

import numpy as np
import torch

from .relational_attention import NeighborTemplate


def _canonical_piece(value: str) -> str:
    value = value.upper().strip().replace("EEG", "")
    value = value.replace("T3", "T7").replace("T4", "T8").replace("T5", "P7").replace("T6", "P8")
    value = re.sub(r"-(REF|LE|AVG)$", "", value)
    return value.strip(" -_")


def _endpoints(label: str, known: set[str]) -> tuple[str | None, str | None]:
    pieces = [_canonical_piece(piece) for piece in label.replace("/", "-").split("-")]
    pieces = [piece for piece in pieces if piece in known]
    if len(pieces) >= 2 and pieces[1] not in {"A1", "A2", "M1", "M2"}:
        return pieces[0], pieces[1]
    if pieces:
        return pieces[0], None
    compact = _canonical_piece(label)
    for name in sorted(known, key=len, reverse=True):
        if name in compact:
            return name, None
    return None, None


def channel_coordinates(labels: list[str]) -> np.ndarray:
    """Return unit-sphere 10-20 coordinates, using midpoints for bipolar channels."""
    import mne

    positions = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
    positions = {name.upper(): np.asarray(pos, dtype=np.float32) for name, pos in positions.items()}
    known = set(positions)
    coordinates = []
    for label in labels:
        first, second = _endpoints(label, known)
        if first in positions and second in positions:
            point = positions[first] + positions[second]
        elif first in positions:
            point = positions[first]
        else:
            coordinates.append(np.full(3, np.nan, dtype=np.float32))
            continue
        coordinates.append(point / max(float(np.linalg.norm(point)), 1e-8))
    return np.stack(coordinates)


def _spatial_groups(coordinates: np.ndarray, center: int, valid_channels: np.ndarray):
    if not valid_channels[center] or not np.isfinite(coordinates[center]).all():
        return [], [], [], {}
    candidates = []
    for other in np.flatnonzero(valid_channels):
        if other == center or not np.isfinite(coordinates[other]).all():
            continue
        distance = float(np.linalg.norm(coordinates[center] - coordinates[other]) / 2.0)
        candidates.append((distance, int(other)))
    candidates.sort()
    near = [channel for _, channel in candidates[:4]]
    remaining = [(d, c) for d, c in candidates if c not in near]
    far = [channel for _, channel in remaining[-2:]] if remaining else []
    remaining = [(d, c) for d, c in remaining if c not in far]
    middle = len(remaining) // 2
    mid = [channel for _, channel in remaining[max(0, middle - 1):middle + 1]]
    return near, mid, far, {channel: distance for distance, channel in candidates}


def build_neighbor_template(token_valid: torch.Tensor, coordinates: torch.Tensor) -> NeighborTemplate:
    """Construct T, S, and geometry-lag ST neighborhoods for a token grid."""
    valid_np = token_valid.detach().cpu().numpy()
    coords_np = coordinates.detach().cpu().numpy()
    batch, channels, times = valid_np.shape
    length = channels * times

    def allocate(slots: int):
        return (
            np.zeros((batch, length, slots), dtype=np.int64),
            np.zeros((batch, length, slots), dtype=bool),
            np.zeros((batch, length, slots, 2), dtype=np.float32),
        )

    idx_t, val_t, rel_t = allocate(8)
    idx_s, val_s, rel_s = allocate(8)
    idx_st, val_st, rel_st = allocate(16)
    lags_t = (-4, -3, -2, -1, 1, 2, 3, 4)
    for b in range(batch):
        channel_valid = valid_np[b].any(axis=1)
        groups = [_spatial_groups(coords_np[b], c, channel_valid) for c in range(channels)]
        for c in range(channels):
            near, mid, far, distances = groups[c]
            spatial = (near + mid + far)[:8]
            st_pairs = (
                [(other, lag) for other in near[:4] for lag in (-1, 1)]
                + [(other, lag) for other in mid[:2] for lag in (-4, 4)]
                + [(other, lag) for other in far[:2] for lag in (-8, 8)]
            )[:16]
            for n in range(times):
                center = c * times + n
                if not valid_np[b, c, n]:
                    continue
                for slot, lag in enumerate(lags_t):
                    other_n = n + lag
                    if 0 <= other_n < times and valid_np[b, c, other_n]:
                        idx_t[b, center, slot] = c * times + other_n
                        val_t[b, center, slot] = True
                        rel_t[b, center, slot, 0] = lag / 4.0
                for slot, other_c in enumerate(spatial):
                    if valid_np[b, other_c, n]:
                        idx_s[b, center, slot] = other_c * times + n
                        val_s[b, center, slot] = True
                        rel_s[b, center, slot, 0] = distances[other_c]
                for slot, (other_c, lag) in enumerate(st_pairs):
                    other_n = n + lag
                    if 0 <= other_n < times and valid_np[b, other_c, other_n]:
                        idx_st[b, center, slot] = other_c * times + other_n
                        val_st[b, center, slot] = True
                        rel_st[b, center, slot] = (distances[other_c], lag / 8.0)
    return NeighborTemplate(
        torch.from_numpy(idx_t), torch.from_numpy(val_t), torch.from_numpy(rel_t),
        torch.from_numpy(idx_s), torch.from_numpy(val_s), torch.from_numpy(rel_s),
        torch.from_numpy(idx_st), torch.from_numpy(val_st), torch.from_numpy(rel_st),
    )

