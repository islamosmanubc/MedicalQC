"""Federated samplers."""

from __future__ import annotations

from typing import Iterator, List

from torch.utils.data import Sampler


class FederatedClientSampler(Sampler[int]):
    """Sampler yielding indices for a specific hospital."""

    def __init__(self, indices: List[int], shuffle: bool = True, seed: int = 0) -> None:
        self.indices = indices
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[int]:
        if not self.shuffle:
            return iter(self.indices)
        rng = _simple_rng(self.seed)
        idx = self.indices.copy()
        rng.shuffle(idx)
        return iter(idx)

    def __len__(self) -> int:
        return len(self.indices)


class _SimpleRng:
    def __init__(self, seed: int) -> None:
        self.state = seed

    def rand(self) -> float:
        # Linear congruential generator for deterministic shuffling without numpy
        self.state = (1103515245 * self.state + 12345) & 0x7FFFFFFF
        return self.state / 0x7FFFFFFF

    def shuffle(self, data: List[int]) -> None:
        for i in range(len(data) - 1, 0, -1):
            j = int(self.rand() * (i + 1))
            data[i], data[j] = data[j], data[i]


def _simple_rng(seed: int) -> _SimpleRng:
    return _SimpleRng(seed)
