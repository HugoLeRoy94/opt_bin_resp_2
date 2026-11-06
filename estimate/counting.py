"""Frequency-based entropy for arbitrary discrete vector symbols, on CPU."""
import math
from typing import Optional, Sequence

import torch


def entropy_from_counts(counts):
    """Plug-in and Miller-Madow estimates, plus the Good-Turing missing mass.

    Returns (plugin, miller_madow, n_unique, log2_n, missing_mass).

    `missing_mass` is f1/n, where f1 counts symbols seen EXACTLY ONCE. It is the
    Good-Turing estimate of the probability mass that was never sampled, and it is
    the quantity that says whether n was large enough for these frequencies to mean
    anything. A plug-in entropy computed where it is large measures the sample
    size, not the distribution.

    Prefer it to n_unique/n: that counts every distinct symbol rather than the
    once-only ones, so it overstates how badly a sample is covered and has no
    corresponding theory.

    Neither correction is a population bound. Miller-Madow adjusts by the symbols
    SEEN, which is itself an underestimate when most of the alphabet is missing.
    """
    n = int(counts.sum())
    if n <= 0:
        raise ValueError("Counting requires at least one observation.")
    p = counts.double() / n
    plugin = -(p * p.log2()).sum().item()
    unique = counts.numel()
    singletons = int((counts == 1).sum())
    return (plugin, plugin + (unique - 1) / (2 * n * math.log(2)),
            unique, math.log2(n), singletons / n)


_MAX_KEY = 1 << 62      # headroom below int64 overflow
_FLUSH_ROWS = 1 << 23   # buffered rows before a reduction (~67 MB of int64 keys)


def _word_plan(radices):
    """Split the columns into consecutive WORDS, each packed into one int64.

    Returns [(start, stop, strides, radices), ...]; within a word the first column
    is the most significant digit, and the words keep their column order, so
    sorting the packed words lexicographically reproduces the row order that
    torch.unique(dim=0) gives on the raw symbols. None when a single column is
    itself too large to pack (nothing can be done for it).
    """
    plan, start = [], 0
    while start < len(radices):
        stop, total = start, 1
        while stop < len(radices) and total * radices[stop] < _MAX_KEY:
            total *= radices[stop]
            stop += 1
        if stop == start:
            return None
        strides, step = [0] * (stop - start), 1
        for j in range(stop - 1, start - 1, -1):
            strides[j - start] = step
            step *= radices[j]
        plan.append((start, stop, torch.tensor(strides, dtype=torch.int64),
                     torch.tensor(radices[start:stop], dtype=torch.int64)))
        start = stop
    return plan


def _unique_counts(rows, weights=None):
    """(unique rows, summed weights) from a lexicographic sort, cheap in memory.

    torch.unique(dim=0) allocates several multiples of its input for a 2-D table —
    measured at ~350 bytes per row, i.e. 20 GB at 6.7e7 two-word responses. Sorting
    the words from least to most significant with a stable sort gives exactly the
    same row order at ~40 bytes per row, and segment-summing over equal-row runs
    gives the counts (weights=None counts one per row).
    """
    n = rows.shape[0]
    if rows.ndim == 1:
        order = torch.argsort(rows, stable=True) if weights is not None else None
        values = rows[order] if order is not None else torch.sort(rows).values
        new = torch.ones(n, dtype=torch.bool)
        new[1:] = values[1:] != values[:-1]
    else:
        order = torch.arange(n)
        for w in range(rows.shape[1] - 1, -1, -1):          # least significant first
            order = order[torch.argsort(rows[order, w], stable=True)]
        values = rows[order]
        new = torch.ones(n, dtype=torch.bool)
        new[1:] = (values[1:] != values[:-1]).any(1)
    edges = torch.cat((new.nonzero().squeeze(1), torch.tensor([n])))
    if weights is None:
        return values[edges[:-1]], torch.diff(edges)
    cumulative = torch.cat((torch.zeros(1, dtype=torch.int64), weights[order].cumsum(0)))
    return values[edges[:-1]], torch.diff(cumulative[edges])


class SymbolCounter:
    """Merge observed integer rows, retaining U unique symbols and frequencies.

    With `radices`, each row is packed into W = ceil over words int64 keys (ONE key
    whenever prod(radices) < 2^62, as for grouped cell counts; two for a 75-receptor
    binary response). Packing happens on the symbols' own device, so only 8W bytes
    per observation cross to CPU and the counting memory is independent of the
    number of columns. Without radices the raw rows are kept, which costs 8J bytes
    per observation; give radices whenever the alphabet is bounded.

    Rows are buffered and reduced when the buffer exceeds both _FLUSH_ROWS and the
    current unique table, so the table is re-sorted a logarithmic number of times
    instead of once per chunk: total cost O(n log n), not quadratic. Peak memory is
    the buffer plus the table, never the full stream of raw rows.

    No thresholding of count values greater than one, and no alphabet-sized
    allocation on either path.
    """

    def __init__(self, radices: Optional[Sequence[int]] = None, flush_rows: int = _FLUSH_ROWS):
        self._plan = self._radices = None
        if radices is not None:
            values = [int(r) for r in radices]
            if min(values, default=1) < 1:
                raise ValueError("Radices must be positive.")
            self._radices = torch.tensor(values, dtype=torch.int64)
            self._plan = _word_plan(values)
        self._flush_rows = int(flush_rows)
        self._device_plan = {}
        self._pending, self._pending_rows = [], 0
        self._keys = None        # packed (U, W) / (U,) keys, or raw (U, J) rows
        self._counts = None
        self._symbols = None
        self._width = None
        self.n_samples = 0

    # --- accumulation -----------------------------------------------------

    def _on(self, device):
        """(plan, radices) on `device`, cached: one host-to-device copy per device."""
        cached = self._device_plan.get(device)
        if cached is None:
            cached = ([(a, b, s.to(device), r) for a, b, s, r in self._plan or []],
                      None if self._radices is None else self._radices.to(device))
            self._device_plan[device] = cached
        return cached

    def _pack(self, symbols, plan):
        if self._plan is None:
            return symbols
        # Weighted sum, NOT a matmul: CUDA has no int64 GEMM ("addmv_impl_cuda not
        # implemented for 'Long'"), while elementwise multiply and sum are supported.
        words = [(symbols[:, a:b] * s).sum(dim=1) for a, b, s, _ in plan]
        return words[0] if len(words) == 1 else torch.stack(words, dim=1)

    def update(self, symbols):
        if symbols.ndim != 2 or symbols.shape[0] == 0:
            raise ValueError("Expected a nonempty matrix of discrete symbols.")
        symbols = symbols.detach().to(torch.int64)
        if self._width is None:
            self._width = symbols.shape[1]
        if symbols.shape[1] != self._width or (
                self._radices is not None and symbols.shape[1] != self._radices.numel()):
            raise ValueError("Symbol dimension changed between chunks.")
        plan, radices = self._on(symbols.device)
        if radices is not None and ((symbols < 0).any() or (symbols >= radices).any()):
            raise ValueError("Symbol value outside the declared radices.")
        self._pending.append(self._pack(symbols, plan).cpu())
        self._pending_rows += symbols.shape[0]
        self.n_samples += symbols.shape[0]
        self._symbols = None
        held = 0 if self._keys is None else self._keys.shape[0]
        if self._pending_rows >= max(self._flush_rows, held):
            self._flush()

    def _flush(self):
        """Fold the buffered rows into (unique keys, counts). Idempotent.

        Only the counts are needed for an entropy, and they cost 8 bytes per unique
        symbol. The (U, J) row table is reconstructed lazily by `symbols`, because
        materialising it would undo the whole point of packing.
        """
        if not self._pending:
            return
        rows = self._pending[0] if len(self._pending) == 1 else torch.cat(self._pending)
        self._pending, self._pending_rows = [], 0
        keys, counts = _unique_counts(rows)
        if self._keys is not None:
            keys, counts = _unique_counts(torch.cat((self._keys, keys)),
                                          torch.cat((self._counts, counts)))
        self._keys, self._counts, self._symbols = keys, counts, None

    # --- results ----------------------------------------------------------

    @property
    def symbols(self):
        self._flush()
        if self._keys is None:
            raise ValueError("Counting requires at least one observation.")
        if self._symbols is None:
            if self._plan is None:
                self._symbols = self._keys
            else:
                cols = []
                for w, (_, _, strides, radices) in enumerate(self._plan):
                    key = self._keys if self._keys.ndim == 1 else self._keys[:, w]
                    cols.append((key[:, None] // strides[None, :]) % radices[None, :])
                self._symbols = torch.cat(cols, dim=1)
        return self._symbols

    @property
    def counts(self):
        self._flush()
        if self._counts is None:
            raise ValueError("Counting requires at least one observation.")
        return self._counts

    def entropy(self):
        return entropy_from_counts(self.counts)
