"""Population Stability Index numeric core adapted from Evidently.

Copyright 2021-2023 Evidently AI, Inc.
Licensed under Apache License 2.0; see EVIDENTLY_LICENSE.txt beside this file.
Source: evidentlyai/evidently, commit
34771fe41c08ca074d96a52aaf4a45cd640cbaf5,
src/evidently/legacy/calculations/stattests/psi.py, function ``_psi``.

MODIFIED in YK Evidence: extracted the NumPy elementwise PSI calculation;
accepts already-binned positive vectors and returns the original per-bin
contributions as well as their sum. Evidently's pandas/stat-test registration,
its combined-sample binning, zero replacement and default 0.1 drift decision
are not included. The caller's frozen reference bins and additive smoothing
are separate project code in ``monitoring.py``.
"""

from __future__ import annotations

import numpy as np


def psi_components(reference_percents: np.ndarray, current_percents: np.ndarray) -> dict:
    """Return Evidently's PSI terms and sum for aligned positive vectors."""
    reference = np.asarray(reference_percents)
    current = np.asarray(current_percents)
    if reference.ndim != 1 or current.ndim != 1 or not len(reference) or len(reference) != len(current):
        raise ValueError("PSI requires aligned nonempty one-dimensional vectors")
    if reference.dtype.kind not in "ifu" or current.dtype.kind not in "ifu":
        raise ValueError("PSI proportions must be numeric without booleans")
    reference = reference.astype(np.float64)
    current = current.astype(np.float64)
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(current)) or np.any(reference <= 0) or np.any(current <= 0):
        raise ValueError("PSI proportions must be finite and strictly positive")

    # Unchanged numerical expression from Evidently's _psi at the pinned commit.
    psi_values = (reference - current) * np.log(reference / current)
    psi_value = np.sum(psi_values)
    return {"contributions": psi_values.tolist(), "psi": float(psi_value)}
