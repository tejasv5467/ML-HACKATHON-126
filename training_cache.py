"""Validate the development-only feature-cache contract before fitting."""
import numpy as np

FEATURE_COUNT = 7
CANDIDATE_COUNT = 5


def validate_partition(X, y, groups, expected_trips):
    if X.ndim != 3 or X.shape[1:] != (CANDIDATE_COUNT, FEATURE_COUNT) or not len(X):
        raise ValueError("Invalid feature dimensions")
    if y.shape != X.shape[:2] or groups.shape != (len(X),):
        raise ValueError("Invalid label or group dimensions")
    if not np.isfinite(X).all() or not np.isfinite(groups).all():
        raise ValueError("Nonfinite features or group identifiers")
    if not np.isin(y, [0, 1]).all() or (y.sum(axis=1) > 1).any():
        raise ValueError("Invalid binary candidate proxy labels")
    if len(np.unique(groups)) != expected_trips:
        raise ValueError("Unexpected development trip count")
    if not np.any(y == 1) or not np.any(y == 0):
        raise ValueError("Both classes are required")


def validate_development(X, y, groups, tx, ty, tg, ids, fn, tn):
    validate_partition(X, y, groups, 80)
    validate_partition(tx, ty, tg, 10)
    if np.intersect1d(groups, tg).size:
        raise ValueError("Group leakage")
    for name, array in (("ids", ids), ("from_node", fn), ("to_node", tn)):
        if array.shape != ty.shape or not np.issubdtype(array.dtype, np.integer) or (array < 0).any():
            raise ValueError("Invalid " + name)
    if not np.any(tg[1:] == tg[:-1]):
        raise ValueError("No within-trip transitions for continuity metric")
