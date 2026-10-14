"""纯数值预处理：小量纲使用真实 IQR，零跨度显式回退。"""
from __future__ import annotations

import numpy as np


def robust_parameters(observed: np.ndarray) -> dict[str, float | int | str]:
    values = np.asarray(observed, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("尺度拟合需要非空、有限的训练观测")
    median = float(np.median(values))
    q1, q3 = np.percentile(values, (25, 75))
    iqr = float(q3 - q1)
    tolerance = max(abs(median), 1.0) * 1e-8
    if iqr > tolerance:
        scale, method = iqr, "iqr"
    else:
        deviation = float(np.std(values))
        scale, method = (deviation, "std_fallback") if deviation > tolerance else (1.0, "constant")
    return {"median": median, "scale": scale, "iqr": iqr,
            "method": method, "training_observations": int(values.size)}
