"""World A 完整观察合同下、基于原因别累计发生概率的截断 C-index。"""

from __future__ import annotations

import numpy as np


def cif_concordance(times: np.ndarray, events: np.ndarray, predicted_cif: np.ndarray,
                    *, cause: int = 1, horizon: int = 365) -> dict[str, float | int]:
    """Wolbers 式可比对：目标事件较早 vs 目标事件较晚或竞争事件。

    无提前失访是此简化实现的必要条件。风险分数固定为 F_cause(horizon)，
    同时发生目标事件的配对不计入；同分风险给半分。重复出院的点估计按住院配对。
    """
    times = np.asarray(times)
    events = np.asarray(events)
    risk = np.asarray(predicted_cif)
    if not (len(times) == len(events) == len(risk)) or not len(times):
        raise ValueError("outcome/risk length mismatch")
    if cause not in (1, 2) or horizon < 1 or not np.all(np.isin(events, (0, 1, 2))):
        raise ValueError("invalid cause, horizon or event type")
    if np.any((events == 0) & (times < horizon)):
        raise ValueError("early censoring requires IPCW concordance")
    if not np.isfinite(risk).all() or np.any((risk < 0) | (risk > 1)):
        raise ValueError("invalid CIF risk")
    case_indices = np.flatnonzero((events == cause) & (times <= horizon))
    total, concordant = 0, 0.0
    for index in case_indices:
        # 竞争事件可早于目标事件；其后已不可能发生目标事件。
        comparable = (events != cause) | (times > times[index])
        comparable[index] = False
        if not np.any(comparable):
            continue
        difference = risk[index] - risk[comparable]
        total += len(difference)
        concordant += float(np.count_nonzero(difference > 1e-10))
        concordant += 0.5 * float(np.count_nonzero(np.abs(difference) <= 1e-10))
    if not total:
        raise ValueError("no comparable event pairs")
    return {"c_index": concordant / total, "target_events": int(len(case_indices)),
            "comparable_pairs": total, "horizon_days": horizon, "cause": cause}
