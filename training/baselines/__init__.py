"""AECOPD 基线与公共统计工具。

历史 B1/B2/B3 使用当时的条件区间合同；连续时间 Cox 和秩 AUC 工具
供当前 V5 调用。两者不共享同一队列／评价定义，不能混报结果。
导入本包不启动训练；完整来源与依赖见 README.md。
"""

from __future__ import annotations

__all__ = ["shared_contract"]
