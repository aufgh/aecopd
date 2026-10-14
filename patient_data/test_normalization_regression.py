"""在无 PyTorch 环境下直接执行旧加载器的实际归一化函数。"""
from __future__ import annotations

import ast
import unittest
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np


@dataclass
class Fixture:
    hadm_ids: np.ndarray
    splits: np.ndarray
    token_values: np.ndarray
    token_masks: np.ndarray
    token_concepts: np.ndarray
    tokens_by_admission: tuple[np.ndarray, ...]
    concept_names: tuple[str, ...]


class NormalizationRegression(unittest.TestCase):
    def test_actual_static_normalizer_preserves_small_iqr(self) -> None:
        path = Path(__file__).resolve().parents[1] / "training/v5/independent_data.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "normalize_on_training_patients")
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function], type_ignores=[])
        namespace = {"np": np, "replace": replace, "robust_parameters": None}
        # 修正后函数从纯数值模块使用同一尺度策略。
        try:
            from training.v5.numeric_preprocessing import robust_parameters
            namespace["robust_parameters"] = robust_parameters
        except ModuleNotFoundError:
            pass
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
        values = np.zeros((5, 5), dtype=np.float32)
        values[:, 0] = [7.30, 7.35, 7.40, 7.45, 90.0]
        masks = np.zeros_like(values)
        masks[:, 0] = 1
        fixture = Fixture(np.array([1, 2]), np.array(["train", "test"]), values, masks,
                          np.ones(5, dtype=np.int64), (np.arange(4), np.array([4])), ("ph",))
        _, parameters = namespace["normalize_on_training_patients"](fixture)
        self.assertAlmostEqual(parameters["ph"]["scale"][0], 0.075, places=5)
        fixture.token_values[:4, 0] = [0.0, 0.0, 0.0, 1.0]
        _, parameters = namespace["normalize_on_training_patients"](fixture)
        self.assertEqual(parameters["ph"]["scale"][0], 1.0)


if __name__ == "__main__":
    unittest.main()
