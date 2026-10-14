"""重载正式检查点并检查新字典、事件排列、药物 mask 和历史因果性。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from training.v5.event_data import EventBatch
from training.v5.event_set_model import EventSetFusionModel
from training.v5.independent_fusion import IndependentBatch


def verify(root: Path) -> dict[str, object]:
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    history = json.loads((root / "history.json").read_text(encoding="utf-8"))
    audit = json.loads((root / "model_data_audit.json").read_text(encoding="utf-8"))
    if report["status"] != "training_completed" or len(history) != report["epochs_run"]:
        raise ValueError("正式训练报告或历史未完成")
    state = torch.load(root / "model.pt", map_location="cpu", weights_only=True)
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError("正式检查点有非有限权重")
    configuration = {"concept_count": state["concept_embedding.weight"].shape[0] - 1,
                     "type_count": state["type_embedding.weight"].shape[0] - 1,
                     "field_count": state["field_embedding.weight"].shape[0],
                     "text_width": state["text_projection.weight"].shape[1],
                     "width": state["concept_embedding.weight"].shape[1],
                     "event_width": state["event_embedding.weight"].shape[1],
                     "event_concept_count": report["event_concept_count"]}
    model = EventSetFusionModel(**configuration).eval()
    model.load_state_dict(state, strict=True)
    if configuration["event_concept_count"] != len(audit["event_concepts"]):
        raise ValueError("模型概念字典与数据审计不一致")
    if sum(p.numel() for p in model.parameters()) != report["parameter_count"]:
        raise ValueError("模型参数数量与报告不一致")
    values = torch.zeros((1, 2, 1, 5))
    masks = torch.zeros_like(values)
    masks[..., 0] = 1
    base = IndependentBatch(values, masks, torch.ones((1, 2, 1), dtype=torch.long),
                            torch.ones((1, 2, 1), dtype=torch.long), torch.ones((1, 2, 1), dtype=torch.bool),
                            torch.ones((2, configuration["text_width"])) / configuration["text_width"]**0.5,
                            torch.tensor([[0, 0], [1, 0]]), torch.zeros((1, 2, 2)),
                            torch.ones((1, 2), dtype=torch.bool), torch.tensor([[5., 365.]]), torch.tensor([[1, 0]]))
    dictionary = report["event_concepts"]
    lab_id, med_id = dictionary.index("ph:arterial") + 1, dictionary.index("saba") + 1
    batch = EventBatch(base, torch.tensor([[0., .5, 0., 0.], [1., .5, 0., 0.], [0., .2, 0., 0.]]),
                       torch.tensor([lab_id, lab_id, med_id]), torch.tensor([[2, 1]]),
                       torch.tensor([True, True, False]), torch.tensor([True, True, False]))
    with torch.no_grad():
        before = model(batch).log_amplitude
        order = torch.tensor([1, 0, 2])
        reordered = EventBatch(base, batch.event_values[order], batch.event_concepts[order], batch.event_counts,
                               batch.event_is_lab[order], batch.event_numeric_mask[order])
        permutation_delta = float((model(reordered).log_amplitude - before).abs().max())
        batch.event_values[2, 0] = 999.
        masked_delta = float((model(batch).log_amplitude - before).abs().max())
        batch.event_values[2, 1] = 999.
        future_delta = float((model(batch).log_amplitude[0, 0] - before[0, 0]).abs().max())
    if permutation_delta > 1e-5 or masked_delta > 1e-6 or future_delta > 1e-6:
        raise ValueError("正式检查点未通过事件类型／排列／因果性检查")
    result = {"status": "checkpoint_reloaded_and_checked", "model_configuration": configuration,
              "same_timestamp_permutation_max_logit_difference": permutation_delta,
              "medication_numeric_mask_max_logit_difference": masked_delta,
              "future_event_change_previous_prediction_difference": future_delta,
              "fixture": "synthetic observations only; not a new cohort evaluation or clinical accuracy test",
              "checkpoint_sha256": hashlib.sha256((root / "model.pt").read_bytes()).hexdigest()}
    (root / "checkpoint_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.run_dir)))
