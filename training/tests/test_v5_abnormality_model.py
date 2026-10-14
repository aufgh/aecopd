"""异常分支必须独立于旧住院内原始事件编码。"""

import unittest
import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from patient_data.expanded_abnormality_features import FEATURE_NAMES
from training.tests.test_v5_independent_fusion import _batch
from training.v5.abnormality_model import AbnormalityFusionModel
from training.v5.event_data import EventBatch, EventData
from training.v5.lab_recovery_data import (ABNORMALITY_GROUPS, CORE_RISK_GROUPS,
                                           SELECTED_AXIS_SCREEN, attach_abnormalities,
                                           _scale_abnormalities_on_train)


class AbnormalityModelTests(unittest.TestCase):
    def test_clinical_groups_cover_each_axis_once(self):
        axes = [axis for group in ABNORMALITY_GROUPS.values() for axis in group]
        self.assertEqual(len(axes), len(set(axes)))
        self.assertEqual(set(axes), set(name.removesuffix("__observed") for name in FEATURE_NAMES[::6]))

    def test_scaling_uses_training_rows_and_preserves_masks(self):
        values = np.zeros((3, len(FEATURE_NAMES)), dtype=np.float32)
        rows = values.reshape(3, -1, 6)
        rows[0, 0] = [1., 1., 2., .5, .5, 2.]
        rows[1, 0] = [1., 2., 4., 1., .5, 4.]
        rows[2, 0] = [1., 100., 200., 50., .5, 200.]
        scaled, parameters = _scale_abnormalities_on_train(
            values, np.array(["train", "train", "validation"]))
        axis = FEATURE_NAMES[0].removesuffix("__observed")
        self.assertLess(parameters["axes"][axis]["severity_log_p95_positive"], 5.)
        self.assertEqual(float(scaled[2, 0]), 1.)
        self.assertTrue(np.array_equal(scaled[:, ::6], values[:, ::6]))
        self.assertTrue(np.array_equal(scaled[:, 4::6], values[:, 4::6]))
        self.assertTrue(np.all(scaled[:, 6:] == 0))

    def test_raw_events_have_no_path_into_abnormality_model(self):
        torch.manual_seed(9)
        model = AbnormalityFusionModel(concept_count=3, type_count=2, field_count=2,
                                       text_width=8, width=32, abnormality_width=12).eval()
        self.assertFalse(hasattr(model, "event_encoder"))
        self.assertFalse(hasattr(model, "event_gru"))
        self.assertFalse(hasattr(model, "event_embedding"))
        features = torch.zeros((1, 2, 12))
        features[0, 1, :6] = torch.tensor([1., .5, 1., 0., 1., .7])
        common = _batch(text=False)
        empty = EventBatch(common, torch.empty((0, 4)), torch.empty((0,), dtype=torch.long),
                           torch.zeros((1, 2), dtype=torch.long), recovery_by_admission=features)
        synthetic = EventBatch(common, torch.tensor([[123., 5., 8., 13.]]),
                               torch.tensor([99]), torch.tensor([[0, 1]]),
                               recovery_by_admission=features)
        with torch.no_grad():
            first = model(empty).log_amplitude
            second = model(synthetic).log_amplitude
        self.assertTrue(torch.equal(first, second))

    def test_masked_arm_has_no_abnormality_signal(self):
        model = AbnormalityFusionModel(concept_count=3, type_count=2, field_count=2,
                                       text_width=8, width=32, abnormality_width=12).eval()
        common = _batch(text=False)
        values = torch.zeros((1, 2, 12))
        batch = EventBatch(common, torch.empty((0, 4)), torch.empty((0,), dtype=torch.long),
                           torch.zeros((1, 2), dtype=torch.long), recovery_by_admission=values)
        encoded, present = model._events(batch)
        self.assertFalse(bool(present.any()))
        self.assertTrue(torch.equal(encoded, torch.zeros_like(encoded)))

    def test_observed_abnormality_has_gradient(self):
        model = AbnormalityFusionModel(concept_count=3, type_count=2, field_count=2,
                                       text_width=8, width=32, abnormality_width=12)
        values = torch.zeros((1, 2, 12))
        values[0, 0, :6] = torch.tensor([1., .4, .8, .2, .75, 1.])
        batch = EventBatch(_batch(text=False), torch.empty((0, 4)),
                           torch.empty((0,), dtype=torch.long),
                           torch.zeros((1, 2), dtype=torch.long),
                           recovery_by_admission=values)
        model(batch).log_amplitude.sum().backward()
        self.assertGreater(float(model.axis_values[0].weight.grad.abs().sum()), 0.)

    def test_control_modes_mask_only_intended_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            features = np.ones((1, len(FEATURE_NAMES)), dtype=np.float32)
            np.savez_compressed(directory / "abnormality.npz", hadm_ids=np.array([1]),
                                features=features, feature_names=np.asarray(FEATURE_NAMES))
            digest = hashlib.sha256((directory / "abnormality.npz").read_bytes()).hexdigest()
            (directory / "manifest.json").write_text(json.dumps({
                "status": "derived_not_yet_validated_by_prediction", "output_sha256": digest,
                "input_sha256": {"processed_arrays": "source"}}), encoding="utf-8")
            data = EventData(SimpleNamespace(hadm_ids=np.array([1])),
                             np.empty((0, 4)), np.empty(0), np.empty(0),
                             (np.empty(0),), {"event_arrays": "source"})
            core = attach_abnormalities(data, directory, mode="core9").recovery_by_admission
            process = attach_abnormalities(data, directory, mode="measurement_only").recovery_by_admission
            self.assertTrue(np.all(core[:, :54] == 1))
            self.assertTrue(np.all(core[:, 54:] == 0))
            self.assertTrue(np.all(process.reshape(1, -1, 6)[:, :, (0, 5)] == 1))
            self.assertTrue(np.all(process.reshape(1, -1, 6)[:, :, 1:5] == 0))
            without = attach_abnormalities(data, directory,
                                           mode="without_cardiac_burden").recovery_by_admission
            self.assertTrue(np.all(without.reshape(1, -1, 6)[:, -1] == 0))
            self.assertTrue(np.all(without.reshape(1, -1, 6)[:, :-1] == 1))
            selected = attach_abnormalities(data, directory, mode="core_risk").recovery_by_admission
            shaped = selected.reshape(1, -1, 6)
            axes = [name.removesuffix("__observed") for name in FEATURE_NAMES[::6]]
            kept = {axis for group in CORE_RISK_GROUPS for axis in ABNORMALITY_GROUPS[group]}
            for index, axis in enumerate(axes):
                self.assertTrue(np.all(shaped[:, index] == (1 if axis in kept else 0)))
            selected = attach_abnormalities(
                data, directory, mode="without_renal_metabolic_perfusion"
            ).recovery_by_admission.reshape(1, -1, 6)
            removed = set(ABNORMALITY_GROUPS["renal"]) | set(ABNORMALITY_GROUPS["metabolic_perfusion"])
            for index, axis in enumerate(axes):
                self.assertTrue(np.all(selected[:, index] == (0 if axis in removed else 1)))
            for candidate in SELECTED_AXIS_SCREEN:
                screened = attach_abnormalities(
                    data, directory, mode=f"without_renal_metabolic_perfusion_plus_{candidate}"
                ).recovery_by_admission.reshape(1, -1, 6)
                for index, axis in enumerate(axes):
                    self.assertTrue(np.all(screened[:, index] == (0 if axis in removed | {candidate} else 1)))
            for candidate in axes:
                with self.subTest(single_axis=candidate):
                    single = attach_abnormalities(
                        data, directory, mode=f"without_axis_{candidate}"
                    ).recovery_by_admission.reshape(1, -1, 6)
                    for index, axis in enumerate(axes):
                        self.assertTrue(np.all(single[:, index] == (0 if axis == candidate else 1)))


if __name__ == "__main__":
    unittest.main()
