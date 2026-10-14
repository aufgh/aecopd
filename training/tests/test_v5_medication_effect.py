"""药理作用组的复方去重、消融隔离及模型通路。"""

import csv
import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from training.tests.test_v5_independent_fusion import _batch
from training.v5.abnormality_model import AbnormalityFusionModel
from training.v5.event_data import EventBatch, EventData
from training.v5.medication_effect_data import EFFECT_GROUPS, attach_medication_effects
from training.v5.medication_effect_model import MedicationEffectFusionModel
from training.v5.train_event_sequence import _build_model_with_matched_text_initialization
from training.v5.train_event_sequence import deterministic_inference


class MedicationEffectTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_gpu_text_pooling_repeats_exactly_with_deterministic_inference(self):
        model = MedicationEffectFusionModel(concept_count=3, type_count=2, field_count=2,
                                             text_width=8, width=32, abnormality_width=12).cuda().eval()
        batch = EventBatch(_batch(text=True), torch.empty((0, 4)),
                           torch.empty((0,), dtype=torch.long), torch.zeros((1, 2), dtype=torch.long),
                           recovery_by_admission=torch.ones((1, 2, 12)),
                           medication_by_admission=torch.ones((1, 2, 48))).to(torch.device("cuda"))
        @deterministic_inference
        @torch.no_grad()
        def inference():
            return model(batch).log_amplitude
        first = inference()
        for _ in range(5):
            self.assertTrue(torch.equal(first, inference()))

    def test_deterministic_inference_restores_flags_on_exception(self):
        original = torch.are_deterministic_algorithms_enabled()
        original_warning = torch.is_deterministic_algorithms_warn_only_enabled()
        @deterministic_inference
        def failing():
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
            raise RuntimeError("intentional")
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            failing()
        self.assertEqual(torch.are_deterministic_algorithms_enabled(), original)
        self.assertEqual(torch.is_deterministic_algorithms_warn_only_enabled(), original_warning)

    def test_qwen_width_matches_bert_shared_initialization(self):
        arguments = dict(concept_count=3, type_count=2, field_count=2,
                         event_arguments={"abnormality_width": 12}, device=torch.device("cpu"))
        torch.manual_seed(2026)
        bert = _build_model_with_matched_text_initialization(
            MedicationEffectFusionModel, text_width=8,
            shared_text_init_width=None, **arguments).state_dict()
        torch.manual_seed(2026)
        qwen = _build_model_with_matched_text_initialization(
            MedicationEffectFusionModel, text_width=16,
            shared_text_init_width=8, **arguments).state_dict()
        self.assertEqual(qwen["text_projection.weight"].shape[1], 16)
        for name, value in bert.items():
            if not name.startswith("text_projection."):
                self.assertTrue(torch.equal(value, qwen[name]), name)

    def test_one_combination_administration_counts_once_per_effect_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            path = folder / "events.csv.gz"
            with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("kind", "concept", "hadm_id", "source_id", "elapsed_days"))
                writer.writeheader()
                for concept in ("macrolide", "beta_lactam", "tetracycline", "macrolide", "saba"):
                    writer.writerow({"kind": "medication", "concept": concept, "hadm_id": 1,
                                     "source_id": "emar:1", "elapsed_days": 2})
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest = folder / "manifest.json"
            manifest.write_text(json.dumps({"status": "preprocessed_and_checked",
                                            "output_sha256": {"events.csv.gz": digest}}), encoding="utf-8")
            data = EventData(SimpleNamespace(hadm_ids=np.array([1])), np.empty((0, 4)),
                             np.empty(0), np.empty(0), (np.empty(0),),
                             {"processed_manifest": hashlib.sha256(manifest.read_bytes()).hexdigest()})
            active = attach_medication_effects(data, folder, mode="active").medication_by_admission
            antibiotic = list(EFFECT_GROUPS).index("common_systemic_antibacterial") * 6
            self.assertAlmostEqual(float(active[0, antibiotic + 1]), float(np.log1p(1)))
            excluded = attach_medication_effects(data, folder,
                mode="without_common_systemic_antibacterial").medication_by_admission
            self.assertTrue(np.all(excluded[0, antibiotic:antibiotic + 6] == 0))
            self.assertGreater(float(excluded[0, 0]), 0.)
            short_pair = attach_medication_effects(data, folder,
                mode="without_short_bronchodilators").medication_by_admission
            self.assertTrue(np.all(short_pair[0, :12] == 0))
            self.assertGreater(float(short_pair[0, antibiotic]), 0.)
            maintenance = attach_medication_effects(data, folder,
                mode="without_maintenance_inhalers").medication_by_admission
            self.assertGreater(float(maintenance[0, 0]), 0.)
            self.assertTrue(np.all(maintenance[0, 12:30] == 0))
            acute = attach_medication_effects(data, folder,
                mode="core_acute").medication_by_admission
            self.assertGreater(float(acute[0, 0]), 0.)
            self.assertGreater(float(acute[0, antibiotic]), 0.)
            self.assertTrue(np.all(acute[0, 12:30] == 0))
            masked = attach_medication_effects(data, folder, mode="masked").medication_by_admission
            self.assertTrue(np.all(masked == 0))
            exposure = attach_medication_effects(data, folder, mode="exposure_only").medication_by_admission
            self.assertTrue(np.array_equal(exposure[:, ::6], active[:, ::6]))
            self.assertTrue(np.all(exposure.reshape(1, -1, 6)[:, :, 1:] == 0))

    def test_medication_changes_prediction_but_raw_events_do_not(self):
        torch.manual_seed(5)
        model = MedicationEffectFusionModel(concept_count=3, type_count=2, field_count=2,
            text_width=8, width=32, abnormality_width=12).eval()
        common = _batch(text=False)
        laboratory = torch.zeros((1, 2, 12))
        medication = torch.zeros((1, 2, len(EFFECT_GROUPS) * 6))
        medication[0, 1, :6] = torch.tensor([1., .69, .69, .0, .0, .0])
        def batch(medication_features, raw_values):
            return EventBatch(common, raw_values, torch.empty((len(raw_values),), dtype=torch.long),
                              torch.zeros((1, 2), dtype=torch.long),
                              recovery_by_admission=laboratory,
                              medication_by_admission=medication_features)
        with torch.no_grad():
            active = model(batch(medication, torch.empty((0, 4)))).log_amplitude
            changed_raw = model(batch(medication, torch.tensor([[5., 6., 7., 8.]]))).log_amplitude
            masked = model(batch(torch.zeros_like(medication), torch.empty((0, 4)))).log_amplitude
        self.assertTrue(torch.equal(active, changed_raw))
        self.assertFalse(torch.equal(active, masked))

    def test_masked_medication_preserves_lab_only_forward(self):
        arguments = dict(concept_count=3, type_count=2, field_count=2,
                         text_width=8, width=32, abnormality_width=12)
        torch.manual_seed(17)
        old = AbnormalityFusionModel(**arguments).eval()
        torch.manual_seed(17)
        new = MedicationEffectFusionModel(**arguments).eval()
        for name, value in old.state_dict().items():
            self.assertTrue(torch.equal(value, new.state_dict()[name]), name)
        lab = torch.zeros((1, 2, 12))
        lab[0, 1, :6] = torch.tensor([1., .4, .8, .2, .75, 1.])
        batch = EventBatch(_batch(text=False), torch.empty((0, 4)),
                           torch.empty((0,), dtype=torch.long),
                           torch.zeros((1, 2), dtype=torch.long),
                           recovery_by_admission=lab,
                           medication_by_admission=torch.zeros((1, 2, len(EFFECT_GROUPS) * 6)))
        with torch.no_grad():
            self.assertTrue(torch.equal(old(batch).log_amplitude, new(batch).log_amplitude))


if __name__ == "__main__":
    unittest.main()
