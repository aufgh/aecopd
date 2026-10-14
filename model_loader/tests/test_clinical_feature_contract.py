from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clinical_feature_contract import ClinicalFeatureContract


CONTRACT_PATH = Path(__file__).resolve().parents[2] / "config" / "clinical_feature_contract.csv"
DIAGNOSIS_V2_CONTRACT_PATH = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "clinical_feature_contract_v2_diagnosis.csv"
)


def event(
    concept_name: str,
    *,
    event_domain: str = "药物",
    model_scope: str = "main_model",
    status: str = "Administered",
) -> dict[str, str]:
    return {
        "event_domain": event_domain,
        "concept_name": concept_name,
        "model_scope": model_scope,
        "status": status,
        "route": "IV",
        "evidence_type": "actual_administration",
        "source_table": "mimiciv_hosp.emar",
    }


class ClinicalFeatureContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = ClinicalFeatureContract.from_csv(CONTRACT_PATH)

    def test_systemic_steroids_share_one_canonical_treatment_class(self) -> None:
        prednisone = self.contract.map_event(event("泼尼松或泼尼松龙"))
        methylprednisolone = self.contract.map_event(event("甲泼尼龙"))

        self.assertEqual([item.canonical_concept for item in prednisone], ["全身糖皮质激素"])
        self.assertEqual(
            [item.canonical_concept for item in methylprednisolone],
            ["全身糖皮质激素"],
        )
        self.assertEqual(prednisone[0].feature_layer, "care_process")
        self.assertEqual(prednisone[0].value_type, "binary_exposure")

    def test_combination_inhaler_can_emit_two_clinical_treatment_classes(self) -> None:
        mapped = self.contract.map_event(event("ICS/LABA复方"))

        self.assertEqual(
            {item.canonical_concept for item in mapped},
            {"支气管扩张剂", "吸入糖皮质激素"},
        )

    def test_uncurated_encoder_drugs_are_excluded_by_default(self) -> None:
        mapped = self.contract.map_event(
            event("药物名称/制剂代码嵌入", model_scope="main_model_encoder")
        )

        self.assertEqual(mapped, ())

    def test_invalid_keep_value_fails_contract_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.csv"
            path.write_text(
                "rule_id,source_domain,source_concept,model_scope,canonical_concept,"
                "feature_layer,value_type,keep\n"
                "BAD,药物,甲泼尼龙,main_model,全身糖皮质激素,"
                "care_process,binary_exposure,treu\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "keep"):
                ClinicalFeatureContract.from_csv(path)


class DiagnosisV2ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = ClinicalFeatureContract.from_csv(DIAGNOSIS_V2_CONTRACT_PATH)

    def test_chronic_comorbidities_are_grouped_without_renaming_as_prior_history(self) -> None:
        asthma = self.contract.map_event(event("哮喘", event_domain="诊断"))
        bronchiectasis = self.contract.map_event(event("支气管扩张症", event_domain="诊断"))
        self.assertEqual([item.canonical_concept for item in asthma], ["其他慢性呼吸疾病"])
        self.assertEqual(
            [item.canonical_concept for item in bronchiectasis],
            ["其他慢性呼吸疾病"],
        )

    def test_acute_diagnoses_are_separate_from_chronic_comorbidities(self) -> None:
        pneumonia = self.contract.map_event(event("肺炎", event_domain="诊断"))
        sepsis = self.contract.map_event(event("脓毒症", event_domain="诊断"))
        heart_failure = self.contract.map_event(event("充血性心力衰竭", event_domain="诊断"))
        self.assertEqual(pneumonia[0].canonical_concept, "本次住院肺部感染或误吸")
        self.assertEqual(sepsis[0].canonical_concept, "本次住院脓毒症")
        self.assertEqual(heart_failure[0].canonical_concept, "心力衰竭")

    def test_mixed_or_redundant_diagnoses_are_excluded(self) -> None:
        for concept in ("肺栓塞", "静脉血栓栓塞", "肺气肿", "贫血"):
            with self.subTest(concept=concept):
                self.assertEqual(
                    self.contract.map_event(event(concept, event_domain="诊断")),
                    (),
                )

    def test_non_diagnosis_treatment_mapping_is_unchanged(self) -> None:
        mapped = self.contract.map_event(event("甲泼尼龙"))
        self.assertEqual([item.canonical_concept for item in mapped], ["全身糖皮质激素"])


if __name__ == "__main__":
    unittest.main()
