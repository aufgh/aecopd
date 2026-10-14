"""药物商品名、复方、局部制剂与不确定剂型的分类边界。"""

import unittest

from training.v5.clinical_concepts import medication_classes


class MedicationMappingScreeningTest(unittest.TestCase):
    def test_inhaled_brands_and_combinations(self):
        expected = {
            "Symbicort": {"laba", "inhaled_steroid"},
            "Trelegy Ellipta": {"laba", "lama", "inhaled_steroid"},
            "Spiriva Respimat": {"lama"},
            "Anoro Ellipta": {"laba", "lama"},
            "DuoNeb": {"saba", "sama"},
        }
        for name, classes in expected.items():
            with self.subTest(name=name):
                self.assertEqual(medication_classes(name), classes)

    def test_local_formulations_not_systemic_or_inhaled(self):
        for name in ("Ciprofloxacin 0.3% Ophth Soln", "Erythromycin 0.5% Ophth Oint",
                     "PrednisoLONE Acetate 1% Ophth. Susp.", "Hydrocortisone Oint 2.5%",
                     "Hydrocortisone Enema", "Hydrocortisone Acetate Suppository", "Vancomycin Enema",
                     "Clindamycin 1% Solution", "prednisolone-moxiflo-nepafenac"):
            with self.subTest(name=name):
                self.assertEqual(medication_classes(name), set())

    def test_antibiotic_word_does_not_trigger_otic_local_filter(self):
        self.assertEqual(medication_classes("Ceftriaxone", product_descriptions=("Antibiotic vial",)),
                         {"beta_lactam"})

    def test_tetracyclines_are_separate_only_in_revised_mapping(self):
        self.assertEqual(medication_classes("Doxycycline Hyclate"), {"other_antibiotic"})
        self.assertEqual(medication_classes("Doxycycline Hyclate", mapping_version="gold_review_v2"),
                         {"tetracycline"})
        self.assertEqual(medication_classes("Minocycline", mapping_version="gold_review_v2"),
                         {"tetracycline"})
        self.assertEqual(medication_classes("Vancomycin", mapping_version="gold_review_v2"),
                         {"other_antibiotic"})
        self.assertEqual(medication_classes("Doxycycline Ophth Solution", mapping_version="gold_review_v2"),
                         set())

    def test_clinically_reviewed_mapping_excludes_nontherapeutic_records(self):
        version = "clinical_reviewed_2026_09"
        self.assertEqual(medication_classes("CefTAZidime-Heparin Lock", mapping_version=version), set())
        self.assertEqual(medication_classes("Cefepime Graded Challenge", mapping_version=version), set())
        self.assertEqual(medication_classes("Hydrocortisone study drug", mapping_version=version), set())
        self.assertEqual(medication_classes("mometasone", mapping_version=version), set())
        self.assertEqual(medication_classes("fluticasone-umeclidin-vilanter", mapping_version=version),
                         {"inhaled_steroid", "lama", "laba"})

    def test_plain_budesonide_needs_formulation(self):
        self.assertEqual(medication_classes("Budesonide"), set())
        self.assertEqual(medication_classes("Budesonide", product_descriptions=("Budesonide Neb Solution",)),
                         {"inhaled_steroid"})
        self.assertEqual(medication_classes("Budesonide", product_descriptions=("Budesonide 3mg Capsule",)),
                         set())

    def test_glycopyrrolate_not_assumed_lama(self):
        self.assertEqual(medication_classes("Glycopyrrolate"), set())
        self.assertEqual(medication_classes("Glycopyrrolate (CVICU Reversal Protocol)"), set())
        self.assertEqual(medication_classes("Glycopyrrolate", routes=("PO",)), set())


if __name__ == "__main__":
    unittest.main()
