"""严格结局重建：不改变转院参照，不把其他住院错误地当作随访结束。"""
from __future__ import annotations

import unittest

from training.v5.strict_endpoint_labels import diagnosis_key, reconstruct


def admission(hadm: str, start: str, end: str, *, strict: bool = False,
              position: int = 1, location: str = "EMERGENCY ROOM") -> dict[str, str]:
    return {"subject_id": "1", "hadm_id": hadm, "admittime": start, "dischtime": end,
            "is_strict_aecopd": "t" if strict else "f", "best_copd_seq_num": str(position),
            "admission_location": location, "discharge_landmark_eligible": "t"}


class StrictEndpointTests(unittest.TestCase):
    def test_fixed_width_code_spaces_dots_and_case(self) -> None:
        self.assertEqual(diagnosis_key({"icd_version": "9", "icd_code": "49121  "}), (9, "49121"))
        self.assertEqual(diagnosis_key({"icd_version": "10", "icd_code": " j44.1  "}), (10, "J441"))
        self.assertNotEqual(diagnosis_key({"icd_version": "9", "icd_code": "491.22 "}), (9, "49121"))

    def test_skip_other_copd_and_use_later_strict_at_any_position(self) -> None:
        cohort = [admission("1", "2020-01-01", "2020-01-03"),
                  admission("2", "2020-01-10", "2020-01-12"),
                  admission("3", "2020-02-01", "2020-02-03", strict=True, position=8)]
        patients = {"1": {"dod": "\\N"}}
        broad = reconstruct(cohort, patients, strict_only=False)
        strict = reconstruct(cohort, patients, strict_only=True)
        self.assertEqual(broad["1"]["time_days"], 7)
        self.assertEqual(strict["1"]["time_days"], 29)
        self.assertEqual(strict["1"]["next_copd_hadm_id"], 3)

    def test_transfer_after_nonstrict_is_still_excluded(self) -> None:
        cohort = [admission("1", "2020-01-01", "2020-01-03"),
                  admission("2", "2020-01-10", "2020-01-12"),
                  admission("3", "2020-01-13", "2020-01-15", strict=True, location="TRANSFER FROM HOSPITAL")]
        strict = reconstruct(cohort, {"1": {"dod": "\\N"}}, strict_only=True)
        self.assertEqual(strict["1"]["event_type"], 0)

    def test_death_between_other_copd_and_strict_becomes_competing_event(self) -> None:
        cohort = [admission("1", "2020-01-01", "2020-01-03"),
                  admission("2", "2020-01-10", "2020-01-12"),
                  admission("3", "2020-02-01", "2020-02-03", strict=True)]
        strict = reconstruct(cohort, {"1": {"dod": "2020-01-20"}}, strict_only=True)
        self.assertEqual(strict["1"]["event_type"], 2)
        self.assertEqual(strict["1"]["time_days"], 17)

    def test_exact_death_day_tie_keeps_original_event_priority(self) -> None:
        cohort = [admission("1", "2020-01-01", "2020-01-03"),
                  admission("2", "2020-01-10", "2020-01-12", strict=True)]
        strict = reconstruct(cohort, {"1": {"dod": "2020-01-10"}}, strict_only=True)
        self.assertEqual(strict["1"]["event_type"], 1)


if __name__ == "__main__":
    unittest.main()
