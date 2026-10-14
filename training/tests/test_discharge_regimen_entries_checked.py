import unittest
from patient_data.discharge_regimen_entries_checked import read_entry


class CanonicalEntryTests(unittest.TestCase):
    def test_salt_and_form_do_not_bypass(self):
        for text in ('Fluticasone propionate 50 mcg daily','Budesonide EC 3 mg daily','Mometasone furoate 100 mcg daily'):
            self.assertEqual(read_entry(text)['status'],'unresolved')

    def test_corrected_combo_priority(self):
        r=read_entry('Anoro Ellipta (umeclidinium-vilanterol) 62.5-25 mcg inhalation daily')
        self.assertEqual(r['canonical_identity'],'umeclidinium_vilanterol')

    def test_original_action_is_retained(self):
        self.assertEqual(read_entry('Held- Spiriva 18 mcg IH daily')['prefix_action'],'held')
        self.assertEqual(read_entry('Discontinued- Spiriva 18 mcg IH daily')['prefix_action'],'discontinued')

    def test_explicit_ih_and_brand_remain_eligible(self):
        for text in ('Budesonide 0.5 mg IH BID','Pulmicort 0.5 mg daily','Flovent HFA 110 mcg BID'):
            self.assertEqual(read_entry(text)['status'],'listed')


if __name__=='__main__':unittest.main()
