import unittest
from pathlib import Path
from patient_data.discharge_regimen_entries import read_entry,read_unit,masked


class RegimenEntryTests(unittest.TestCase):
    def test_list_combination(self):
        r=read_entry('4. Advair Diskus 250-50 mcg/dose Sig: one puff inhalation BID')
        self.assertEqual(r['status'],'listed');self.assertEqual(r['classes'],['inhaled_steroid','laba'])
        self.assertEqual(r['product_name'],'Advair Diskus')

    def test_explicit_ih_budesonide(self):
        r=read_entry('1. Budesonide 0.5 mg IH BID')
        self.assertEqual(r['classes'],['inhaled_steroid']);self.assertEqual(r['status'],'listed')
        self.assertEqual(r['route_spans'][0][2],'IH')

    def test_direct_stopped(self):
        r=read_entry('2. Discontinued- Spiriva Respimat 2.5 mcg inhalation daily')
        self.assertEqual(r['status'],'explicit_stop');self.assertEqual(r['classes'],['lama'])

    def test_prn_does_not_stop(self):
        r=read_entry('3. Tiotropium Bromide 18 mcg IH daily:PRN')
        self.assertEqual(r['status'],'listed');self.assertTrue(r['as_needed'])

    def test_conditional_unknown(self):
        self.assertEqual(read_entry('Hold Advair Diskus 250 mcg if instructed')['status'],'conditional_unknown')

    def test_do_not_stop_not_stopped(self):
        self.assertNotEqual(read_entry('Do not stop Advair Diskus inhalation BID')['status'],'explicit_stop')

    def test_nasal_or_oral(self):
        for s in ('Fluticasone 50 mcg nasal daily','Budesonide 3 mg PO daily','Glycopyrrolate tablet PO daily'):
            self.assertEqual(read_entry(s)['classes'],[])

    def test_bare_forms_unknown(self):
        for s in ('Budesonide 3 mg daily','Fluticasone 50 mcg daily','Glycopyrrolate 1 mg daily'):
            self.assertEqual(read_entry(s)['status'],'unresolved')

    def test_multiple_header_not_combo(self):
        r=read_entry('Advair and Spiriva inhalation daily')
        self.assertEqual(r['status'],'unresolved')

    def test_other_po_after_semicolon_not_applied(self):
        r=read_entry('Advair Diskus 250 mcg IH BID; Prednisone 20 mg PO daily')
        self.assertEqual(r['classes'],['inhaled_steroid','laba']);self.assertEqual(r['status'],'listed')

    def test_other_stop_after_semicolon_not_applied(self):
        self.assertEqual(read_entry('Advair Diskus 250 mcg IH BID; stop prednisone')['status'],'listed')

    def test_numbered_entries_separate_route(self):
        rr=read_unit('1. Advair Diskus 250 mcg IH BID\n2. Budesonide 3 mg PO daily')
        self.assertEqual(len(rr),2);self.assertEqual(rr[0]['status'],'listed');self.assertEqual(rr[1]['classes'],[])

    def test_name_and_route_offsets(self):
        s=' 2. Tiotropium Bromide 18 mcg IH daily';r=read_entry(s)
        self.assertEqual(s[slice(*r['product_span'])],r['product_name'])
        for a,b,t in r['route_spans']:self.assertEqual(s[a:b],t)

    def test_no_dose_total_or_prescribed_claim(self):
        r=read_entry('Spiriva 18 mcg IH daily')
        self.assertFalse(r['clinically_verified']);self.assertNotIn('prescribed',r);self.assertNotIn('total_dose',r)

    def test_mask_hides_other_words_and_digits(self):
        s=masked('PrivateName 2040-01-01 Advair IH')
        self.assertNotIn('PrivateName',s);self.assertNotIn('2040',s);self.assertIn('advair',s)

    def test_absolute_parent_relative_child_hash(self):
        from patient_data.discharge_regimen_entry_audit import inherited_hash
        p=Path('patient_data/discharge_regimen_audit.py')
        self.assertEqual(inherited_hash({str(p.resolve()):'frozen'},p),'frozen')

    def test_missing_or_ambiguous_hash_key(self):
        from patient_data.discharge_regimen_entry_audit import inherited_hash
        p=Path('patient_data/discharge_regimen_audit.py')
        for mapping in ({},{str(p):'a',str(p.resolve()):'b'}):
            with self.assertRaises(ValueError):inherited_hash(mapping,p)


if __name__=='__main__':unittest.main()
