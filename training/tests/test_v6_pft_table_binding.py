import unittest
from patient_data.pft_table_binding import header_roles,row_values,find_header,line_ranges


class Tests(unittest.TestCase):
    def test_actual_predicted_percent_header(self):
        self.assertEqual(header_roles('Actual Pred % Pred'),('actual','predicted','percent_predicted'))

    def test_explicit_permuted_columns(self):
        self.assertEqual(header_roles('Predicted Actual %Pred'),('predicted','actual','percent_predicted'))

    def test_narrative_and_change_and_duplicate_headers_rejected(self):
        for text in ('Actual was predicted to improve','Pre Post %Change','Actual Actual %Pred','Actual Pred %Pred 2112'):
            self.assertIsNone(header_roles(text))

    def test_three_cells_with_unit(self):
        r=row_values('FEV1 (L) 1.2 3.0 40')
        self.assertEqual([c['value'] for c in r['cells']],[1.2,3,40]);self.assertEqual(r['unit_literal'],'L')

    def test_no_default_volume_unit(self):
        self.assertIsNone(row_values('FEV1 1.2 3.0 40')['unit_literal'])

    def test_percent_marker_is_preserved_and_ratio_volume_rejected(self):
        self.assertTrue(row_values('FEV1 1.2 3.0 40%')['trailing_percent'])
        self.assertIsNone(row_values('FEV1/FVC (L) .6 .8 75'))

    def test_raw_ratio_not_predicted_percent(self):
        self.assertEqual(row_values('FEV1/FVC .6 .8 75')['metric'],'FEV1/FVC')

    def test_ranges_bounds_extra_marker_not_scalar_row(self):
        for row in ('FEV1 <1.2 3.0 40','FEV1 1.2-1.4 3.0 40','FEV1 1.2 3.0 40 +','FEV1 1.2 3.0 40 1.5'):
            self.assertIsNone(row_values(row))

    def test_no_header_cross_narrative(self):
        self.assertIsNone(find_header(['Actual Pred %Pred','The patient improved','FEV1 1.2 3.0 40'],2))

    def test_contiguous_rows_and_blank_lines_allowed(self):
        self.assertEqual(find_header(['Actual Pred %Pred','','FVC 2.4 4.0 60','FEV1 1.2 3.0 40'],3)[0],0)

    def test_incomplete_row_or_other_table_blocks_header(self):
        self.assertIsNone(find_header(['Actual Pred %Pred','FVC missing','FEV1 1.2 3.0 40'],2))

    def test_twelve_line_limit(self):
        self.assertIsNone(find_header(['Actual Pred %Pred']+['']*12+['FEV1 1.2 3.0 40'],13))

    def test_line_original_offsets(self):
        text='head\nFEV1 1.2 3.0 40\r\nFVC 2.4 4.0 60'
        ranges=line_ranges(text,5,len(text))
        for a,z,line in ranges:self.assertEqual(text[a:a+len(line)],line)
        self.assertEqual(ranges[-1][1],len(text))


if __name__=='__main__':unittest.main()
