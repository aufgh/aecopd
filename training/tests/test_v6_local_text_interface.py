import unittest
import numpy as np
import json
from patient_data.local_text_interface import overlaps,token_span,weighted_vector


class Tests(unittest.TestCase):
    def test_actual_sentence_crosses_blocks(self):
        self.assertEqual(overlaps(3,9,[(4,0,5),(5,5,10)]),[(4,2),(5,4)])

    def test_saved_index_mapping_JSON_roundtrip(self):
        # 真实落盘索引不能保留tuple，使独立源重建与JSON list永远不同。
        from patient_data.local_text_interface import index_mapping
        record={'containing_rows_and_tokens':index_mapping([(4,2),(5,4)])}
        self.assertEqual(record,json.loads(json.dumps(record)))

    def test_missing_and_duplicate_tokens_rejected(self):
        for blocks in [[(1,0,4),(2,5,10)],[(1,0,6),(2,5,10)]]:
            with self.assertRaises(ValueError):overlaps(3,9,blocks)

    def test_sentence_offset_range(self):
        self.assertEqual(token_span([(0,3),(4,6),(7,10)],4,10),(1,3))
        with self.assertRaises(ValueError):token_span([(0,3)],4,5)

    def test_weighted_view_order_invariant_and_unit_norm(self):
        a=weighted_vector([[1,0],[0,1]],[1,3]);b=weighted_vector([[0,1],[1,0]],[3,1])
        np.testing.assert_allclose(a,b);self.assertAlmostEqual(float(np.linalg.norm(a)),1,places=6)
        self.assertAlmostEqual(float(a[1]/a[0]),3,places=6)

    def test_invalid_vector_rejected(self):
        with self.assertRaises(ValueError):weighted_vector([[0,0]],[1])
        with self.assertRaises(ValueError):weighted_vector([[1,0]],[0])


if __name__=='__main__':unittest.main()
