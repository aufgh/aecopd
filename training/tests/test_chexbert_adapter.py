"""无需真实患者或检查点的token/四状态接口回归。"""
import unittest
import numpy as np
from transformers import BertTokenizer
from training.v6.chexbert_inference import encode_ids, one_hot_states


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer=BertTokenizer.from_pretrained('training/model_cache/CheXbert_stanford_2026-10-09/bert-base-uncased',local_files_only=True)

    def test_official_token_contract_empty_normalized_and_truncated(self):
        t=self.tokenizer
        ids,_=encode_ids('  No\n evidence   of pneumonia. ',t)
        # transformers5已移除旧encode_plus；与公开BERT词表的固定token及标准接口核对。
        expected=[101,2053,3350,1997,18583,1012,102]
        self.assertEqual(t('No evidence of pneumonia.')['input_ids'],expected)
        self.assertEqual(ids,expected)
        self.assertEqual(encode_ids('',t)[0],[101,102])
        ids,n=encode_ids('pneumonia '*800,t)
        self.assertEqual((len(ids),n,ids[-1]),(512,802,102))

    def test_four_states_not_ordinal_and_no_report_not_blank(self):
        codes=np.array([[0]*7,[1]*7,[2]*7,[3]*7,[0]*7],np.int8)
        result=one_hot_states(codes,np.array([1,1,1,1,0],bool)).reshape(5,7,4)
        for i in range(4): np.testing.assert_array_equal(result[i,:,i],np.ones(7))
        self.assertEqual(result[-1].sum(),0)
        self.assertFalse(np.array_equal(result[0],result[-1]))

    def test_invalid_class_rejected(self):
        with self.assertRaises(ValueError): one_hot_states(np.full((1,7),4),np.ones(1,bool))
        with self.assertRaises(ValueError): one_hot_states(np.zeros((1,7)),np.ones(1,float))


if __name__=='__main__': unittest.main()
