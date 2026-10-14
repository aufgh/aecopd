"""文本维度改变不得改变非文本初始化或空文本随机流。"""
import copy
import unittest
import torch
from training.v6.model import HistoryBaseline, ClinicalBatch
from training.v6.exploration_model import ExpandedBaseline
from training.v6.compact_context_data import CompactContextBatch
from training.v6.compact_model import CompactAnchoredModel
from training.v6.text_encoder_comparison import shared_parameters, assert_empty_equal, arm_source
from training.v6.verify_text_encoder_comparison import metric_difference


def empty_batch(width):
    lab=torch.zeros(3,22,9); lab[:,0,0]=.5; lab[:,0,6]=1
    clinical=ClinicalBatch(torch.zeros(3,14),torch.zeros(3,14,dtype=torch.bool),lab,
        torch.zeros(3,7),torch.ones(3,7,dtype=torch.bool),torch.empty(0,width),torch.empty(0,2,dtype=torch.long))
    med=torch.zeros(3,7,4); med[:,:,2]=1
    return CompactContextBatch(clinical,med,torch.zeros(3,5),torch.ones(3,5,dtype=torch.bool),
                               torch.zeros(3,3),torch.ones(3,3,dtype=torch.bool))


class EncoderComparisonTests(unittest.TestCase):
    def test_nontext_initialization_and_empty_training(self):
        torch.manual_seed(42); base=ExpandedBaseline(HistoryBaseline())
        models=[]
        for width in (768,2560):
            torch.manual_seed(7); models.append(CompactAnchoredModel(copy.deepcopy(base),width))
        a,b=(shared_parameters(m) for m in models)
        self.assertEqual(a.keys(),b.keys())
        self.assertTrue(all(torch.equal(v,b[k]) for k,v in a.items()))
        output=[]; hist=torch.zeros(3,34)
        for model,width in zip(models,(768,2560),strict=True):
            torch.manual_seed(99)
            optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4)
            losses=[]
            for _ in range(3):
                optimizer.zero_grad(); pred=model(hist,empty_batch(width))
                loss=(pred.log_amplitude+.5).square().mean()+pred.shape.square().mean()
                losses.append(float(loss.detach())); loss.backward(); optimizer.step()
            model.eval(); output.append((losses,model(hist,empty_batch(width)).log_amplitude))
            self.assertTrue(all(p.grad is None for p in model.baseline.parameters()))
        self.assertEqual(output[0][0],output[1][0]); self.assertTrue(torch.equal(output[0][1],output[1][1]))

    def test_history_equivalence_excludes_runtime(self):
        row=dict(epoch=0,train_objective=None,validation_objective=.4,validation_brier90=.1,validation_auc90=.7,validation_ap=.2,seconds=1)
        assert_empty_equal([row],[{**row,'seconds':2}])
        with self.assertRaises(ValueError): assert_empty_equal([row],[{**row,'validation_brier90':.11}])

    def test_source_routing(self):
        self.assertEqual(arm_source('bert'),'bert'); self.assertEqual(arm_source('none_768'),'bert')
        self.assertEqual(arm_source('qwen'),'qwen'); self.assertEqual(arm_source('none_2560'),'qwen')

    def test_metric_tolerance_preserves_counts_and_structure(self):
        self.assertAlmostEqual(metric_difference({'n':3,'brier':.1},{'n':3,'brier':.1+1e-9}),1e-9)
        for wrong in ({'n':4,'brier':.1},{'n':3,'brier':.101},{'n':3},{'n':3,'brier':float('nan')}):
            with self.assertRaises(ValueError): metric_difference({'n':3,'brier':.1},wrong)


if __name__=='__main__': unittest.main()
