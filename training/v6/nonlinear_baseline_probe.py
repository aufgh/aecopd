"""线性扩展B未达到等效证据时的小型非线性承接检查；两种子、旧H固定。"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from patient_data.export_mimiciv import sha256
from training.v6 import architecture_experiment as experiment
from training.v6.exploration_model import ExpandedBaseline


class AdditiveAmplitude(nn.Module):
    def __init__(self, history):
        super().__init__()
        self.history=nn.Linear(6,2)
        self.history.load_state_dict(history.amplitude.state_dict())
        self.current=nn.Sequential(nn.Linear(34,32),nn.GELU(),nn.Linear(32,2))
        nn.init.zeros_(self.current[-1].weight);nn.init.zeros_(self.current[-1].bias)

    def forward(self,x): return self.history(x[:,:6])+self.current(x)


class NonlinearExpandedBaseline(ExpandedBaseline):
    initialization_seed=42
    def __init__(self,history):
        nn.Module.__init__(self)
        torch.manual_seed(self.initialization_seed)
        self.raw_shape=nn.Parameter(history.raw_shape.detach().clone())
        self.amplitude=AdditiveAmplitude(history)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--verify',action='store_true');args=p.parse_args()
    NonlinearExpandedBaseline.initialization_seed=args.seed
    original=experiment.ExpandedBaseline;experiment.ExpandedBaseline=NonlinearExpandedBaseline
    try:
        if not args.verify:
            common=dict(reference=Path('training_runs/v6_history_anchored_bert_seed42_2026-10-09'),
                compact_dir=Path('patient_data/processed/v6_compact_multisource_2026-10-09'),
                combinations=Path('patient_data/processed/v6_cbc_combinations_2026-10-10'),
                candidate=Path('patient_data/processed/v6_screen_candidate_bert_2026-10-10'),
                epochs=60,batch_size=512,bootstrap=200,seed=args.seed,verify=False,baseline_run=args.output)
            experiment.run(SimpleNamespace(**common,phase='baseline',output=args.output))
            r=json.loads((args.output/'report.json').read_text())
            r['contract']['code_sha256'][str(Path(__file__))]=sha256(Path(__file__))
            r['contract']['baseline_architecture']='H affine amplitude skip + 34->32->2 zero-final-init current MLP, global two shapes; joint trained then frozen'
            r['contract']['limits'].append('Follow-up capacity probe after linear-B result, not a prespecified blind test')
            experiment.write_json(args.output/'report.json',r);experiment.write_json(args.output/'contract.json',r['contract'])
        experiment.verify(SimpleNamespace(output=args.output))
    finally: experiment.ExpandedBaseline=original


if __name__=='__main__': main()
