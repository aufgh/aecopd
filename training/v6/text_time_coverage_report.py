"""审计的英文覆盖图和阶段状态；不产生性能结论或训练。"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from patient_data.export_mimiciv import sha256
from patient_data.text_time_coverage import ROOT
from training.v6.train import write_json


def main():
    report=json.loads((ROOT/'audit.json').read_text())
    for tag in ('a','b'):
        proof=json.loads((ROOT/f'verification_{tag}.json').read_text())
        if proof['report_sha256']!=sha256(ROOT/'audit.json') or proof['maximum_count_or_index_error']!=0:raise ValueError('source audit not independently verified')
        enc=json.loads((ROOT/f'encoding_check_{tag}.json').read_text());att=json.loads((ROOT/f'attention_audit_{tag}.json').read_text())
        if enc['source_audit_sha256']!=sha256(ROOT/'audit.json') or att['audit_sha256']!=sha256(ROOT/'audit.json'):raise ValueError('GPU proof stale')
    if json.loads((ROOT/'encoding_check_a.json').read_text())!=json.loads((ROOT/'encoding_check_b.json').read_text()):raise ValueError('GPU encoding process differs')
    if json.loads((ROOT/'attention_audit_a.json').read_text())!=json.loads((ROOT/'attention_audit_b.json').read_text()):raise ValueError('GPU attention process differs')
    roles=('hospital_course','discharge_condition','discharge_physical_exam','discharge_instructions','transitional_issues')
    names=('Hospital course','Discharge condition','Discharge physical exam','Discharge instructions','Transitional issues')
    totals=np.array([report['by_role'][r]['priority_candidates'] for r in roles]);y=np.arange(len(roles))
    fig,ax=plt.subplots(figsize=(10,5.6))
    for offset,key,name,color in [(-.17,'B_union_full','Original B: full span retained','#1f77b4'),(.17,'encoded_candidate_source_unit','Saved candidate encoding: source retained','#2ca02c')]:
        counts=np.array([report['by_role'][r].get(key,0) for r in roles]);bars=ax.barh(y+offset,counts/totals*100,height=.3,label=name,color=color)
        for bar,n,total in zip(bars,counts,totals):ax.text(bar.get_width()+1,bar.get_y()+bar.get_height()/2,f'{n}/{total}',va='center',fontsize=9)
    ax.set_yticks(y,names);ax.invert_yaxis();ax.set_xlim(0,125);ax.set_xticks([0,25,50,75,100])
    ax.set_xlabel('Percent of lexical priority candidate units retained');ax.grid(axis='x',alpha=.2)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.17),fontsize=8,frameon=False)
    fig.suptitle('Training-only original-span audit; not semantic accuracy or predictive gain',fontsize=11)
    fig.tight_layout(rect=(0,.06,1,.94));fig.savefig(ROOT/'source_coverage.png',dpi=150);plt.close(fig)
    write_json(ROOT/'state.json',dict(status='source_token_and_frozen_attention_audit_complete_independently_verified',
        audit_sha256=sha256(ROOT/'audit.json'),verification_tags=['a','b'],unit_tests=8,new_fits=0,new_vectors=0,
        figures={'source_coverage.png':sha256(ROOT/'source_coverage.png')},report_code_sha256=sha256(Path(__file__)),
        original_model_or_input_changed=False,goal_achieved=False,
        next='Predeclare matched same-sentence local versus containing-block/availability views; no repeated extraction/full-document rerun'))
    print(json.dumps(dict(status='audit_complete',new_fits=0,new_vectors=0,priority_candidates=report['priority_candidate_units'])),flush=True)


if __name__=='__main__':main()
