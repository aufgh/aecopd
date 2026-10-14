"""阶段性原值覆盖与已核验训练图，不重建来源或训练。"""
import json
import numpy as np
from patient_data.export_mimiciv import sha256
from training.v6.core_lab_continuous import ROOT,AUDIT,AXES,ARMS

def run():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    proof=json.loads((ROOT/'verification.json').read_text())
    if proof['maximum_error'] or proof['screening_sha256']!=sha256(ROOT/'screening.json'):
        raise ValueError('risk proof changed')
    audit=json.loads((AUDIT/'audit.json').read_text());stats=audit['axis_statistics']
    fig,ax=plt.subplots(figsize=(10,4.6));x=np.arange(5)
    for j,(key,label,color) in enumerate((('coverage','Valid observations','#3274a1'),
        ('all_zero_direction_admissions','Entire stay: zero directional severity','#999999'),
        ('all_zero_with_original_value_variation','Zero severity but physical values vary','#e1812c'))):
        ax.bar(x+(j-1)*.25,[stats[a][key] for a in AXES],width=.25,label=label,color=color)
    ax.set_xticks(x,['Arterial pH','Arterial pCO2','Hemoglobin','Albumin','Creatinine'])
    ax.set_ylabel('Training discharges');ax.set_ylim(0,15000)
    ax.set_title('Core lab representation audit: 13,592 training discharges\nZero directional severity is not globally normal clinical status')
    ax.legend(loc='upper left',fontsize=9);ax.grid(axis='y',alpha=.2);fig.tight_layout()
    fig.savefig(ROOT/'value_coverage.png',dpi=180);plt.close(fig)
    r=json.loads((ROOT/'screening.json').read_text());fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    for arm in ARMS:
        history=json.loads((ROOT/arm/'history.json').read_text());tr=[z for z in history if z['train_nll'] is not None]
        axes[0].plot([z['epoch'] for z in tr],[z['train_nll'] for z in tr],label=arm)
        axes[1].plot([z['epoch'] for z in history],[z['tuning_ibs'] for z in history],label=arm)
    axes[0].set_title('Training joint NLL (lower better)')
    axes[1].set_title('Tuning readmission IBS7-365 (lower better)')
    axes[1].axhline(r['reference']['readmission']['integrated_brier_7_365'],ls='--',color='gray',label='Original V6')
    for ax in axes:ax.set_xlabel('Epoch');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Core physical values added to frozen V6; seed42\nBoth selected epoch0: no selected daily-risk gain')
    fig.tight_layout();fig.savefig(ROOT/'training.png',dpi=180);plt.close(fig)

if __name__=='__main__':run()
