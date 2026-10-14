"""仅已核验历史的训练图；不重建数据或拟合。"""
import json
from pathlib import Path
from patient_data.export_mimiciv import sha256
from training.v6.disposition_context import ROOT,ARMS

def run():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    proof=json.loads((ROOT/'verification.json').read_text())
    if proof['maximum_error'] or proof['screening_sha256']!=sha256(ROOT/'screening.json'):
        raise ValueError('proof changed')
    r=json.loads((ROOT/'screening.json').read_text());fig,axes=plt.subplots(1,2,figsize=(11,4))
    for arm in ARMS:
        history=json.loads((ROOT/arm/'history.json').read_text())
        train=[p for p in history if p['train_nll'] is not None]
        axes[0].plot([p['epoch'] for p in train],[p['train_nll'] for p in train],label=arm)
        axes[1].plot([p['epoch'] for p in history],[p['tuning_ibs'] for p in history],label=arm)
    axes[0].set_title('Training joint NLL (lower better)')
    axes[1].set_title('Tuning readmission IBS7-365 (lower better)')
    axes[1].axhline(r['reference']['readmission']['integrated_brier_7_365'],color='gray',linestyle='--',label='Original V6')
    for ax in axes:ax.set_xlabel('Epoch');ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Fine discharge disposition; frozen original V6; seed42\nBoth selected epoch0; daily selected predictions unchanged')
    fig.tight_layout();fig.savefig(ROOT/'training.png',dpi=180);plt.close(fig)

if __name__=='__main__':run()
