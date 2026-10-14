"""经独立重放后生成英文方向图；非训练性能/验证AUC图。"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from patient_data.export_mimiciv import sha256
from training.v6.train import write_json


def run():
    root=Path('training_runs/v6_objective_direction_2026-10-13')
    network=Path('training_runs/v6_objective_network_2026-10-13')
    for folder in (root,network):
        proof=json.loads((folder/'independent_verification.json').read_text())
        if proof['report_sha256']!=sha256(folder/'diagnostic.json') or proof['maximum_error']>1e-10:
            raise ValueError('independent diagnostic proof required')
    head=json.loads((root/'diagnostic.json').read_text());full=json.loads((network/'diagnostic.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(14,8.4),layout='constrained')
    pairs=[('nll','readmission'),('nll','death'),('nll','joint'),('readmission','death')]
    for row,seed in enumerate(('42','7')):
        scopes=full['models'][seed]['scopes']
        order=['all_clinical','encoder','risk_head','last_projection']
        values=np.array([[scopes[s]['cosines'][a][b] for a,b in pairs] for s in order])
        ax=axes[row,0];im=ax.imshow(values,vmin=-1,vmax=1,cmap='RdBu',aspect='auto')
        ax.set_xticks(range(4),['NLL vs\nreadmission','NLL vs\ndeath','NLL vs\njoint score','Readmission\nvs death'])
        ax.set_yticks(range(4),['All clinical parameters','Encoder','Risk correction head','Final projection'])
        for i in range(4):
            for j in range(4):ax.text(j,i,f'{values[i,j]:.3f}',ha='center',va='center',color='white' if abs(values[i,j])>.7 else 'black')
        ax.set_title(f'Source {seed}: gradient cosine (not AUC)')
        fig.colorbar(im,ax=ax,shrink=.8,label='-1 opposite | 0 orthogonal | +1 aligned')
        ax=axes[row,1];names=['nll','readmission','death','joint'];x=np.arange(4)
        delta=head['models'][seed]['local_descent']
        for offset,key,label,color in [(-.18,'readmission','Readmission IBS','#1f77b4'),(.18,'death','Death IBS','#ff7f0e')]:
            ax.bar(x+offset,[delta[n][1]['changes'][key]*1e5 for n in names],width=.36,label=label,color=color)
        ax.axhline(0,color='black',lw=.8);ax.set_xticks(x,['NLL','Readmission IBS','Death IBS','Joint score'])
        ax.set_ylabel('Training IBS change (x 1e-5; lower is better)')
        ax.set_xlabel('One local descent direction; matched raw-output RMS = 0.01')
        ax.set_title(f'Source {seed}: final-projection local step');ax.legend()
    fig.suptitle('Fixed original V6 states; all 13,592 training discharges; no optimization or validation selection',fontsize=13)
    fig.savefig(root/'objective_directions.png',dpi=160);plt.close(fig)
    write_json(root/'state.json',dict(status='diagnostic_complete_and_independently_replayed',new_model_fits=0,
        training_records=head['admissions'],teachers=2,goal_achieved=False,
        report_sha256=sha256(root/'diagnostic.json'),network_report_sha256=sha256(network/'diagnostic.json'),
        figure_sha256=sha256(root/'objective_directions.png')))
    print('Saved verified gradient/local-direction figure; no new performance result.')


if __name__=='__main__':run()
