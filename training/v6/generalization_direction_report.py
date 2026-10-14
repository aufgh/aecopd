"""已独立重放的训练/调参方向图；不是模型性能或特征重要性图。"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from patient_data.export_mimiciv import sha256
from training.v6.generalization_direction import ROOT,NAMES
from training.v6.train import write_json


def heatmap(ax,values,rows,columns,title):
    data=np.array([[np.nan if v is None else v for v in r] for r in values],float)
    palette=plt.get_cmap('RdBu').copy();palette.set_bad('#dddddd')
    im=ax.imshow(np.ma.masked_invalid(data),vmin=-1,vmax=1,cmap=palette,aspect='auto')
    ax.set_xticks(range(len(columns)),columns);ax.set_yticks(range(len(rows)),rows)
    ax.set_title(title)
    for i in range(len(rows)):
        for j in range(len(columns)):
            value=data[i,j]
            ax.text(j,i,'N/A' if not np.isfinite(value) else f'{value:.3f}',ha='center',va='center',
                    color='white' if np.isfinite(value) and abs(value)>.7 else 'black',fontsize=9)
    return im


def run():
    proof=json.loads((ROOT/'independent_verification.json').read_text())
    if (proof['report_sha256']!=sha256(ROOT/'diagnostic.json') or proof['maximum_gradient_error']>1e-10
            or proof['maximum_report_error']>1e-10):raise ValueError('fresh independent proof required')
    report=json.loads((ROOT/'diagnostic.json').read_text())
    order=['all_clinical','encoder','risk_head','last_projection','labs','additional_labs','medications',
           'microbiology','support','text','fusion','inactive_legacy_current','inactive_fine_medications']
    labels=['All clinical parameters','Encoder','Risk correction head','Final projection','Original laboratory axes',
            'Additional laboratories','Medication classes','Microbiology','Respiratory support','BERT text branch',
            'Fusion','Legacy current (masked; in H34)','Fine medications (zero input)']
    columns=['NLL','Readmission\nIBS','Death\nIBS','Joint\nscore']
    fig,axes=plt.subplots(1,2,figsize=(14,9),layout='constrained')
    for j,seed in enumerate(('42','7')):
        scopes=report['models'][seed]['scopes']
        values=[[scopes[s]['same_objective'][name]['cosine'] for name in NAMES] for s in order]
        im=heatmap(axes[j],values,labels if j==0 else ['']*len(order),columns,f'Source {seed}: same-objective train vs tuning')
    fig.colorbar(im,ax=axes,shrink=.75,label='Gradient cosine: -1 opposing | 0 orthogonal | +1 aligned')
    fig.suptitle('Fixed original V6 states; fold-mean gradients; no parameter updates; not AUC or feature importance')
    fig.savefig(ROOT/'branch_directions.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for j,seed in enumerate(('42','7')):
        cross=report['models'][seed]['scopes']['all_clinical']['cross_objective_cosines']
        im=heatmap(axes[j],[[cross[a][b] for b in NAMES] for a in NAMES],columns,columns,f'Source {seed}: all clinical parameters')
        axes[j].set_xlabel('Tuning objective');axes[j].set_ylabel('Training objective')
    fig.colorbar(im,ax=axes,shrink=.8,label='Train-gradient vs tuning-gradient cosine')
    fig.suptitle('Cross-objective local generalization directions; previously exposed tuning patients; not a new performance test')
    fig.savefig(ROOT/'cross_objective_directions.png',dpi=160);plt.close(fig)
    write_json(ROOT/'figure_verification.json',dict(status='generated_from_independently_verified_gradients',
        source_sha256=sha256(Path(__file__)),report_sha256=sha256(ROOT/'diagnostic.json'),goal_achieved=False))
    print('Saved independently verified direction figures; no new model performance claim.')


if __name__=='__main__':run()
