"""仅绘制已完成配对训练记录，不读取病历或重新拟合。"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('training_runs/v6_discharge_regimen_screen_2026-10-14')


def main():
    r=json.loads((ROOT/'screening.json').read_text());base=r['reference']['readmission']['integrated_brier_7_365']
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for arm,label in (('availability','Source availability'),('content','Regimen content')):
        rows=json.loads((ROOT/arm/'history.json').read_text())
        axes[0].plot([x['epoch'] for x in rows[1:]],[x['train_nll'] for x in rows[1:]],label=label)
        axes[1].plot([x['epoch'] for x in rows],[x['tuning_ibs']-base for x in rows],label=label)
    axes[0].set(title='Training joint NLL (lower better)',ylabel='NLL')
    axes[1].axhline(0,color='black',linewidth=.8);axes[1].set(title='Tuning readmission IBS minus original',ylabel='IBS difference (lower better)')
    for ax in axes:ax.set_xlabel('Epoch');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Seed 42; original V6 frozen; both selected epoch 0; no risk improvement',fontsize=11)
    fig.savefig(ROOT/'training.png',dpi=150);plt.close(fig)


if __name__=='__main__':main()
