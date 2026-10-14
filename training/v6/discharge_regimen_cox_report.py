"""只读已完成Cox概率校准描述；无拟合/新选点。"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('training_runs/v6_discharge_regimen_cox_2026-10-14')


def main():
    report=json.loads((ROOT/'report.json').read_text())
    items=[('H34 Cox',report['h34_reference']),('H34 + source',report['results']['availability']),
           ('H34 + source + regimen',report['results']['content'])]
    fig,axes=plt.subplots(1,2,figsize=(10,4.1),layout='constrained')
    for ax,cause,label in zip(axes,('readmission','death'),('Strict readmission','Death before readmission')):
        maximum=.5 if cause=='readmission' else .35
        ax.plot([0,maximum],[0,maximum],'k--',linewidth=1,label='Ideal calibration')
        for name,result in items:
            groups=result['probability_readout'][cause]['90']['bins']
            ax.plot([g['predicted'] for g in groups],[g['observed'] for g in groups],marker='o',markersize=4,label=name)
        ax.set(xlim=(0,maximum),ylim=(0,maximum),title=label,xlabel='Mean predicted 90-day cumulative risk',ylabel='Observed World-A fraction')
        ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Exposed tuning set; partial-input Cox; descriptive bins, not clinical validation',fontsize=10)
    fig.savefig(ROOT/'calibration90.png',dpi=150);plt.close(fig)


if __name__=='__main__':main()
