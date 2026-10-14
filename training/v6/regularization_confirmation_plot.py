"""已独立核验的锁定确认差值图；不再选择/改写名单。"""
import json
from pathlib import Path
from patient_data.export_mimiciv import sha256
from training.v6.regularization_search import ROOT


def run():
    r=json.loads((ROOT/'confirmation_report.json').read_text())
    proof=json.loads((ROOT/'confirmation_verification.json').read_text())
    if proof['confirmation_report_sha256']!=sha256(ROOT/'confirmation_report.json') or proof['maximum_error']>3e-7:
        raise ValueError('verified confirmation required')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=r['lock']['nominated']
    labels={'v6_shrink10':'Correction penalty 0.1','v6_shrink01':'Correction penalty 0.01','v6_wd10':'Weight decay 0.1'}
    fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True)
    for ax,key,ci,title in zip(axes,('integrated_brier_difference','auc90_difference'),
                              ('integrated_brier_95ci','auc90_95ci'),
                              ('IBS7-365 difference (negative better)','AUC90 difference (positive better)')):
        for j,name in enumerate(names):
            for ref,offset,color,marker in (('v6_control',-.1,'C0','o'),('official_V6',.1,'C1','D')):
                p=r['comparisons'][f'{name} minus {ref}'];low,high=p[ci]
                ax.hlines(j+offset,low,high,color=color,lw=2)
                ax.scatter(p[key],j+offset,color=color,marker=marker,
                           label='Versus matched V6' if j==0 and ref=='v6_control' else 'Versus official V6' if j==0 else None)
        ax.axvline(0,color='black',lw=1);ax.grid(axis='x',alpha=.25);ax.set_xlabel(title)
    axes[0].set_yticks(range(len(names)),[labels[n] for n in names]);axes[0].invert_yaxis()
    fig.suptitle('Locked regularization confirmation; seed 42; 1,418 discharges / 395 patients\n'
                 'Patient-cluster bootstrap 200; exploratory 95% intervals; no multiplicity adjustment')
    handles,legend_labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc='upper center',bbox_to_anchor=(.5,.86),ncol=2,fontsize=9)
    fig.tight_layout(rect=(0,0,1,.81));target=ROOT/'confirmation_increment.png';fig.savefig(target,dpi=180);plt.close(fig)
    print(json.dumps(dict(status='verified_confirmation_figure_saved',path=str(target))),flush=True)


if __name__=='__main__':run()
