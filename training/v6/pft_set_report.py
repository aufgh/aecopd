"""背景PFT集合数值/来源配对读出；不改选点/默认。"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from patient_data.export_mimiciv import sha256
from training.v6.pft_set_experiment import ROOT,SEEDS,ARMS,inputs,augmented,new_model,prediction
from training.v6.evaluation import paired_interval,metrics,public_metrics,calibration_bins
from training.v6.train import write_json


def build_report():
    saved=json.loads((ROOT/'screening.json').read_text())
    for tag in ('a','b'):
        proof=json.loads((ROOT/f'verification_{tag}.json').read_text())
        if proof['report_sha256']!=sha256(ROOT/'screening.json') or proof['maximum_error']!=0:raise ValueError('not independently verified')
    data,_,arrays,_=inputs();d,_,_,_,_,_,tune,_,_,_=data
    observed=arrays['observed'][tune];counts=np.rint(np.expm1(d.history_raw[tune,2])).astype(int)
    groups={'source_observed':observed,'source_missing':~observed,'first':counts==0,'one_two':(counts>=1)&(counts<=2),'three_plus':counts>=3}
    report=dict(pairs={},layers={},means={},calibration={},actual_epochs={},unobserved_max_error={},no_confirmation_test=True,
        screening_sha256=sha256(ROOT/'screening.json'),code_sha256=sha256(Path(__file__)))
    fig,axs=plt.subplots(1,3,figsize=(15,4.5));colors={'availability':'#1f77b4','content':'#2ca02c'}
    differences=[]
    calfig,calaxs=plt.subplots(2,2,figsize=(9,8))
    for j,seed in enumerate(SEEDS):
        cache,_=augmented(data,tune,seed,arrays);original=new_model('content',seed);_,_,ref,_=prediction(original,cache);del original,cache
        curves={'original_V6':ref};report['actual_epochs'][str(seed)]={}
        for arm in ARMS:
            history=json.loads((ROOT/f'seed{seed}'/arm/'history.json').read_text());info=saved['fits'][str(seed)][arm]
            report['actual_epochs'][str(seed)][arm]={'actual':info['completed_epochs'],'best':info['best_epoch']}
            axs[j].plot([r['epoch'] for r in history],[r['tuning_ibs'] for r in history],label=arm,color=colors[arm])
            best=info['best_epoch'];axs[j].scatter(best,history[best]['tuning_ibs'],color=colors[arm],marker='D',s=30)
            with np.load(ROOT/f'seed{seed}'/arm/'tuning_predictions.npz') as z:curves[arm]=z['curve']
        axs[j].set(title=f'Seed {seed}: actual early-stopped epochs',xlabel='Epoch',ylabel='Tuning IBS7-365 (lower better)');axs[j].grid(alpha=.2);axs[j].legend(fontsize=8)
        report['calibration'][str(seed)]={}
        for name,curve in curves.items():
            report['calibration'][str(seed)][name]={}
            for cause in (1,2):
                for day in (30,90,180,365):
                    bins=calibration_bins(d.times[tune],d.events[tune],curve[:,:,cause],day=day,cause=cause)
                    report['calibration'][str(seed)][name][f'{cause}/{day}']=bins
                    if j==0 and day in (90,365):
                        ax=calaxs[cause-1,0 if day==90 else 1]
                        ax.plot([r['predicted'] for r in bins],[r['observed'] for r in bins],marker='o',markersize=3,
                            color='#777777' if name=='original_V6' else colors[name],label=name)
        for name in ('original_V6','availability'):
            report['pairs'][f'{seed}/content minus {name}']=paired_interval(d.arrays['subject_ids'][tune],d.times[tune],d.events[tune],
                curves['content'][:,:,1],curves[name][:,:,1],replicates=200,seed=seed)
        report['unobserved_max_error'][str(seed)]=float(np.abs(curves['content'][~observed]-ref[~observed]).max())
        if report['unobserved_max_error'][str(seed)]!=0:raise ValueError('unobserved source changed original risks')
        report['layers'][str(seed)]={}
        for group,mask in groups.items():
            ix=tune[mask];values={}
            for name in ('original_V6','availability','content'):
                values[name]={'readmission':public_metrics(metrics(d.times[ix],d.events[ix],curves[name][mask,:,1])),
                    'death':public_metrics(metrics(d.times[ix],d.events[ix],curves[name][mask,:,2],cause=2))}
            report['layers'][str(seed)][group]=dict(admissions=int(mask.sum()),patients=len(np.unique(d.arrays['subject_ids'][ix])),models=values)
        p=report['pairs'][f'{seed}/content minus original_V6'];differences.append((p['integrated_brier_difference'],p['auc90_difference']))
    for arm in (*ARMS,'original_V6'):
        ms=[saved['references'][str(s)] if arm=='original_V6' else saved['fits'][str(s)][arm]['metrics'] for s in SEEDS]
        report['means'][arm]=dict(ibs=float(np.mean([m['readmission']['integrated_brier_7_365'] for m in ms])),
            auc90=float(np.mean([m['readmission']['points']['90']['auc'] for m in ms])),
            death_ibs=float(np.mean([m['death']['integrated_brier_7_365'] for m in ms])),
            timepoints={str(t):dict(auc=float(np.mean([m['readmission']['points'][str(t)]['auc'] for m in ms])),
                brier=float(np.mean([m['readmission']['points'][str(t)]['brier'] for m in ms]))) for t in (7,14,30,60,90,180,365)})
    fig.suptitle('Frozen original V6; PFT availability / numeric content; early stopping',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.92));fig.savefig(ROOT/'early_stopping.png',dpi=150);plt.close(fig)
    for i in range(2):
        for k,day in enumerate((90,365)):
            ax=calaxs[i,k];ax.plot([0,1],[0,1],'k--',linewidth=1)
            ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted cumulative probability',ylabel='Observed World-A fraction',
                   title=f'{"Readmission" if i==0 else "Death"}: day {day}');ax.grid(alpha=.2);ax.legend(fontsize=8)
    calfig.suptitle('Specified tuning set; predeclared seed 42; risk-bin descriptive calibration',fontsize=11)
    calfig.tight_layout(rect=(0,0,1,.95));calfig.savefig(ROOT/'calibration.png',dpi=150);plt.close(calfig)
    fig,axs=plt.subplots(1,2,figsize=(10,3.8));diff=np.asarray(differences)
    for k,ax in enumerate(axs):
        ax.axvline(0,color='black',linewidth=1);ax.scatter(diff[:,k],[-.12,0,.12],color=colors['content'],alpha=.6)
        ax.scatter(diff[:,k].mean(),0,marker='D',s=75,color=colors['content']);ax.set_yticks([0],['background PFT numeric content'])
        ax.set_xlabel('Paired difference across three seeds');ax.grid(alpha=.2)
    axs[0].set_title('IBS: content minus original\n(lower better)');axs[1].set_title('AUC90: content minus original\n(higher better)')
    axs[0].set_xlim(-.002,.002);axs[1].set_xlim(-.02,.02)
    axs[0].plot([],[],marker='o',linestyle='',color=colors['content'],alpha=.6,label='Clinical seed')
    axs[0].plot([],[],marker='D',linestyle='',color=colors['content'],label='Mean across three seeds')
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,fontsize=8,loc='lower center',ncol=2)
    fig.suptitle('Specified tuning set; fixed teacher; three clinical seeds / two teacher sources',fontsize=11)
    fig.tight_layout(rect=(0,.12,1,.9));fig.savefig(ROOT/'paired_differences.png',dpi=150);plt.close(fig)
    report['limits']=['tuning set exposed; selected best checkpoints','fixed-model bootstrap not inner refitting or independent cohort',
        'three clinical seeds reuse two teacher sources','fixed PFT concepts; unknown measurement dates; not medical gold',
        'World-A observed fractions not full out-of-system ascertainment; descriptive bins not prospective calibration proof']
    write_json(ROOT/'paired_readout.json',report)
    print(json.dumps(dict(means=report['means'],unobserved_max_error=report['unobserved_max_error'])),flush=True)


if __name__=='__main__':build_report()






