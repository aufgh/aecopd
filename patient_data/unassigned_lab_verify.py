"""一次独立原源重建及异常表示合成反例；不拟合、不修改既有派生。"""
import gzip
import json
from pathlib import Path
import numpy as np
from patient_data.unassigned_lab_audit import ROOT,derive
from patient_data.lab_recovery_features import abnormal_severity
from patient_data.v6_lab_state import summarize
from patient_data.export_mimiciv import sha256

def representation_examples():
    pairs=(('arterial_low_ph','units',7.38,7.42,7.35,7.45),
           ('arterial_high_pco2','mm Hg',35.,44.,35.,45.),
           ('hemoglobin_low','g/dL',13.,15.,12.,16.),
           ('albumin_low','g/dL',3.6,4.7,3.5,5.),
           ('creatinine_high','mg/dL',.7,1.1,.5,1.2))
    output=[]
    for axis,unit,left,right,lower,upper in pairs:
        a=abnormal_severity(axis,left,lower,upper,severity_cap=None)
        b=abnormal_severity(axis,right,lower,upper,severity_cap=None)
        x,_=summarize([(.25,a),(2.,a)],3.)
        y,_=summarize([(.25,b),(2.,b)],3.)
        np.testing.assert_array_equal(x,y)
        output.append(dict(axis=axis,unit=unit,illustrative_values=[left,right],
            illustrative_reference=[lower,upper],severity_values=[a,b],nine_raw_channels_equal=True))
    return dict(status='synthetic_encoding_noninjectivity_not_patient_outcome_evidence',examples=output,
        measured_patient_coverage=False,new_fits=0,
        limits='same severity-axis input does not imply complete H34/labs/text patient inputs identical')

def run():
    if (ROOT/'verification.json').exists():raise FileExistsError('already independently verified')
    old=json.loads((ROOT/'audit.json').read_text());new,records=derive()
    seconds=new.pop('actual_seconds');old.pop('actual_seconds')
    evidence_hash=old.pop('local_evidence_sha256')
    if evidence_hash!=sha256(ROOT/'local_evidence.jsonl.gz') or old!=new:raise ValueError('source aggregation differs')
    with gzip.open(ROOT/'local_evidence.jsonl.gz','rt',encoding='utf-8') as f:
        expected=[json.loads(line) for line in f]
    if expected!=records:raise ValueError('source binding/rejection record differs')
    proof=dict(status='one_independent_full_source_and_binding_replay',maximum_error=0.,
        rebuilt_source_rows=new['assigned_source_rows_scanned']+new['total_unassigned_rows'],
        local_records_replayed=len(records),actual_rebuild_seconds=seconds,new_fits=0,
        audit_sha256=sha256(ROOT/'audit.json'),verification_code_sha256=sha256(Path(__file__)))
    (ROOT/'verification.json').write_text(json.dumps(proof,indent=2)+'\n')
    example=representation_examples()
    example['code_sha256']={p:sha256(Path(p)) for p in ('patient_data/lab_recovery_features.py','patient_data/v6_lab_state.py')}
    (ROOT/'representation_examples.json').write_text(json.dumps(example,indent=2)+'\n')
    (ROOT/'state.json').write_text(json.dumps(dict(status='source_audit_independently_verified_no_binding',
        new_fits=0,goal_achieved=False,next_stage=new['next_stage']))+'\n')
    print(json.dumps(proof),flush=True)

if __name__=='__main__':run()
