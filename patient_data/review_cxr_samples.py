"""基于已阅读原文的少量非专家标签复核；不是临床金标准或总体准确率。"""
import argparse
import json
from pathlib import Path
from patient_data.export_mimiciv import sha256

# 先阅读r2固定原文、再给出预期，未查看CheXbert输出时确定。
# 只选含义明确的项目；不强行给“无急性异常”等泛化句填七项阴性。
SELECTION_SHA='c455efde9b0c433d3288a8220e7793c95e6d28747079a347673e4cd00504f3a9'
CASES=((3167,'Edema',1),(3167,'Cardiomegaly',1),(3167,'Pleural Effusion',2),(3167,'Pneumonia',2),
       (2073,'Edema',2),(4152,'Pneumothorax',2),
       (2097,'Pneumonia',2),(2097,'Edema',2),(2097,'Pleural Effusion',2),
       (417,'Pneumothorax',2),(3612,'Cardiomegaly',2),(3612,'Pneumothorax',2),
       (2579,'Pneumonia',2),(2579,'Edema',2),(2579,'Atelectasis',1),
       (3412,'Pleural Effusion',1),(3412,'Atelectasis',1),
       (3510,'Pleural Effusion',1),(3510,'Atelectasis',1),
       (3777,'Atelectasis',1),(3777,'Cardiomegaly',2),
       (4682,'Pneumonia',2),(4682,'Edema',2),(4682,'Cardiomegaly',1))


def review(selected,labeled,output):
    if output.exists(): raise FileExistsError(output)
    if sha256(selected/'selected_reports.jsonl')!=SELECTION_SHA: raise ValueError('review source changed')
    records=list(map(json.loads,(selected/'selected_reports.jsonl').read_text(encoding='utf-8').splitlines()))
    labels={r['note_id']:r for r in map(json.loads,(labeled/'labeled_reports.jsonl').read_text(encoding='utf-8').splitlines())}
    details=[]
    for index,condition,expected in CASES:
        original=records[index]; result=labels[original['note_id']]
        if original['section_sha256']!=result['section_sha256']: raise ValueError('review text changed')
        predicted=result['labels14'][condition]
        details.append({'source_index':index,'note_id':original['note_id'],'condition':condition,'expected':expected,
                        'predicted':predicted,'matches':predicted==expected,'source_text':original['text'],
                        'section_sha256':original['section_sha256']})
    report={'status':'limited_nonexpert_source_review_completed','reports':len({c[0] for c in CASES}),
            'clear_attribute_checks':len(details),'matches':sum(d['matches'] for d in details),'details':details,
            'source_review':'One assistant reviewer read original report sections before predictions; excludes ambiguous attributes',
            'limits':['Not clinician-adjudicated gold standard; targeted small sample not representative sensitivity/precision',
                      'No claim about all7 attributes, uncertain classes, rare conditions or entire cohort accuracy']}
    output.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='details'},ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selected',type=Path,default=Path('patient_data/processed/v6_cxr_selected_2026-10-09_r2'))
    p.add_argument('--labeled',type=Path,default=Path('patient_data/processed/v6_chexbert_cxr_2026-10-09'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); review(a.selected,a.labeled,a.output)
