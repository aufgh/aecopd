"""构建公开Git白名单清单；不读取患者目录，不回显秘密，不修改原文件。"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'work/public_release_2026-10-14'
DOCS = {
    'V6_SYMMETRIC_CONTEXT_PROTOCOL_2026-10-14.md', 'V6_SYMMETRIC_CONTEXT_RESULTS_2026-10-14.md',
    'V6_UNIFIED_LONGITUDINAL_F1_PRD_2026-10-14.md', 'V6_UNIFIED_LONGITUDINAL_F1_RESULTS_2026-10-14.md',
    'V6_MULTIHORIZON_READOUT_PROTOCOL_2026-10-14.md', 'V6_MULTIHORIZON_READOUT_RESULTS_2026-10-14.md',
    'V6_HISTORY_AND_ALERT_EXPLAINED_2026-10-14.md', 'V6_CURRENT_INPUT_CATALOG_2026-10-11.md',
    'COPD_PREDICTION_PRIMARY_STUDIES_REVIEW_2026-10-14.md', 'V6_FAST_SCREENING_POLICY_2026-10-14.md',
    'PUBLIC_README.md', 'PUBLIC_NAVIGATION.md', 'PUBLIC_GITIGNORE.txt', 'PUBLICATION_BOUNDARY_2026-10-14.md',
}
TEXT = {'.py','.ps1','.sql','.md','.yaml','.yml','.toml','.txt','.mjs','.css','.csv',''}
CODE_ROOTS = {'training','cohort_pipeline','model_loader'}
ID = re.compile(r'(?<![\w.])(?:[1-9]\d{7})(?![\w.])')
SECRET = re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|(?:postgres(?:ql)?|https?)://[^\s/:]+:[^\s/@]+@)')
LITERAL = re.compile(r'(?:password|passwd|pgpassword|api_key|access_token|secret)[^\n]{0,60}?(?:=\s*|\.get\([^\n]{0,60}?,\s*)[\"\']([^\"\'\n]+)[\"\']',re.I)
# 人工核对过的假阳性：日期随机种子/容器标签/官方表行数、交互式密码读取。
# 逐文件hash绑定，修改后自动失效；不是允许旧硬编码密码或病例ID。
REVIEWED_PATHS = {
    'cohort_pipeline/official_mimiciii_postgres/postgres_checks.sql',
    'model_loader/validate_loader.py', 'training/validate_batching.py',
    'training/v5/finish_all_lab_axes.ps1', 'training/v6/regularization_search.py',
    'training/v6/task_gradient_diagnostic.py', 'training/export_mimiciv_discharge_all.py',
}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def candidate(p):
    top=p.parts[0];s=p.as_posix()
    if top in CODE_ROOTS:
        return ((p.suffix in TEXT - {'.md', '.csv'} and '/runtime/' not in s)
                or s=='training/runtime/public_tabular_requirements.txt')
    if top=='patient_data':
        return ((len(p.parts)==2 and p.suffix=='.py') or
                (s.startswith('patient_data/clinical_facts/') and p.suffix=='.py'))
    if top=='config':return s in ('config/clinical_feature_contract.csv','config/clinical_feature_contract_v2_diagnosis.csv')
    if top=='docs':return len(p.parts)==2 and p.name in DOCS
    return s in ('.gitignore','work/prepare_public_snapshot.py')


def risk(text):
    reasons=[]
    if SECRET.search(text):reasons.append('secret_pattern')
    for m in LITERAL.finditer(text):
        value=m[1]
        if value.strip() and not any(t in value.lower() for t in ('example','placeholder','changeme','${','your_','env:','\\n','postgresql password','postgres password')):
            reasons.append('literal_credential_review');break
    for line in text.splitlines():
        # PMID/publication URLs and documented numeric schema bounds are not patient examples.
        clean=re.sub(r'https?://[^\s)>\]]+','',line)
        clean=re.sub(r'PMID\s*[:：]?\s*\d+','',clean,flags=re.I)
        hits=ID.findall(clean)
        if any(int(x)%1000000!=0 for x in hits):
            reasons.append('eight_digit_identifier_review');break
    return sorted(set(reasons))


def scan():
    raw=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=ROOT)
    paths=sorted(set(x.decode('utf-8') for x in raw.split(b'\0') if x))
    accepted=[];excluded=[]
    review_path=OUT/'reviewed_false_positives.json'
    reviewed=json.loads(review_path.read_text()) if review_path.exists() else {}
    for s in paths:
        p=Path(s);f=ROOT/p
        if not f.is_file():continue
        if not candidate(p):
            excluded.append(dict(path=s,reason=['outside_public_allowlist']));continue
        if f.stat().st_size>1024*1024:
            excluded.append(dict(path=s,reason=['oversized']));continue
        text=f.read_text(encoding='utf-8-sig',errors='strict')
        reasons=risk(text)
        if s in REVIEWED_PATHS and reviewed.get(s)==sha(f):reasons=[]
        if reasons:excluded.append(dict(path=s,reason=reasons))
        else:accepted.append(dict(path=s,sha256=sha(f),bytes=f.stat().st_size))
    OUT.mkdir(exist_ok=True)
    report=dict(accepted=accepted,excluded=excluded,publication='public source and reviewed aggregate docs only',
                history_policy='new parentless public snapshot; old local Git history never pushed',
                limits='heuristic scan plus explicit allowlist; no patient artifacts or legacy prose published')
    (OUT/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(accepted=len(accepted),bytes=sum(x['bytes'] for x in accepted),
        excluded_risk=[x for x in excluded if x['reason']!=['outside_public_allowlist']]),ensure_ascii=True),flush=True)
    return report


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    scan()
