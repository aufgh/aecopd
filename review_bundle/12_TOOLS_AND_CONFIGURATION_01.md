# 12_TOOLS_AND_CONFIGURATION · 1

源文件正文或经过字段白名单提取的汇总数据，来源身份见每段元数据。

## .gitignore · lines 1–90

<!-- REVIEW_FRAGMENT {"path":".gitignore","part":1,"start_line":1,"end_line":90,"source_sha256":"0b228e7850e17b86534260fc63f07e24ee3d4b9ff01359b9d8e48775988f6861","fragment_sha256":"0b228e7850e17b86534260fc63f07e24ee3d4b9ff01359b9d8e48775988f6861","added_terminal_newline":false} -->
````text
# Public repository: source and aggregate documents only.
# Never add MIMIC data, identifiers, predictions, embeddings or model weights.
.env
.env.*
*.env
*.pem
*.key
*.p12
*.pfx
.pgpass
pgpass.conf
*credentials*
*secrets*
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
.venv/
venv/
node_modules/
.idea/
.vscode/
*.log
*.out
*.tmp
*.bak
*.csv
*.tsv
*.json
*.jsonl
*.ndjson
*.parquet
*.arrow
*.feather
*.sqlite*
*.db*
*.gz
*.bz2
*.xz
*.zip
*.7z
*.tar
*.rar
*.npy
*.npz
*.pt
*.pth
*.ckpt
*.safetensors
*.bin
*.gguf
*.onnx
*.pkl
*.pickle
*.joblib
*.h5
*.hdf5
*.png
*.jpg
*.jpeg
*.webp
*.pdf
*.pptx
*.docx
*.xlsx
/outputs/
/training_runs/
/transfer/
/Bonsai2_Extraction_Pipeline/
/.agents/
/.build_architecture/
/.codex-finalizer-architecture/
/training/model_cache/
/training/cv/
/training/reports/
/training/runtime/*/
/patient_data/**
!/patient_data/**/
!/patient_data/*.py
!/patient_data/clinical_facts/*.py
/work/**
!/work/prepare_public_snapshot.py
!/config/clinical_feature_contract.csv
!/config/clinical_feature_contract_v2_diagnosis.csv

# Reviewed aggregate-only V7 figures, not patient images or record-level outputs.
!/docs/figures/v7_first_experiment_2026-10-15.png
!/docs/figures/v7_loss_comparison_2026-10-15.png
!/docs/figures/v7_sequential_research_2026-10-15.png
````
<!-- END REVIEW_FRAGMENT -->

## tools/build_markdown_review.py · lines 1–148

<!-- REVIEW_FRAGMENT {"path":"tools/build_markdown_review.py","part":1,"start_line":1,"end_line":148,"source_sha256":"550b8551e7ffeb095d557d0da8df07924caf4eedddb12b7912685ab41fdc9739","fragment_sha256":"550b8551e7ffeb095d557d0da8df07924caf4eedddb12b7912685ab41fdc9739","added_terminal_newline":false} -->
````python
"""按已审查源码清单构建或核验 Markdown 阅读包；不自动发现私有数据。"""
import argparse
import hashlib
import json
from pathlib import Path
import re

MAX_BYTES = 800_000
FOLDER = 'review_bundle'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def chunks(raw):
    lines = raw.splitlines(keepends=True)
    if not lines:
        yield 1, 0, b''
        return
    start = 1; pending = []; size = 0
    for number, line in enumerate(lines, 1):
        if len(line) > MAX_BYTES-4096:
            raise ValueError('single source line exceeds fragment limit')
        if pending and size+len(line) > MAX_BYTES-4096:
            yield start, number-1, b''.join(pending)
            start = number; pending = []; size = 0
        pending.append(line); size += len(line)
    yield start, len(lines), b''.join(pending)


def render(root, catalog):
    groups = {}; locations = {}
    for item in catalog['sources']:
        path = root/item['path']
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('catalog path escapes public checkout')
        raw = path.read_bytes()
        if digest(raw) != item['sha256'] or len(raw) != item['bytes']:
            raise ValueError('reviewed source changed: '+item['path'])
        raw.decode('utf-8')
        locations[item['path']] = []
        for part, (start, end, fragment) in enumerate(chunks(raw), 1):
            text = fragment.decode('utf-8')
            runs = [len(v) for v in re.findall(r'`+', text)]
            fence = '`'*max(4, max(runs, default=0)+1)
            language = {'.py':'python', '.sql':'sql', '.json':'json', '.md':'markdown',
                        '.yaml':'yaml', '.yml':'yaml', '.ps1':'powershell', '.csv':'csv'}.get(path.suffix, 'text')
            added = not fragment.endswith(b'\n')
            meta = dict(path=item['path'], part=part, start_line=start, end_line=end,
                        source_sha256=item['sha256'], fragment_sha256=digest(fragment),
                        added_terminal_newline=added)
            block = ('\n## '+item['path']+f' · lines {start}–{end}\n\n'
                     +'<!-- REVIEW_FRAGMENT '+json.dumps(meta, ensure_ascii=False, separators=(',',':'))+' -->\n'
                     +fence+language+'\n'+text+('\n' if added else '')+fence+'\n'
                     +'<!-- END REVIEW_FRAGMENT -->\n')
            groups.setdefault(item['group'], []).append((item['path'], block))
    result = {}
    for group, blocks in sorted(groups.items()):
        part = 1; current = f'# {group} · {part}\n\n源文件正文或经过字段白名单提取的汇总数据，来源身份见每段元数据。\n'
        used = []
        for path, block in blocks:
            if used and len((current+block).encode('utf-8')) > MAX_BYTES:
                name = f'{group}_{part:02d}.md'; result[name] = current.encode('utf-8')
                for source in set(used):locations[source].append(name)
                part += 1; current = f'# {group} · {part}\n\n续篇；文件碎片按part序号拼接可还原原文。\n'; used = []
            current += block; used.append(path)
        if used:
            name = f'{group}_{part:02d}.md'; result[name] = current.encode('utf-8')
            for source in set(used):locations[source].append(name)
    index = ['# AECOPD 完整公开源码、汇总结果与数据结构快照', '',
        '这是替代旧00–07 TXT摘录的Markdown阅读包。每段都是可直接读取的正文；不需要解压ZIP，也不把文件目录当成已读源码。', '',
        '版本父提交：`'+catalog['base_public_commit']+'`。本次新增/更新内容以source_catalog.json中的SHA256绑定；最终发布身份以当前Git提交为准。', '',
        f"收录{len(catalog['sources'])}份文本文件，分成{len(result)}个正文分卷；所有分卷合计正文可逐字节重建。", '',
        '**阅读顺序**', '',
        '先读数据快照和结果，再按任务读取V7/V6及其依赖；历史V5/基线/SQL/测试全部保留。若会话上下文不够，可分批上传主题分卷。只有确实读取的正文才能作为源码证据。', '',
        '**边界与口径**', '',
        '“数据快照”包含表规模、划分统计、字段、数组shape/dtype、文件级hash；不包含患者行、ID取值、逐人标签/划分/预测、文本、向量或权重。'
        '历史汇总JSON经指标字段白名单提取，明确不是原始完整报告；不同任务/划分/标签的结果不可混排。合成测试不能证明独立临床有效性。', '',
        '当前任务总计19,401次合格出院/5,351患者：训练13,592/3,766，调参1,531/402，确认1,418/395，历史测试2,860/788。'
        '同患者不跨划分；历史确认/测试有项目早期曝光。本次发布不重训、不重新评估其结局。', '',
        '**分卷**', '', '| 文件 | 字节数 |', '|---|---:|']
    for name, raw in result.items():index.append(f'| [{name}]({name}) | {len(raw):,} |')
    index += ['', '**完整文件定位**', '', '| 原路径 | 正文分卷 |', '|---|---|']
    for path, names in sorted(locations.items()):
        index.append('| `'+path+'` | '+'、'.join(f'[{n}]({n})' for n in names)+' |')
    index += ['', '**不在公开范围内的源码**', '',
              '此前审查中以下9份原型含历史病例引用，保留本地，不把它们伪装成完整公开代码：', '']
    index.extend('- `'+name+'`' for name in catalog['withheld_code'])
    index += ['', '可重装依赖、缓存、重复发布现场、私有数据目录和本地旧Git历史不属于此公开源码包。'
        'PNG等既有汇总图仍在仓库原路径，本包索引列为二进制附件，不将其编码为伪Markdown数据。', '',
        '**仓库中保留的二进制汇总附件**', '']
    index.extend('- `'+item['path']+'` · '+str(item['bytes'])+' bytes · SHA256 `'+item['sha256']+'`' for item in catalog['binary_assets'])
    index += ['', '核验：`python tools/build_markdown_review.py --verify`。该工具仅读取已审查的清单路径，不递归收集本地数据。', '']
    result['00_READ_FIRST.md'] = '\n'.join(index).encode('utf-8')
    manifest = dict(base_public_commit=catalog['base_public_commit'], source_catalog_sha256=digest(json.dumps(catalog,ensure_ascii=False,indent=2).encode('utf-8')+b'\n'),
                    source_count=len(catalog['sources']), markdown_files=len(result),
                    outputs={name:dict(bytes=len(raw),sha256=digest(raw)) for name,raw in result.items()},
                    payload_bytes=sum(item['bytes'] for item in catalog['sources']),
                    all_source_bytes_reconstructible=True, contains_patient_values=False)
    result['manifest.json'] = (json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
    return result


def reconstruct(root, catalog):
    collected = {}
    pattern = re.compile(r'<!-- REVIEW_FRAGMENT (\{[^\n]+\}) -->\n(?P<fence>`{4,})[^\n]*\n(?P<body>.*?)(?P=fence)\n<!-- END REVIEW_FRAGMENT -->', re.S)
    for p in sorted((root/FOLDER).glob('*.md')):
        for match in pattern.finditer(p.read_bytes().decode('utf-8')):
            meta = json.loads(match[1]); body = match['body'].encode('utf-8')
            if meta['added_terminal_newline']:
                if not body.endswith(b'\n'):raise ValueError('fragment terminator missing')
                body = body[:-1]
            if digest(body) != meta['fragment_sha256']:
                raise ValueError('fragment bytes differ: '+meta['path'])
            collected.setdefault(meta['path'], []).append((meta['part'], body))
    if set(collected) != {item['path'] for item in catalog['sources']}:
        raise ValueError('source coverage differs')
    for item in catalog['sources']:
        pieces = sorted(collected[item['path']])
        if [i for i,_ in pieces] != list(range(1,len(pieces)+1)):
            raise ValueError('missing or duplicate fragment')
        raw = b''.join(body for _,body in pieces)
        if raw != (root/item['path']).read_bytes() or digest(raw) != item['sha256']:
            raise ValueError('reconstructed source differs: '+item['path'])


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--verify', action='store_true'); args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    catalog = json.loads((root/FOLDER/'source_catalog.json').read_text(encoding='utf-8'))
    files = render(root, catalog)
    for name, raw in files.items():
        target = root/FOLDER/name
        if args.verify:
            if not target.is_file() or target.read_bytes() != raw:
                raise ValueError('bundle differs: '+name)
        else:
            if target.exists():raise FileExistsError('existing bundle; use --verify')
            target.write_bytes(raw)
    reconstruct(root, catalog)
    print(json.dumps(dict(source_files=len(catalog['sources']), markdown_files=sum(n.endswith('.md') for n in files),
                          markdown_bytes=sum(len(v) for n,v in files.items() if n.endswith('.md')),
                          reconstructed_source_bytes=True), ensure_ascii=False))


if __name__ == '__main__':
    main()
````
<!-- END REVIEW_FRAGMENT -->

## work/prepare_public_snapshot.py · lines 1–98

<!-- REVIEW_FRAGMENT {"path":"work/prepare_public_snapshot.py","part":1,"start_line":1,"end_line":98,"source_sha256":"e72b8b002f4791936f39109d004f398c0845536c3529da520c8f1358748118e2","fragment_sha256":"e72b8b002f4791936f39109d004f398c0845536c3529da520c8f1358748118e2","added_terminal_newline":false} -->
````python
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
````
<!-- END REVIEW_FRAGMENT -->
