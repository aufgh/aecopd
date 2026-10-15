# 05_PATIENT_DATA_CODE · 1

源文件正文或经过字段白名单提取的汇总数据，来源身份见每段元数据。

## patient_data/align_saved_sections.py · lines 1–155

<!-- REVIEW_FRAGMENT {"path":"patient_data/align_saved_sections.py","part":1,"start_line":1,"end_line":155,"source_sha256":"b84646f33384fe8b0be6c37b07e3118ad64f451241f50f07a7aa4414c2a6e19b","fragment_sha256":"b84646f33384fe8b0be6c37b07e3118ad64f451241f50f07a7aa4414c2a6e19b","added_terminal_newline":false} -->
````python
"""核对保存的A章节向量与本次原始小结；只生成派生索引，不改原始数据。"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import time
import numpy as np
from patient_data.export_mimiciv import rows, sha256
# 旧包__init__导入已停用的模型。单独读取纯解析模块，不恢复旧训练依赖。
_parser_path = Path(__file__).resolve().parents[1]/'training/aecopd_training/text_sections.py'
_spec = importlib.util.spec_from_file_location('_saved_section_parser', _parser_path)
_parser = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _parser
_spec.loader.exec_module(_parser)
split_clinical_document = _parser.split_clinical_document
build_section_embedding_text = _parser.build_section_embedding_text

ROLES = ('chief_complaint', 'history_of_present_illness', 'past_medical_history',
         'hospital_course', 'pertinent_results', 'discharge_diagnosis',
         'discharge_condition', 'discharge_medications', 'follow_up', 'other')
FOUR = (1, 2, 3, 6)


def section_inputs(raw):
    parsed = split_clinical_document('current_discharge', raw['note_type'], raw['text'])
    result = {}
    normalized = raw['text'].replace('\r\n', '\n').replace('\r', '\n').strip()
    for section in parsed.sections:
        # parser省略部分标题，但每个正文仍必须是规范化原文的连续片段。
        start = normalized.find(section.text)
        end = start + len(section.text)
        if start < 0:
            # 历史解析器对标题同一行正文/other标题作strip，可能删去第一行尾部空格。
            # 只允许这个明确的空白变换，不容许文本改写或跨段拼接。
            first, separator, rest = section.text.partition('\n')
            pattern = re.escape(first) + (r'[ \t]*\n'+re.escape(rest) if separator else '')
            match = re.search(pattern, normalized)
            if match is None: raise ValueError('section is not a normalized source span')
            start, end = match.span()
        if not 0 <= start <= end <= len(normalized): raise ValueError('invalid source span')
        source = normalized[start:end]
        first, separator, rest = source.partition('\n')
        if source != section.text and (first.rstrip()+separator+rest) != section.text:
            raise ValueError('source span changed beyond first-line whitespace')
        result[section.section_order] = (
            section, build_section_embedding_text('current_discharge', parsed.modality, section, raw['note_type']))
    return result


def build(args):
    if args.output.exists(): raise FileExistsError(args.output)
    started = time.perf_counter()
    manifest = json.loads((args.snapshot/'metadata/manifest.json').read_text())
    raw_path = args.snapshot/'raw/discharge.csv.gz'
    if manifest['status'] != 'validated' or sha256(raw_path) != manifest['sources']['raw/discharge']['sha256']:
        raise ValueError('raw discharge snapshot changed')
    with np.load(args.state/'clinical_state.npz') as z:
        a = {k:z[k] for k in ('hadm_ids', 'subject_ids', 'patient_splits')}
    by_h = {str(h):i for i,h in enumerate(a['hadm_ids'])}
    index_path = args.assets/'model_v2/qwen_section_embedding_index.csv.gz'
    chunks_path = args.assets/'model_v2/qwen_section_text_chunks.csv.gz'
    selected, notes = {}, {}
    for count, r in enumerate(rows(index_path)):
        if int(r['embedding_row']) != count: raise ValueError('A index is not sequential')
        if r['document_role'] != 'current_discharge' or r['hadm_id'] not in by_h: continue
        p = by_h[r['hadm_id']]
        if r['data_split'] != a['patient_splits'][p] or r['section_role'] not in ROLES:
            raise ValueError('patient split/role differs')
        if int(r['chunk_index']) != 0: raise ValueError('unexpected multi-chunk A section')
        note = r['source_document_key'].rsplit(':', 1)[1]
        selected[count] = r
        notes.setdefault(note, (p, r['source_document_key']))
        if notes[note] != (p, r['source_document_key']): raise ValueError('note ownership conflict')
    print(json.dumps({'stage':'index', 'rows':len(selected), 'notes':len(notes)}), flush=True)
    expected, note_hashes = {}, {}
    for r in rows(raw_path):
        if r['note_id'] not in notes: continue
        p, key = notes[r['note_id']]
        if r['hadm_id'] != str(a['hadm_ids'][p]) or r['subject_id'] != str(a['subject_ids'][p]):
            raise ValueError('raw note ownership differs')
        if key in expected: raise ValueError('duplicate raw note')
        expected[key] = (r, section_inputs(r))
        note_hashes[r['note_id']] = hashlib.sha256(r['text'].encode()).hexdigest()
    if len(expected) != len(notes): raise ValueError('saved note absent from current snapshot')
    kept, refs, seen, roles, fold_counts = [], [], set(), Counter(), Counter()
    for number, r in enumerate(rows(chunks_path)):
        if number not in selected: continue
        idx = selected[number]
        for key in ('document_key','source_document_key','hadm_id','data_split','document_role',
                    'section_role','section_order','section_instance','chunk_index','modality'):
            if r[key] != idx[key]: raise ValueError('A chunk/index mismatch: '+key)
        raw, sections = expected[r['source_document_key']]
        section, text = sections[int(r['section_order'])]
        start, end = int(r['char_start']), int(r['char_end'])
        if (r['source_hadm_id'] != r['hadm_id'] or r['subject_id'] != raw['subject_id']
            or r['charttime'] != raw['charttime'] or r['storetime'] != raw['storetime']
            or r['section_role'] != section.section_role or int(r['section_instance']) != section.section_instance
            or not 0 <= start <= end <= len(text) or r['chunk_text'] != text[start:end]
            or text[:start].strip() or text[end:].strip()):
            raise ValueError('raw source/span/prefix mismatch')
        key = r['source_document_key'], int(r['section_order'])
        if key in seen: raise ValueError('duplicate current section')
        seen.add(key)
        p = by_h[r['hadm_id']]
        refs.append((p, ROLES.index(r['section_role']), len(kept)))
        kept.append(number); roles[r['section_role']] += 1; fold_counts[r['data_split']] += 1
    expected_keys = {(key, order) for key,(_,sections) in expected.items() for order in sections}
    if seen != expected_keys or len(kept) != len(selected):
        raise ValueError('not all source sections represented exactly once')
    vectors_path = args.assets/'bioclinicalbert_sections_2026-09-24/bioclinicalbert_section_embeddings.float16.npy'
    state_path = vectors_path.parent/'encoding_state.json'
    state = json.loads(state_path.read_text())
    if state['status'] != 'complete' or state['completed_rows'] != count+1:
        raise ValueError('incomplete A BERT')
    v = np.load(vectors_path, mmap_mode='r')
    if v.shape != (count+1,768) or v.dtype != np.float16: raise ValueError('invalid saved shape')
    subset = np.asarray(v[kept])
    norms = np.linalg.norm(subset.astype(np.float32), axis=1)
    if not np.isfinite(subset).all() or not np.all((norms > .99) & (norms < 1.01)):
        raise ValueError('invalid selected encoding')
    args.output.mkdir(parents=True)
    np.save(args.output/'vectors.float16.npy', subset)
    np.savez_compressed(args.output/'alignment.npz', **a, refs=np.asarray(refs,np.int64), source_rows=np.asarray(kept,np.int64))
    matched = np.unique(np.asarray(refs)[:,0])
    audit = dict(status='current_discharge_source_and_complete_section_coverage_verified',
        roles=list(ROLES), four_ids=list(FOUR), rows=len(kept), admissions=len(matched),
        patients=len(np.unique(a['subject_ids'][matched])), source_notes=len(notes),
        sections=dict(roles), section_rows_by_fold=dict(fold_counts),
        members_by_fold={f:int(np.sum(a['patient_splits'][matched]==f)) for f in ('train','validation','test')},
        norm_min=float(norms.min()), norm_max=float(norms.max()),
        note_text_sha256=note_hashes,
        source_sha256={str(p):sha256(p) for p in (raw_path,index_path,chunks_path,vectors_path,state_path)},
        state_manifest_sha256=sha256(args.state/'manifest.json'),
        output_sha256={f:sha256(args.output/f) for f in ('vectors.float16.npy','alignment.npz')},
        code_sha256={str(p):sha256(p) for p in (Path(__file__),Path('training/aecopd_training/text_sections.py'))},
        limitations=['Full means all natural sections of the selected saved current note, not every database note.',
            'Subset comes from an old strict cohort and then-time note selection; not population-wide coverage.',
            'Final stored discharge summary is treated as discharge information; not real-time deployment.',
            'Vectors are reused from historical completion state; this audit does not independently rerun BERT.'],
        elapsed_seconds=time.perf_counter()-started)
    (args.output/'manifest.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps({k:audit[k] for k in ('status','rows','admissions','members_by_fold','elapsed_seconds')}),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',type=Path,default=Path('patient_data/mimiciv_copd/snapshot_2026-09-27'))
    p.add_argument('--state',type=Path,default=Path('patient_data/processed/v6_clinical_state_2026-10-09_r2'))
    p.add_argument('--assets',type=Path,default=Path('outputs/01a08b96-2988-7001-af1d-b40d901f3400/private'))
    p.add_argument('--output',type=Path,default=Path('patient_data/processed/v6_saved_sections_aligned_2026-10-11'))
    build(p.parse_args())
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/audit_cxr_labels.py · lines 1–65

<!-- REVIEW_FRAGMENT {"path":"patient_data/audit_cxr_labels.py","part":1,"start_line":1,"end_line":65,"source_sha256":"d56750c66a6d1d00f2521d3845fbb120e243f82bcf601eeeb76107524833bca0","fragment_sha256":"d56750c66a6d1d00f2521d3845fbb120e243f82bcf601eeeb76107524833bca0","added_terminal_newline":false} -->
````python
"""独立只读回查CheXbert输入原文、时间、成员及四状态落盘一致性。"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
from patient_data.export_mimiciv import rows, sha256

CONDITIONS=('Pneumonia','Consolidation','Edema','Cardiomegaly','Pleural Effusion','Pneumothorax','Atelectasis')


def verify(directory,snapshot,output):
    if output.exists(): raise FileExistsError(output)
    manifest=json.loads((directory/'manifest.json').read_text())
    for name,digest in manifest['output_sha256'].items():
        if sha256(directory/name)!=digest: raise ValueError('CheXbert output changed')
    reports=[json.loads(line) for line in (directory/'labeled_reports.jsonl').read_text(encoding='utf-8').splitlines()]
    by_note={r['note_id']:r for r in reports}
    if len(by_note)!=len(reports): raise ValueError('duplicate selected note')
    admissions={r['hadm_id']:r for r in rows(snapshot/'cohort/admissions.csv.gz')}
    matched=0
    for original in rows(snapshot/'raw/radiology.csv.gz'):
        r=by_note.get(original['note_id'])
        if not r: continue
        if any(original[key]!=r[key] for key in ('subject_id','hadm_id','charttime','storetime')):
            raise ValueError('original identity/time differs')
        if hashlib.sha256(original['text'].encode()).hexdigest()!=r['original_sha256']:
            raise ValueError('original text hash differs')
        start,end=r['start_char'],r['end_char']
        if not 0<=start<=end<=len(original['text']) or original['text'][start:end]!=r['text']:
            raise ValueError('original section span differs')
        if hashlib.sha256(r['text'].encode()).hexdigest()!=r['section_sha256']: raise ValueError('section hash differs')
        adm=admissions[r['hadm_id']]
        begin,chart,store,discharge=map(datetime.fromisoformat,(adm['admittime'],r['charttime'],r['storetime'],adm['dischtime']))
        if not begin<=chart<=store<=discharge or adm['discharge_landmark_eligible']!='t': raise ValueError('prediction time violated')
        matched+=1
    if matched!=len(reports): raise ValueError('original source missing')
    with np.load(directory/'cxr_context.npz',allow_pickle=False) as archive: arrays={k:archive[k] for k in archive.files}
    position={str(h):i for i,h in enumerate(arrays['hadm_ids'])}
    seen=set()
    for r in reports:
        i=position[r['hadm_id']]
        if i in seen or not arrays['cxr_present'][i]: raise ValueError('duplicate or unobserved image row')
        if str(arrays['subject_ids'][i])!=r['subject_id'] or str(arrays['patient_splits'][i])!=r['patient_split']:
            raise ValueError('subject/split inconsistent')
        expected=np.array([r['labels14'][c] for c in CONDITIONS])
        if not np.array_equal(arrays['cxr_states'][i],expected): raise ValueError('labels do not match archive')
        seen.add(i)
    present=arrays['cxr_present']; states=arrays['cxr_states']
    onehot=(np.eye(4,dtype=np.float32)[states]*present[:,None,None]).reshape(len(states),28)
    if len(seen)!=int(present.sum()) or not np.array_equal(onehot,arrays['cxr_onehot']): raise ValueError('missingness/onehot inconsistent')
    result={'status':'all_selected_reports_original_time_identity_state_verified','reports':matched,
            'admissions':len(states),'manifest_sha256':sha256(directory/'manifest.json'),
            'limits':['Engineering provenance verification, not clinical semantic accuracy',
                      'Report classifier false positives/negatives and CheXbert semantic errors need separate review']}
    output.write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,default=Path('patient_data/processed/v6_chexbert_cxr_2026-10-09'))
    p.add_argument('--snapshot',type=Path,default=Path('patient_data/mimiciv_copd/snapshot_2026-09-27'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); verify(a.directory,a.snapshot,a.output)
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/audit_feature_candidates.py · lines 1–367

<!-- REVIEW_FRAGMENT {"path":"patient_data/audit_feature_candidates.py","part":1,"start_line":1,"end_line":367,"source_sha256":"4fd4b332ade0f88d662e73c31267bc8185ad18f28cb2961bbee341d48d4bb9c7","fragment_sha256":"4fd4b332ade0f88d662e73c31267bc8185ad18f28cb2961bbee341d48d4bb9c7","added_terminal_newline":false} -->
````python
"""只读审计 V5 的原始候选、现行事件概念与静态/文本覆盖，不改训练输入。"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

from patient_data.export_mimiciv import NULL, rows, sha256
from training.v5.clinical_concepts import medication_classes

CURRENT_LABS = {"50818", "50820", "50821", "50862", "50882", "50912", "51222", "51301"}
LAB_ITEM_FOR_CONCEPT = {"pco2": "50818", "ph": "50820", "po2": "50821",
                        "albumin": "50862", "bicarbonate": "50882", "creatinine": "50912",
                        "hemoglobin": "51222", "wbc": "51301"}
REVIEW_LABS = {
    "50813": "乳酸/组织灌注", "50817": "血气氧饱和度/氧合", "50889": "CRP/炎症",
    "50802": "碱剩余/酸碱状态，与碳酸氢盐和pH重叠",
    "50804": "血气计算总二氧化碳，与碳酸氢盐重叠",
    "50868": "阴离子间隙/酸碱状态，和氯/碳酸氢盐相关",
    "50902": "血氯/酸碱状态，和钠/碳酸氢盐相关",
    "50931": "血糖/糖尿病与应激激素治疗",
    "50960": "血镁/支气管扩张剂及利尿剂治疗安全",
    "50963": "NT-proBNP/心衰鉴别", "50971": "血钾/治疗安全", "50983": "血钠/整体病情",
    "50924": "铁蛋白/炎症及贫血，非COPD特异且相对稀疏",
    "51003": "肌钙蛋白T/心脏事件", "51199": "嗜酸粒细胞绝对数/治疗表型",
    "51006": "尿素氮/肾功能及容量状态，与肌酐相关",
    "51200": "嗜酸粒细胞百分比/治疗表型", "51265": "血小板/全身病情",
    "51221": "红细胞压积/贫血，与血红蛋白高度相关",
    "51256": "中性粒细胞百分比/感染和激素反应，与白细胞相关",
    "51277": "RDW/贫血与衰弱代理，非COPD特异",
    "52075": "绝对中性粒细胞/感染，与百分比及白细胞相关",
    "52073": "嗜酸粒细胞绝对数/治疗表型",
}
BRAND_REVIEW = ("duoneb", "combivent", "symbicort", "advair", "breo", "trelegy",
                "spiriva", "incruse", "anoro", "stiolto", "pulmicort", "ventolin",
                "proair", "xopenex", "atrovent", "daliresp", "roflumilast")
RATIONALE = {
    "ph": "酸碱状态与通气衰竭；标本不同不能混作动脉血气",
    "pco2": "二氧化碳潴留/通气衰竭；静脉值不等于PaCO2",
    "po2": "氧合严重度；只有已核实的动脉标本才能称PaO2",
    "bicarbonate": "酸碱代偿及长期高碳酸血症线索，受肾功能等影响",
    "wbc": "感染/炎症的非特异线索，受激素影响",
    "hemoglobin": "贫血与携氧能力，也反映共病",
    "creatinine": "肾功能、合并症与治疗耐受性，非COPD特异",
    "albumin": "营养/衰弱及炎症负担，非COPD特异",
    "saba": "急性加重短效β2激动剂治疗；给药是诊疗行为而非疗效",
    "sama": "急性加重短效抗胆碱能治疗；给药是诊疗行为而非疗效",
    "laba": "长期维持支气管扩张治疗；需复核复方/商品名",
    "lama": "长期维持支气管扩张治疗；需复核复方/商品名",
    "inhaled_steroid": "预防加重相关维持治疗；需复核吸入途径与复方",
    "systemic_steroid": "急性加重系统激素治疗；需排除局部制剂",
    "macrolide": "有指征的抗菌治疗；也可能用于其他感染",
    "beta_lactam": "有指征的抗菌治疗；也可能用于其他感染",
    "fluoroquinolone": "有指征的抗菌治疗；也可能用于其他感染",
    "other_antibiotic": "宽泛抗菌药组合，异质性高，需审查具体药物",
    "loop_diuretic": "心衰/容量负荷治疗，属于共病过程信号",
    "other_diuretic": "异质性利尿药组合，非AECOPD特异，需审查",
}
STATIC_RATIONALE = {
    "context:age_at_admission": "年龄影响COPD、共病和死亡风险",
    "context:male": "性别是基本人口学变量，需看独立增量而非假定因果",
    "context:prior_coded_admissions": "既往COPD编码住院频次是再入院基线风险",
    "context:prior_strict_aecopd_admissions": "既往严格加重住院频次是严重加重史",
    "aecopd_index_code:strict_aecopd": "本次严格加重编码反映本次事件表型，不是未来标签",
    "chronic_comorbidity:heart_failure": "心衰可模仿/加重气促并改变再入院风险",
    "chronic_comorbidity:cardiac_other": "缺血性心脏病/房颤等心脏共病，编码组较宽",
    "chronic_comorbidity:ckd": "肾病影响全身风险与治疗耐受性",
    "chronic_comorbidity:diabetes": "糖尿病影响感染及整体健康风险",
    "chronic_comorbidity:malignancy": "肿瘤反映非COPD疾病负担与竞争死亡",
    "chronic_comorbidity:chronic_respiratory": "慢性呼吸衰竭/睡眠呼吸障碍反映呼吸储备",
    "acute_diagnosis:acute_resp_failure": "本次急性呼吸衰竭是严重程度线索",
    "acute_diagnosis:pneumonia_or_aspiration": "肺炎/误吸提示感染与可能的再入院机制",
    "acute_diagnosis:sepsis": "脓毒症是本次全身严重度与共病过程",
    "acute_diagnosis:aki": "急性肾损伤反映本次器官损害",
    "disposition:home": "出院去向及照护需求代理，可能受制度和医生行为影响",
    "disposition:facility": "出院去向及照护需求代理，可能受制度和医生行为影响",
    "disposition:other": "出院去向及照护需求代理，可能受制度和医生行为影响",
}
TEXT_RATIONALE = {
    "history_of_present_illness": "入院病情叙述，可能含症状/诱因但与诊断编码重叠",
    "past_medical_history": "既往病史语义，可能补充未编码疾病但与共病编码重叠",
    "hospital_course": "住院期间病情与治疗反应叙述，可能补充数值和药物事件",
    "discharge_condition": "出院时残余症状/功能状态，可能与近期再入院有关",
}


def _write_csv(path: Path, names: tuple[str, ...], records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(records)


def _family(concept: str) -> str:
    return concept.split(":", 1)[0]


def _static_source(concept: str) -> str:
    if concept == "context:age_at_admission":
        return "raw/patients.anchor_age,anchor_year + cohort/admissions.admittime"
    if concept == "context:male":
        return "raw/patients.gender"
    if concept.startswith("context:prior_"):
        return "earlier eligible cohort/admissions + raw/diagnoses_icd"
    if concept.startswith("chronic_comorbidity:"):
        return "raw/diagnoses_icd current and prior eligible admissions"
    if concept.startswith(("acute_diagnosis:", "aecopd_index_code:")):
        return "raw/diagnoses_icd current admission"
    if concept.startswith("disposition:"):
        return "cohort/admissions.discharge_location"
    raise ValueError(f"unknown static concept source: {concept}")


def audit(raw_root: Path, processed_root: Path, structured_tokens: Path,
          shared_text: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    raw_manifest = json.loads((raw_root / "metadata/manifest.json").read_text(encoding="utf-8"))
    if raw_manifest["status"] != "validated":
        raise ValueError("raw ledger is not validated")
    processed_manifest = json.loads((processed_root / "manifest.json").read_text(encoding="utf-8"))
    if processed_manifest["status"] != "preprocessed_and_checked":
        raise ValueError("processed event set is not validated")
    expected = processed_manifest["output_sha256"]["events.csv.gz"]
    if sha256(processed_root / "events.csv.gz") != expected:
        raise ValueError("processed events changed")
    patient_splits = {r["subject_id"]: r["patient_split"]
                      for r in rows(raw_root / "cohort/patient_splits.csv.gz")}
    eligible_records = {r["hadm_id"]: r for r in rows(raw_root / "cohort/admissions.csv.gz")
                        if r["discharge_landmark_eligible"] == "t"}
    eligible = {hadm: (r["subject_id"], patient_splits[r["subject_id"]])
                for hadm, r in eligible_records.items()}
    if len(eligible) != 19401:
        raise ValueError("unexpected discharge cohort")
    patients = {subject for subject, _ in eligible.values()}

    profiles: dict[str, dict[str, object]] = defaultdict(lambda: {
        "events": 0, "patients": set(), "admissions": set(), "train_admissions": set(),
        "sources": set(), "same_time_collision": 0, "same_source_collision": 0,
        "numeric_present": 0, "units": Counter(),
    })
    seen_time: set[tuple[str, str, str]] = set()
    seen_source: set[tuple[str, str]] = set()
    med_source_concepts: dict[str, set[str]] = defaultdict(set)
    for r in rows(processed_root / "events.csv.gz"):
        hadm, concept = r["hadm_id"], r["concept"]
        if hadm not in eligible or concept not in processed_manifest["concepts"]:
            raise ValueError("processed event outside frozen cohort/dictionary")
        subject, split = eligible[hadm]
        if r["subject_id"] != subject or r["patient_split"] != split:
            raise ValueError("processed event patient/split mismatch")
        p = profiles[concept]
        p["events"] += 1
        p["patients"].add(subject)
        p["admissions"].add(hadm)
        if split == "train":
            p["train_admissions"].add(hadm)
        p["sources"].add(r["source_id"])
        p["numeric_present"] += r["numeric_mask"] == "1"
        p["units"][r["value_unit"]] += 1
        time_key = (hadm, concept, r["charttime"])
        if time_key in seen_time:
            p["same_time_collision"] += 1
        seen_time.add(time_key)
        source_key = (concept, r["source_id"])
        if source_key in seen_source:
            p["same_source_collision"] += 1
        seen_source.add(source_key)
        if r["kind"] == "medication":
            med_source_concepts[r["source_id"]].add(concept)

    model_rows = []
    for concept in processed_manifest["concepts"]:
        p = profiles[concept]
        family = _family(concept)
        is_medication = family not in {"ph", "pco2", "po2", "bicarbonate", "wbc",
                                       "hemoglobin", "creatinine", "albumin"}
        model_rows.append({
            "concept": concept, "kind": "medication" if is_medication else "laboratory",
            "source": "raw/emar confirmed administrations + raw/emar_detail product/route" if is_medication else
                f"raw/labevents.itemid={LAB_ITEM_FOR_CONCEPT[family]}" +
                (" + specimen itemid=52033" if family in {"ph", "pco2", "po2"} else ""),
            "events": p["events"], "patients": len(p["patients"]),
            "admissions": len(p["admissions"]), "admission_coverage_pct": round(100 * len(p["admissions"]) / len(eligible), 2),
            "unobserved_admissions": len(eligible) - len(p["admissions"]),
            "train_admissions": len(p["train_admissions"]),
            "same_timestamp_collisions_not_proven_duplicates": p["same_time_collision"],
            "repeated_source_ids": p["same_source_collision"],
            "numeric_observations": p["numeric_present"],
            "unit_counts": json.dumps(p["units"], ensure_ascii=False),
            "clinical_rationale": RATIONALE.get(family, "需人工补充临床理由"),
        })

    lab_dictionary = {r["itemid"]: r for r in rows(raw_root / "raw/d_labitems.csv.gz")}
    lab_profiles: dict[str, dict[str, object]] = defaultdict(lambda: {
        "rows": 0, "numeric": 0, "admissions": set(), "patients": set(),
        "numeric_admissions": set(), "numeric_patients": set(), "units": Counter(),
        "same_time_collision": 0, "source_collision": 0})
    reviewed_times: set[tuple[str, str, str]] = set()
    reviewed_ids: set[str] = set()
    for r in rows(raw_root / "raw/labevents.csv.gz"):
        admission = eligible_records.get(r["hadm_id"])
        if admission is None or not (admission["admittime"] <= r["charttime"] <= r["storetime"] <= admission["dischtime"]):
            continue
        p = lab_profiles[r["itemid"]]
        p["rows"] += 1
        is_numeric = r["valuenum"] not in (NULL, "")
        p["numeric"] += is_numeric
        p["admissions"].add(r["hadm_id"])
        p["patients"].add(r["subject_id"])
        if is_numeric:
            p["numeric_admissions"].add(r["hadm_id"])
            p["numeric_patients"].add(r["subject_id"])
        p["units"][r["valueuom"]] += 1
        if r["itemid"] in REVIEW_LABS:
            time_key = (r["hadm_id"], r["itemid"], r["charttime"])
            p["same_time_collision"] += time_key in reviewed_times
            p["source_collision"] += r["labevent_id"] in reviewed_ids
            reviewed_times.add(time_key)
            reviewed_ids.add(r["labevent_id"])
    raw_labs = []
    for itemid, p in sorted(lab_profiles.items(), key=lambda x: -x[1]["rows"]):
        dictionary = lab_dictionary[itemid]
        raw_labs.append({
            "itemid": itemid, "label": dictionary["label"], "fluid": dictionary["fluid"],
            "category": dictionary["category"], "review_status":
                "current_input" if itemid in CURRENT_LABS else
                "clinical_candidate_not_in_input" if itemid in REVIEW_LABS else "unreviewed",
            "review_reason": REVIEW_LABS.get(itemid, ""), "raw_rows": p["rows"],
            "numeric_rows": p["numeric"], "patients": len(p["patients"]),
            "admissions": len(p["admissions"]),
            "numeric_patients": len(p["numeric_patients"]),
            "numeric_admissions": len(p["numeric_admissions"]),
            "numeric_unobserved_admissions": len(eligible) - len(p["numeric_admissions"]),
            "same_timestamp_collisions_not_proven_duplicates":
                p["same_time_collision"] if itemid in REVIEW_LABS else "",
            "repeated_source_ids": p["source_collision"] if itemid in REVIEW_LABS else "",
            "admission_coverage_pct": round(100 * len(p["admissions"]) / len(eligible), 2),
            "unit_counts": json.dumps(p["units"].most_common(5), ensure_ascii=False),
        })

    drug_profiles: dict[str, dict[str, object]] = defaultdict(lambda: {
        "rows": 0, "confirmed": 0, "accepted_sources": 0,
        "admissions": set(), "patients": set(), "mapped": set(), "actual_classes": set()})
    selected_emar_names: dict[str, str] = {}
    for r in rows(raw_root / "raw/emar.csv.gz"):
        hadm = r["hadm_id"]
        admission = eligible_records.get(hadm)
        if admission is None or not (admission["admittime"] <= r["charttime"] <= r["storetime"] <= admission["dischtime"]):
            continue
        name = r["medication"].strip()
        p = drug_profiles[name]
        p["rows"] += 1
        p["admissions"].add(hadm)
        p["patients"].add(r["subject_id"])
        p["mapped"].update(medication_classes(name) if name != NULL else ())
        p["confirmed"] += r["event_txt"] in {"Administered", "Started", "Applied", "Delayed Administered"}
        key = r["emar_id"] + ":" + r["emar_seq"]
        if key in med_source_concepts:
            selected_emar_names[key] = name
            p["accepted_sources"] += 1
            p["actual_classes"].update(med_source_concepts[key])
    if len(selected_emar_names) != len(med_source_concepts):
        raise ValueError("processed medication sources not linked to raw eMAR")
    drug_rows = []
    for name, p in sorted(drug_profiles.items(), key=lambda x: -x[1]["confirmed"]):
        brand_hit = any(brand in name.lower() for brand in BRAND_REVIEW)
        drug_rows.append({"medication_name": name, "emar_rows": p["rows"],
                          "confirmed_rows": p["confirmed"], "patients": len(p["patients"]),
                          "admissions": len(p["admissions"]), "name_only_classes": "|".join(sorted(p["mapped"])),
                          "accepted_emar_sources": p["accepted_sources"],
                          "actual_in_model_classes": "|".join(sorted(p["actual_classes"])),
                          "brand_review": int(brand_hit),
                          "potential_missing_brand": int(brand_hit and not p["actual_classes"])})

    source_routes: dict[str, set[str]] = defaultdict(set)
    for r in rows(raw_root / "raw/emar_detail.csv.gz"):
        key = r["emar_id"] + ":" + r["emar_seq"]
        if key in med_source_concepts and r["route"] not in (NULL, ""):
            source_routes[key].add(r["route"].strip().upper())
    route_by_concept: dict[str, Counter[str]] = defaultdict(Counter)
    name_by_concept: dict[str, Counter[str]] = defaultdict(Counter)
    for key, concepts in med_source_concepts.items():
        route = "|".join(sorted(source_routes[key])) if source_routes[key] else "UNKNOWN"
        for concept in concepts:
            route_by_concept[concept][route] += 1
            name_by_concept[concept][selected_emar_names[key]] += 1
    for record in model_rows:
        if record["concept"] in route_by_concept:
            record["routes_top10"] = json.dumps(route_by_concept[record["concept"]].most_common(10), ensure_ascii=False)
            record["drug_names_top10"] = json.dumps(name_by_concept[record["concept"]].most_common(10), ensure_ascii=False)
        else:
            record["routes_top10"] = ""
            record["drug_names_top10"] = ""

    static_profiles: dict[str, dict[str, object]] = defaultdict(lambda: {
        "rows": 0, "admissions": set(), "patients": set(), "positive_current": 0, "positive_prior": 0})
    for r in rows(structured_tokens):
        if r["hadm_id"] not in eligible:
            continue
        if r["token_type"] in {"laboratory", "administered_medication"}:
            continue
        key = r["token_type"] + ":" + r["concept"]
        p = static_profiles[key]
        p["rows"] += 1
        p["admissions"].add(r["hadm_id"])
        p["patients"].add(r["subject_id"])
        p["positive_current"] += r["mask0"] == "1" and float(r["value0"]) > 0
        p["positive_prior"] += r["mask1"] == "1" and float(r["value1"]) > 0
    static_rows = [{"concept": name, "source": _static_source(name),
                    "rows": p["rows"], "patients": len(p["patients"]),
                    "admissions": len(p["admissions"]), "unobserved_admissions": len(eligible) - len(p["admissions"]),
                    "positive_current_rows": p["positive_current"], "positive_prior_rows": p["positive_prior"],
                    "duplicate_same_admission_rows": p["rows"] - len(p["admissions"]),
                    "clinical_rationale": STATIC_RATIONALE.get(name, "需人工复核")}
                   for name, p in sorted(static_profiles.items())]

    text_profiles: dict[str, dict[str, object]] = defaultdict(lambda: {
        "chunks": 0, "admissions": set(), "patients": set()})
    for r in rows(shared_text):
        if r["hadm_id"] not in eligible:
            raise ValueError("text field outside frozen cohort")
        p = text_profiles[r["field"]]
        p["chunks"] += 1
        p["admissions"].add(r["hadm_id"])
        p["patients"].add(r["subject_id"])
    text_rows = [{"field": name, "source": "shared_text + discharge note field extraction",
                  "chunks": p["chunks"], "patients": len(p["patients"]),
                  "admissions": len(p["admissions"]), "unobserved_admissions": len(eligible) - len(p["admissions"]),
                  "multiple_chunks_above_one_per_admission": p["chunks"] - len(p["admissions"]),
                  "clinical_rationale": TEXT_RATIONALE.get(name, "需人工复核")}
                 for name, p in sorted(text_profiles.items())]

    output.mkdir(parents=True)
    _write_csv(output / "current_event_concepts.csv", tuple(model_rows[0]), model_rows)
    _write_csv(output / "all_raw_lab_items.csv", tuple(raw_labs[0]), raw_labs)
    _write_csv(output / "all_emar_medication_names.csv", tuple(drug_rows[0]), drug_rows)
    _write_csv(output / "current_static_concepts.csv", tuple(static_rows[0]), static_rows)
    _write_csv(output / "current_text_fields.csv", tuple(text_rows[0]), text_rows)
    summary = {"status": "audited", "eligible_admissions": len(eligible), "eligible_patients": len(patients),
               "current_event_concepts": len(model_rows), "raw_lab_items_with_rows": len(raw_labs),
               "raw_medication_names": len(drug_rows), "current_static_concepts": len(static_rows),
               "current_text_fields": len(text_rows),
               "unmapped_brand_names": [r for r in drug_rows if r["potential_missing_brand"] and r["confirmed_rows"] >= 10],
               "inputs_sha256": {"raw_manifest": sha256(raw_root / "metadata/manifest.json"),
                                 "processed_manifest": sha256(processed_root / "manifest.json"),
                                 "structured_tokens": sha256(structured_tokens), "shared_text": sha256(shared_text)}}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw_root", "processed_root", "structured_tokens", "shared_text", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.raw_root, args.processed_root, args.structured_tokens, args.shared_text, args.output)
    print(json.dumps({key: result[key] for key in ("status", "eligible_admissions", "current_event_concepts",
                                                    "raw_lab_items_with_rows", "raw_medication_names")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/audit_lab_abnormality_feasibility.py · lines 1–76

<!-- REVIEW_FRAGMENT {"path":"patient_data/audit_lab_abnormality_feasibility.py","part":1,"start_line":1,"end_line":76,"source_sha256":"4c85df3f96b781b6d5929f8ab6ff53e5e8a725d19ee2aabf346ab6a66de3206a","fragment_sha256":"4c85df3f96b781b6d5929f8ab6ff53e5e8a725d19ee2aabf346ab6a66de3206a","added_terminal_newline":false} -->
````python
"""核对当前已入模检验的参考区间与可用于恢复轨迹的重复观测。"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path


def rows(path: Path):
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def numeric(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def audit(processed: Path, raw: Path) -> dict[str, object]:
    accepted = {}
    for row in rows(processed / "events.csv.gz"):
        if row["kind"] == "lab":
            accepted[row["source_id"]] = (row["hadm_id"], row["concept"],
                                           float(row["elapsed_days"]), float(row["value_raw"]))
    result = defaultdict(Counter)
    by_stay = defaultdict(list)
    found = 0
    for row in rows(raw / "raw/labevents.csv.gz"):
        key = row["labevent_id"]
        if key not in accepted:
            continue
        hadm, concept, days, value = accepted[key]
        if row["hadm_id"] != hadm or numeric(row["valuenum"]) != value:
            raise ValueError(f"accepted/raw laboratory row disagrees: {key}")
        found += 1
        counts = result[concept]
        counts["observations"] += 1
        lower, upper = numeric(row["ref_range_lower"]), numeric(row["ref_range_upper"])
        if lower is not None and upper is not None and lower < upper:
            counts["valid_two_sided_reference"] += 1
            state = "low" if value < lower else "high" if value > upper else "normal"
            counts[f"reference_{state}"] += 1
        if row["flag"].strip():
            counts["nonempty_flag"] += 1
            counts["flag_" + row["flag"].strip().lower()] += 1
        by_stay[(hadm, concept)].append((days, value))
    if found != len(accepted):
        raise ValueError(f"only {found} of {len(accepted)} accepted laboratory IDs found")
    for (_, concept), observations in by_stay.items():
        counts = result[concept]
        counts["admissions_with_any"] += 1
        if len(observations) >= 2:
            counts["admissions_with_at_least_two"] += 1
            days = [day for day, _ in observations]
            if max(days) - min(days) >= 1:
                counts["admissions_with_24h_span"] += 1
    return {"accepted_lab_events": found,
            "concepts": {concept: dict(counts) for concept, counts in sorted(result.items())}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", required=True, type=Path)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(audit(args.processed, args.raw), ensure_ascii=False, indent=2),
                           encoding="utf-8")
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/audit_medication_effect_mapping.py · lines 1–76

<!-- REVIEW_FRAGMENT {"path":"patient_data/audit_medication_effect_mapping.py","part":1,"start_line":1,"end_line":76,"source_sha256":"c2b9659080e5ce453fa327d29fd4423a604f844b1f8c78b28fd2b824508234cb","fragment_sha256":"c2b9659080e5ce453fa327d29fd4423a604f844b1f8c78b28fd2b824508234cb","added_terminal_newline":false} -->
````python
"""审计药理映射的原始药名、途径可见性和类别；不修改源记录。"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def _rows(path: Path):
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def audit(raw: Path, processed: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((processed / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "preprocessed_and_checked" or manifest.get("medication_mapping") not in (
            "gold_review_v2", "clinical_reviewed_2026_09"):
        raise ValueError("expected revised, validated medication mapping")
    digest = hashlib.sha256((processed / "events.csv.gz").read_bytes()).hexdigest()
    if digest != manifest["output_sha256"]["events.csv.gz"]:
        raise ValueError("processed medication events changed after validation")
    accepted: dict[str, set[str]] = defaultdict(set)
    for row in _rows(processed / "events.csv.gz"):
        if row["kind"] == "medication":
            accepted[row["source_id"]].add(row["concept"])
    names: dict[str, tuple[str, str]] = {}
    for row in _rows(raw / "raw/emar.csv.gz"):
        key = row["emar_id"] + ":" + row["emar_seq"]
        if key in accepted:
            if key in names:
                raise ValueError("duplicate accepted eMAR key")
            names[key] = row["medication"], row["hadm_id"]
    if set(names) != set(accepted):
        raise ValueError("accepted eMAR sources missing from original data")
    routes: dict[str, set[str]] = defaultdict(set)
    for row in _rows(raw / "raw/emar_detail.csv.gz"):
        key = row["emar_id"] + ":" + row["emar_seq"]
        if key in names and row["route"] not in ("", "\\N"):
            routes[key].add(row["route"].strip().upper())
    counts: Counter[tuple[str, str]] = Counter()
    admissions: dict[tuple[str, str], set[str]] = defaultdict(set)
    missing_routes: Counter[tuple[str, str]] = Counter()
    for key, (name, hadm) in names.items():
        group = (name, "|".join(sorted(accepted[key])))
        counts[group] += 1
        admissions[group].add(hadm)
        missing_routes[group] += not routes[key]
    output.mkdir(parents=True)
    with (output / "accepted_medication_names.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("原始药名", "映射类别", "实际给药来源数", "覆盖住院数", "无途径记录来源数"))
        for group, count in counts.most_common():
            writer.writerow((*group, count, len(admissions[group]), missing_routes[group]))
    summary = {"mapping": manifest["medication_mapping"],
               "processed_events_sha256": manifest["output_sha256"]["events.csv.gz"],
               "accepted_unique_administrations": len(names),
               "accepted_names": len(counts), "administrations_without_route": sum(missing_routes.values()),
               "concept_events": dict(Counter(concept for classes in accepted.values() for concept in classes))}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.raw, args.processed, args.output), ensure_ascii=False))
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/audit_medication_formulations.py · lines 1–46

<!-- REVIEW_FRAGMENT {"path":"patient_data/audit_medication_formulations.py","part":1,"start_line":1,"end_line":46,"source_sha256":"8118c48c5e264bfaa036dc35bb46f482f4af43a3bec438e68b671c23a1b866b5","fragment_sha256":"8118c48c5e264bfaa036dc35bb46f482f4af43a3bec438e68b671c23a1b866b5","added_terminal_newline":false} -->
````python
"""聚合审查现行 eMAR 药名在详情表中的制剂和途径，不输出患者行。"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from patient_data.export_mimiciv import NULL, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    names = {}
    for row in rows(args.raw_root / "raw/emar.csv.gz"):
        if row["event_txt"] in {"Administered", "Started", "Applied", "Delayed Administered"}:
            names[row["emar_id"], row["emar_seq"]] = row["medication"].strip()
    combinations: Counter[tuple[str, str, str]] = Counter()
    per_source_seen: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    for row in rows(args.raw_root / "raw/emar_detail.csv.gz"):
        key = row["emar_id"], row["emar_seq"]
        if key not in names:
            continue
        route = row["route"].strip().upper() if row["route"] not in (NULL, "") else "UNKNOWN"
        product = row["product_description"].strip() if row["product_description"] not in (NULL, "") else "UNKNOWN"
        pair = route, product
        if pair not in per_source_seen[key]:
            combinations[names[key], route, product] += 1
            per_source_seen[key].add(pair)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("medication_name", "route", "product_description", "emar_sources"))
        for (name, route, product), count in sorted(combinations.items(), key=lambda kv: -kv[1]):
            writer.writerow((name, route, product, count))
    print(f"wrote {len(combinations)} aggregate formulations")


if __name__ == "__main__":
    main()
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/audit_saved_data.py · lines 1–49

<!-- REVIEW_FRAGMENT {"path":"patient_data/audit_saved_data.py","part":1,"start_line":1,"end_line":49,"source_sha256":"b79d0d99364a125ca922798d2f430541e6788a2ee94af9437a6a9e9117cc7da0","fragment_sha256":"b79d0d99364a125ca922798d2f430541e6788a2ee94af9437a6a9e9117cc7da0","added_terminal_newline":false} -->
````python
"""复核原始检验字段与标本可追溯性，仅输出聚合计数。"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from export_mimiciv import NULL, rows


def audit(root: Path) -> dict[str, object]:
    specimen_types: dict[tuple[str, str], str] = {}
    blood_gases: list[tuple[str, str, str]] = []
    items: Counter[str] = Counter()
    fields: set[str] = set()
    for row in rows(root / "raw/labevents.csv.gz"):
        fields.update(row)
        items[row["itemid"]] += 1
        key = (row["subject_id"], row["specimen_id"])
        if row["itemid"] == "52033" and row["value"] != NULL:
            value = row["value"].strip().upper()
            previous = specimen_types.get(key)
            # 一份标本类型冲突时不能按最后一条覆盖。
            specimen_types[key] = value if previous in (None, value) else "CONFLICT"
        elif row["itemid"] in ("50818", "50820", "50821"):
            blood_gases.append((row["itemid"], *key))
    required = {"labevent_id", "subject_id", "hadm_id", "specimen_id", "itemid",
                "charttime", "storetime", "value", "valuenum", "valueuom",
                "ref_range_lower", "ref_range_upper", "flag", "comments"}
    if not required.issubset(fields):
        raise ValueError("检验原始字段不完整")
    by_type: dict[str, Counter[str]] = {item: Counter() for item in ("50818", "50820", "50821")}
    for item, subject, specimen in blood_gases:
        by_type[item][specimen_types.get((subject, specimen), "UNKNOWN")] += 1
    if not specimen_types:
        raise ValueError("未保存标本类型，不能将血气视为动脉血气")
    result = {"status": "checked", "distinct_lab_items": len(items), "lab_records": sum(items.values()),
              "required_lab_fields_present": True, "blood_gas_specimen_type_counts": by_type,
              "note": "本报告仅审查原始底账，不清洗或重建模型输入；UNKNOWN/CONFLICT 必须由派生层明确处理。"}
    (root / "metadata/raw_lab_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.root), ensure_ascii=False))
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/audit_saved_text_assets.py · lines 1–39

<!-- REVIEW_FRAGMENT {"path":"patient_data/audit_saved_text_assets.py","part":1,"start_line":1,"end_line":39,"source_sha256":"245ec180c4251dbc3d317306bcde7ca407fd016358bb344a0f56401df629839e","fragment_sha256":"245ec180c4251dbc3d317306bcde7ca407fd016358bb344a0f56401df629839e","added_terminal_newline":false} -->
````python
"""只审计A/B旧向量索引覆盖，不将旧索引强行用作本次训练行号。"""
from collections import Counter
import argparse
import json
from pathlib import Path
import numpy as np
from patient_data.export_mimiciv import rows,sha256


def audit(state,index,output):
    if output.exists(): raise FileExistsError(output)
    with np.load(state) as z:
        ids=set(map(str,z['hadm_ids'])); splits=dict(zip(map(str,z['hadm_ids']),map(str,z['patient_splits'])))
    coverage=set(); counts=Counter(); sections=Counter(); disagreements=Counter()
    for r in rows(index):
        counts['all_rows']+=1
        if r['document_role']=='current_discharge':
            counts['current_discharge_rows']+=1
            if r['hadm_id'] in ids:
                coverage.add(r['hadm_id']); counts['matched_current_discharge_rows']+=1
                sections[r['section_role']]+=1
                disagreements[(splits[r['hadm_id']],r['data_split'])]+=1
    result=dict(status='index_coverage_audit_not_content_or_runtime_alignment',
        source_index_sha256=sha256(index),current_cohort_admissions=len(ids),
        matched_current_discharge_admissions=len(coverage),counts=dict(counts),sections=dict(sections),
        fold_pairs={str(k):v for k,v in disagreements.items()},
        limits=['Only role/current-hadm coverage checked; original document hashes not matched yet.',
            'Fold labels match on overlap, but full cohort/prefix/section representation differs; no controlled full-text comparison.',
            'History-role arrays cannot be placed into current B chunk indices.'])
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--state',type=Path,default=Path('patient_data/processed/v6_clinical_state_2026-10-09_r2/clinical_state.npz'))
    p.add_argument('--index',type=Path,default=Path('outputs/01a08b96-2988-7001-af1d-b40d901f3400/private/model_v2/qwen_section_embedding_index.csv.gz'))
    p.add_argument('--output',type=Path,default=Path('patient_data/text_encodings/A_full_document_sections/current_cohort_overlap_2026-10-11.json'))
    a=p.parse_args(); audit(a.state,a.index,a.output)
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/audit_v6_input_candidates.py · lines 1–237

<!-- REVIEW_FRAGMENT {"path":"patient_data/audit_v6_input_candidates.py","part":1,"start_line":1,"end_line":237,"source_sha256":"81ece27498740fe5f96d6011f269fff8f4281faa4d395571cbc31eebee49ad96","fragment_sha256":"81ece27498740fe5f96d6011f269fff8f4281faa4d395571cbc31eebee49ad96","added_terminal_newline":false} -->
````python
"""只读核对 V6 未用候选与复测时间；输出聚合统计，不导出患者文本。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
import re
import time

from patient_data.export_mimiciv import rows, sha256
from patient_data.preprocess_events import specimen_name


CANDIDATES = {
    "eosinophils_percent": ("51200",),
    "eosinophils_absolute": ("51199", "52073"),
    "neutrophils_percent": ("51256",),
    "neutrophils_absolute": ("52075",),
    "blood_gas_oxygen_saturation": ("50817",),
    "blood_gas_po2": ("50821",),
    "base_excess": ("50802",),
    "anion_gap": ("50868",),
    "chloride": ("50902",),
    "rdw": ("51277",),
    "hematocrit": ("51221",),
    "troponin": ("51002", "51003", "52642"),
}
TRACKS = {
    "arterial_ph": ("50820",), "arterial_pco2": ("50818",),
    "bicarbonate": ("50882",), "wbc": ("51301",),
    "albumin": ("50862",), "hemoglobin": ("51222",),
    "creatinine": ("50912",), "sodium": ("50983",),
    "potassium": ("50971",), "urea_nitrogen": ("51006",),
    "glucose": ("50931",), "platelet": ("51265",),
    "magnesium": ("50960",), "lactate": ("50813",),
    "crp": ("50889",), "ntprobnp": ("50963",),
}
# 只识别检查名称/标题，不依据 body 中“建议未来CT”判定当前确有CT。
CHEST_CT = re.compile(
    r"(?:\bCT\s+(?:OF\s+(?:THE\s+)?)?(?:CHEST|THORAX)\b|"
    r"\bCHEST\s+CT\b|\bCTA\s+(?:CHEST|PULMONARY)\b|"
    r"\bCT\s+(?:PULMONARY\s+)?ANGIOGRAM\b)", re.I)


def number(value: str) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def measurement_stats(samples, admissions, subjects):
    observed = len(samples)
    repeated = late_absent = early_and_late = 0
    for hadm, timestamps in samples.items():
        start, end = admissions[hadm]
        ordered = sorted(timestamps)
        repeated += len(ordered) >= 2
        late_absent += (end - ordered[-1]).total_seconds() > 48 * 3600
        early_and_late += ((ordered[0] - start).total_seconds() <= 24 * 3600
                           and (end - ordered[-1]).total_seconds() <= 48 * 3600)
    return {"observed_admissions": observed,
            "observed_patients": len({subjects[h] for h in samples}),
            "two_distinct_timestamps": repeated,
            "last_measurement_older_than_48h": late_absent,
            "first_in_24h_and_last_in_48h": early_and_late}


def audit(snapshot: Path, labels: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    manifest = json.loads((snapshot / "metadata/manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "validated":
        raise ValueError("snapshot not validated")
    admissions, subjects = {}, {}
    for row in rows(snapshot / "cohort/admissions.csv.gz"):
        if row["discharge_landmark_eligible"].lower() not in ("1", "t", "true"):
            continue
        hadm = row["hadm_id"]
        admissions[hadm] = (datetime.fromisoformat(row["admittime"]),
                            datetime.fromisoformat(row["dischtime"]))
        subjects[hadm] = row["subject_id"]
    if len(admissions) != 19401:
        raise ValueError("eligible cohort drift")
    dictionary = {r["itemid"]: r for r in rows(snapshot / "raw/d_labitems.csv.gz")}
    groups = {**CANDIDATES, **TRACKS}
    procalcitonin = tuple(k for k, v in dictionary.items()
                         if "procalciton" in v["label"].lower())
    if procalcitonin:
        groups["procalcitonin"] = procalcitonin
    by_item = defaultdict(list)
    for name, items in groups.items():
        for item in items:
            by_item[item].append(name)
    measurements = {name: defaultdict(set) for name in groups}
    units = {name: Counter() for name in groups}
    rejects = {name: Counter() for name in groups}
    specimen_types = defaultdict(set)
    gas = []
    raw_count = 0
    for row in rows(snapshot / "raw/labevents.csv.gz"):
        raw_count += 1
        hadm, item = row["hadm_id"], row["itemid"]
        if hadm not in admissions or (item not in by_item and item != "52033"):
            continue
        start, end = admissions[hadm]
        try:
            chart = datetime.fromisoformat(row["charttime"])
            store = datetime.fromisoformat(row["storetime"])
            available = start <= chart <= store <= end
        except ValueError:
            available = False
        if item == "52033":
            if available:
                specimen_types[(hadm, row["specimen_id"])].add(specimen_name(row["value"]))
            continue
        for name in by_item[item]:
            if not available:
                rejects[name]["time_unavailable"] += 1
                continue
            if number(row["valuenum"]) is None:
                rejects[name]["no_finite_numeric_value"] += 1
                continue
            if row["valueuom"] in ("", "\\N"):
                rejects[name]["no_unit"] += 1
                continue
            if name in ("arterial_ph", "arterial_pco2"):
                gas.append((name, hadm, row["specimen_id"], chart, row["valueuom"]))
            else:
                measurements[name][hadm].add(chart)
                units[name][row["valueuom"]] += 1
    if raw_count != manifest["sources"]["raw/labevents"]["rows"]:
        raise ValueError("raw laboratory row count drift")
    for name, hadm, specimen, chart, unit in gas:
        types = specimen_types.get((hadm, specimen), set())
        if types == {"arterial"}:
            measurements[name][hadm].add(chart)
            units[name][unit] += 1
        else:
            rejects[name]["not_confirmed_arterial_or_conflict"] += 1
    print(json.dumps({"stage": "raw_labs_scanned", "rows": raw_count}), flush=True)
    if not measurements["arterial_ph"] or not measurements["arterial_pco2"]:
        raise ValueError("arterial mapping missing despite existing validated arterial inputs")
    lab_results = {}
    for name, items in groups.items():
        lab_results[name] = {
            "items": [dictionary[item] for item in items],
            "units_rows": dict(units[name]), "excluded_rows": dict(rejects[name]),
            **measurement_stats(measurements[name], admissions, subjects)}
    ct_hadms, ct_subjects = set(), set()
    ct_count = report_count = 0
    for row in rows(snapshot / "raw/radiology.csv.gz"):
        report_count += 1
        hadm = row["hadm_id"]
        if hadm not in admissions:
            continue
        start, end = admissions[hadm]
        try:
            if not start <= datetime.fromisoformat(row["charttime"]) <= datetime.fromisoformat(row["storetime"]) <= end:
                continue
        except ValueError:
            continue
        # 常见当前检查标题在全文开头；启发式候选，不宣称临床真值/召回率。
        if CHEST_CT.search(row["text"][:900]):
            ct_count += 1
            ct_hadms.add(hadm); ct_subjects.add(subjects[hadm])
    if report_count != manifest["sources"]["raw/radiology"]["rows"]:
        raise ValueError("raw radiology row count drift")
    splits = {r["subject_id"]: r["patient_split"]
              for r in rows(snapshot / "cohort/patient_splits.csv.gz")}
    boundaries = [0, 7, 14, 30, 60, 90, 180, 365]
    events = {split: {str(cause): [0] * 7 for cause in (1, 2)}
              for split in ("train", "validation", "test")}
    target_sets = {split: [set() for _ in range(7)] for split in events}
    label_rows = list(rows(labels))
    if {r["hadm_id"] for r in label_rows} != set(admissions) or len(label_rows) != len(admissions):
        raise ValueError("strict label membership drift")
    for row in label_rows:
        cause, day = int(row["event_type"]), float(row["time_days"])
        if cause == 0:
            continue
        # 与候选模型一致，段为 [a,b)，365 位于末段；同日 0 在首段。
        k = min(sum(day >= edge for edge in boundaries[1:-1]), 6)
        split = splits[row["subject_id"]]
        events[split][str(cause)][k] += 1
        if cause == 1:
            target = row["next_copd_hadm_id"]
            if target in ("", "\\N"):
                raise ValueError("positive readmission label has no target admission")
            target_sets[split][k].add(target)
    result = {
        "status": "aggregate_readonly_input_audit", "eligible_admissions": len(admissions),
        "eligible_patients": len(set(subjects.values())),
        "source_manifest_sha256": sha256(snapshot / "metadata/manifest.json"),
        "source_file_sha256_verified": {
            name: sha256(snapshot / (name + ".csv.gz")) == manifest["sources"][name]["sha256"]
            for name in ("raw/labevents", "raw/radiology", "raw/d_labitems")},
        "strict_labels_sha256": sha256(labels),
        "lab_scan_rows": raw_count, "lab_candidates_and_tracks": lab_results,
        "procalcitonin_dictionary_items": list(procalcitonin),
        "chest_ct_title_candidate": {"all_radiology_rows": report_count,
            "candidate_reports": ct_count, "candidate_admissions": len(ct_hadms),
            "candidate_patients": len(ct_subjects),
            "method": "first 900 chars title regex; chart/store available by discharge; not adjudicated CT truth"},
        "strict_event_counts_by_time_segment": events,
        "distinct_readmission_targets_per_segment": {
            split: [len(targets) for targets in bins] for split, bins in target_sets.items()},
        "distinct_readmission_targets_per_split": {
            split: len(set().union(*bins)) for split, bins in target_sets.items()},
        "segment_boundaries": boundaries,
        "limitations": ["Numeric/time/unit presence is not full model inclusion QA",
                        "Candidate lab codes not equivalent to approved predictors",
                        "Repeat timestamp counts do not prove abnormality or recovery",
                        "One future admission can label multiple landmarks and appear in different time segments",
                        "Whole dataset coverage is not supervised validation evidence",
                        "No live PostgreSQL access or microbiology export in this audit"],
        "elapsed_seconds": round(time.perf_counter() - started, 2)}
    if not all(result["source_file_sha256_verified"].values()):
        raise ValueError("source SHA mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(output),
                      "seconds": result["elapsed_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.snapshot, args.labels, args.output)
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/build_feature_catalog.py · lines 1–100

<!-- REVIEW_FRAGMENT {"path":"patient_data/build_feature_catalog.py","part":1,"start_line":1,"end_line":100,"source_sha256":"e01de9752699ba657bc18a3003158f9265eddbb75c56f4393e377f0a93bb3fa1","fragment_sha256":"e01de9752699ba657bc18a3003158f9265eddbb75c56f4393e377f0a93bb3fa1","added_terminal_newline":false} -->
````python
"""把已验证的事件、静态和文本审计合成一张中文概念总表。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SPECIMEN_ZH = {"arterial": "动脉血", "venous": "静脉血", "central_venous": "中心静脉血",
               "mixed": "混合标本", "unknown": "标本未确定"}
EVENT_ZH = {
    "ph": "酸碱度 pH", "pco2": "二氧化碳分压 pCO₂", "po2": "氧分压 pO₂",
    "albumin": "白蛋白", "bicarbonate": "碳酸氢盐", "creatinine": "肌酐",
    "hemoglobin": "血红蛋白", "wbc": "白细胞计数",
    "saba": "短效 β₂ 激动剂 SABA", "sama": "短效抗胆碱药 SAMA",
    "laba": "长效 β₂ 激动剂 LABA", "lama": "长效抗胆碱药 LAMA",
    "inhaled_steroid": "吸入糖皮质激素", "systemic_steroid": "全身糖皮质激素",
    "macrolide": "大环内酯类抗菌药", "beta_lactam": "β-内酰胺类抗菌药",
    "fluoroquinolone": "氟喹诺酮类抗菌药", "other_antibiotic": "其他抗菌药（异质组）",
    "loop_diuretic": "袢利尿剂", "other_diuretic": "其他利尿剂（异质组）",
}
STATIC_ZH = {
    "context:age_at_admission": "本次入院年龄", "context:male": "男性",
    "context:prior_coded_admissions": "既往符合编码的住院次数",
    "context:prior_strict_aecopd_admissions": "既往严格急性加重编码住院次数",
    "aecopd_index_code:strict_aecopd": "本次严格 AECOPD 编码",
    "chronic_comorbidity:heart_failure": "心力衰竭共病",
    "chronic_comorbidity:cardiac_other": "其他心脏共病",
    "chronic_comorbidity:ckd": "慢性肾病共病",
    "chronic_comorbidity:diabetes": "糖尿病共病",
    "chronic_comorbidity:malignancy": "恶性肿瘤共病",
    "chronic_comorbidity:chronic_respiratory": "其他慢性呼吸共病",
    "acute_diagnosis:acute_resp_failure": "本次急性呼吸衰竭",
    "acute_diagnosis:pneumonia_or_aspiration": "本次肺炎或误吸",
    "acute_diagnosis:sepsis": "本次脓毒症",
    "acute_diagnosis:aki": "本次急性肾损伤",
    "disposition:home": "出院去向：家庭",
    "disposition:facility": "出院去向：机构",
    "disposition:other": "出院去向：其他",
}
TEXT_ZH = {
    "history_of_present_illness": "现病史文本", "past_medical_history": "既往史文本",
    "hospital_course": "住院经过文本", "discharge_condition": "出院时状况文本",
}
FIELDS = ("类别", "中文临床概念", "模型字段", "原始来源", "覆盖人数", "覆盖住院数",
          "未观察住院数", "原始行或片段数", "同刻碰撞或多片段数", "重复原始ID或同住院重复行数",
          "临床理由", "计数解释")


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build(directory: Path) -> Path:
    output = directory / "current_concept_catalog_zh.csv"
    if output.exists():
        raise FileExistsError(output)
    result = []
    for row in _read(directory / "current_event_concepts.csv"):
        concept = row["concept"]
        family, _, specimen = concept.partition(":")
        if family not in EVENT_ZH or (specimen and specimen not in SPECIMEN_ZH):
            raise ValueError(f"untranslated event concept: {concept}")
        name = SPECIMEN_ZH[specimen] + EVENT_ZH[family] if specimen else EVENT_ZH[family]
        result.append(("检验事件" if row["kind"] == "laboratory" else "实际给药事件",
                       name, concept, row["source"], row["patients"], row["admissions"],
                       row["unobserved_admissions"], row["events"],
                       row["same_timestamp_collisions_not_proven_duplicates"], row["repeated_source_ids"],
                       row["clinical_rationale"], "同刻多条未证实重复；复方可跨类别映射"))
    for row in _read(directory / "current_static_concepts.csv"):
        concept = row["concept"]
        if concept not in STATIC_ZH:
            raise ValueError(f"untranslated static concept: {concept}")
        result.append(("结构化静态", STATIC_ZH[concept], concept, row["source"], row["patients"],
                       row["admissions"], row["unobserved_admissions"], row["rows"], "不适用",
                       row["duplicate_same_admission_rows"], row["clinical_rationale"],
                       "既往次数为数值，诊断及去向为类别；零值不等于缺失"))
    for row in _read(directory / "current_text_fields.csv"):
        field = row["field"]
        if field not in TEXT_ZH:
            raise ValueError(f"untranslated text field: {field}")
        result.append(("出院小结文本", TEXT_ZH[field], field, row["source"], row["patients"],
                       row["admissions"], row["unobserved_admissions"], row["chunks"],
                       row["multiple_chunks_above_one_per_admission"], "不适用", row["clinical_rationale"],
                       "同住院多片段是分块，不是重复文书"))
    if len(result) != 54:
        raise ValueError(f"unexpected concept count: {len(result)}")
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        writer.writerows(result)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_dir", type=Path)
    print(build(parser.parse_args().audit_dir))
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/clinical_combinations.py · lines 1–133

<!-- REVIEW_FRAGMENT {"path":"patient_data/clinical_combinations.py","part":1,"start_line":1,"end_line":133,"source_sha256":"8714f65bf5a01ebf2807f6669a3422e9c3948c4d83e88f2e7279e566ab38016b","fragment_sha256":"8714f65bf5a01ebf2807f6669a3422e9c3948c4d83e88f2e7279e566ab38016b","added_terminal_newline":false} -->
````python
"""同采样CBC的原成分/NLR/SII；只读冻结底账，不读结局，训练折定尺度。"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
import numpy as np
from patient_data.export_mimiciv import rows, sha256
from patient_data.v6_lab_state import summarize
from training.v5.numeric_preprocessing import robust_parameters

NAMES = ('neutrophil_fraction', 'lymphocyte_fraction', 'platelets_k_per_ul', 'nlr', 'sii_k_per_ul')
ITEMS = {'51256': 'npct', '51244': 'lpct', '52075': 'nabs', '51133': 'labs', '51265': 'plt'}


def ratios(neutrophils, lymphocytes, platelets=None):
    if not all(math.isfinite(x) and x >= 0 for x in (neutrophils, lymphocytes)) or lymphocytes <= 0:
        return None, None
    ratio = neutrophils/lymphocytes
    sii = platelets*ratio if platelets is not None and math.isfinite(platelets) and platelets >= 0 else None
    if not math.isfinite(ratio): return None, None
    if sii is not None and not math.isfinite(sii): sii = None
    return ratio, sii


def scale(raw, splits):
    out = raw.copy(); parameters = {}
    for a, name in enumerate(NAMES):
        for j in range(6):
            valid = raw[:, a, 6] > 0
            if j in (3, 5): valid = valid & (raw[:, a, 7] > 0)
            value = np.log1p(raw[:, a, j]) if j < 4 else raw[:, a, j]
            train = valid & (splits == 'train')
            if train.any():
                p = robust_parameters(value[train]); parameters[f'{name}/{j}'] = p
                out[:, a, j] = np.where(valid, np.clip((value-p['median'])/p['scale'], -8, 8), 0)
            else:
                parameters[f'{name}/{j}'] = {'training_observations': 0}
                out[:, a, j] = 0
    return out.astype(np.float32), parameters


def build(snapshot, state_dir, output):
    if output.exists(): raise FileExistsError(output)
    sm = json.loads((snapshot/'metadata/manifest.json').read_text())
    if sm['status'] != 'validated': raise ValueError('unvalidated snapshot')
    hashes = {}
    for name in ('raw/labevents', 'raw/d_labitems', 'cohort/admissions'):
        path = snapshot/(name+'.csv.gz'); hashes[name] = sha256(path)
        if hashes[name] != sm['sources'][name]['sha256']: raise ValueError('raw source changed')
    manifest = json.loads((state_dir/'manifest.json').read_text())
    if sha256(state_dir/'clinical_state.npz') != manifest['output_sha256']['clinical_state.npz']:
        raise ValueError('state changed')
    with np.load(state_dir/'clinical_state.npz') as z:
        ids = {k: z[k] for k in ('hadm_ids', 'subject_ids', 'patient_splits')}
    positions = {str(h): i for i, h in enumerate(ids['hadm_ids'])}
    adm = {r['hadm_id']: (datetime.fromisoformat(r['admittime']), datetime.fromisoformat(r['dischtime']), r['subject_id'])
           for r in rows(snapshot/'cohort/admissions.csv.gz') if r['hadm_id'] in positions}
    labs = defaultdict(lambda: defaultdict(list)); counts = Counter()
    for r in rows(snapshot/'raw/labevents.csv.gz'):
        if r['itemid'] not in ITEMS or r['hadm_id'] not in adm: continue
        start, end, subject = adm[r['hadm_id']]
        if r['subject_id'] != subject: raise ValueError('identity mismatch')
        try: t = datetime.fromisoformat(r['charttime']); value = float(r['valuenum'])
        except (ValueError, TypeError): counts['missing_numeric_or_time'] += 1; continue
        if not start <= t <= end or not math.isfinite(value) or value < 0: counts['invalid_time_value'] += 1; continue
        if '<' in r.get('value', '') or '>' in r.get('value', ''):
            counts['quantification_bound_excluded'] += 1; continue
        unit = r['valueuom'].strip().lower().replace(' ', '').replace('μ', 'u').replace('µ', 'u')
        key = ITEMS[r['itemid']]
        if (key.endswith('pct') and (unit != '%' or value > 100)) or (not key.endswith('pct') and unit not in ('k/ul', '10^3/ul')):
            counts['invalid_unit_or_percent'] += 1; continue
        specimen = r['specimen_id'].strip()
        if specimen in ('', '\\N'): counts['missing_specimen_excluded'] += 1; continue
        labs[(r['hadm_id'], t, specimen)][key].append(value)
    samples = defaultdict(list); pairing = Counter()
    for (h, t, specimen), values in labs.items():
        v = {k: float(np.median(x)) for k, x in values.items()}
        counts['duplicate_same_item_sample'] += sum(len(x)-1 for x in values.values())
        if 'npct' in v: samples[h, 0].append(((t-adm[h][0]).total_seconds()/86400, v['npct']/100))
        if 'lpct' in v: samples[h, 1].append(((t-adm[h][0]).total_seconds()/86400, v['lpct']/100))
        if 'plt' in v: samples[h, 2].append(((t-adm[h][0]).total_seconds()/86400, v['plt']))
        if 'nabs' in v and 'labs' in v:
            n, l, source = v['nabs'], v['labs'], 'absolute'
        elif 'npct' in v and 'lpct' in v:
            n, l, source = v['npct'], v['lpct'], 'percentage'
        else: pairing['no_paired_differential'] += 1; continue
        nlr, sii = ratios(n, l, v.get('plt'))
        day = (t-adm[h][0]).total_seconds()/86400
        if nlr is not None: samples[h, 3].append((day, nlr)); pairing[source+'_valid_nlr'] += 1
        else: pairing['zero_lymphocyte_denominator_excluded'] += 1
        if sii is not None: samples[h, 4].append((day, sii)); pairing[source+'_valid_sii'] += 1
    raw = np.zeros((len(positions), 5, 9))
    for (h, a), values in samples.items():
        duration = (adm[h][1]-adm[h][0]).total_seconds()/86400
        raw[positions[h], a] = summarize(values, duration)[0]
    # 原成分对照同时包含绝对分类的N/L，避免比值臂独享绝对数来源。
    # 百分比与绝对数不能相加混作同一概念；用另行可核验的absolute ratio输入替代会混淆。
    # 因此比值仅在同标本有百分比成分时使用，绝对源只用于配对一致性统计。
    pct_samples = defaultdict(list)
    for (h, t, specimen), values in labs.items():
        v = {k: float(np.median(x)) for k, x in values.items()}
        if 'npct' not in v or 'lpct' not in v: continue
        nlr, sii = ratios(v['npct'], v['lpct'], v.get('plt'))
        day = (t-adm[h][0]).total_seconds()/86400
        if nlr is not None: pct_samples[h, 3].append((day, nlr))
        if sii is not None: pct_samples[h, 4].append((day, sii))
    raw[:, 3:] = 0
    for (h, a), values in pct_samples.items(): raw[positions[h], a] = summarize(values, (adm[h][1]-adm[h][0]).total_seconds()/86400)[0]
    scaled, parameters = scale(raw, ids['patient_splits'])
    output.mkdir(parents=True)
    np.savez_compressed(output/'combinations.npz', **ids, values=scaled, raw=raw, names=np.asarray(NAMES))
    report = dict(status='combinations_source_checked', source_hashes=hashes,
                  state_manifest_sha256=sha256(state_dir/'manifest.json'), names=NAMES, parameters=parameters,
                  counts=dict(counts), pairing=dict(pairing),
                  coverage={name: {'admissions': int((raw[:, j, 6] > 0).sum()),
                                  'patients': len(set(ids['subject_ids'][raw[:, j, 6] > 0].tolist())),
                                  'by_fold': {f: int(((ids['patient_splits']==f)&(raw[:, j, 6] > 0)).sum()) for f in ('train','validation','test')}} for j,name in enumerate(NAMES)},
                  formula='NLR=neutrophil_percent/lymphocyte_percent; SII=platelets(K/uL)*NLR; exact charttime AND specimen_id',
                  input_contract='5 axes x 9: first/max/latest/max-minus-latest/log_last_age/log_peak_gap/observed/recheck/latest_at_max; maxima are not clinical abnormality thresholds',
                  limits=['Not causal recovery or baseline inflammation; steroids/infection affect differential',
                          'No imputation of absent ratios; ratios restricted to same percentage-source components used by control',
                          'Exact same specimen/time is conservative, not all CBCs; no cross-day division'],
                  output_sha256=sha256(output/'combinations.npz'))
    (output/'manifest.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report['coverage']), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('snapshot', 'state_dir', 'output'): p.add_argument('--'+name.replace('_','-'), type=Path, required=True)
    a = p.parse_args(); build(a.snapshot, a.state_dir, a.output)
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/clinical_facts/__init__.py · lines 1–1

<!-- REVIEW_FRAGMENT {"path":"patient_data/clinical_facts/__init__.py","part":1,"start_line":1,"end_line":1,"source_sha256":"72c755eaedee989d8f463f1c06ed999e18ab51106c4e99d88a805b2d942babdf","fragment_sha256":"72c755eaedee989d8f463f1c06ed999e18ab51106c4e99d88a805b2d942babdf","added_terminal_newline":false} -->
````python
"""V6 出院前影像与有证据临床事实；原始底账只读。"""
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/clinical_facts/audit.py · lines 1–96

<!-- REVIEW_FRAGMENT {"path":"patient_data/clinical_facts/audit.py","part":1,"start_line":1,"end_line":96,"source_sha256":"cdd66ad75bc8f86391a749564d6d5519365d6afee51b74dcbbe885973c9140cf","fragment_sha256":"cdd66ad75bc8f86391a749564d6d5519365d6afee51b74dcbbe885973c9140cf","added_terminal_newline":false} -->
````python
"""直接回查原始快照与全部报告/字符/成员，不把解析率当临床准确度。"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from patient_data.export_mimiciv import rows,sha256
from patient_data.clinical_facts.llm import digest
from patient_data.cxr_report_selection import usable
from patient_data.clinical_facts.cxr_trajectory import SELECTED

def audit_cxr(args):
    m=json.loads((args.cxr/'manifest.json').read_text())
    if m['status']!='all_cxr_labeled_and_aggregated': raise ValueError('not completed')
    for name,value in m['output_sha256'].items():
        if sha256(args.cxr/name)!=value: raise ValueError('changed CXR files')
    raw={r['note_id']:r for r in rows(args.snapshot/'raw/radiology.csv.gz')}
    admissions={r['hadm_id']:r for r in rows(args.snapshot/'cohort/admissions.csv.gz')}
    with (args.cxr/'labeled_reports.jsonl').open(encoding='utf-8') as f: reports=list(map(json.loads,f))
    grouped=defaultdict(list); repeats=0
    for r in reports:
        source=raw[r['note_id']]
        if any(r[k]!=source[k] for k in ('hadm_id','subject_id','charttime','storetime')): raise ValueError('raw identity mismatch')
        if digest(source['text'])!=r['original_sha256'] or not usable(source,admissions[r['hadm_id']])[0]: raise ValueError('raw source/time failed')
        for s in r['sections']:
            if source['text'][s['start_char']:s['end_char']]!=s['text']: raise ValueError('source span mismatch')
        if r['text']!='\n'.join(s['text'] for s in r['sections']): raise ValueError('combined report differs')
        end=0
        for chunk in r['chunks']:
            if chunk['start_char']!=end or chunk['end_char']<=end or chunk['tokens']>512: raise ValueError('lost or truncated chunk')
            end=chunk['end_char']
        if end!=len(r['text']): raise ValueError('missing report tail')
        if not all(v in (0,1,2,3) for v in r['labels14'].values()): raise ValueError('invalid labels')
        grouped[r['hadm_id']].append(r)
    with np.load(args.cxr/'cxr_trajectory.npz') as f: a={k:f[k] for k in f.files}
    for i,(h,s) in enumerate(zip(a['hadm_ids'],a['subject_ids'],strict=True)):
        rs=grouped.get(str(h),[]); expected=bool(rs)
        if a['cxr_present'][i]!=expected: raise ValueError('availability incorrectly includes absent reports')
        if not rs:
            if a['cxr_values'][i].any(): raise ValueError('absent report has nonzero input')
            continue
        if any(r['subject_id']!=str(s) or r['patient_split']!=a['patient_splits'][i] for r in rs): raise ValueError('patient/split failed')
        times=sorted({r['charttime'] for r in rs}); repeats+=len(times)>1
        x=a['cxr_values'][i].reshape(7,14)
        for j,field in enumerate(SELECTED):
            sequence=[]
            for t in times:
                states={r['labels14'][field] for r in rs if r['charttime']==t}-{0}
                code=3 if {1,2}<=states else 1 if 1 in states else 3 if 3 in states else 2 if 2 in states else 0
                sequence.append(code)
            if x[j,sequence[0]]!=1 or x[j,4+sequence[-1]]!=1 or x[j,:4].sum()!=1 or x[j,4:8].sum()!=1:
                raise ValueError('first/latest four-state vector failed')
            neg=any(sequence[k]==2 and 1 in sequence[:k] for k in range(len(sequence)))
            if x[j,10]!=neg or x[j,8]!=(1 in sequence): raise ValueError('temporal change failed')
    if len(reports)!=m['reports'] or int(a['cxr_present'].sum())!=m['admissions_with_cxr']: raise ValueError('coverage differs')
    result={'status':'all_CXR_raw_source_and_trajectory_checked','reports':len(reports),'admissions_with_cxr':len(grouped),
            'repeat_timestamp_admissions':repeats,'source_chars_time_members_masks_checked':True,
            'semantic_accuracy':'not an expert gold standard','manifest_sha256':sha256(args.cxr/'manifest.json')}
    (args.cxr/'independent_source_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result),flush=True)

def audit_tasks(args):
    m=json.loads((args.tasks/'manifest.json').read_text())
    for name,value in m['output_sha256'].items():
        if sha256(args.tasks/name)!=value: raise ValueError('task/source changed')
    with (args.tasks/'sources.jsonl').open(encoding='utf-8') as f: sources={r['note_id']:r for r in map(json.loads,f)}
    verified=0
    for name in ('radiology','discharge'):
        for raw in rows(args.snapshot/('raw/'+name+'.csv.gz')):
            if raw['note_id'] not in sources: continue
            source=sources[raw['note_id']]
            if any(source[k]!=raw[k] for k in ('subject_id','hadm_id','charttime','storetime','text')):
                raise ValueError('task reference is not complete original note')
            if digest(raw['text'])!=source['original_sha256']: raise ValueError('original note hash failed')
            verified+=1
    if verified!=len(sources): raise ValueError('source note missing from frozen snapshot')
    total=0; maps=0; seen=set()
    with (args.tasks/'tasks.jsonl').open(encoding='utf-8') as f:
        for task in map(json.loads,f):
            if task['task_key'] in seen: raise ValueError('duplicate task key')
            seen.add(task['task_key']); source=sources[task['note_id']]
            for mapping in task['source_maps']:
                actual=source['text'][mapping['original_start']:mapping['original_end']]
                if actual!=task['text'][mapping['input_start']:mapping['input_end']]: raise ValueError('input/original offset failed')
                maps+=1
            total+=1
    if total!=m['tasks']: raise ValueError('task count differs')
    result={'status':'all_tasks_original_notes_and_offsets_checked','original_notes':verified,'tasks':total,'mapped_spans':maps,
            'manifest_sha256':sha256(args.tasks/'manifest.json'),'limits':'Source correctness, not semantic extraction accuracy'}
    (args.tasks/'independent_source_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',type=Path,default=Path('patient_data/mimiciv_copd/snapshot_2026-09-27'))
    p.add_argument('--cxr',type=Path,default=Path('patient_data/processed/v6_cxr_trajectory_2026-10-10_r2'))
    p.add_argument('--tasks',type=Path); args=p.parse_args()
    audit_tasks(args) if args.tasks else audit_cxr(args)
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/clinical_facts/audit_radiology_comparison.py · lines 1–79

<!-- REVIEW_FRAGMENT {"path":"patient_data/clinical_facts/audit_radiology_comparison.py","part":1,"start_line":1,"end_line":79,"source_sha256":"9b163613b25bbc3227643c9f5271cba5465a7d3e961ea5e8a0d453e5dc4c517b","fragment_sha256":"9b163613b25bbc3227643c9f5271cba5465a7d3e961ea5e8a0d453e5dc4c517b","added_terminal_newline":false} -->
````python
"""影像比较的独立原文、时间、成员和张量回查；不冒充医学准确率。"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np
from patient_data.export_mimiciv import rows, sha256
from patient_data.clinical_facts.llm import digest
from patient_data.cxr_report_selection import usable
from patient_data.clinical_facts.radiology_compare import records
from patient_data.clinical_facts.cxr_trajectory import SELECTED, aggregate
from patient_data.clinical_facts.features import ct_vector


def audit(args):
    data, members, selection = records(args.selected)
    sources = {r['note_id']: r for r in rows(args.snapshot / 'raw/radiology.csv.gz')}
    admissions = {r['hadm_id']: r for r in rows(args.snapshot / 'cohort/admissions.csv.gz')}
    by_id = {r['note_id']: r for r in data}; index = {str(h): i for i, h in enumerate(members['hadm_ids'])}
    report = {}; matched_presence = None
    for name, folder in (('radbert', args.radbert), ('radgraph', args.radgraph)):
        m = json.loads((folder / 'manifest.json').read_text())
        if m['status'] != 'full_radiology_processed': raise ValueError('unfinished imaging')
        for file, checksum in m['output_sha256'].items():
            if sha256(folder / file) != checksum: raise ValueError('processed input changed')
        with (folder / 'reports.jsonl').open(encoding='utf-8') as f: predictions = list(map(json.loads, f))
        if len(predictions) != len(by_id) or {r['note_id'] for r in predictions} != by_id.keys(): raise ValueError('report set differs')
        a = dict(np.load(folder / 'imaging.npz')); expected = np.zeros_like(a['present']); evidence = []; groups = defaultdict(list)
        for key in ('hadm_ids', 'subject_ids', 'patient_splits'):
            if not np.array_equal(a[key], members[key]): raise ValueError('members or splits changed')
        for p in predictions:
            source = sources[p['note_id']]; selected = by_id[p['note_id']]
            if digest(source['text']) != p['original_sha256'] or not usable(source, admissions[p['hadm_id']])[0]: raise ValueError('raw source or availability differs')
            if any(p[k] != selected[k] for k in ('hadm_id', 'subject_id', 'charttime', 'modality')): raise ValueError('report identity differs')
            kind = 0 if p['modality'] == 'cxr' else 1; pos = index[p['hadm_id']]
            expected[pos, kind] = True; groups[pos, kind].append(p)
            if name == 'radgraph':
                # 合并结果章节的偏移，再映射回原报告；跨章节的实体应当拒绝。
                maps = []; cursor = 0
                for s in selected['sections']:
                    if source['text'][s['start_char']:s['end_char']] != s['text']: raise ValueError('selected source span changed')
                    maps.append((cursor, cursor + len(s['text']), s['start_char'], s['kind'])); cursor += len(s['text']) + 1
                for e in p['evidence']:
                    if selected['text'][e['start']:e['end']] != e['quote']: raise ValueError('selected evidence mismatch')
                    span = next((s for s in maps if s[0] <= e['start'] < e['end'] <= s[1]), None)
                    if span is None: raise ValueError('entity crosses original section boundary')
                    lo = span[2] + e['start'] - span[0]; hi = span[2] + e['end'] - span[0]
                    if source['text'][lo:hi] != e['quote']: raise ValueError('original evidence mismatch')
                    evidence.append(dict(e, hadm_id=p['hadm_id'], note_id=p['note_id'], original_start=lo, original_end=hi, section=span[3]))
        if not np.array_equal(a['present'], expected) or (a['values'][~expected] != 0).any(): raise ValueError('incorrect observation mask')
        if matched_presence is not None and not np.array_equal(matched_presence, expected): raise ValueError('candidate coverage differs')
        matched_presence = expected
        if name == 'radgraph':
            for (pos, kind), predictions in groups.items():
                expected_vector = aggregate(predictions).reshape(-1) if kind == 0 else ct_vector([f for r in predictions for f in r['ct_facts']])
                if not np.array_equal(a['values'][pos, kind, :len(expected_vector)], expected_vector): raise ValueError('fact aggregation mismatch')
            evidence_path = folder / 'original_evidence_audit.jsonl'
            if evidence_path.exists(): raise FileExistsError(evidence_path)
            with evidence_path.open('x', encoding='utf-8') as f:
                for e in evidence: f.write(json.dumps(e, ensure_ascii=False) + '\n')
        else:
            lengths = np.linalg.norm(a['values'][expected], axis=1)
            if not np.allclose(lengths, np.sqrt(2), atol=2e-6): raise ValueError('dense embedding normalization differs')
        report[name] = dict(reports=len(by_id), cxr_admissions=int(expected[:, 0].sum()), ct_admissions=int(expected[:, 1].sum()),
                            exact_original_evidence=len(evidence), manifest_sha256=sha256(folder / 'manifest.json'))
    value = dict(status='full_original_source_time_members_and_arrays_checked', candidates=report,
                 identical_report_sets_and_source_masks=True, limitations=['Engineering and source roundtrip, not expert semantic accuracy', 'Frozen representations, not evidence of readmission benefit'])
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.write_text(json.dumps(value, indent=2), encoding='utf-8'); print(json.dumps(value), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selected', type=Path, default=Path('patient_data/processed/v6_imaging_all_2026-10-10_r2'))
    p.add_argument('--snapshot', type=Path, default=Path('patient_data/mimiciv_copd/snapshot_2026-09-27'))
    p.add_argument('--radbert', type=Path, default=Path('patient_data/processed/v6_radbert_imaging_2026-10-10_r2'))
    p.add_argument('--radgraph', type=Path, default=Path('patient_data/processed/v6_radgraph_imaging_source_checked_2026-10-10'))
    p.add_argument('--output', type=Path, required=True)
    audit(p.parse_args())
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/clinical_facts/audit_radiology_repetition.py · lines 1–48

<!-- REVIEW_FRAGMENT {"path":"patient_data/clinical_facts/audit_radiology_repetition.py","part":1,"start_line":1,"end_line":48,"source_sha256":"a2107f32598f8bac2d274fdb17b37043d8e80381b2ee09cc4a5a8a96d6c0bb9d","fragment_sha256":"a2107f32598f8bac2d274fdb17b37043d8e80381b2ee09cc4a5a8a96d6c0bb9d","added_terminal_newline":false} -->
````python
"""多胸片复查审计；只输出匿名统计，不以报告次数推断病情恶化。"""
import json
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path
import re
from patient_data.export_mimiciv import rows, sha256
from patient_data.clinical_facts.radiology_compare import records


def main():
    selected = Path('patient_data/processed/v6_imaging_all_2026-10-10_r2')
    snapshot = Path('patient_data/mimiciv_copd/snapshot_2026-09-27')
    data, members, manifest = records(selected)
    grouped = defaultdict(list)
    for r in data:
        if r['modality'] == 'cxr': grouped[r['hadm_id']].append(r)
    admissions = {r['hadm_id']: r for r in rows(snapshot / 'cohort/admissions.csv.gz')}
    top = sorted(grouped.values(), key=len, reverse=True)[:10]
    needed = {r['note_id'] for reports in top for r in reports}
    originals = {r['note_id']: r['text'] for r in rows(snapshot / 'raw/radiology.csv.gz') if r['note_id'] in needed}
    result = []
    for reports in top:
        dates = Counter(r['charttime'][:10] for r in reports)
        texts = [originals[r['note_id']] for r in reports]
        indications = []
        for text in texts:
            match = re.search(r'(?:INDICATION|REASON FOR (?:EXAM|EXAMINATION))\s*:\s*(.*?)(?=\n\s*(?:TECHNIQUE|COMPARISON|FINDINGS|IMPRESSION)\s*:|\Z)', text, re.I | re.S)
            indications.append(match.group(1) if match else '')
        admission = admissions[reports[0]['hadm_id']]
        los = (datetime.fromisoformat(admission['dischtime']) - datetime.fromisoformat(admission['admittime'])).total_seconds() / 86400
        result.append(dict(reports=len(reports), length_of_stay_days=round(los, 1),
                           distinct_charttimes=len({r['charttime'] for r in reports}), report_dates=len(dates), maximum_reports_per_day=max(dates.values()),
                           exact_full_text_duplicates=len(texts) - len(set(texts)),
                           indications_mentioning_device_check=sum(bool(re.search(r'line|tube|intubat|extubat|catheter', s, re.I)) for s in indications),
                           reports_mentioning_device_terms=sum(bool(re.search(r'endotracheal|chest tube|central venous|\bPICC\b|catheter', s, re.I)) for s in texts)))
    counts = Counter(len(v) for v in grouped.values())
    value = dict(status='source_checked_anonymous_report_statistics', report_counts=dict(sorted(counts.items())),
                 covered_admissions=len(grouped), multiple_report_admissions=sum(v for k, v in counts.items() if k > 1), top_ten=result,
                 selected_manifest_sha256=sha256(selected / 'manifest.json'),
                 limitations=['Device-term matches are mentions, not verified placement or causation', 'Distinct full texts can still contain redundant clinical content', 'Higher report frequency is not equivalent to clinical worsening'])
    output = Path('patient_data/processed/v6_radiology_repetition_audit_2026-10-10.json')
    if output.exists(): raise FileExistsError(output)
    output.write_text(json.dumps(value, indent=2), encoding='utf-8')
    print(json.dumps(value), flush=True)


if __name__ == '__main__': main()
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/clinical_facts/audit_sentence_screen.py · lines 1–73

<!-- REVIEW_FRAGMENT {"path":"patient_data/clinical_facts/audit_sentence_screen.py","part":1,"start_line":1,"end_line":73,"source_sha256":"067df668c168f9f213d3e491f675765b969510de665ab55881a158dc65554aa7","fragment_sha256":"067df668c168f9f213d3e491f675765b969510de665ab55881a158dc65554aa7","added_terminal_newline":false} -->
````python
"""从原始快照独立回查全部句段、上下文、成员及章节字符覆盖；不估计医学准确率。"""
import argparse
from collections import Counter, defaultdict
import json
import gzip
from pathlib import Path
import re
import time
from patient_data.export_mimiciv import rows, sha256
from patient_data.clinical_facts.llm import digest


def audit(args):
    result = args.output / 'source_audit.json'
    if result.exists(): raise FileExistsError(result)
    start = time.perf_counter()
    m = json.loads((args.output / 'manifest.json').read_text())
    for name, expected in m['output_sha256'].items():
        if sha256(args.output / name) != expected: raise ValueError('screen output changed')
    with (args.output / 'documents.jsonl').open(encoding='utf-8') as stream:
        docs = {r['note_id']: r for r in map(json.loads, stream)}
    # 原文仅读一次，审计原句与覆盖，不重新运行抽取器作为唯一证据。
    raw = {}
    for kind in ('discharge', 'radiology'):
        for r in rows(args.snapshot / ('raw/' + kind + '.csv.gz')):
            if r['note_id'] in docs: raw[r['note_id']] = r
    if raw.keys() != docs.keys(): raise ValueError('missing original document')
    for key, d in docs.items():
        r = raw[key]
        if r['hadm_id'] != d['hadm_id'] or r['subject_id'] != d['subject_id'] or digest(r['text']) != d['original_sha256']:
            raise ValueError('source identity or SHA mismatch')
    counts = Counter(); selected_keys = set(); covered = defaultdict(list)
    with gzip.open(args.output / 'all_units.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for row in map(json.loads, stream):
            text = raw[row['note_id']]['text']; a, b = row['start_char'], row['end_char']
            ca, cb = row['context_start'], row['context_end']
            if not 0 <= ca <= a < b <= cb <= len(text) or text[a:b] != row['quote']:
                raise ValueError('original quote/context offsets invalid')
            section = docs[row['note_id']]['section_spans'][row['section_ordinal']]
            if not section['start_char'] <= ca <= cb <= section['end_char']:
                raise ValueError('context crossed section boundary')
            covered[(row['note_id'], row['section_ordinal'])].append((a, b))
            counts[row['kind'] + '_units'] += 1
            if row['selected']:
                if row['unit_key'] in selected_keys: raise ValueError('duplicate selected unit')
                selected_keys.add(row['unit_key'])
    for key, d in docs.items():
        for j, section in enumerate(d['section_spans']):
            spans = covered[key, j]
            if any(b > c for (_, b), (c, _) in zip(spans, spans[1:])): raise ValueError('overlapping units')
            original = raw[key]['text'][section['start_char']:section['end_char']]
            actual = ''.join(raw[key]['text'][a:b] for a, b in spans)
            if re.sub(r'\s', '', original) != re.sub(r'\s', '', actual): raise ValueError('nonwhitespace section text lost')
    selected_count = 0
    with gzip.open(args.output / 'selected_sentences.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for row in map(json.loads, stream):
            if not row['selected'] or row['unit_key'] not in selected_keys: raise ValueError('selected output differs')
            selected_keys.remove(row['unit_key']); selected_count += 1
    if selected_keys: raise ValueError('selected output incomplete')
    report = dict(status='all_original_quotes_contexts_and_section_coverage_checked', documents=len(docs),
        units=dict(counts), selected_units=selected_count, manifest_sha256=sha256(args.output / 'manifest.json'),
        audit_code_sha256=sha256(Path(__file__)), seconds=time.perf_counter()-start,
        limits=['Exact source and coverage validation is not semantic precision or recall',
                'Section headers/preamble outside parser sections are not included in coverage denominator',
                'Source report selection and retrospective final-summary policy remain inherited'])
    result.write_text(json.dumps(report, indent=2), encoding='utf-8'); print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, default=Path('patient_data/mimiciv_copd/snapshot_2026-09-27'))
    parser.add_argument('--output', type=Path, required=True)
    audit(parser.parse_args())
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/clinical_facts/catalog.py · lines 1–78

<!-- REVIEW_FRAGMENT {"path":"patient_data/clinical_facts/catalog.py","part":1,"start_line":1,"end_line":78,"source_sha256":"77e54c839473f42e8438d170d7a554fb6e50e300093220aa29b1c8deccbb32f3","fragment_sha256":"77e54c839473f42e8438d170d7a554fb6e50e300093220aa29b1c8deccbb32f3","added_terminal_newline":false} -->
````python
"""固定临床事实字典；类别描述而非风险标签。未提及不输出事实。"""
import hashlib
import json

CT_FIELDS = {name: ('present','absent','uncertain') for name in (
    'emphysema','centrilobular_emphysema','paraseptal_emphysema','bullae',
    'bronchiectasis','airway_wall_thickening','fibrosis','consolidation',
    'pleural_effusion','enlarged_pulmonary_artery')}
DISCHARGE_FIELDS = {
    'residual_dyspnea': ('present','absent','improved','not_at_baseline'),
    'oxygen_at_discharge': ('room_air','baseline_oxygen','new_or_increased_oxygen','home_oxygen'),
    'ventilation_at_discharge': ('planned','poor_tolerance','refused','weaned'),
    'exertional_desaturation': ('present','absent'),
    'steroid_discharge_course': ('planned','completed','stopped'),
    'respiratory_antibiotic_course': ('planned','completed'),
    'inhaled_ics': ('prescribed','stopped'),
    'inhaled_laba': ('prescribed','stopped'),
    'inhaled_lama': ('prescribed','stopped'),
    'adherence_or_device_barrier': ('present','explicitly_assessed_no_barrier'),
    'functional_dependence': ('dependent','independent','improved'),
    'living_alone': ('present','absent'),
    'discharge_care': ('home','home_services','snf','rehab','hospice'),
    'care_limits': ('dnr','dni','comfort_care','ama'),
}
FIELDS = dict(CT_FIELDS, **DISCHARGE_FIELDS)
ROUTES = {
    'respiratory': {
        'fields': tuple(DISCHARGE_FIELDS)[:4],
        'roles': ('hospital_course','discharge_condition','discharge_physical_exam','discharge_instructions','transitional_issues'),
    },
    'medications': {
        'fields': tuple(DISCHARGE_FIELDS)[4:9],
        'roles': ('discharge_medications','hospital_course','discharge_instructions','transitional_issues'),
    },
    'function_care': {
        'fields': tuple(DISCHARGE_FIELDS)[9:],
        'roles': ('social_history','hospital_course','discharge_condition','discharge_disposition','discharge_instructions','transitional_issues'),
    },
    'chest_ct': {'fields': tuple(CT_FIELDS), 'roles': ('FINDINGS','IMPRESSION','CONCLUSION')},
}
ROUTES['discharge_all']={'fields':tuple(DISCHARGE_FIELDS),
                         'roles':tuple(dict.fromkeys(role for name in ('respiratory','medications','function_care') for role in ROUTES[name]['roles']))}
VERSION = 'clinical-facts-v1-2026-10-10'
CATALOG_SHA = hashlib.sha256(json.dumps({'fields':FIELDS,'routes':ROUTES,'version':VERSION},sort_keys=True).encode()).hexdigest()

INSTRUCTIONS = """You extract explicit clinical facts, NOT predictions. Return only JSON {"facts": [[field,value,verbatim_quote], ...]}.
Use only the supplied section. Omit unmentioned fields. Copy a continuous exact quote from SOURCE for every fact; no paraphrase or ellipses.
Keep negation and uncertainty. Do not infer diagnosis, response, drug indication, normality, or care from silence.
For discharge respiratory fields use EXPLICIT discharge/latest clinical status, not admission symptoms or generic return-to-ER advice.
Room air and independent function may be extracted when explicitly documented; do not invent them.
For discharge medications only extract actual discharge plans/completion, not admission drugs. Respiratory antibiotics require an explicit pulmonary indication in the evidence. ICS/LABA/LAMA refer to pulmonary inhaled medication components, not nasal sprays; combination products can have multiple components.
Do not classify intraoperative intubation or routine OSA CPAP as acute respiratory failure. DNR and DNI are separate facts.
In CT, comparison/history alone is not a current finding. Extract explicit morphology; uncertainty stays uncertain. Bronchial wall thickening is NOT bronchiectasis. Ground-glass opacity or pneumonia alone is NOT consolidation. Edema/septal thickening alone is NOT fibrosis. Do not force an unlisted finding into the closest listed field. 'Likely', 'suggestive of', 'possible', 'may represent' indicate uncertain diagnosis, unless only its cause (not the finding itself) is uncertain.
Use the allowed field/value catalog below. Output [] if no applicable facts. No analysis in JSON."""
DEFINITIONS={
    'residual_dyspnea':'Explicit shortness of breath or breathlessness at discharge. No acute distress alone is not absence of dyspnea.',
    'oxygen_at_discharge':'Explicit oxygen use, flow, or room air (RA) at discharge. Clear lungs or independent walking say nothing about oxygen.',
    'ventilation_at_discharge':'Mechanical ventilation/intubation or NIV/BiPAP/CPAP support. Oxygen nasal cannula is not ventilation. Unable to wean is NOT weaned.',
    'exertional_desaturation':'Explicit oxygenation problem during exertion/walking, not walking limitation alone.',
    'steroid_discharge_course':'Explicit systemic steroid drug and discharge course/taper/completion. Antibiotic treatment is not a steroid. Discontinued/stopped because not indicated is stopped, NOT completed.',
    'respiratory_antibiotic_course':'Explicit pulmonary infection antibiotic plan/completion, not UTI antibiotics.',
    'inhaled_ics':'Pulmonary inhaled corticosteroid, e.g. budesonide or fluticasone in inhaler, not nasal spray.',
    'inhaled_laba':'Long-acting beta agonist, e.g. salmeterol/formoterol/vilanterol, NOT albuterol.',
    'inhaled_lama':'Long-acting muscarinic antagonist, e.g. tiotropium/umeclidinium, NOT ipratropium.',
    'adherence_or_device_barrier':'Explicit missed/stopped medication, poor inhaler technique/tolerance, affordability/access/device use barrier; mucus plugging alone is not adherence.',
    'functional_dependence':'Specific assessed mobility/ADL dependence or recovery. Omit generic Ambulatory - Independent template.',
    'living_alone':'Explicit living alone or with another person. Home/Home With Service/SNF do NOT establish living alone.',
    'discharge_care':'Explicit actual discharge destination/support, not planned destination or admission origin.',
    'care_limits':'Explicit DNR/DNI/comfort-care/against-medical-advice. Full Code is NOT DNR/DNI.',
}
# 字段含义与提示词属于数据合同，不能只对字段key求哈希。
CATALOG_SHA=hashlib.sha256(json.dumps({'fields':FIELDS,'routes':ROUTES,'version':VERSION,
                                    'instructions':INSTRUCTIONS,'definitions':DEFINITIONS},sort_keys=True).encode()).hexdigest()

def prompt(route, role, text):
    allowed={name:FIELDS[name] for name in ROUTES[route]['fields']}
    definitions={k:v for k,v in DEFINITIONS.items() if k in allowed}
    return INSTRUCTIONS+'\nCATALOG: '+json.dumps(allowed)+'\nDEFINITIONS: '+json.dumps(definitions)+'\nSECTION: '+role+'\nSOURCE:\n'+text
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/clinical_facts/cxr_trajectory.py · lines 1–108

<!-- REVIEW_FRAGMENT {"path":"patient_data/clinical_facts/cxr_trajectory.py","part":1,"start_line":1,"end_line":108,"source_sha256":"c00d8fe58b1b0b7822da41e54ec2bc269b9d613a57a7e6e2be6c7f0caff95b06","fragment_sha256":"c00d8fe58b1b0b7822da41e54ec2bc269b9d613a57a7e6e2be6c7f0caff95b06","added_terminal_newline":false} -->
````python
"""全部合格胸片的七项观测轨迹；不将未提及当作阴性或恢复。"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
import numpy as np
from patient_data.export_mimiciv import sha256

SELECTED=('Pneumonia','Consolidation','Edema','Cardiomegaly','Pleural Effusion','Pneumothorax','Atelectasis')
FEATURES=('first_blank','first_positive','first_negative','first_uncertain',
          'latest_blank','latest_positive','latest_negative','latest_uncertain',
          'ever_positive','ever_uncertain','later_explicit_negative','new_positive',
          'first_conflict','latest_conflict')

def combine(states):
    states=set(map(int,states))-{0}
    conflict=1 in states and 2 in states
    return (3 if conflict else 1 if 1 in states else 3 if 3 in states else 2 if 2 in states else 0), conflict

def aggregate(records):
    """同charttime是集合而非序列；storetime不制造病理先后。"""
    if not records: return np.zeros((7,14),np.float32)
    by_time=defaultdict(list)
    for r in records: by_time[r['charttime']].append(r)
    buckets=[]
    for timestamp in sorted(by_time):
        rows=by_time[timestamp]; codes=[]; conflicts=[]
        for field in SELECTED:
            code,conflict=combine(r['labels14'][field] for r in rows)
            codes.append(code); conflicts.append(conflict or any(r['chunk_conflicts'].get(field,False) for r in rows))
        buckets.append((codes,conflicts))
    result=np.zeros((7,14),np.float32)
    for j in range(7):
        sequence=[b[0][j] for b in buckets]; first,last=sequence[0],sequence[-1]
        result[j,first]=1; result[j,4+last]=1
        result[j,8]=1 in sequence; result[j,9]=3 in sequence
        # 必须存在较晚的明确阴性；同一时刻、空白或疑似不作为改善。
        result[j,10]=any(v==2 and 1 in sequence[:i] for i,v in enumerate(sequence))
        result[j,11]=first==2 and 1 in sequence[1:]
        result[j,12]=buckets[0][1][j]; result[j,13]=buckets[-1][1][j]
    return result

def chunks(text,tokenizer):
    """全字符覆盖、保留首尾；超长段逐字缩短至官方512预算。"""
    from training.v6.chexbert_inference import encode_ids
    start=0; result=[]
    while start<len(text):
        end=min(len(text),start+1800)
        while encode_ids(text[start:end],tokenizer)[1]>500:
            end=start+(end-start)//2
            if end<=start: raise ValueError('single character exceeds token budget')
        # 在预算内的自然句边界结束；无边界则保留硬边界的审计偏移。
        if end<len(text):
            boundary=max(text.rfind('\n',start+max(1,(end-start)//2),end),text.rfind('. ',start+max(1,(end-start)//2),end))
            if boundary>start: end=boundary+1
        result.append((start,end,text[start:end])); start=end
    if ''.join(c[2] for c in result)!=text: raise ValueError('lost report text')
    return result

def build(args):
    from training.v6.chexbert_inference import load_model,label_texts,CONDITIONS
    import torch
    if args.output.exists(): raise FileExistsError(args.output)
    selection=json.loads((args.selected/'manifest.json').read_text())
    for name,value in selection['output_sha256'].items():
        if sha256(args.selected/name)!=value: raise ValueError('selection hash changed')
    records=[json.loads(s) for s in (args.selected/'cxr_reports.jsonl').read_text(encoding='utf-8').splitlines()]
    model,tokenizer,asset=load_model(args.assets)
    spans=[chunks(r['text'],tokenizer) for r in records]; texts=[c[2] for group in spans for c in group]
    started=time.perf_counter(); labels,counts=label_texts(texts,model,tokenizer,args.batch_size)
    if (counts>512).any(): raise ValueError('truncated chunk')
    elapsed=time.perf_counter()-started; pointer=0
    for record,group in zip(records,spans,strict=True):
        subset=labels[pointer:pointer+len(group)]; pointer+=len(group)
        combined=[combine(subset[:,j]) for j in range(14)]
        record.update(labels14={name:v[0] for name,v in zip(CONDITIONS,combined,strict=True)},
                      chunk_conflicts={name:bool(v[1]) for name,v in zip(CONDITIONS,combined,strict=True)},
                      chunks=[{'start_char':a,'end_char':b,'tokens':int(counts[pointer-len(group)+i]),
                               'labels14':list(map(int,subset[i]))} for i,(a,b,_) in enumerate(group)])
    with np.load(args.selected/'members.npz') as f: members={k:f[k] for k in f.files}
    grouped=defaultdict(list)
    for record in records: grouped[record['hadm_id']].append(record)
    values=np.stack([aggregate(grouped.get(str(h),[])) for h in members['hadm_ids']])
    present=np.asarray([str(h) in grouped for h in members['hadm_ids']],bool)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'cxr_trajectory.npz',**members,cxr_values=values.reshape(-1,98),
                        cxr_present=present,condition_names=np.array(SELECTED),feature_names=np.array(FEATURES))
    with (args.output/'labeled_reports.jsonl').open('x',encoding='utf-8') as f:
        for record in records: f.write(json.dumps(record,ensure_ascii=False)+'\n')
    result={'status':'all_cxr_labeled_and_aggregated','reports':len(records),'admissions_with_cxr':int(present.sum()),
            'admissions_with_repeat_timestamps':sum(len({r['charttime'] for r in group})>1 for group in grouped.values()),
            'chunks':len(texts),'seconds':elapsed,'reports_per_second':len(records)/elapsed,'truncated_chunks':0,
            'parent_state_manifest_sha256':selection['parent_state_manifest_sha256'],
            'selection_manifest_sha256':sha256(args.selected/'manifest.json'),'asset_manifest_sha256':sha256(args.assets/'asset_manifest.json'),
            'feature_names':FEATURES,'condition_names':SELECTED,
            'trajectory_counts':{field:{key:int(values[:,j,k].sum()) for k,key in enumerate(FEATURES[8:],8)} for j,field in enumerate(SELECTED)},
            'code_sha256':sha256(Path(__file__)),'output_sha256':{n:sha256(args.output/n) for n in ('cxr_trajectory.npz','labeled_reports.jsonl')},
            'limits':['Report labels not image pixels or expert gold standard','Later explicit negative is an observation, not proof of treatment effect',
                      'Unknown modality excluded; no assumption that all thoracic findings are CXR','Possible pretrained MIMIC-CXR overlap']}
    (args.output/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selected',type=Path,default=Path('patient_data/processed/v6_imaging_all_2026-10-10_r2'))
    p.add_argument('--assets',type=Path,default=Path('training/model_cache/CheXbert_stanford_2026-10-09'))
    p.add_argument('--output',type=Path,required=True); p.add_argument('--batch-size',type=int,default=32)
    build(p.parse_args())
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/clinical_facts/features.py · lines 1–106

<!-- REVIEW_FRAGMENT {"path":"patient_data/clinical_facts/features.py","part":1,"start_line":1,"end_line":106,"source_sha256":"57a2ac01f16eae50e620357fe8fde5215c96c9920aa817a5e91a3dbbc40348e3","fragment_sha256":"57a2ac01f16eae50e620357fe8fde5215c96c9920aa817a5e91a3dbbc40348e3","added_terminal_newline":false} -->
````python
"""事实转类别输入：未知/阴性/冲突分离；禁止将半完成抽取送入正式训练。"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from patient_data.export_mimiciv import sha256
from patient_data.clinical_facts.catalog import CT_FIELDS,DISCHARGE_FIELDS,CATALOG_SHA

CT_NAMES=tuple(name+':'+value for name in CT_FIELDS for value in (*CT_FIELDS[name],'observed','conflict','ever_positive'))
DISCHARGE_NAMES=tuple(name+':'+value for name in DISCHARGE_FIELDS for value in (*DISCHARGE_FIELDS[name],'observed','conflict'))

def ct_vector(facts):
    output=[]
    for field in CT_FIELDS:
        selected=[f for f in facts if f['field']==field]
        latest=max((f['charttime'] for f in selected),default=None)
        values={f['value'] for f in selected if f['charttime']==latest}
        conflict='present' in values and 'absent' in values
        # 同时刻明确相反，不能挑一份当正常。保留疑似与冲突。
        if conflict: values={'uncertain'}
        output.extend([float(v in values) for v in CT_FIELDS[field]])
        output.extend([float(bool(selected)),float(conflict),float(any(f['value']=='present' for f in selected))])
    return np.array(output,np.float32)

def discharge_vector(facts):
    output=[]
    priorities={'discharge_medications':4,'discharge_physical_exam':4,'discharge_condition':4,
                'discharge_disposition':4,'discharge_instructions':3,'transitional_issues':3,'social_history':2,'hospital_course':1}
    for field,possible in DISCHARGE_FIELDS.items():
        selected=[f for f in facts if f['field']==field]
        best=max((priorities.get(f['section_role'],0) for f in selected),default=None)
        values={f['value'] for f in selected if priorities.get(f['section_role'],0)==best}
        # DNR+DNI和残余/改善可共存，其余互斥状态不静默选一个。
        conflict=len(values)>1 and field not in ('care_limits','residual_dyspnea')
        output.extend(float(v in values) for v in possible)
        output.extend([float(bool(selected)),float(conflict)])
    return np.array(output,np.float32)

def read_results(directory,tasks):
    status=json.loads((directory/'status.json').read_text())
    contract=json.loads((directory/'contract.json').read_text())
    if status['status']!='completed' or contract['limit'] or contract['catalog_sha256']!=CATALOG_SHA:
        raise ValueError('full extraction has not finished; no partial facts in formal input')
    for name,value in contract['code_sha256'].items():
        if sha256(Path(__file__).parent/name)!=value: raise ValueError('extraction code changed during/after run')
    if sha256(directory/'results.jsonl')!=status['results_sha256']: raise ValueError('result hash differs')
    with (directory/'results.jsonl').open(encoding='utf-8') as f: results={r['task_key']:r for r in map(json.loads,f)}
    expected={t['task_key'] for t in tasks}
    if set(results)!=expected or any(r['status']=='failed' for r in results.values()): raise ValueError('incomplete task results')
    return results,contract

def build(args):
    if args.output.exists(): raise FileExistsError(args.output)
    manifest=json.loads((args.tasks/'manifest.json').read_text())
    for name,value in manifest['output_sha256'].items():
        if sha256(args.tasks/name)!=value: raise ValueError('task source changed')
    with (args.tasks/'tasks.jsonl').open(encoding='utf-8') as f: tasks=list(map(json.loads,f))
    with (args.tasks/'sources.jsonl').open(encoding='utf-8') as f: sources={r['note_id']:r for r in map(json.loads,f)}
    ct_tasks=[t for t in tasks if t['route']=='chest_ct']; ds_tasks=[t for t in tasks if t['route']=='discharge_all']
    ct,ctc=read_results(args.ct,ct_tasks); ds,dsc=read_results(args.discharge,ds_tasks)
    expected_digest=sha256(args.tasks/'manifest.json')
    if ctc['tasks_manifest_sha256']!=expected_digest or dsc['tasks_manifest_sha256']!=expected_digest: raise ValueError('wrong task contract')
    facts=defaultdict(list); evidence=[]
    for task in tasks:
        r=(ct if task['route']=='chest_ct' else ds)[task['task_key']]; source=sources[task['note_id']]
        if source['original_sha256']!=task['original_sha256']: raise ValueError('source identity mismatch')
        for fact in r['facts']:
            if source['text'][fact['original_start']:fact['original_end']]!=fact['quote']: raise ValueError('original evidence failed')
            enriched=dict(fact,hadm_id=task['hadm_id'],note_id=task['note_id'],charttime=task['charttime'],
                          channel=source['channel'],task_key=task['task_key'],original_sha256=task['original_sha256'])
            facts[task['hadm_id']].append(enriched); evidence.append(enriched)
            # 确认阳性亚型必然属于肺气肿；规则推导保存原文且标为derived。
            if fact['field'] in ('centrilobular_emphysema','paraseptal_emphysema') and fact['value']=='present':
                parent=dict(enriched,field='emphysema',derived='positive_emphysema_subtype_implies_emphysema')
                facts[task['hadm_id']].append(parent); evidence.append(parent)
    with np.load(args.tasks/'members.npz') as f: members={k:f[k] for k in f.files}
    ct_hadm={r['hadm_id'] for r in sources.values() if r['channel']=='ct'}
    ds_hadm={r['hadm_id'] for r in sources.values() if r['channel']=='discharge'}
    ct_values=[]; ds_values=[]
    for h in members['hadm_ids']:
        fs=facts[str(h)]; ct_values.append(ct_vector([f for f in fs if f['channel']=='ct']))
        ds_values.append(discharge_vector([f for f in fs if f['channel']=='discharge']))
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'facts.npz',**members,ct_values=np.stack(ct_values),discharge_values=np.stack(ds_values),
                        ct_present=np.array([str(h) in ct_hadm for h in members['hadm_ids']]),
                        discharge_present=np.array([str(h) in ds_hadm for h in members['hadm_ids']]),
                        ct_names=np.array(CT_NAMES),discharge_names=np.array(DISCHARGE_NAMES))
    with (args.output/'evidence.jsonl').open('x',encoding='utf-8') as f:
        for e in evidence: f.write(json.dumps(e,ensure_ascii=False)+'\n')
    result={'status':'full_facts_source_checked_not_prediction_validated','admissions':len(ct_values),
            'ct_sources':len(ct_hadm),'discharge_sources':len(ds_hadm),'evidence_facts':len(evidence),
            'ct_columns':CT_NAMES,'discharge_columns':DISCHARGE_NAMES,'catalog_sha256':CATALOG_SHA,
            'parent_state_manifest_sha256':manifest['parent_state_manifest_sha256'],'tasks_manifest_sha256':expected_digest,
            'ct_status_sha256':sha256(args.ct/'status.json'),'discharge_status_sha256':sha256(args.discharge/'status.json'),
            'code_sha256':sha256(Path(__file__)),'output_sha256':{n:sha256(args.output/n) for n in ('facts.npz','evidence.jsonl')},
            'limits':['Programmatic evidence validation is not expert semantic accuracy','Final discharge note is retrospectively assumed available at discharge',
                      'Facts describe observations and plans, not causal treatment effects','No new embeddings; existing BERT unchanged']}
    (args.output/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps({k:v for k,v in result.items() if 'columns' not in k}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks',type=Path,default=Path('patient_data/processed/v6_bonsai_tasks_2026-10-10'))
    p.add_argument('--ct',type=Path,default=Path('patient_data/processed/v6_bonsai_ct_full_2026-10-10'))
    p.add_argument('--discharge',type=Path,default=Path('patient_data/processed/v6_bonsai_discharge_full_2026-10-10'))
    p.add_argument('--output',type=Path,required=True); build(p.parse_args())
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/clinical_facts/guards.py · lines 1–79

<!-- REVIEW_FRAGMENT {"path":"patient_data/clinical_facts/guards.py","part":1,"start_line":1,"end_line":79,"source_sha256":"c18f5ca13f7abbb1ff41f23f452de3191b31de29480aadfad7310cc2f96c3d2b","fragment_sha256":"c18f5ca13f7abbb1ff41f23f452de3191b31de29480aadfad7310cc2f96c3d2b","added_terminal_newline":false} -->
````python
"""位置、章节、时相和药理证据的保守门禁；不更改LLM事实的极性。"""
import re
from patient_data.clinical_facts.catalog import ROUTES

DRUGS={
    'inhaled_ics':r'\b(?:fluticasone|budesonide|beclomethasone|mometasone|ciclesonide|advair|symbicort|dulera|breo|trelegy|breztri|ICS)\b',
    'inhaled_laba':r'\b(?:salmeterol|formoterol|arformoterol|vilanterol|olodaterol|indacaterol|advair|symbicort|dulera|breo|trelegy|breztri|anoro|stiolto|LABA)\b',
    'inhaled_lama':r'\b(?:tiotropium|umeclidinium|glycopyrrolate|aclidinium|revefenacin|spiriva|incruse|seebri|tudorza|yupelri|anoro|stiolto|trelegy|breztri|LAMA)\b',
}
CT_CONCEPTS={
    'emphysema':r'emphysem', 'centrilobular_emphysema':r'centrilobular.*emphysem',
    'paraseptal_emphysema':r'paraseptal.*emphysem', 'bullae':r'\b(?:bullae|bulla|bullous|blebs?)\b',
    'bronchiectasis':r'bronchiecta|(?:bronchial|airway)\s+(?:dilatation|dilation)',
    'airway_wall_thickening':r'(?:bronchi\w*|airway\w*)\s+(?:wall\s+)?thicken|thicken\w*\s+(?:along|of)\s+(?:the\s+)?(?:major\s+)?airway',
    'fibrosis':r'fibro(?:sis|tic)', 'consolidation':r'consolidat',
    'pleural_effusion':r'pleural\s+(?:effusion|fluid)',
    'enlarged_pulmonary_artery':r'pulmonary\s+(?:arter(?:y|ies|ial)(?:\s+trunk)?)',
}
def allowed_roles(field):
    if field in ROUTES['chest_ct']['fields']: return ROUTES['chest_ct']['roles']
    return next(route['roles'] for key,route in ROUTES.items() if key!='discharge_all' and field in route['fields'])

def guard(fact,task):
    start,end=fact['local_start'],fact['local_end']
    mapping=next((m for m in task['source_maps'] if m['input_start']<=start<end<=m['input_end']),None)
    if mapping is None or mapping['role'] not in allowed_roles(fact['field']): return None,'wrong_source_section'
    quote=task['text'][start:end]
    if quote!=fact['quote']: raise ValueError('local quote mismatch')
    field=fact['field']; role=mapping['role']
    if field in CT_CONCEPTS:
        normalized=' '.join(quote.split())
        if not re.search(CT_CONCEPTS[field],normalized,re.I): return None,'unsupported_CT_concept'
        if field=='enlarged_pulmonary_artery' and not re.search(r'\b(?:enlarg\w*|dilat\w*|prominent|caliber|calibre|diameter|measur\w*)\b',normalized,re.I):
            return None,'pulmonary_artery_patency_not_size'
        if field=='fibrosis' and fact['value']=='present' and re.search(r'\b(?:likely|possible|suggest\w*|may|probable)\b',normalized,re.I):
            return None,'uncertain_fibrosis_called_definite'
    if re.search(r'\b(?:if you (?:develop|experience)|call 911|return to (?:the )?(?:ER|emergency))\b',quote,re.I):
        return None,'generic_warning_template'
    if field=='oxygen_at_discharge' and not re.search(r'\b(?:oxygen|O2|RA|room air|cannula|NC|LPM)\b',quote,re.I):
        return None,'no_oxygen_evidence'
    if field=='oxygen_at_discharge' and fact['value']=='room_air' and not re.search(r'\b(?:RA|room air)\b',quote,re.I):
        return None,'no_home_oxygen_not_room_air'
    if field=='ventilation_at_discharge':
        if not re.search(r'\b(?:ventilat\w*|intubat\w*|extubat\w*|BiPAP|CPAP|NIV|NIPPV)\b',quote,re.I): return None,'oxygen_is_not_ventilation'
        if fact['value']=='weaned' and re.search(r'\b(?:unable|not|failed|failure)\b.{0,50}\b(?:wean|extubat)',quote,re.I|re.S): return None,'failed_wean_not_success'
    if field=='residual_dyspnea' and not re.search(r'\b(?:dyspn\w*|SOB|breathless\w*|shortness of breath|breathing comfortably)\b',quote,re.I):
        return None,'no_dyspnea_evidence'
    if field=='living_alone' and not re.search(r'\b(?:liv\w*|resid\w*|stays?|alone|by (?:him|her|them)self)\b',quote,re.I):
        return None,'destination_not_living_arrangement'
    if field=='care_limits':
        words={'dnr':r'\bDNR\b|do not resuscitate','dni':r'\bDNI\b|do not intubate',
               'comfort_care':r'hospice|comfort (?:care|measures)|palliative','ama':r'\bAMA\b|against medical advice'}
        if not re.search(words[fact['value']],quote,re.I): return None,'unsupported_care_limit'
    if field=='steroid_discharge_course' and not re.search(r'\b(?:pred\w*|methylpred\w*|hydrocortisone|dexamethasone|decadron|solu[- ]?medrol|steroid\w*)\b',quote,re.I):
        return None,'no_systemic_steroid_evidence'
    if field=='steroid_discharge_course' and fact['value']=='completed' and re.search(r'\b(?:discontinued|stopped)\b',quote,re.I):
        return None,'stopped_not_completed_steroid_course'
    if field=='adherence_or_device_barrier' and not re.search(r'\b(?:inhaler\w*|MDI|DPI|spacer|adher\w*|complian\w*|afford\w*|access\w*|refus\w*|noncompli\w*|medication\w*|device\w*)\b',quote,re.I):
        return None,'physiologic_burden_not_adherence'
    if field=='functional_dependence' and fact['value']=='independent' and re.fullmatch(r'(?:Activity Status:\s*)?Ambulatory\s*-\s*Independent\.?',quote.strip(),re.I):
        return None,'default_independent_template'
    if field in ('residual_dyspnea','exertional_desaturation','oxygen_at_discharge') and role not in ('discharge_condition','discharge_physical_exam'):
        context=task['text'][max(mapping['input_start'],start-150):min(mapping['input_end'],end+150)]
        if not re.search(r'\b(?:discharg\w+|leaving (?:the )?hospital|at home|home oxygen|home O2)\b',context,re.I):
            return None,'no_explicit_discharge_time_anchor'
    if field in DRUGS:
        if not re.search(DRUGS[field],quote,re.I): return None,'unsupported_inhaler_component'
        if re.search(r'\b(?:nasal|nostril|flonase|nasonex)\b',quote,re.I): return None,'nasal_route_not_pulmonary'
    if field in ROUTES['medications']['fields'] and role=='hospital_course' and fact['value'] in ('planned','prescribed','stopped'):
        context=task['text'][max(mapping['input_start'],start-200):min(mapping['input_end'],end+200)]
        if not re.search(r'\b(?:discharg\w+|home|taper|continue|remaining|complete\w*)\b',context,re.I):
            return None,'inpatient_drug_not_discharge_plan'
    if field=='respiratory_antibiotic_course':
        context=task['text'][max(mapping['input_start'],start-250):min(mapping['input_end'],end+250)]
        if not re.search(r'\b(?:COPD|AECOPD|pneumonia|respiratory|bronchitis|pulmonary|lung infection)\b',context,re.I):
            return None,'antibiotic_indication_not_pulmonary'
    output=dict(fact,quote=quote,section_role=role,original_start=mapping['original_start']+start-mapping['input_start'],
                original_end=mapping['original_start']+end-mapping['input_start'])
    return output,None
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/clinical_facts/imaging.py · lines 1–116

<!-- REVIEW_FRAGMENT {"path":"patient_data/clinical_facts/imaging.py","part":1,"start_line":1,"end_line":116,"source_sha256":"dad4c512856fde8271340995e4dd56569e1eae8c9c031ac6315fe2dca72613c2","fragment_sha256":"dad4c512856fde8271340995e4dd56569e1eae8c9c031ac6315fe2dca72613c2","added_terminal_newline":false} -->
````python
"""追回明确胸片，分离胸部CT，保存全部合格报告的逐字章节。"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from patient_data.cxr_report_selection import identify, HEADER, usable
from patient_data.export_mimiciv import rows, sha256

CHEST = re.compile(r'\b(chest|thorax|thoracic|lungs?)\b', re.I)
CT = re.compile(r'\b(CT|CTA|MDCT|HRCT|COMPUTED TOMOGRAPHY)\b', re.I)
OTHER = re.compile(r'\b(MRI|MR|ULTRASOUND|SONOGRAM|FLUOROSCOPY|BIOPSY)\b', re.I)
VIEW = re.compile(r'\b(radiographs?|x[ -]?rays?|portable|AP|PA|lateral|frontal|views?|port)\b',re.I)
EXPLICIT_CXR = re.compile(r'^(?:(?:portable\s+)?(?:AP|PA|frontal)(?:\s+(?:and\s+)?lateral)?\s+(?:portable\s+)?(?:chest|radiograph)|chest\s+(?:radiographs?|x[ -]?rays?|views?))\s+(?:was\s+|were\s+|is\s+|are\s+)?(?:obtained|performed|demonstrates?|shows?|reveals?)\b', re.I)


def sections(text):
    result=[]; headings=list(HEADER.finditer(text))
    for i,h in enumerate(headings):
        kind=h.group(1).upper()
        if kind not in ('FINDINGS','IMPRESSION','CONCLUSION'): continue
        start=h.end(); end=headings[i+1].start() if i+1<len(headings) else len(text)
        left=text[start:end]; offset=len(left)-len(left.lstrip()); value=left.strip()
        if value:
            start+=offset; end=start+len(value)
            result.append({'kind':kind,'start_char':start,'end_char':end,'text':value})
    return result


def modality(text):
    category,title=identify(text)
    headings=list(HEADER.finditer(text)); actual=[]
    for i,h in enumerate(headings):
        if h.group(1).upper() in ('EXAM','EXAMINATION','TECHNIQUE'):
            end=headings[i+1].start() if i+1<len(headings) else len(text)
            actual.append(text[h.end():end].strip())
    if title: actual.append(title)
    for segment in actual:
        if CT.search(segment):
            return ('chest_ct' if CHEST.search(segment) else 'other'), 'actual_CT_exam_or_technique'
        if OTHER.search(segment): return 'other','actual_non_CXR_exam_or_technique'
        if CHEST.search(segment) and re.search(r'\b(?:axial|helical)\s+(?:CT\s+)?images?\b',segment,re.I):
            return 'chest_ct','actual_axial_chest_technique'
    if category.startswith('cxr_'): return 'cxr',category
    stop=min((h.start() for h in headings),default=len(text))
    current_titles=actual+[line.strip() for line in text[:stop].splitlines()[:12]]
    for title in current_titles:
        if CHEST.search(title) and VIEW.search(title) and not CT.search(title) and not OTHER.search(title):
            return 'cxr','explicit_current_chest_views'
    # 未识别的明确非胸部EXAM不得被正文比较/指征提到的胸片改写。
    if category=='unrecognized_exam':
        if title and not CHEST.search(title): return 'other','explicit_non_chest_exam'
    for section in sections(text):
        if section['kind']=='FINDINGS' and EXPLICIT_CXR.search(' '.join(section['text'][:800].split())):
            return 'cxr','explicit_CXR_in_current_findings'
    return ('other' if category.startswith('non_cxr') else 'unknown'),category


def build(snapshot,state,output):
    if output.exists(): raise FileExistsError(output)
    raw=json.loads((snapshot/'metadata/manifest.json').read_text(encoding='utf-8'))
    sm=json.loads((state/'manifest.json').read_text(encoding='utf-8'))
    if raw['status']!='validated' or sm['status']!='derived_and_roundtrip_checked_not_prediction_validated':
        raise ValueError('unvalidated source')
    for source in ('raw/radiology','cohort/admissions'):
        if sha256(snapshot/(source+'.csv.gz'))!=raw['sources'][source]['sha256']: raise ValueError('source changed')
    if sha256(state/'clinical_state.npz')!=sm['output_sha256']['clinical_state.npz']: raise ValueError('state changed')
    with np.load(state/'clinical_state.npz',allow_pickle=False) as f: members={k:f[k] for k in ('hadm_ids','subject_ids','patient_splits')}
    admissions={r['hadm_id']:r for r in rows(snapshot/'cohort/admissions.csv.gz') if r['discharge_landmark_eligible']=='t'}
    if set(admissions)!=set(map(str,members['hadm_ids'])): raise ValueError('members differ')
    split={str(h):str(s) for h,s in zip(members['hadm_ids'],members['patient_splits'],strict=True)}
    records={'cxr':[],'chest_ct':[]}; audit=Counter(); seen=set()
    for row in rows(snapshot/'raw/radiology.csv.gz'):
        audit['raw_reports']+=1
        if row['note_id'] in seen: raise ValueError('duplicate radiology note')
        seen.add(row['note_id'])
        if row['hadm_id'] not in admissions: audit['not_eligible']+=1; continue
        kind,why=modality(row['text']); audit['modality_'+kind]+=1
        okay,reason=usable(row,admissions[row['hadm_id']])
        if not okay: audit[reason]+=1; continue
        audit['available_'+kind]+=1
        if kind not in records: continue
        spans=sections(row['text'])
        if not spans: audit['no_result_section_'+kind]+=1; continue
        for span in spans:
            if not 0<=span['start_char']<=span['end_char']<=len(row['text']) or row['text'][span['start_char']:span['end_char']]!=span['text']:
                raise ValueError('invalid span')
        record={k:row[k] for k in ('subject_id','hadm_id','note_id','charttime','storetime')}
        record.update(modality=kind,classification_basis=why,original_sha256=hashlib.sha256(row['text'].encode()).hexdigest(),
                      patient_split=split[row['hadm_id']],sections=spans,text='\n'.join(s['text'] for s in spans))
        records[kind].append(record)
    if audit['raw_reports']!=raw['sources']['raw/radiology']['rows']: raise ValueError('raw count differs')
    output.mkdir(parents=True)
    for kind,values in records.items():
        values.sort(key=lambda r:(int(r['hadm_id']),r['charttime'],r['storetime'],r['note_id']))
        with (output/(kind+'_reports.jsonl')).open('x',encoding='utf-8') as f:
            for record in values: f.write(json.dumps(record,ensure_ascii=False)+'\n')
    np.savez_compressed(output/'members.npz',**members)
    result={'status':'all_pre_discharge_imaging_selected','parent_state_manifest_sha256':sha256(state/'manifest.json'),
            'source_manifest_sha256':sha256(snapshot/'metadata/manifest.json'),'raw_radiology_sha256':sha256(snapshot/'raw/radiology.csv.gz'),
            'admissions':len(admissions),'audit':dict(audit),'code_sha256':sha256(Path(__file__)),
            'coverage':{k:{'reports':len(v),'admissions':len({r['hadm_id'] for r in v})} for k,v in records.items()},
            'output_sha256':{n:sha256(output/n) for n in ('cxr_reports.jsonl','chest_ct_reports.jsonl','members.npz')},
            'limits':['Rule-based modality classification is not an expert gold standard','All results sections; no future reports; unknown not forced to CXR']}
    (output/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',type=Path,default=Path('patient_data/mimiciv_copd/snapshot_2026-09-27'))
    p.add_argument('--state',type=Path,default=Path('patient_data/processed/v6_clinical_state_2026-10-09_r2'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); build(a.snapshot,a.state,a.output)
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/clinical_facts/llm.py · lines 1–106

<!-- REVIEW_FRAGMENT {"path":"patient_data/clinical_facts/llm.py","part":1,"start_line":1,"end_line":106,"source_sha256":"9655bd536573213cbd27ba20e1d95f508cf7992e3e4178f22c83b2a299e4dffd","fragment_sha256":"9655bd536573213cbd27ba20e1d95f508cf7992e3e4178f22c83b2a299e4dffd","added_terminal_newline":false} -->
````python
"""仅loopback Bonsai客户端；拒绝截断JSON、幻觉证据与错误类别。"""
import argparse
import hashlib
import json
import re
import time
from urllib.parse import urlparse
import requests
from patient_data.clinical_facts.catalog import FIELDS, ROUTES, prompt


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def validate(payload, route, source):
    if not isinstance(payload,dict) or set(payload)!={'facts'} or not isinstance(payload['facts'],list):
        raise ValueError('invalid_schema')
    accepted=[]; rejected=[]
    for fact in payload['facts']:
        if not isinstance(fact,list) or len(fact)!=3 or not all(isinstance(x,str) for x in fact):
            rejected.append('invalid_fact_schema'); continue
        field,value,quote=fact
        if field not in ROUTES[route]['fields'] or value not in FIELDS[field]:
            rejected.append('invalid_field_or_value'); continue
        # 完全逐字匹配；不做大小写/标点/省略号的模糊替换。
        if not quote.strip() or len(quote)<4:
            rejected.append('nonverbatim_evidence'); continue
        matches=[(m.start(),m.end()) for m in re.finditer(re.escape(quote),source)]
        if not matches:
            # 仅允许CRLF/换行/连续空白等价，词、大小写、标点仍完全一致。
            parts=list(re.finditer(r'\S+',source)); normalized=' '.join(p.group() for p in parts)
            normalized_quote=' '.join(quote.split()); positions=[]
            for p in parts:
                positions.extend(range(p.start(),p.end())); positions.append(p.end())
            matches=[(positions[m.start()],positions[m.end()-1]+1)
                     for m in re.finditer(re.escape(normalized_quote),normalized)]
        if not matches:
            rejected.append('nonverbatim_evidence'); continue
        start,end=matches[0]; exact=source[start:end]
        accepted.append({'field':field,'value':value,'quote':exact,'generated_quote':quote,
                         'local_start':start,'local_end':end,'ambiguous_quote_location':len(matches)>1})
    # 重复不重复计数；相反类别都保留，由聚合层显式标冲突。
    return list({(f['field'],f['value'],f['quote']):f for f in accepted}.values()),rejected


class LocalBonsai:
    def __init__(self, url='http://127.0.0.1:8080', thinking=False, timeout=180):
        parsed=urlparse(url)
        if parsed.scheme!='http' or parsed.hostname not in ('127.0.0.1','localhost','::1'):
            raise ValueError('only local loopback is allowed for patient text')
        self.url=url.rstrip('/'); self.thinking=thinking; self.timeout=timeout
        self.settings={'thinking':thinking,'temperature':0,'max_tokens':1600 if thinking else 1000,'slot_context':8192}

    def extract(self, route, role, text):
        message=prompt(route,role,text)
        tokenized=requests.post(self.url+'/tokenize',json={'content':message,'add_special':True},timeout=30)
        tokenized.raise_for_status(); source_tokens=len(tokenized.json()['tokens'])
        if source_tokens>min(6800,8192-self.settings['max_tokens']-200):
            raise ExtractionError('input_exceeds_safe_slot_budget',{'source_tokens':source_tokens})
        body={'messages':[{'role':'user','content':message}], 'temperature':0,'max_tokens':self.settings['max_tokens'],
              'chat_template_kwargs':{'enable_thinking':self.thinking},'stream':False}
        if self.thinking: body['reasoning_effort']='medium'
        started=time.perf_counter()
        response=requests.post(self.url+'/v1/chat/completions',json=body,timeout=self.timeout)
        response.raise_for_status(); result=response.json()
        choice=result['choices'][0]; content=choice['message'].get('content') or ''
        output={'elapsed_seconds':time.perf_counter()-started,'usage':result.get('usage',{}),
                'preflight_tokens':source_tokens,
                'finish_reason':choice.get('finish_reason'),'raw_content':content,
                'reasoning':choice['message'].get('reasoning_content'), 'prompt_sha256':digest(message)}
        if choice.get('finish_reason')!='stop': raise ExtractionError('output_truncated',output)
        try:
            # 可去掉完整外层Markdown围栏，但不修补缺括号/截断输出。
            cleaned=content.strip()
            if cleaned.startswith('```') and cleaned.endswith('```'):
                cleaned=re.sub(r'^```(?:json)?\s*','',cleaned)[:-3].strip()
            facts,rejected=validate(json.loads(cleaned),route,text)
        except (ValueError,TypeError) as error:
            raise ExtractionError('parse_or_schema_error: '+str(error),output) from error
        output.update(facts=facts,rejected=rejected)
        return output


class ExtractionError(RuntimeError):
    def __init__(self, message, response):
        super().__init__(message); self.response=response


def synthetic():
    client=LocalBonsai()
    cases=[('chest_ct','IMPRESSION','Severe centrilobular emphysema. No pleural effusion. Possible bronchiectasis.',
            {('emphysema','present'),('centrilobular_emphysema','present'),('pleural_effusion','absent'),('bronchiectasis','uncertain')}),
           ('respiratory','discharge_condition','At discharge breathing comfortably on room air. If you develop shortness of breath call 911.',
            {('oxygen_at_discharge','room_air')}),
           ('medications','discharge_medications','Tiotropium 18 mcg inhaled daily. Fluticasone-salmeterol inhaled twice daily. Fluticasone nasal spray daily.',
            {('inhaled_lama','prescribed'),('inhaled_ics','prescribed'),('inhaled_laba','prescribed')})]
    for route,role,text,expected in cases:
        result=client.extract(route,role,text)
        actual={(f['field'],f['value']) for f in result['facts']}
        print(json.dumps({'case':route,'expected':sorted(expected),'actual':sorted(actual),
                          'matches':expected==actual,'seconds':result['elapsed_seconds'],
                          'usage':result['usage'],'rejected':result['rejected'],'synthetic_facts':result['facts']}),flush=True)

if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args(); synthetic()
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/clinical_facts/radiology_compare.py · lines 1–242

<!-- REVIEW_FRAGMENT {"path":"patient_data/clinical_facts/radiology_compare.py","part":1,"start_line":1,"end_line":242,"source_sha256":"b4e67ea67d0b1e1db3c74104b8f0f76df53b0f6bcd72bbc58d335f433b4115c6","fragment_sha256":"b4e67ea67d0b1e1db3c74104b8f0f76df53b0f6bcd72bbc58d335f433b4115c6","added_terminal_newline":false} -->
````python
"""固定同一胸片/CT清单，冻结RadBERT与真实RadGraph-XL本地GPU处理。"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import time
import numpy as np
from patient_data.export_mimiciv import sha256
from patient_data.clinical_facts.cxr_trajectory import aggregate, SELECTED
from patient_data.clinical_facts.catalog import CT_FIELDS
from patient_data.clinical_facts.features import ct_vector
from patient_data.clinical_facts.guards import CT_CONCEPTS


def records(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, digest in manifest['output_sha256'].items():
        if sha256(root / name) != digest:
            raise ValueError('imaging input hash changed')
    result = []
    for kind in ('cxr', 'chest_ct'):
        with (root / (kind + '_reports.jsonl')).open(encoding='utf-8') as f:
            result.extend(json.loads(line) for line in f)
    with np.load(root / 'members.npz') as f:
        members = {k: f[k] for k in f.files}
    return result, members, manifest


def graph_chunks(text, limit=1400):
    """字符完整覆盖；尽量在句子末分界，绝不截掉长报告的后半部分。"""
    output = []; start = 0
    while start < len(text):
        end = min(start + limit, len(text))
        if end < len(text):
            boundary = max(text.rfind('\n', start + limit // 2, end), text.rfind('. ', start + limit // 2, end))
            if boundary > start:
                end = boundary + 1
        output.append((start, end, text[start:end])); start = end
    return output


def graph_facts(annotation, text):
    """专科实体+关系映射固定临床概念；类别来自模型，非关键词猜肯否。"""
    entities = annotation['entities']; tokens = annotation['text'].split()
    cursor = 0; offsets = []
    for token in tokens:
        pos = text.find(token, cursor)
        if pos < 0:
            raise ValueError('RadGraph token cannot be roundtripped to supplied source')
        offsets.append((pos, pos + len(token))); cursor = pos + len(token)
    polarity = {'Observation::definitely present': 1, 'Observation::definitely absent': 2,
                'Observation::uncertain': 3}
    cxr = {k: [] for k in SELECTED}; ct = []; evidence = []
    patterns = {'Pneumonia': r'pneumonia', 'Consolidation': r'consolidat', 'Edema': r'\bedema\b',
                'Cardiomegaly': r'cardiomegal|enlarged\s+heart', 'Pleural Effusion': r'effusion',
                'Pneumothorax': r'pneumothora', 'Atelectasis': r'atelecta'}
    for key, entity in entities.items():
        if entity['label'] not in polarity:
            continue
        state = polarity[entity['label']]
        related = [v['tokens'] for v in entities.values()
                   if any(rel == 'modify' and target == key for rel, target in v['relations'])]
        # 不能把“increase/acute”等修饰节点沿modify边复制成目标疾病并继承修饰词极性。
        related += [entities[target]['tokens'] for rel, target in entity['relations'] if rel == 'located_at']
        context = ' '.join(related + [entity['tokens']])
        lo, hi = offsets[entity['start_ix']][0], offsets[entity['end_ix']][1]
        quote = text[lo:hi]
        found = []
        for name, pattern in patterns.items():
            if re.search(pattern, context, re.I):
                cxr[name].append(state); found.append(name)
        for field, pattern in CT_CONCEPTS.items():
            if re.search(pattern, context, re.I):
                if field == 'enlarged_pulmonary_artery' and not re.search(r'enlarg|dilat|prominent|caliber|calibre|diameter|measur', context, re.I):
                    continue
                value = {1: 'present', 2: 'absent', 3: 'uncertain'}[state]
                ct.append({'field': field, 'value': value}); found.append(field)
        if found:
            evidence.append({'fields': found, 'state': state, 'quote': quote, 'start': lo, 'end': hi,
                             'entity': entity, 'related_context': context})
    def combine(values):
        if 1 in values and 2 in values: return 3
        return 1 if 1 in values else 3 if 3 in values else 2 if 2 in values else 0
    return {k: combine(v) for k, v in cxr.items()}, ct, evidence


def build(args):
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('GPU required')
    torch.set_num_threads(2)
    if args.output.exists() and not args.resume: raise FileExistsError(args.output)
    if args.resume and (args.kind != 'radgraph' or (args.output / 'manifest.json').exists()):
        raise ValueError('resume only unfinished RadGraph output')
    data, members, manifest = records(args.selected)
    if args.limit: data = data[:args.limit]
    args.output.mkdir(parents=True, exist_ok=args.resume)
    n = len(members['hadm_ids']); index = {str(h): i for i, h in enumerate(members['hadm_ids'])}
    all_present = np.zeros((n, 2), bool)
    vectors = np.zeros((n, 2, 1536), np.float32)
    graphs = np.zeros_like(vectors)
    grouped = defaultdict(list); started = time.perf_counter(); chunks = 0
    completed = 0
    if args.resume:
        origin = json.loads((args.output / 'resume_origin.json').read_text())
        if origin['source_manifest_sha256'] != sha256(args.selected / 'manifest.json'): raise ValueError('resume source changed')
        with (args.output / 'reports.jsonl').open(encoding='utf-8') as f: previous = list(map(json.loads, f))
        for saved, source in zip(previous, data, strict=False):
            if any(saved[k] != source[k] for k in ('hadm_id', 'subject_id', 'note_id', 'charttime', 'original_sha256', 'modality')):
                raise ValueError('resume is not an exact matching report prefix')
            for e in saved['evidence']:
                if source['text'][e['start']:e['end']] != e['quote']: raise ValueError('resume evidence mismatch')
            kind = 0 if saved['modality'] == 'cxr' else 1; pos = index[saved['hadm_id']]
            all_present[pos, kind] = True; grouped[pos, kind].append(saved); chunks += saved['chunks']
        completed = len(previous)
        if completed > len(data): raise ValueError('resume has extra reports')
        print(json.dumps({'resuming_completed_reports': completed}), flush=True)
    if args.kind == 'radbert':
        from transformers import AutoModel, AutoTokenizer
        assets = Path('training/model_cache/RadBERT_2026-10-10')
        asset = json.loads((assets / 'asset_manifest.json').read_text())
        for name, value in asset['files'].items():
            if sha256(assets / name) != value['sha256']: raise ValueError('RadBERT assets changed')
        tokenizer = AutoTokenizer.from_pretrained(str(assets), local_files_only=True)
        # 不使用checkpoint没有训练过的随机pooler；只取冻结token隐藏状态。
        model = AutoModel.from_pretrained(str(assets), local_files_only=True, add_pooling_layer=False).eval().requires_grad_(False).cuda()
        if model.config.hidden_size != 768: raise ValueError('unexpected RadBERT width')
        boundary_ids = tokenizer.encode('', add_special_tokens=True)
        if len(boundary_ids) != 2: raise ValueError('unexpected RoBERTa special token pattern')
    else:
        from radgraph import RadGraph
        assets = Path('training/model_cache/RadGraph_XL_2026-10-10')
        asset = json.loads((assets / 'asset_manifest.json').read_text())
        # 原始配置保存在已校验的公开压缩包中；运行配置仅把骨干路径绑定本地资产。
        import tarfile
        with tarfile.open(assets / 'modern-radgraph-xl.tar.gz') as archive:
            config = json.load(archive.extractfile('config.json'))
        backbone = Path('training/model_cache/ModernBERT_base_2026-10-10')
        backbone_manifest = json.loads((backbone / 'asset_manifest.json').read_text())
        for name, meta in backbone_manifest['files'].items():
            if sha256(backbone / name) != meta['sha256']: raise ValueError('ModernBERT asset changed')
        config['dataset_reader']['token_indexers']['bert']['model_name'] = str(backbone.resolve())
        config['model']['embedder']['token_embedders']['bert']['model_name'] = str(backbone.resolve())
        (assets / 'modern-radgraph-xl/config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
        model = RadGraph(model_type='modern-radgraph-xl', cuda=0, model_cache_dir=str(assets),
                         tokenizer_cache_dir='/workspace/training/model_cache/huggingface/hub').eval().requires_grad_(False)
        # 短报告/变长输入不值得为每种形状触发首次compile；权重不变，仍在GPU前向。
        for module in model.modules():
            if hasattr(module, 'config') and hasattr(module.config, 'reference_compile'):
                module.config.reference_compile = False
    with (args.output / 'reports.jsonl').open('a' if args.resume else 'x', encoding='utf-8') as out:
        for ordinal, record in enumerate(data):
            if ordinal < completed: continue
            kind = 0 if record['modality'] == 'cxr' else 1
            pos = index[record['hadm_id']]; all_present[pos, kind] = True
            saved = {k: record[k] for k in ('hadm_id', 'subject_id', 'note_id', 'charttime', 'original_sha256', 'modality')}
            if args.kind == 'radbert':
                ids = tokenizer.encode(record['text'], add_special_tokens=False)
                pieces = [ids[i:i + 510] for i in range(0, len(ids), 510)]
                sums = np.zeros(768, np.float64); count = 0
                for start in range(0, len(pieces), args.batch_size):
                    # transformers5不再提供build_inputs_with_special_tokens；边界由真实tokenizer验收。
                    batch = tokenizer.pad([{'input_ids': [boundary_ids[0]] + p + [boundary_ids[1]]}
                                           for p in pieces[start:start + args.batch_size]], return_tensors='pt')
                    batch = {k: v.cuda() for k, v in batch.items()}
                    with torch.inference_mode():
                        hidden = model(**batch).last_hidden_state
                        # 排除特殊token；最后短块按实际词token数加权，不被等块平均放大。
                        mask = batch['attention_mask'].clone()
                        mask[:, 0] = 0
                        for j, piece in enumerate(pieces[start:start + args.batch_size]): mask[j, len(piece) + 1:] = 0
                        sums += (hidden * mask[:, :, None]).sum((0, 1)).cpu().numpy()
                        count += int(mask.sum())
                if count != len(ids): raise ValueError('RadBERT token coverage differs')
                vector = (sums / max(1, count)).astype(np.float32)
                vector /= max(float(np.linalg.norm(vector)), 1e-12)
                grouped[pos, kind].append((record['charttime'], vector))
                saved.update(input_tokens=len(ids), chunks=len(pieces)); chunks += len(pieces)
            else:
                labels = {k: [] for k in SELECTED}; facts = []; evidence = []
                pieces = graph_chunks(record['text']); chunks += len(pieces)
                for offset, end, piece in pieces:
                    with torch.inference_mode(): annotation = model([piece])['0']
                    cxr, ct, quotes = graph_facts(annotation, piece)
                    for name, value in cxr.items(): labels[name].append(value)
                    facts.extend(dict(f, charttime=record['charttime']) for f in ct)
                    for q in quotes:
                        q.update(start=q['start'] + offset, end=q['end'] + offset)
                        if record['text'][q['start']:q['end']] != q['quote']: raise ValueError('quote mismatch')
                    evidence.extend(quotes)
                def merge(v):
                    return 3 if 1 in v and 2 in v else 1 if 1 in v else 3 if 3 in v else 2 if 2 in v else 0
                saved.update(labels14={k: merge(v) for k, v in labels.items()}, chunk_conflicts={}, ct_facts=facts,
                             evidence=evidence, chunks=len(pieces))
                grouped[pos, kind].append(saved)
            out.write(json.dumps(saved, ensure_ascii=False) + '\n'); out.flush()
            if (ordinal + 1) % 500 == 0 or ordinal == 0:
                print(json.dumps({'kind': args.kind, 'reports': ordinal + 1, 'total': len(data),
                                  'seconds': time.perf_counter() - started}), flush=True)
    for (pos, kind), values in grouped.items():
        if args.kind == 'radbert':
            by_time = defaultdict(list)
            for t, v in values: by_time[t].append(v)
            latest = np.mean(by_time[max(by_time)], axis=0)
            average = np.mean([np.mean(v, axis=0) for v in by_time.values()], axis=0)
            for v in (latest, average): v /= max(float(np.linalg.norm(v)), 1e-12)
            vectors[pos, kind] = np.concatenate([latest, average])
        elif kind == 0:
            graphs[pos, kind, :98] = aggregate(values).reshape(-1)
        else:
            fs = [f for r in values for f in r['ct_facts']]
            graphs[pos, kind, :60] = ct_vector(fs)
    values = vectors if args.kind == 'radbert' else graphs
    np.savez_compressed(args.output / 'imaging.npz', **members, values=values, present=all_present)
    result = {'status': 'full_radiology_processed' if not args.limit else 'pilot_not_full', 'kind': args.kind,
              'reports': len(data), 'chunks': chunks, 'seconds': time.perf_counter() - started,
              'resumed_completed_reports': completed, 'seconds_scope': 'current process only; excludes prior inference and pause' if completed else 'complete current process',
              'resume_origin_sha256': sha256(args.output / 'resume_origin.json') if completed else None,
              'source_manifest_sha256': sha256(args.selected / 'manifest.json'),
              'parent_state_manifest_sha256': manifest['parent_state_manifest_sha256'],
              'asset_manifest_sha256': sha256(assets / 'asset_manifest.json'),
              'code_sha256': sha256(Path(__file__)),
              'radgraph_backbone_manifest_sha256': sha256(backbone / 'asset_manifest.json') if args.kind == 'radgraph' else None,
              'output_sha256': {s: sha256(args.output / s) for s in ('reports.jsonl', 'imaging.npz')},
              'dimensions': [2, 1536], 'encoder_frozen': True,
              'runtime': {'torch': torch.__version__, 'transformers': __import__('transformers').__version__,
                          'device': torch.cuda.get_device_name(0), 'radgraph_reference_compile': False if args.kind == 'radgraph' else None},
              'limits': ['Not expert accuracy; RadGraph concepts mapped from model entities and relations',
                         'Clinical fact padding and dense embedding are different representations',
                         'No report count given as an extra numerical feature']}
    (args.output / 'manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selected', type=Path, default=Path('patient_data/processed/v6_imaging_all_2026-10-10_r2'))
    p.add_argument('--kind', choices=('radbert', 'radgraph'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=8); p.add_argument('--limit', type=int, default=0)
    p.add_argument('--resume', action='store_true')
    build(p.parse_args())
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/clinical_facts/refine_sentence_screen.py · lines 1–74

<!-- REVIEW_FRAGMENT {"path":"patient_data/clinical_facts/refine_sentence_screen.py","part":1,"start_line":1,"end_line":74,"source_sha256":"4a3d5ab6e5cf0a498d1b9b43bb71dfa75b772fa1efe6d4b31c5ad9ef37f8bc8e","fragment_sha256":"4a3d5ab6e5cf0a498d1b9b43bb71dfa75b772fa1efe6d4b31c5ad9ef37f8bc8e","added_terminal_newline":false} -->
````python
"""复用已核验句段边界，局部补充规则再筛；全量原句仍保留，不重读/重切原文。"""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import random
import re
import time
from patient_data.export_mimiciv import rows, sha256
from patient_data.clinical_facts.sentence_rules import classify, VERSION, RULE_SHA
from patient_data.clinical_facts.sentence_screen import write


def build(args):
    if args.output.exists(): raise FileExistsError(args.output)
    parent = json.loads((args.parent / 'manifest.json').read_text())
    for name, value in parent['output_sha256'].items():
        if sha256(args.parent / name) != value: raise ValueError('parent changed')
    excluded = set()
    for path in args.development_review:
        with path.open(encoding='utf-8') as stream: excluded.update(r['note_id'] for r in map(json.loads, stream))
    started = time.perf_counter(); args.output.mkdir(parents=True)
    counts = defaultdict(Counter); topics = defaultdict(Counter); changed = 0; per_doc = Counter()
    review = defaultdict(list); seen = Counter(); rng = random.Random(44)
    # 本次规则改动只影响这些词/已选句；没有命中的未选句复用原判断。
    changed_terms = re.compile(r'opaci|scar|colonic|colon\b|bowel|rectal|vitals?|\bBP\b|\bHR\b|\bRR\b|O2\s+sat|no pathologic|ventilation|home\s+with|cream|ointment|topical|Flonase|nasal\s+spray', re.I)
    with gzip.open(args.parent / 'all_units.jsonl.gz', 'rt', encoding='utf-8') as source, \
         gzip.open(args.output / 'all_units.jsonl.gz', 'xt', encoding='utf-8', compresslevel=1) as audit, \
         gzip.open(args.output / 'selected_sentences.jsonl.gz', 'xt', encoding='utf-8', compresslevel=1) as selected:
        for row in map(json.loads, source):
            if row['selected'] or changed_terms.search(row['quote']):
                decision = classify(row['quote'], row['section'], row['kind'])
                changed += int(any(row[k] != decision[k] for k in decision)); row.update(decision)
            c = counts[row['kind']]; c['units'] += 1; c['unit_chars'] += len(row['quote']); c[row['reason']] += 1
            write(audit, row)
            if row['selected']:
                write(selected, row); c['selected_chars'] += len(row['quote']); topics[row['kind']].update(row['topics']); per_doc[row['note_id']] += 1
            if row['note_id'] in excluded: continue
            bucket = (row['kind'], row['selected']); seen[bucket] += 1
            if len(review[bucket]) < 4: review[bucket].append(row.copy())
            else:
                pos = rng.randrange(seen[bucket])
                if pos < 4: review[bucket][pos] = row.copy()
    with (args.parent / 'documents.jsonl').open(encoding='utf-8') as source, (args.output / 'documents.jsonl').open('x', encoding='utf-8') as dest:
        for row in map(json.loads, source):
            row['selected'] = per_doc[row['note_id']]; c = counts[row['kind']]
            c['documents'] += 1; c['documents_with_selection'] += int(row['selected'] > 0); write(dest, row)
    wanted = {r['note_id'] for batch in review.values() for r in batch}; originals = {}
    for name in ('discharge', 'radiology'):
        for row in rows(args.snapshot / ('raw/' + name + '.csv.gz')):
            if row['note_id'] in wanted: originals[row['note_id']] = row['text']
    if originals.keys() != wanted: raise ValueError('missing review original')
    with (args.output / 'review_sample.jsonl').open('x', encoding='utf-8') as stream:
        for bucket in sorted(review):
            for row in review[bucket]:
                row['context'] = originals[row['note_id']][row['context_start']:row['context_end']]; write(stream, row)
    m = dict(status='completed_rule_candidates_not_semantic_gold', version=VERSION, rule_sha256=RULE_SHA,
        counts={k: dict(v) for k, v in counts.items()}, topics={k: dict(v) for k, v in topics.items()},
        code_sha256=sha256(Path(__file__)), rule_code_sha256=sha256(Path(__file__).with_name('sentence_rules.py')),
        source_manifest_sha256=parent['source_manifest_sha256'], parent_manifest_sha256=sha256(args.parent / 'manifest.json'),
        elapsed_seconds=time.perf_counter()-started, changed_units=changed, development_review_excluded_notes=len(excluded),
        policy=parent['policy'] + '; source-reviewed local rule refinement reuses source-audited boundaries',
        output_sha256={n: sha256(args.output / n) for n in ('all_units.jsonl.gz','selected_sentences.jsonl.gz','documents.jsonl','review_sample.jsonl')})
    (args.output / 'manifest.json').write_text(json.dumps(m, indent=2), encoding='utf-8')
    print(json.dumps({k: m[k] for k in ('status','counts','elapsed_seconds','changed_units')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--development-review', nargs='+', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, default=Path('patient_data/mimiciv_copd/snapshot_2026-09-27'))
    build(parser.parse_args())
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/clinical_facts/runner.py · lines 1–117

<!-- REVIEW_FRAGMENT {"path":"patient_data/clinical_facts/runner.py","part":1,"start_line":1,"end_line":117,"source_sha256":"4c1318b2fc7b9449464b41d82258f955e9e24b27905a5800d0266cc69e122b9a","fragment_sha256":"4c1318b2fc7b9449464b41d82258f955e9e24b27905a5800d0266cc69e122b9a","added_terminal_newline":false} -->
````python
"""本地四并发、成功键续跑、有限失败重试、逐条落盘的事实抽取器。"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import json
import os
from pathlib import Path
import time
import requests
from patient_data.export_mimiciv import sha256
from patient_data.clinical_facts.catalog import CATALOG_SHA
from patient_data.clinical_facts.llm import LocalBonsai,ExtractionError,digest
from patient_data.clinical_facts.guards import guard

def atomic_json(path,value):
    temp=path.with_suffix('.tmp'); temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    # Windows读状态/杀毒软件可短暂持有无DELETE共享的句柄；仅重试原子替换，不重跑抽取。
    for attempt in range(40):
        try:
            os.replace(temp,path)
            return
        except PermissionError:
            if attempt == 39: raise
            time.sleep(.05)

def perform(task,client):
    errors=[]; started=time.perf_counter(); response=None
    for attempt in range(1,3):
        try:
            response=client.extract(task['route'],' | '.join(dict.fromkeys(m['role'] for m in task['source_maps'])),task['text'])
            facts=[]; rejects=list(response['rejected'])
            for fact in response['facts']:
                accepted,error=guard(fact,task)
                if accepted is not None: facts.append(accepted)
                else: rejects.append(error)
            # 有无法核对的字段不能被静默宣布完整；先重试一次。
            if any(r in ('nonverbatim_evidence','invalid_fact_schema','invalid_field_or_value') for r in rejects) and attempt<2:
                errors.append('rejected_evidence_retry'); continue
            status='success_with_rejections' if rejects else 'success'
            return {'task_key':task['task_key'],'status':status,'facts':facts,'rejected':rejects,
                    'attempts':attempt,'errors':errors,'seconds':time.perf_counter()-started,'response':response}
        except (requests.RequestException,ExtractionError,KeyError,ValueError) as error:
            errors.append(type(error).__name__+': '+str(error))
            if isinstance(error,ExtractionError): response=error.response
    return {'task_key':task['task_key'],'status':'failed','facts':[],'rejected':[],
            'attempts':2,'errors':errors,'seconds':time.perf_counter()-started,'response':response}

def run(args):
    manifest=json.loads((args.tasks/'manifest.json').read_text())
    if manifest['catalog_sha256']!=CATALOG_SHA: raise ValueError('catalog changed')
    for name,value in manifest['output_sha256'].items():
        if sha256(args.tasks/name)!=value: raise ValueError('task source changed')
    with (args.tasks/'tasks.jsonl').open(encoding='utf-8') as f: tasks=[json.loads(line) for line in f]
    if len(tasks)!=manifest['tasks']: raise ValueError('task count changed')
    if args.channel!='all': tasks=[t for t in tasks if t['route']==args.channel]
    if args.limit: tasks=tasks[:args.limit]
    client=LocalBonsai(thinking=args.thinking); args.output.mkdir(parents=True,exist_ok=True)
    code={n:sha256(Path(__file__).parent/n) for n in ('catalog.py','llm.py','guards.py','runner.py')}
    model_path=Path('D:/ywj/Bonsai-demo/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PTQ1_0.gguf')
    model_sha=sha256(model_path)
    if model_sha!='53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3': raise ValueError('GGUF identity changed')
    contract={'tasks_manifest_sha256':sha256(args.tasks/'manifest.json'),'catalog_sha256':CATALOG_SHA,
              'model_sha256':model_sha,'settings':client.settings,'code_sha256':code,'parallel':args.parallel,
              'channel':args.channel,'limit':args.limit,'tasks':len(tasks)}
    contract_path=args.output/'contract.json'
    if contract_path.exists() and json.loads(contract_path.read_text())!=contract: raise ValueError('resume contract differs')
    atomic_json(contract_path,contract)
    # 独占锁避免两个本机进程同时写结果；崩溃保留PID供人工查证，不自动删锁。
    lock=args.output/'running.lock'
    with lock.open('x') as f: f.write(str(os.getpid()))
    records={}; destination=args.output/'results.jsonl'
    if destination.exists():
        with destination.open(encoding='utf-8') as f:
            for line in f:
                result=json.loads(line); records[result['task_key']]=result
    pending=[t for t in tasks if records.get(t['task_key'],{}).get('status') not in ('success','success_with_rejections')]
    started=time.perf_counter(); initial=len(tasks)-len(pending)
    def status(state):
        relevant=[records[t['task_key']] for t in tasks if t['task_key'] in records]
        counts=Counter(r['status'] for r in relevant)
        valid=sum(counts[s] for s in ('success','success_with_rejections'))
        return {'status':state,'pid':os.getpid(),'total':len(tasks),'successful':valid,'failed':counts['failed'],
                'with_rejections':counts['success_with_rejections'],'new_completed':len(relevant)-initial,
                'elapsed_seconds':time.perf_counter()-started,'effective_tasks_per_second':max(0,len(relevant)-initial)/max(.001,time.perf_counter()-started),
                'rejection_counts':dict(Counter(reason for r in relevant for reason in r['rejected'])),
                'accepted_facts':sum(len(r['facts']) for r in relevant)}
    try:
        atomic_json(args.output/'status.json',status('running'))
        # 持续填满四槽但只持有四个Future；不等同批最慢任务才发下一批。
        with destination.open('a',encoding='utf-8') as f,ThreadPoolExecutor(max_workers=args.parallel) as pool:
            iterator=iter(pending); futures=set()
            for _ in range(args.parallel):
                task=next(iterator,None)
                if task is not None: futures.add(pool.submit(perform,task,client))
            completed=0
            while futures:
                done,futures=wait(futures,return_when=FIRST_COMPLETED)
                for future in done:
                    result=future.result(); records[result['task_key']]=result
                    f.write(json.dumps(result,ensure_ascii=False)+'\n'); f.flush(); os.fsync(f.fileno())
                    completed+=1
                    task=next(iterator,None)
                    if task is not None: futures.add(pool.submit(perform,task,client))
                progress=status('running'); atomic_json(args.output/'status.json',progress)
                if completed%100==0 or completed==1: print(json.dumps(progress),flush=True)
        final=status('completed' if all(records[t['task_key']]['status'] in ('success','success_with_rejections') for t in tasks) else 'finished_with_failed_tasks')
        final.update(results_sha256=sha256(destination),contract_sha256=sha256(contract_path))
        atomic_json(args.output/'status.json',final); print(json.dumps(final),flush=True)
    finally:
        lock.unlink()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks',type=Path,default=Path('patient_data/processed/v6_bonsai_tasks_2026-10-10'))
    p.add_argument('--output',type=Path,required=True); p.add_argument('--parallel',type=int,choices=range(1,5),default=4)
    p.add_argument('--channel',choices=('all','chest_ct','discharge_all'),default='all'); p.add_argument('--limit',type=int,default=0)
    p.add_argument('--thinking',action='store_true'); args=p.parse_args(); run(args)
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/clinical_facts/sanitize_radgraph.py · lines 1–66

<!-- REVIEW_FRAGMENT {"path":"patient_data/clinical_facts/sanitize_radgraph.py","part":1,"start_line":1,"end_line":66,"source_sha256":"136aa93f92018f486cacbd3ea1e9098892a1fc1aad0999af22a16da994b95adc","fragment_sha256":"136aa93f92018f486cacbd3ea1e9098892a1fc1aad0999af22a16da994b95adc","added_terminal_newline":false} -->
````python
"""剔除跨自然章节的实体后重聚合；保留完整公开模型原输出，不重跑GPU。"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from patient_data.export_mimiciv import sha256
from patient_data.clinical_facts.radiology_compare import records
from patient_data.clinical_facts.cxr_trajectory import SELECTED, aggregate
from patient_data.clinical_facts.features import ct_vector
from patient_data.clinical_facts.catalog import CT_FIELDS


def build(args):
    if args.output.exists(): raise FileExistsError(args.output)
    source, members, selected = records(args.selected); by_id = {r['note_id']: r for r in source}
    manifest = json.loads((args.source / 'manifest.json').read_text())
    if manifest['status'] != 'full_radiology_processed' or manifest['kind'] != 'radgraph': raise ValueError('not full RadGraph')
    for name, checksum in manifest['output_sha256'].items():
        if sha256(args.source / name) != checksum: raise ValueError('input changed')
    with (args.source / 'reports.jsonl').open(encoding='utf-8') as f: reports = list(map(json.loads, f))
    if len(reports) != len(source) or {r['note_id'] for r in reports} != by_id.keys(): raise ValueError('reports differ')
    dropped = []; groups = defaultdict(list)
    for r in reports:
        maps = []; cursor = 0
        for s in by_id[r['note_id']]['sections']:
            maps.append((cursor, cursor + len(s['text']))); cursor += len(s['text']) + 1
        valid = []
        for e in r['evidence']:
            if by_id[r['note_id']]['text'][e['start']:e['end']] != e['quote']: raise ValueError('input evidence differs')
            if any(a <= e['start'] < e['end'] <= b for a, b in maps): valid.append(e)
            else: dropped.append(dict(e, note_id=r['note_id'], reason='entity_spans_distinct_natural_sections'))
        # 必须在剔除无效实体后重建标签，而非仅改引用使旧错误标签仍然进入模型。
        labels = {k: [] for k in SELECTED}; facts = []
        for e in valid:
            for field in e['fields']:
                if field in SELECTED: labels[field].append(e['state'])
                elif field in CT_FIELDS: facts.append(dict(field=field, value={1: 'present', 2: 'absent', 3: 'uncertain'}[e['state']], charttime=r['charttime']))
        def merged(values):
            return 3 if 1 in values and 2 in values else 1 if 1 in values else 3 if 3 in values else 2 if 2 in values else 0
        r.update(evidence=valid, ct_facts=facts, labels14={k: merged(v) for k, v in labels.items()},
                 chunk_conflicts={k: 1 in v and 2 in v for k, v in labels.items()})
        groups[r['hadm_id'], 0 if r['modality'] == 'cxr' else 1].append(r)
    n = len(members['hadm_ids']); index = {str(h): i for i, h in enumerate(members['hadm_ids'])}
    values = np.zeros((n, 2, 1536), np.float32); present = np.zeros((n, 2), bool)
    for (hadm, kind), rs in groups.items():
        i = index[hadm]; present[i, kind] = True
        vector = aggregate(rs).reshape(-1) if kind == 0 else ct_vector([f for r in rs for f in r['ct_facts']])
        values[i, kind, :len(vector)] = vector
    args.output.mkdir(parents=True)
    for name, values_out in (('reports.jsonl', reports), ('excluded_entities.jsonl', dropped)):
        with (args.output / name).open('x', encoding='utf-8') as f:
            for r in values_out: f.write(json.dumps(r, ensure_ascii=False) + '\n')
    np.savez_compressed(args.output / 'imaging.npz', **members, values=values, present=present)
    manifest.update(parent_radgraph_manifest_sha256=sha256(args.source / 'manifest.json'),
                    excluded_cross_section_entities=len(dropped), sanitization_code_sha256=sha256(Path(__file__)),
                    output_sha256={name: sha256(args.output / name) for name in ('reports.jsonl', 'excluded_entities.jsonl', 'imaging.npz')})
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'status': manifest['status'], 'reports': len(reports), 'excluded_cross_section_entities': len(dropped)}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selected', type=Path, default=Path('patient_data/processed/v6_imaging_all_2026-10-10_r2'))
    p.add_argument('--source', type=Path, default=Path('patient_data/processed/v6_radgraph_imaging_2026-10-10_r2'))
    p.add_argument('--output', type=Path, required=True); build(p.parse_args())
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/clinical_facts/sentence_rules.py · lines 1–86

<!-- REVIEW_FRAGMENT {"path":"patient_data/clinical_facts/sentence_rules.py","part":1,"start_line":1,"end_line":86,"source_sha256":"74fdb09cd35738b64d80a5857a6587fa3dd953d0997c0bdd942943c415c3b3b0","fragment_sha256":"74fdb09cd35738b64d80a5857a6587fa3dd953d0997c0bdd942943c415c3b3b0","added_terminal_newline":false} -->
````python
"""仅筛选临床原句的可审计规则；主题/词面线索不是疾病事实或预测变量。"""
import hashlib
import json
import re

VERSION = 'clinical-sentence-screen-v4-source-reviewed'
# 不沿用旧8/14字段的出院残余约束；允许原文多主题，时点另存。
TOPICS = {
    'respiratory_status': r'\b(?:COPD|AECOPD|asthma|dyspn\w*|SOB|DOE|breathless\w*|shortness\s+of\s+breath|wheez\w*|cough\w*|sputum|phlegm|respiratory\s+(?:failure|distress|insufficiency)|work of breathing|accessory muscles?|mucus\s+plug\w*)\b',
    'oxygen_ventilation': r'\b(?:oxygen|O2|SpO2|room air|nasal cannula|desatur\w*|hypox\w*|hypercap\w*|BiPAP|CPAP|NIV|NIPPV|intubat\w*|extubat\w*|ventilat\w*|wean\w*|on RA|\d+\s*L\s*NC)\b',
    'respiratory_treatment': r'\b(?:predni\w*|methylpred\w*|hydrocortisone|dexamethasone|steroid\w*|solu.?medrol|albuterol|salbutamol|ipratropium|duoneb|salmeterol|formoterol|arformoterol|tiotropium|umeclidinium|budesonide|fluticasone|advair|symbicort|spiriva|incruse|trelegy|breo|anoro|combivent|inhaler\w*|nebuliz\w*|azithro\w*|doxycycl\w*|levoflox\w*|moxiflox\w*|ceftriaxone|cefepime|amoxicillin|augmentin|piperacillin|tazobactam|vancomycin|meropenem|antibiotic\w*)\b',
    'respiratory_microbiology': r'\b(?:sputum\s+cultur\w*|BAL|bronchoalveolar|tracheal\s+aspirat\w*|pseudomon\w*|haemophilus|streptococcus pneumoniae|legionella|influenza|RSV|rhinovirus|respiratory syncytial|coloniz\w*|resistan\w*)\b',
    'pulmonary_function': r'\b(?:FEV\s*1|FVC|DLCO|PFTs?|spirometr\w*|pulmonary\s+function)\b',
    'cardiopulmonary_context': r'\b(?:CHF|heart failure|pulmonary\s+hypertension|cor pulmonale|cardiomyopath\w*|chest pain|myocardial infarct\w*|acute coronary|smok\w*|tobacco|pack.?years?)\b',
    'function_frailty': r'\b(?:ADLs?|frail\w*|cachex\w*|decondition\w*|bedbound|wheelchair|walker|ambulat\w*|mobility|functional\s+(?:status|dependence)|requires?\s+assist\w*|unable\s+to\s+walk)\b',
    'management_barriers': r'\b(?:adher\w*|non.?complian\w*|not\s+(?:taking|using)|inhaler technique|spacer|cannot afford|unable to afford|financial barrier|medication access|lives?\s+(?:alone|with)|homeless|caregiver|home\s+(?:with\s+)?(?:services?|health|nursing|oxygen)|visiting nurse|VNA|SNF|skilled nursing|rehab\w*)\b',
    'care_limits': r'\b(?:DNR|DNI|do not resuscitate|do not intubate|hospice|comfort\s+(?:care|measures)|AMA|against medical advice)\b',
    'thoracic_parenchyma': r'\b(?:emphysem\w*|hyperinfl\w*|overinfl\w*|bullae|bulla|bullous|blebs?|consolidat\w*|infiltrat\w*|pneumonia|atelecta\w*|ground.?glass|opaci\w*|scar\w*|fibro\w*|reticul\w*|interstitial|(?:pulmonary|re.?expansion)\s+edema|vascular\s+congestion|lung\s+volumes?|volume\s+loss|ventilation\s+(?:at|of|in)\s+the\s+(?:left|right|bilateral)\s+lung)\b',
    'thoracic_airway': r'\b(?:bronchiecta\w*|bronchi\w*\s+(?:dilat\w*|wall|thicken\w*)|airway\s+(?:wall|thicken\w*)|mucus\s+plug\w*|mucoid\s+impaction|endobronchial)\b',
    'thoracic_pleura_vascular': r'\b(?:effusion\w*|pneumothora\w*|pleural\s+(?:fluid|thicken\w*)|pulmonary\s+(?:arter\w*|embol\w*)|cardiomegal\w*|enlarged\s+heart)\b',
    'thoracic_other': r'\b(?:nodules?|mass(?:es)?|neoplasm\w*|malignan\w*)\b',
    'thoracic_cardiac': r'\b(?:pericardial\s+effusion|heart\s+(?:size|is|appears)|cardiac silhouette)\b',
}
COMPILED = {k: re.compile(v, re.I) for k, v in TOPICS.items()}
IMAGING_TOPICS = {k for k in TOPICS if k.startswith('thoracic_')}
CUES = {
    'negation_words': r'\b(?:no|not|without|denies?|negative for|resolved|absent)\b',
    'uncertainty_words': r'\b(?:possible|possibly|may|might|probable|likely|suggest\w*|cannot exclude)\b',
    'change_words': r'\b(?:new|worsen\w*|improv\w*|resolv\w*|persist\w*|unchanged|increas\w*|decreas\w*|wean\w*|taper\w*)\b',
    'discharge_words': r'\b(?:discharg\w*|at home|home oxygen)\b',
    'historical_words': r'\b(?:previous\w*|history of|baseline|prior|formerly)\b',
    'plan_words': r'\b(?:will|plan\w*|continue|remaining|to complete|follow.?up|recommend\w*)\b',
}
WARNING = re.compile(r'\b(?:if you (?:develop|experience|have)|call\s+911|return to (?:the )?(?:ER|emergency)|seek immediate medical)\b', re.I)
FUTURE_OUTCOME = re.compile(r'\b(?:subsequently readmitted|readmitted after discharge|died after discharge|post.?discharge admission)\b', re.I)
DEVICE = re.compile(r'\b(?:tube|line|catheter|pacemaker|port|endotracheal|ETT|PICC|lead)\b', re.I)
TECHNIQUE = re.compile(r'\b(?:portable|technique|AP view|PA and lateral|contrast|dose|motion artifact)\b', re.I)
# 所有章节都保留审计，不把实验室表、家族史、联系方式变成正文证据。
EXCLUDED_ROLES = {'metadata', 'allergies', 'family_history', 'admission_labs', 'discharge_labs'}
NONTHORACIC = re.compile(r'\b(?:adrenal|renal|kidney|hepatic|liver|pancrea\w*|splenic|spleen|thyroid|prostat\w*|pelvic|uter\w*|ovari\w*|abdom\w*|colonic|colon|bowel|rectal)\b', re.I)
THORACIC_LOCATION = re.compile(r'\b(?:lung\w*|pulmonary|bronchi\w*|airway|pleur\w*|thorac\w*|chest|mediastin\w*|hilar|lobes?|basilar|apical|pericardial|heart|cardiac)\b', re.I)
NONRESP_INDICATION = re.compile(r'\b(?:UTI|urinary|renal\s+(?:collection|abscess|infection)|pyelonephritis|cellulitis|abdominal\s+infection|adrenal insufficiency|gout|skin rash)\b', re.I)
VITAL_TABLE = re.compile(r'\b(?:vitals?|BP\s*\d|HR\s*\d|RR\s*\d|O2\s+sat\s*\d)\b', re.I)
TOPICAL_DRUG = re.compile(r'\b(?:cream|ointment|topical|pruriti\w*|eczema|Flonase|nasal\s+spray)\b', re.I)
RULE_SHA = hashlib.sha256(json.dumps({'version': VERSION, 'topics': TOPICS, 'cues': CUES,
    'excluded_roles': sorted(EXCLUDED_ROLES), 'warning': WARNING.pattern,
    'future_outcome': FUTURE_OUTCOME.pattern, 'device': DEVICE.pattern,
    'technique': TECHNIQUE.pattern, 'nonthoracic': NONTHORACIC.pattern,
    'thoracic_location': THORACIC_LOCATION.pattern, 'nonresp_indication': NONRESP_INDICATION.pattern,
    'vital_table': VITAL_TABLE.pattern, 'topical_drug': TOPICAL_DRUG.pattern}, sort_keys=True).encode()).hexdigest()


def classify(text, role, kind):
    allowed = IMAGING_TOPICS if kind != 'discharge' else TOPICS.keys()
    topics = sorted(k for k in allowed if COMPILED[k].search(text))
    flags = []
    if DEVICE.search(text): flags.append('device_mentioned')
    if TECHNIQUE.search(text): flags.append('technical_mentioned')
    if WARNING.search(text): flags.append('warning_or_hypothetical')
    reason = 'selected_candidate' if topics else 'no_topic_match'
    if NONTHORACIC.search(text) and not THORACIC_LOCATION.search(text) and topics and all(k.startswith('thoracic_') for k in topics):
        reason = 'explicit_nonthoracic_finding'
    if 'respiratory_treatment' in topics:
        flags.append('drug_indication_not_inferred')
        if NONRESP_INDICATION.search(text) and set(topics) == {'respiratory_treatment'}:
            reason = 'explicit_nonrespiratory_indication'
        if TOPICAL_DRUG.search(text) and set(topics) == {'respiratory_treatment'}:
            # 不把外用/鼻用制剂当作下气道治疗；同句含其他临床主题仍保留原句。
            reason = 'topical_or_nasal_only'
    if role in EXCLUDED_ROLES: reason = 'excluded_section'
    if kind == 'discharge' and VITAL_TABLE.search(text) and set(topics) <= {'oxygen_ventilation'}:
        if not re.search(r'\b(?:wean\w*|require\w*|increas\w*|decreas\w*|discharg\w*|exert\w*|ambulat\w*|BiPAP|CPAP|intubat\w*)\b', text, re.I):
            reason = 'isolated_vital_table'
    if FUTURE_OUTCOME.search(text): reason = 'observed_future_outcome'
    # 单纯的防范模板才过滤；有其他实际陈述的混合句保留待复核。
    if WARNING.search(text) and not re.search(r'\b(?:patient|he|she|currently|still|discharged|weaned)\b', text, re.I):
        reason = 'generic_warning'
    if kind != 'discharge' and not topics and reason == 'no_topic_match':
        if DEVICE.search(text): reason = 'device_only'
        elif TECHNIQUE.search(text): reason = 'technical_only'
        elif re.search(r'\b(?:no acute|clear lungs?|lungs? (?:are )?clear|unremarkable chest|no pathologic changes in the lung)\b', text, re.I):
            topics = ['thoracic_explicit_normal']; reason = 'selected_candidate'
    cues = {name: [m.group() for m in re.finditer(pattern, text, re.I)] for name, pattern in CUES.items()}
    # 同一句的词面线索并不代表每个主题都拥有相同的肯否/时相。
    return dict(topics=topics, selected=reason == 'selected_candidate', reason=reason,
                flags=flags, context_cues={k: v for k, v in cues.items() if v})
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/clinical_facts/sentence_screen.py · lines 1–172

<!-- REVIEW_FRAGMENT {"path":"patient_data/clinical_facts/sentence_screen.py","part":1,"start_line":1,"end_line":172,"source_sha256":"3dc4d9fccc7bbf0a7555f27b73d55f28815d2e6420156b04b1e10f8a249d16ac","fragment_sha256":"3dc4d9fccc7bbf0a7555f27b73d55f28815d2e6420156b04b1e10f8a249d16ac","added_terminal_newline":false} -->
````python
"""三类文书逐原句筛选：全量未选句也留审计；不调用LLM、不编码、不训练。"""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
import gzip
from pathlib import Path
import random
import re
import time
import numpy as np
from patient_data.export_mimiciv import rows, sha256
from patient_data.clinical_facts.llm import digest
from patient_data.clinical_facts.sentence_rules import classify, VERSION, RULE_SHA
from training.v5.section_alignment_data.section_parser import parse_sections


def units(text):
    """保持原文字符索引；小数/常用缩写不切开，列表和空行分界。"""
    boundaries = {0, len(text)}
    for match in re.finditer(r'\n[ \t\r]*\n|\n(?=[ \t]*(?:[-*#]|\d+[.)]|[A-Za-z][A-Za-z /]{1,35}:)[ \t]*)', text):
        boundaries.add(match.start()); boundaries.add(match.end())
    for match in re.finditer(r'[.!?](?=\s+[A-Z0-9])', text):
        pos = match.start()
        before = text[max(0, pos - 18):pos + 1]
        if re.search(r'(?:^|\n)[ \t]*\d+\.$', text[:pos + 1]):
            continue  # 编号应与其条目保持同一语义单元。
        if text[pos] == '.' and (re.search(r'\b(?:Dr|Mr|Mrs|Ms|vs|e\.g|i\.e|q\.d|b\.i\.d|t\.i\.d)\.$', before, re.I)
                                 or re.search(r'\b[A-Z]\.$', before)):
            continue
        boundaries.add(pos + 1)
    result = []
    ordered = sorted(boundaries)
    for left, right in zip(ordered, ordered[1:]):
        while left < right and text[left].isspace(): left += 1
        while right > left and text[right - 1].isspace(): right -= 1
        if left < right: result.append((left, right))
    # 非空白字符必须且只能覆盖一次；标题不在传入body里。
    if ''.join(text[a:b] for a, b in result).split() != text.split():
        # 标点末尾相邻片段会影响split token；比较去空白字符串才是覆盖判据。
        if re.sub(r'\s', '', ''.join(text[a:b] for a, b in result)) != re.sub(r'\s', '', text):
            raise ValueError('sentence segmentation lost characters')
    return result


def screen(original, sections, kind):
    output = []
    for ordinal, section in enumerate(sections):
        start, end, body = section['start_char'], section['end_char'], section['text']
        role = section.get('role', section.get('kind', 'unclassified'))
        if not 0 <= start <= end <= len(original) or original[start:end] != body:
            raise ValueError('invalid original section offsets')
        spans = units(body)
        for i, (a, b) in enumerate(spans):
            # 相邻语句只作上下文，主题筛选根据主句。跨章节不能合并。
            ca = spans[max(0, i - 1)][0]; cb = spans[min(len(spans) - 1, i + 1)][1]
            row = dict(section=role, section_ordinal=ordinal, unit_ordinal=i,
                       start_char=start+a, end_char=start+b, quote=body[a:b],
                       context_start=start+ca, context_end=start+cb)
            row.update(classify(row['quote'], role, kind))
            output.append(row)
    return output


def checked_manifest(directory):
    m = json.loads((directory / 'manifest.json').read_text())
    for name, value in m['output_sha256'].items():
        if sha256(directory / name) != value: raise ValueError('source file changed: ' + name)
    return m


def sources(args):
    """旧任务清单只用来核对最终文书ID，全文章节重新切分，非旧主题裁剪。"""
    checked_manifest(args.notes); checked_manifest(args.imaging)
    with np.load(args.imaging / 'members.npz') as data:
        people = {str(h): str(s) for h, s in zip(data['hadm_ids'], data['subject_ids'], strict=True)}
        folds = {str(h): str(f) for h, f in zip(data['hadm_ids'], data['patient_splits'], strict=True)}
    with (args.notes / 'sources.jsonl').open(encoding='utf-8') as stream:
        summaries = [r for r in map(json.loads, stream) if r['channel'] == 'discharge']
    wanted = {}
    for kind in ('cxr', 'chest_ct'):
        with (args.imaging / (kind + '_reports.jsonl')).open(encoding='utf-8') as stream:
            for r in map(json.loads, stream): wanted[r['note_id']] = (kind, r)
    raw_m = json.loads((args.snapshot / 'metadata/manifest.json').read_text())
    for name in ('raw/radiology', 'raw/discharge'):
        path = args.snapshot / (name + '.csv.gz')
        if sha256(path) != raw_m['sources'][name]['sha256']: raise ValueError('original snapshot changed')
    expected = {r['note_id']: r for r in summaries}
    for r in rows(args.snapshot / 'raw/discharge.csv.gz'):
        if r['note_id'] not in expected: continue
        selected = expected.pop(r['note_id'])
        if digest(r['text']) != selected['original_sha256'] or r['text'] != selected['text']:
            raise ValueError('summary original changed')
        sections = [asdict(s) for s in parse_sections(r['text'])]
        if not sections: sections = [dict(role='unclassified', start_char=0, end_char=len(r['text']), text=r['text'])]
        yield 'discharge', r, sections, people, folds
    if expected: raise ValueError('summary not found in raw source')
    for r in rows(args.snapshot / 'raw/radiology.csv.gz'):
        if r['note_id'] not in wanted: continue
        kind, selected = wanted.pop(r['note_id'])
        if digest(r['text']) != selected['original_sha256']: raise ValueError('radiology original changed')
        yield kind, r, selected['sections'], people, folds
    if wanted: raise ValueError('radiology not found in raw source')


def write(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def build(args):
    if args.output.exists(): raise FileExistsError(args.output)
    started = time.perf_counter(); args.output.mkdir(parents=True)
    counts = defaultdict(Counter); selected_topics = defaultdict(Counter); review = defaultdict(list)
    rng = random.Random(43); reservoir_seen = Counter(); notes = []; unique = set()
    excluded_review_notes = set()
    if args.development_review:
        with args.development_review.open(encoding='utf-8') as stream:
            excluded_review_notes = {r['note_id'] for r in map(json.loads, stream)}
    with gzip.open(args.output / 'all_units.jsonl.gz', 'xt', encoding='utf-8', compresslevel=1) as audit, \
         gzip.open(args.output / 'selected_sentences.jsonl.gz', 'xt', encoding='utf-8', compresslevel=1) as chosen:
        for kind, raw, sections, people, folds in sources(args):
            h = raw['hadm_id']
            if h not in people or people[h] != raw['subject_id'] or raw['note_id'] in unique:
                raise ValueError('member/identity/duplicate mismatch')
            unique.add(raw['note_id']); stats = counts[kind]; stats['documents'] += 1
            doc_units = screen(raw['text'], sections, kind)
            meta = {k: raw[k] for k in ('subject_id', 'hadm_id', 'note_id', 'charttime', 'storetime')}
            meta.update(kind=kind, patient_split=folds[h], original_sha256=digest(raw['text']))
            notes.append(dict(meta, section_spans=[{k: s[k] for k in ('start_char', 'end_char')} for s in sections],
                              units=len(doc_units), selected=sum(v['selected'] for v in doc_units)))
            for unit in doc_units:
                row = dict(meta, **unit)
                row['unit_key'] = digest(raw['note_id'] + ':' + str(unit['start_char']) + ':' + str(unit['end_char']))
                if raw['text'][row['start_char']:row['end_char']] != row['quote']: raise ValueError('quote differs')
                stats['units'] += 1; stats['unit_chars'] += len(row['quote']); stats[row['reason']] += 1
                write(audit, row)
                if row['selected']:
                    write(chosen, row); stats['selected_chars'] += len(row['quote']); selected_topics[kind].update(row['topics'])
                if row['note_id'] in excluded_review_notes: continue
                bucket = (kind, row['selected'])
                reservoir_seen[bucket] += 1
                sample = dict(row, context=raw['text'][row['context_start']:row['context_end']])
                if len(review[bucket]) < 6: review[bucket].append(sample)
                else:
                    position = rng.randrange(reservoir_seen[bucket])
                    if position < 6: review[bucket][position] = sample
            stats['documents_with_selection'] += int(any(v['selected'] for v in doc_units))
    with (args.output / 'documents.jsonl').open('x', encoding='utf-8') as stream:
        for r in notes: write(stream, r)
    with (args.output / 'review_sample.jsonl').open('x', encoding='utf-8') as stream:
        for bucket in sorted(review):
            for row in review[bucket]: write(stream, row)
    result = dict(status='completed_rule_candidates_not_semantic_gold', version=VERSION, rule_sha256=RULE_SHA,
        counts={k: dict(v) for k, v in counts.items()}, topics={k: dict(v) for k, v in selected_topics.items()},
        elapsed_seconds=time.perf_counter()-started, code_sha256=sha256(Path(__file__)),
        source_manifest_sha256={name: sha256(directory / 'manifest.json') for name, directory in
                               (('notes', args.notes), ('imaging', args.imaging))},
        policy='No LLM/encoding/training; final summary policy unchanged; all unselected units retained; cues are lexical, not clinical assertion',
        development_review_excluded_notes=len(excluded_review_notes),
        output_sha256={n: sha256(args.output / n) for n in ('all_units.jsonl.gz', 'selected_sentences.jsonl.gz', 'documents.jsonl', 'review_sample.jsonl')})
    (args.output / 'manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('status', 'counts', 'elapsed_seconds')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, default=Path('patient_data/mimiciv_copd/snapshot_2026-09-27'))
    parser.add_argument('--notes', type=Path, default=Path('patient_data/processed/v6_bonsai_tasks_2026-10-10_validated'))
    parser.add_argument('--imaging', type=Path, default=Path('patient_data/processed/v6_imaging_all_2026-10-10_r2'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--development-review', type=Path)
    build(parser.parse_args())
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/clinical_facts/short_topic_audit.py · lines 1–105

<!-- REVIEW_FRAGMENT {"path":"patient_data/clinical_facts/short_topic_audit.py","part":1,"start_line":1,"end_line":105,"source_sha256":"8933fbce4da44b16ac65d8902aba3008c7d433f4e777e912afe3c0c8b4c74ca7","fragment_sha256":"8933fbce4da44b16ac65d8902aba3008c7d433f4e777e912afe3c0c8b4c74ca7","added_terminal_newline":false} -->
````python
"""已完成短主题先导的独立回查与保守二次门禁；不改原始LLM输出。"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import numpy as np
from patient_data.export_mimiciv import sha256
from patient_data.clinical_facts.features import discharge_vector, DISCHARGE_NAMES
from patient_data.clinical_facts.catalog import CATALOG_SHA


def extra_guard(fact, task):
    field, value, quote = fact['field'], fact['value'], fact['quote']
    if field == 'exertional_desaturation':
        if not re.search(r'\b(?:ambulat\w*|walk\w*|exert\w*|exerc\w*|activity)\b', quote, re.I):
            return 'oxygenation_without_exertional_context'
        if not re.search(r'desatur|\b(?:SpO2|O2|sat(?:uration|ted|ting)?s?|hypox\w*|oxygen)\b', quote, re.I):
            return 'dyspnea_or_exercise_limitation_not_desaturation'
    if field in ('steroid_discharge_course', 'respiratory_antibiotic_course') and value == 'completed':
        if re.search(r'\b(?:will|remaining|to complete|your last day|completes|to finish)\b', quote, re.I):
            return 'future_completion_not_already_completed'
        if re.search(r'discontinu|stopped|not indicated', quote, re.I):
            return 'stopped_not_completed_course'
        if not re.search(r'\b(?:completed|finished|received.*course)\b', quote, re.I):
            return 'completion_requires_explicit_past_evidence'
    if field == 'steroid_discharge_course' and value == 'planned':
        if not re.search(r'\b(?:taper|continue|prescrib\w*|\d+\s*mg|remaining|for.*days?)\b|\b(?:discharg\w*|home)\b.{0,40}\b(?:on|with)\b', quote, re.I):
            return 'inpatient_response_not_discharge_steroid_plan'
    if field in ('inhaled_ics', 'inhaled_laba', 'inhaled_lama') and fact['section_role'] == 'hospital_course':
        if re.search(r'formulary|instead of home', quote, re.I) and not re.search(r'\bdischarg\w*\b', quote, re.I):
            return 'inpatient_substitution_not_discharge_regimen'
    return None


def build(args):
    if args.output.exists(): raise FileExistsError(args.output)
    status = json.loads((args.results / 'status.json').read_text())
    if status['status'] != 'completed': raise ValueError('pilot is unfinished')
    manifest = json.loads((args.tasks / 'manifest.json').read_text())
    contract = json.loads((args.results / 'contract.json').read_text())
    if contract['tasks_manifest_sha256'] != sha256(args.tasks / 'manifest.json') or contract['catalog_sha256'] != CATALOG_SHA:
        raise ValueError('pilot source contract differs')
    for name, digest in manifest['output_sha256'].items():
        if sha256(args.tasks / name) != digest: raise ValueError('pilot source changed')
    if sha256(args.results / 'results.jsonl') != status['results_sha256']: raise ValueError('pilot results changed')
    with (args.tasks / 'tasks.jsonl').open(encoding='utf-8') as f: tasks = {r['task_key']: r for r in map(json.loads, f)}
    with (args.tasks / 'sources.jsonl').open(encoding='utf-8') as f: sources = {r['note_id']: r for r in map(json.loads, f)}
    with (args.results / 'results.jsonl').open(encoding='utf-8') as f: results = {r['task_key']: r for r in map(json.loads, f)}
    if tasks.keys() != results.keys(): raise ValueError('pilot tasks incomplete')
    accepted = []; rejected = []; verified = 0
    task_seconds = []; token_counts = Counter(); reason_chars = 0; route_stats = {}
    for key, task in tasks.items():
        result = results[key]; task_seconds.append(result['seconds'])
        usage = result['response'].get('usage', {})
        token_counts.update({k: v for k, v in usage.items() if isinstance(v, (int, float))})
        for usage_key, values in usage.items():
            if isinstance(values, dict):
                token_counts.update({usage_key + '.' + k: v for k, v in values.items() if isinstance(v, (int, float))})
        reason_chars += len(result['response'].get('reasoning') or '')
        for fact in result['facts']:
            if sources[task['note_id']]['text'][fact['original_start']:fact['original_end']] != fact['quote']:
                raise ValueError('verbatim original evidence mismatch')
            verified += 1
            row = dict(fact, task_key=key, hadm_id=task['hadm_id'], note_id=task['note_id'])
            error = extra_guard(fact, task)
            if error: rejected.append(dict(row, reason=error))
            else: accepted.append(row)
        route_stats.setdefault(task['route'], Counter())
        route_stats[task['route']]['tasks'] += 1
        route_stats[task['route']]['accepted_before_postcheck'] += len(result['facts'])
    members = {k: v for k, v in np.load(args.tasks / 'members.npz').items()}
    hadms = sorted({r['hadm_id'] for r in sources.values()}, key=int)
    by_hadm = {h: [f for f in accepted if f['hadm_id'] == h] for h in hadms}
    values = np.stack([discharge_vector(by_hadm[h]) for h in hadms])
    args.output.mkdir(parents=True)
    for name, rows in (('accepted_evidence.jsonl', accepted), ('rejected_evidence.jsonl', rejected)):
        with (args.output / name).open('x', encoding='utf-8') as f:
            for row in rows: f.write(json.dumps(row, ensure_ascii=False) + '\n')
    np.savez_compressed(args.output / 'pilot_facts.npz', hadm_ids=np.array(hadms), values=values, names=np.array(DISCHARGE_NAMES))
    report = dict(status='completed_short_topic_pilot_source_audit_not_prediction_validated', notes=len(hadms), tasks=len(tasks),
                  thinking=contract['settings']['thinking'], parallel=contract['parallel'], accepted_before_postcheck=verified,
                  all_retained_quotes_source_checked=True, accepted_after_postcheck=len(accepted),
                  source_notes_with_any_fact=sum(bool(fs) for fs in by_hadm.values()), postcheck_reasons=dict(Counter(r['reason'] for r in rejected)),
                  by_field=dict(Counter(f['field'] for f in accepted)), by_route={k: dict(v) for k, v in route_stats.items()},
                  request_seconds_median=float(np.median(task_seconds)), request_seconds_p90=float(np.quantile(task_seconds, .9)),
                  total_request_seconds=float(sum(task_seconds)), completion_rate=status['successful'] / status['total'],
                  raw_output_rejections=status['rejection_counts'], usage=dict(token_counts), reasoning_characters=reason_chars,
                  source_manifest_sha256=sha256(args.tasks / 'manifest.json'), result_sha256=sha256(args.results / 'results.jsonl'),
                  code_sha256=sha256(Path(__file__)),
                  output_sha256={name: sha256(args.output / name) for name in ('accepted_evidence.jsonl', 'rejected_evidence.jsonl', 'pilot_facts.npz')},
                  limitations=['No expert accuracy estimate or recall gold standard', '96-note pilot cannot establish predictive benefit',
                               'Postcheck developed from error inspection and is not an independent validation accuracy',
                               'Raw source/output preserved; uncertain facts excluded, never converted to normal',
                               'Explicit normality and unmentioned remain distinct; no BERT fine-tuning'])
    (args.output / 'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks', type=Path, default=Path('patient_data/processed/v6_short_topics_96_2026-10-10'))
    p.add_argument('--results', type=Path, default=Path('patient_data/processed/v6_short_topics_results_96_2026-10-10_r2'))
    p.add_argument('--output', type=Path, required=True)
    build(p.parse_args())
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/clinical_facts/short_topics.py · lines 1–85

<!-- REVIEW_FRAGMENT {"path":"patient_data/clinical_facts/short_topics.py","part":1,"start_line":1,"end_line":85,"source_sha256":"61897e09a6754da1a7262d932759f4f3bb362aa2dee2e0083b976db950654ef0","fragment_sha256":"61897e09a6754da1a7262d932759f4f3bb362aa2dee2e0083b976db950654ef0","added_terminal_newline":false} -->
````python
"""小结非思考短主题先导：固定随机选择、按章节限域、保留候选片段完整偏移。"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import numpy as np
from patient_data.export_mimiciv import sha256
from patient_data.clinical_facts.catalog import ROUTES, CATALOG_SHA
from patient_data.clinical_facts.tasks import blocks
from patient_data.clinical_facts.llm import digest

PATTERNS = {
    'respiratory': r'dyspn|shortness of breath|breathless|\bSOB\b|oxygen|\bO2\b|room air|cannula|desatur|BiPAP|CPAP|NIV|NIPPV|ventilat|intubat|extubat',
    'medications': r'predn|methylpred|steroid|hydrocortisone|dexamethasone|antibiotic|azithro|levoflox|moxiflox|doxycycl|ceftriax|amoxic|augmentin|tiotrop|spiriva|salmeterol|formoterol|budesonide|fluticasone|advair|symbicort|inhaler|COPD|pneumonia|respiratory',
    'function_care': r'ADL|depend|assist|ambulat|walker|wheelchair|mobility|rehab|SNF|home|lives?|living|alone|adher|complian|afford|spacer|device|inhaler|DNR|DNI|hospice|comfort|AMA|against medical',
}


def candidates(section, route):
    """优先整段保留；这是可审计的候选筛选，不宣称语义完备或无漏检。"""
    text = section['text']
    if section['role'] not in ('hospital_course', 'discharge_instructions', 'transitional_issues'):
        return [section]
    result = []
    for match in re.finditer(r'[^\n]+(?:\n(?!\s*\n)[^\n]+)*', text):
        if re.search(PATTERNS[route], match.group(), re.I):
            result.append(dict(section, text=match.group(), start_char=section['start_char'] + match.start(),
                               end_char=section['start_char'] + match.end()))
    return result


def build(args):
    if args.output.exists(): raise FileExistsError(args.output)
    m = json.loads((args.source / 'manifest.json').read_text())
    for name, value in m['output_sha256'].items():
        if sha256(args.source / name) != value: raise ValueError('source changed')
    with np.load(args.source / 'members.npz') as f: members = {k: f[k] for k in f.files}
    with (args.source / 'sources.jsonl').open(encoding='utf-8') as f:
        sources = [r for r in map(json.loads, f) if r['channel'] == 'discharge']
    fold = {str(h): s for h, s in zip(members['hadm_ids'], members['patient_splits'], strict=True)}
    rng = np.random.default_rng(42); selected = []
    for name in ('train', 'validation', 'test'):
        eligible = [r for r in sources if fold[r['hadm_id']] == name]
        positions = rng.choice(len(eligible), min(args.per_fold, len(eligible)), replace=False) if args.per_fold else range(len(eligible))
        selected.extend(eligible[i] for i in positions)
    tasks = []; audit = Counter()
    for record in selected:
        for route in PATTERNS:
            sections = [s for s in record['sections'] if s['role'] in ROUTES[route]['roles']]
            spans = [p for s in sections for p in candidates(s, route)]
            audit[route + '_source_chars'] += sum(len(s['text']) for s in sections)
            audit[route + '_selected_chars'] += sum(len(s['text']) for s in spans)
            for b in blocks(spans, limit=2400):
                task = {k: record[k] for k in ('hadm_id', 'subject_id', 'note_id', 'charttime', 'storetime', 'original_sha256')}
                task.update(route=route, **b)
                task['task_key'] = digest(json.dumps(task, sort_keys=True) + CATALOG_SHA)
                for p in task['source_maps']:
                    if record['text'][p['original_start']:p['original_end']] != task['text'][p['input_start']:p['input_end']]:
                        raise ValueError('source offset mismatch')
                tasks.append(task)
    args.output.mkdir(parents=True)
    for name, values in (('tasks.jsonl', tasks), ('sources.jsonl', selected)):
        with (args.output / name).open('x', encoding='utf-8') as f:
            for r in values: f.write(json.dumps(r, ensure_ascii=False) + '\n')
    np.savez_compressed(args.output / 'members.npz', **members)
    result = dict(status='llm_tasks_source_checked', catalog_sha256=CATALOG_SHA, tasks=len(tasks),
                  source_counts={'discharge': len(selected)}, task_counts=dict(Counter(t['route'] for t in tasks)),
                  selected_folds=dict(Counter(fold[r['hadm_id']] for r in selected)), audit=dict(audit),
                  parent_state_manifest_sha256=m['parent_state_manifest_sha256'],
                  source_manifest_sha256=sha256(args.source / 'manifest.json'), code_sha256=sha256(Path(__file__)),
                  per_fold=args.per_fold, policy='Non-thinking short topics; 96-note pilot by default, not a full predictive experiment; keyword preselection can miss evidence',
                  output_sha256={n: sha256(args.output / n) for n in ('tasks.jsonl', 'sources.jsonl', 'members.npz')})
    (args.output / 'manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('patient_data/processed/v6_bonsai_tasks_2026-10-10_validated'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--per-fold', type=int, default=32)
    a = p.parse_args()
    if a.per_fold < 0: p.error('invalid per-fold count')
    build(a)
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/clinical_facts/tasks.py · lines 1–108

<!-- REVIEW_FRAGMENT {"path":"patient_data/clinical_facts/tasks.py","part":1,"start_line":1,"end_line":108,"source_sha256":"87909ba47e7e53226792205d3c2371451ae181fb8ea2110b81acfa9693c3f466","fragment_sha256":"87909ba47e7e53226792205d3c2371451ae181fb8ea2110b81acfa9693c3f466","added_terminal_newline":false} -->
````python
"""固定成员，选择最终小结，逐章节覆盖的可续跑LLM任务清单。"""
import argparse
from collections import Counter
import json
from pathlib import Path
from datetime import datetime
import numpy as np
from patient_data.export_mimiciv import rows,sha256
from patient_data.clinical_facts.catalog import ROUTES,CATALOG_SHA
from patient_data.clinical_facts.llm import digest
from training.v5.section_alignment_data.section_parser import parse_sections

def blocks(sections, limit=12000):
    """完整覆盖，不用任意首尾；输入位置映射支持代码端原文回查。"""
    text=''; maps=[]; output=[]
    for section in sections:
        source=section['text']; pos=0
        while pos<len(source):
            header='\nSECTION: '+section['role']+'\n'
            room=limit-len(text)-len(header)
            if room<100:
                output.append({'text':text,'source_maps':maps}); text=''; maps=[]; continue
            end=min(len(source),pos+room)
            if end<len(source):
                boundary=max(source.rfind('\n',pos+room//2,end),source.rfind('. ',pos+room//2,end))
                if boundary>pos: end=boundary+1
            text+=header; start=len(text); text+=source[pos:end]
            maps.append({'role':section['role'],'input_start':start,'input_end':len(text),
                         'original_start':section['start_char']+pos,'original_end':section['start_char']+end})
            pos=end
    if text: output.append({'text':text,'source_maps':maps})
    # 检验每个原文章节完整且只覆盖一次。
    for section in sections:
        spans=[m for b in output for m in b['source_maps'] if m['role']==section['role']
               and section['start_char']<=m['original_start']<section['end_char']]
        if sum(m['original_end']-m['original_start'] for m in spans)!=len(section['text']):
            raise ValueError('lost source section characters')
    return output

def make_tasks(record,sections,route):
    result=[]
    for block in blocks(sections):
        task={k:record[k] for k in ('hadm_id','subject_id','note_id','charttime','storetime','original_sha256')}
        task.update(route=route,**block)
        task['task_key']=digest(json.dumps(task,sort_keys=True)+CATALOG_SHA)
        result.append(task)
    return result

def build(args):
    if args.output.exists(): raise FileExistsError(args.output)
    raw=json.loads((args.snapshot/'metadata/manifest.json').read_text())
    selected=json.loads((args.imaging/'manifest.json').read_text())
    if sha256(args.snapshot/'raw/discharge.csv.gz')!=raw['sources']['raw/discharge']['sha256']: raise ValueError('discharge source changed')
    if sha256(args.snapshot/'raw/radiology.csv.gz')!=raw['sources']['raw/radiology']['sha256']: raise ValueError('radiology source changed')
    for name,value in selected['output_sha256'].items():
        if sha256(args.imaging/name)!=value: raise ValueError('imaging source changed')
    with np.load(args.imaging/'members.npz') as f: members={k:f[k] for k in f.files}
    identities={str(h):str(s) for h,s in zip(members['hadm_ids'],members['subject_ids'],strict=True)}
    admissions={r['hadm_id']:r for r in rows(args.snapshot/'cohort/admissions.csv.gz') if r['hadm_id'] in identities}
    final={}; audit=Counter()
    for r in rows(args.snapshot/'raw/discharge.csv.gz'):
        audit['raw_discharge']+=1
        if r['hadm_id'] not in identities: continue
        if r['subject_id']!=identities[r['hadm_id']]: raise ValueError('discharge subject differs')
        for key in ('charttime','storetime'): datetime.fromisoformat(r[key])
        if not r['text'].strip(): continue
        old=final.get(r['hadm_id'])
        if old is None or (r['charttime'],r['storetime'],r['note_id'])>(old['charttime'],old['storetime'],old['note_id']): final[r['hadm_id']]=r
    tasks=[]; references=[]; eligible=set()
    original_radiology={r['note_id']:r for r in rows(args.snapshot/'raw/radiology.csv.gz')}
    for line in (args.imaging/'chest_ct_reports.jsonl').read_text(encoding='utf-8').splitlines():
        r=json.loads(line)
        sections=[dict(s,role=s['kind']) for s in r['sections']]
        original=original_radiology[r['note_id']]
        if digest(original['text'])!=r['original_sha256']: raise ValueError('CT original hash differs')
        for s in sections:
            if original['text'][s['start_char']:s['end_char']]!=s['text']: raise ValueError('CT original span differs')
        tasks.extend(make_tasks(r,sections,'chest_ct'))
        # 偏移指向完整原始报告，而非合并后的Findings/Impression文本。
        references.append(dict(r,text=original['text'],channel='ct'))
    for hadm,r in sorted(final.items(),key=lambda v:int(v[0])):
        r=dict(r,original_sha256=digest(r['text']))
        spans=[dict(role=s.role,start_char=s.start_char,end_char=s.end_char,text=s.text)
               for s in parse_sections(r['text']) if s.role in ROUTES['discharge_all']['roles'] and s.content_status=='informative']
        if r['storetime']>admissions[hadm]['dischtime']: audit['final_discharge_stored_after_discharge']+=1
        audit['selected_discharge']+=1
        if spans:
            eligible.add(hadm); tasks.extend(make_tasks(r,spans,'discharge_all'))
        else: audit['no_routed_sections']+=1
        references.append(dict(r,sections=spans,channel='discharge'))
    args.output.mkdir(parents=True)
    for name,records in (('tasks.jsonl',tasks),('sources.jsonl',references)):
        with (args.output/name).open('x',encoding='utf-8') as f:
            for r in records: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    np.savez_compressed(args.output/'members.npz',**members)
    result={'status':'llm_tasks_source_checked','tasks':len(tasks),'task_counts':dict(Counter(t['route'] for t in tasks)),
            'source_counts':dict(Counter(r['channel'] for r in references)), 'discharge_routed_admissions':len(eligible),
            'catalog_sha256':CATALOG_SHA,'audit':dict(audit),'snapshot_manifest_sha256':sha256(args.snapshot/'metadata/manifest.json'),
            'imaging_manifest_sha256':sha256(args.imaging/'manifest.json'),'parent_state_manifest_sha256':selected['parent_state_manifest_sha256'],
            'code_sha256':sha256(Path(__file__)),'output_sha256':{n:sha256(args.output/n) for n in ('tasks.jsonl','sources.jsonl','members.npz')},
            'policy':'Retrospective FINAL summary as discharge data, including later storage; unsuitable for claimed prospective real-time availability'}
    (args.output/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',type=Path,default=Path('patient_data/mimiciv_copd/snapshot_2026-09-27'))
    p.add_argument('--imaging',type=Path,default=Path('patient_data/processed/v6_imaging_all_2026-10-10_r2'))
    p.add_argument('--output',type=Path,required=True); build(p.parse_args())
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/compact_multisource.py · lines 1–320

<!-- REVIEW_FRAGMENT {"path":"patient_data/compact_multisource.py","part":1,"start_line":1,"end_line":320,"source_sha256":"27fa9091c831134b92dcccbd0660c0c204c09830dbd36ab8cb98190c8f62d8dc","fragment_sha256":"27fa9091c831134b92dcccbd0660c0c204c09830dbd36ab8cb98190c8f62d8dc","added_terminal_newline":false} -->
````python
"""七类多来源药物、五类微生物及三类呼吸支持；只读底账，不训练。"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np

from patient_data.export_mimiciv import rows, sha256
from patient_data.eos_med_context import SYSTEMIC_ROUTES
from patient_data.resp_micro_context import available_micro, invalid_micro, micro_scope, viral_positive
from patient_data.v6_lab_state import MED_GROUPS, AXES
from training.v5.clinical_concepts import medication_classes

MED_CHANNELS = ('execution_evidence', 'order_only_evidence', 'execution_source_observed', 'order_source_observed')
MICRO_NAMES = ('common_copd_bacteria', 'pseudomonas_aeruginosa', 'other_named_respiratory_bacteria',
               'respiratory_virus_positive', 'named_bacteria_resistance')
SUPPORT_NAMES = ('recorded_invasive_support', 'recorded_noninvasive_support', 'recorded_supplemental_oxygen')
INHALED_ROUTES = {'IH', 'NEB', 'INH', 'INHALATION', 'ORAL INHALATION'}
ORDER_SYSTEMIC_ROUTES = SYSTEMIC_ROUTES | {'IV BOLUS', 'SC', 'SUBCUT', 'SUBCUTANEOUS', 'OGT'}
OTHER_BACTERIA = {
    'STAPH AUREUS COAG +', 'STAPHYLOCOCCUS AUREUS', 'ESCHERICHIA COLI', 'KLEBSIELLA PNEUMONIAE',
    'KLEBSIELLA OXYTOCA', 'SERRATIA MARCESCENS', 'ENTEROBACTER CLOACAE', 'ENTEROBACTER CLOACAE COMPLEX',
    'ENTEROBACTER AEROGENES', 'ACINETOBACTER BAUMANNII COMPLEX', 'ACINETOBACTER LWOFFII',
    'STENOTROPHOMONAS MALTOPHILIA', 'PROTEUS MIRABILIS', 'PROVIDENCIA STUARTII', 'PROVIDENCIA RETTGERI',
    'MORGANELLA MORGANII', 'HAFNIA ALVEI', 'ACHROMOBACTER SP.', 'ACHROMOBACTER SPECIES',
    'ACHROMOBACTER XYLOSOXIDANS', 'ACHROMOBACTER (ALCALIGENES) DENTRIFICANS',
    'ELIZABETHKINGIA (CHRYSEOBACTERIUM) MENIGOSEPTICA', 'PSEUDOMONAS SPECIES', 'PSEUDOMONAS FLUORESCENS',
}
OXYGEN_DEVICES = {'NASAL CANNULA', 'HIGH FLOW NASAL CANNULA', 'NON-REBREATHER', 'VENTI MASK',
                  'OXYMIZER', 'MEDIUM CONC MASK'}


def parse_time(value):
    try: return datetime.fromisoformat(value)
    except (TypeError, ValueError): return None


def order_available(row, admission, *, entered=None, status=None):
    """仅接受出院前已录入且治疗区间与住院相交的医嘱；不把起始日期当录入日期。"""
    start, end = admission
    first = parse_time(row.get('starttime'))
    stop = parse_time(row.get('stoptime'))
    entered = parse_time(row.get('entertime')) if entered is None else entered
    status = row.get('status', '') if status is None else status
    if entered is None: return False, 'entry_time_unknown'
    if entered > end: return False, 'entered_after_discharge'
    if status.strip().upper() in {'H', 'HOLD', 'HELD', 'CANCELLED', 'CANCELED', 'VOID', 'REJECTED'}:
        return False, 'held_or_cancelled_order'
    if first is None: return False, 'start_time_unknown'
    if row.get('stoptime') not in (None, '', r'\N') and stop is None: return False, 'invalid_stop_time'
    if stop is not None and stop < first: return False, 'negative_order_interval'
    if first > end or (stop is not None and stop < start): return False, 'outside_admission'
    return True, 'available_order'


def order_groups(name, route, descriptions=()):
    """剂型/途径仅上游判定；复方可映射多个药效，未知途径不伪造系统治疗。"""
    route = route.strip().upper()
    descriptions = tuple(descriptions) + (('inhalation',) if route in INHALED_ROUTES else ())
    classes = medication_classes(name, routes=(route,), product_descriptions=descriptions,
                                 mapping_version='clinical_reviewed_2026_09')
    result = set()
    for j, (_, constituents) in enumerate(MED_GROUPS.items()):
        if not classes.intersection(constituents): continue
        allowed = INHALED_ROUTES if j < 5 else ORDER_SYSTEMIC_ROUTES
        if route in allowed: result.add(j)
    return result


def bacteria_group(name):
    name = name.strip().upper()
    if (name in {'STREPTOCOCCUS PNEUMONIAE', 'MORAXELLA CATARRHALIS', 'HAEMOPHILUS INFLUENZAE'}
            or name.startswith('HAEMOPHILUS INFLUENZAE,')): return 0
    if name == 'PSEUDOMONAS AERUGINOSA': return 1
    if name in OTHER_BACTERIA: return 2
    return None


def compact_micro_scope(row):
    """真菌/分枝杆菌/军团菌专项培养，不能证明常规细菌组被检查。"""
    scope = micro_scope(row)
    test = row['test_name'].strip().upper()
    if scope == 'lower':
        if test == 'RESPIRATORY CULTURE': return 'bacteria'
        if 'RESPIRATORY VIRAL' in test or 'INFLUENZA' in test: return 'virus'
        return 'excluded_targeted_or_nonculture_test'
    return 'virus' if scope == 'resp_viral' else 'other'


def compact_support(support, support_masks, numeric, numeric_masks, device_rows):
    n = len(support)
    values = np.zeros((n, 3), np.float32)
    masks = np.zeros((n, 3), bool)
    # 气管导管是支持证据，不声称急性COPD适应证；气切单独存在不算在通气。
    values[:, 0] = np.maximum(support[:, 0], support[:, 2])
    masks[:, 0] = support_masks[:, 0] | support_masks[:, 2]
    values[:, 1] = np.maximum(support[:, 1], support[:, 3])
    masks[:, 1] = support_masks[:, 1] | support_masks[:, 3]
    # 氧流量/FIO2只用于上游证据，不产生新时间/剂量输入。
    for k in (2, 3, 4):
        masks[:, 2] |= numeric_masks[:, k, 2]
        values[:, 2] = np.maximum(values[:, 2], (numeric_masks[:, k, 2] & (numeric[:, k, 2] > 0)).astype(np.float32))
    masks[:, 2] |= numeric_masks[:, 5, 2]
    values[:, 2] = np.maximum(values[:, 2], (numeric_masks[:, 5, 2] & (numeric[:, 5, 2] > 21)).astype(np.float32))
    for i, item, raw in device_rows:
        name = raw.strip().upper()
        if item == '226732':
            if name not in ('', r'\N', 'OTHER'): masks[i, 2] = True
            if name in OXYGEN_DEVICES: values[i, 2] = 1
        if item == '229314' and name in {'NIV', 'NIV-ST'}:
            values[i, 1] = 1; masks[i, 1] = True
    # CPAP/PSV、CPAP面罩和呼吸机品牌不能独立证明急性NIV。
    return values, masks


def load_checked(directory, filename, status):
    meta = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if meta['status'] != status: raise ValueError('unfinished parent: '+str(directory))
    for name, digest in meta['output_sha256'].items():
        if sha256(directory/name) != digest: raise ValueError('parent file changed: '+name)
    with np.load(directory/filename, allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    return arrays, meta


def build(snapshot, supplement, state_dir, eos_dir, resp_micro_dir, output):
    if output.exists(): raise FileExistsError(output)
    started = time.perf_counter()
    state, st = load_checked(state_dir, 'clinical_state.npz', 'derived_and_roundtrip_checked_not_prediction_validated')
    eos, em = load_checked(eos_dir, 'eos_med_context.npz', 'eos_med_context_validated_not_trained')
    rm, mm = load_checked(resp_micro_dir, 'resp_micro_context.npz', 'resp_micro_context_validated_not_trained')
    sm = json.loads((snapshot/'metadata/manifest.json').read_text(encoding='utf-8'))
    sup = json.loads((supplement/'metadata/manifest.json').read_text(encoding='utf-8'))
    if (sm['status'] != 'validated' or sup['status'] != 'validated_restricted_source_export_not_model_validated'
            or not sup['cohort_membership_exactly_matches_parent']
            or sha256(snapshot/'metadata/manifest.json') != st['source_manifest_sha256']
            or sha256(snapshot/'metadata/manifest.json') != sup['parent_snapshot_manifest_sha256']
            or sha256(supplement/'metadata/manifest.json') != mm['parent_supplement_manifest_sha256']):
        raise ValueError('raw/state lineage mismatch')
    parent_hash = sha256(state_dir/'manifest.json')
    for arrays, meta in ((eos, em), (rm, mm)):
        if meta.get('parent_state_manifest_sha256', meta.get('parent_clinical_state_manifest_sha256')) != parent_hash:
            raise ValueError('derived lineage mismatch')
        for key in ('hadm_ids', 'subject_ids', 'patient_splits'):
            if not np.array_equal(arrays[key], state[key]): raise ValueError('member/order/split mismatch')
    if tuple(state['axis_names']) != AXES or tuple(state['med_group_names']) != tuple(MED_GROUPS):
        raise ValueError('vocabulary drift')
    if not np.array_equal(state['med_exposure'], eos['med_exposure']): raise ValueError('execution exposure drift')
    source_hashes = {}
    for name in ('raw/pharmacy', 'raw/prescriptions', 'cohort/admissions'):
        path = snapshot/(name+'.csv.gz'); digest = sha256(path)
        if digest != sm['sources'][name]['sha256']: raise ValueError('raw changed: '+name)
        source_hashes[str(path)] = digest
    for name in ('microbiologyevents', 'respiratory_chartevents'):
        path = supplement/'raw'/(name+'.csv.gz'); digest = sha256(path)
        if digest != sup['sources'][name]['sha256']: raise ValueError('supplement changed: '+name)
        source_hashes[str(path)] = digest
    hadms = [str(h) for h in state['hadm_ids']]; positions = {h: i for i, h in enumerate(hadms)}
    members = {r['hadm_id']: r for r in rows(snapshot/'cohort/admissions.csv.gz')}
    admissions = {h: (parse_time(members[h]['admittime']), parse_time(members[h]['dischtime'])) for h in hadms}
    if len(hadms) != 19401 or len(positions) != 19401: raise ValueError('cohort drift')
    for h, i in positions.items():
        if int(members[h]['subject_id']) != state['subject_ids'][i]: raise ValueError('identity drift')
    def checked_rows(path, expected_count):
        count = 0
        for row in rows(path):
            count += 1
            h = row['hadm_id']
            if h not in members or row['subject_id'] != members[h]['subject_id']: raise ValueError('raw association mismatch')
            if h in positions: yield row
        if count != expected_count: raise ValueError('row count changed: '+str(path))
    n = len(hadms); audit = Counter()
    ordered = np.zeros((n, 7), np.float32); pharma_flag = np.zeros_like(ordered)
    rx_flag = np.zeros_like(ordered); rx_unverified = np.zeros_like(ordered)
    order_source = np.zeros(n, bool); pharma = {}; unique_orders = set()
    for r in checked_rows(snapshot/'raw/pharmacy.csv.gz', sm['sources']['raw/pharmacy']['rows']):
        h = r['hadm_id']; i = positions[h]; pid = r['pharmacy_id']
        if pid in pharma: raise ValueError('duplicate pharmacy_id')
        pharma[pid] = (h, parse_time(r['entertime']), r['status'], r['route'])
        ok, reason = order_available(r, admissions[h]); audit['pharmacy:'+reason] += 1
        if not ok: continue
        order_source[i] = True
        groups = order_groups(r['medication'], r['route'])
        if not groups: audit['pharmacy:no_target_class_or_incompatible_route'] += 1
        for j in groups:
            pharma_flag[i, j] = 1; ordered[i, j] = 1
            unique_orders.add((h, pid, j))
    for r in checked_rows(snapshot/'raw/prescriptions.csv.gz', sm['sources']['raw/prescriptions']['rows']):
        h = r['hadm_id']; i = positions[h]; linked = pharma.get(r['pharmacy_id'])
        if linked is not None and linked[0] != h: raise ValueError('prescription/pharmacy association mismatch')
        descriptions = (r['prod_strength'], r['form_rx'])
        route = r['route'] if r['route'] not in ('', r'\N') else linked[3] if linked else ''
        groups = order_groups(r['drug'], route, descriptions)
        if linked is None:
            # 未关联处方没有录入时间，仅起始日期不足以证明出院前可见；仅审计，不进入主输入。
            audit['prescription:unlinked_entry_time_unknown'] += 1
            for j in groups: rx_unverified[i, j] = 1
            continue
        ok, reason = order_available(r, admissions[h], entered=linked[1], status=linked[2])
        audit['prescription:'+reason] += 1
        if not ok: continue
        order_source[i] = True
        if not groups: audit['prescription:no_target_class_or_incompatible_route'] += 1
        for j in groups:
            rx_flag[i, j] = 1; ordered[i, j] = 1
            key = (h, r['pharmacy_id'], j)
            if key in unique_orders: audit['order_class_duplicate_collapsed'] += 1
            unique_orders.add(key)
    med = np.zeros((n, 7, 4), np.float32)
    med[:, :, 0] = state['med_exposure']
    med[:, :, 1] = ordered * (1 - state['med_exposure'])
    med[:, :, 2] = eos['med_source_mask']
    med[:, :, 3] = order_source[:, None]
    micro = np.zeros((n, 5), np.float32); micro_masks = np.zeros_like(micro, bool)
    lower_available = np.zeros(n, bool); viral_available = np.zeros(n, bool)
    isolates = set(); sir_isolates = set(); micro_ids = set()
    for r in checked_rows(supplement/'raw/microbiologyevents.csv.gz', sup['sources']['microbiologyevents']['rows']):
        if r['microevent_id'] in micro_ids: raise ValueError('duplicate microevent_id')
        micro_ids.add(r['microevent_id']); scope = compact_micro_scope(r)
        if scope not in ('bacteria', 'virus'):
            audit['micro:'+scope] += 1; continue
        ok, reason = available_micro(r, *admissions[r['hadm_id']])
        audit['micro:'+reason] += 1
        if not ok: continue
        if invalid_micro(r): audit['micro:invalid_or_cancelled'] += 1; continue
        i = positions[r['hadm_id']]
        if scope == 'bacteria':
            lower_available[i] = True
            j = bacteria_group(r['org_name'])
            if j is None: audit['micro:unmapped_organism_rows'] += 1; continue
            micro[i, j] = 1
            key = tuple(r[k] for k in ('hadm_id', 'micro_specimen_id', 'test_seq', 'test_itemid', 'org_name', 'isolate_num'))
            if key in isolates: audit['micro:duplicate_isolate_rows_collapsed'] += 1
            isolates.add(key)
            if r['interpretation'] in ('S', 'I', 'R'):
                micro_masks[i, 4] = True; sir_isolates.add(key)
                if r['interpretation'] == 'R': micro[i, 4] = 1
        else:
            viral_available[i] = True
            if viral_positive(r['org_name'], r['comments']): micro[i, 3] = 1
    micro_masks[:, :3] = lower_available[:, None]; micro_masks[:, 3] = viral_available
    device_rows = []
    for r in checked_rows(supplement/'raw/respiratory_chartevents.csv.gz', sup['sources']['respiratory_chartevents']['rows']):
        h = r['hadm_id']; t = parse_time(r['charttime']); stored = parse_time(r['storetime'])
        if t is None or stored is None or not admissions[h][0] <= t <= stored <= admissions[h][1]:
            raise ValueError('respiratory source availability drift')
        if r['selection_role'] == 'state_record': device_rows.append((positions[h], r['itemid'], r['value']))
    support, support_masks = compact_support(rm['support_values'], rm['support_masks'],
                                             rm['resp_numeric_raw'], rm['resp_numeric_masks'], device_rows)
    for values, masks in ((micro, micro_masks), (support, support_masks)):
        if not np.isin(values, (0, 1)).all() or ((values > 0) & ~masks).any(): raise ValueError('invalid evidence/mask')
    if ((med[:, :, 0] > 0) & (med[:, :, 2] == 0)).any() or ((med[:, :, 1] > 0) & (med[:, :, 3] == 0)).any():
        raise ValueError('medication positive without source')
    if (med[:, :, 0] * med[:, :, 1]).any(): raise ValueError('execution and order-only overlap')
    def coverage(present):
        return {'admissions': int(present.sum()), 'patients': len(set(state['subject_ids'][present].tolist()))}
    def evidence(values, masks):
        return {'positive': coverage(values > 0), 'source_observed': coverage(masks), 'unknown_admissions': int((~masks).sum())}
    report = {'status': 'compact_multisource_validated_not_trained', 'admissions': n,
              'patients': len(set(state['subject_ids'].tolist())), 'lab_axes_kept': list(AXES),
              'parent_state_manifest_sha256': parent_hash,
              'parent_eos_manifest_sha256': sha256(eos_dir/'manifest.json'),
              'parent_resp_micro_manifest_sha256': sha256(resp_micro_dir/'manifest.json'),
              'source_sha256': source_hashes, 'code_sha256': sha256(Path(__file__)), 'audit': dict(audit),
              'med_group_names': list(MED_GROUPS), 'med_channel_names': MED_CHANNELS,
              'micro_names': MICRO_NAMES, 'support_names': SUPPORT_NAMES,
              'med_coverage': {g: {'execution': coverage(med[:, j, 0] > 0),
                                  'order_only': coverage(med[:, j, 1] > 0),
                                  'pharmacy': coverage(pharma_flag[:, j] > 0),
                                  'linked_prescription': coverage(rx_flag[:, j] > 0)} for j, g in enumerate(MED_GROUPS)},
              'med_any_coverage': {'execution': coverage(med[:, :, 0].any(axis=1)),
                                   'order_only': coverage(med[:, :, 1].any(axis=1)),
                                   'execution_or_order': coverage(med[:, :, :2].any(axis=(1, 2))),
                                   'no_emar_with_order': coverage(~eos['med_source_mask'].any(axis=1) & ordered.any(axis=1)),
                                   'order_source_observed': coverage(order_source),
                                   'unlinked_rx_audit_only': coverage(rx_unverified.any(axis=1))},
              'micro_coverage': {g: evidence(micro[:, j], micro_masks[:, j]) for j, g in enumerate(MICRO_NAMES)},
              'support_coverage': {g: evidence(support[:, j], support_masks[:, j]) for j, g in enumerate(SUPPORT_NAMES)},
              'unique_order_classes': len(unique_orders), 'unique_controlled_bacterial_isolates': len(isolates),
              'bacterial_isolates_with_SIR': len(sir_isolates),
              'limitations': ['Order evidence is not administration, dose, discharge medication or respiratory indication',
                             'Source-observed mask does not establish complete treatment coverage',
                             'Unlinked prescription without entered time retained only for audit',
                             'Controlled positive culture is not proof of active infection or colonization',
                             'Resistance means at least one R result, not multidrug resistance',
                             'Respiratory support is ICU recorded evidence; acute indication and non-ICU absence unknown',
                             'CPAP alone is not NIV; tracheostomy alone is not invasive ventilation',
                             'No medication time/count/dose or support duration enters the compact interface',
                             'All 22 lab axes retained unchanged for future group ablation; no training this run']}
    for path, digest in source_hashes.items():
        if sha256(Path(path)) != digest: raise ValueError('source changed during build')
    output.mkdir(parents=True)
    filename = 'compact_context.npz'
    arrays = {k: state[k] for k in ('hadm_ids', 'subject_ids', 'patient_splits', 'axis_names')}
    arrays.update(med_features=med, med_group_names=np.asarray(tuple(MED_GROUPS)), med_channel_names=np.asarray(MED_CHANNELS),
                  med_pharmacy_audit=pharma_flag, med_linked_rx_audit=rx_flag, med_unlinked_rx_audit=rx_unverified,
                  micro_values=micro, micro_masks=micro_masks, micro_names=np.asarray(MICRO_NAMES),
                  support_values=support, support_masks=support_masks, support_names=np.asarray(SUPPORT_NAMES))
    np.savez_compressed(output/filename, **arrays)
    with np.load(output/filename, allow_pickle=False) as archive:
        for key, value in arrays.items():
            if not np.array_equal(archive[key], value): raise ValueError('roundtrip failed: '+key)
    report['elapsed_seconds'] = round(time.perf_counter()-started, 2)
    report['output_sha256'] = {filename: sha256(output/filename)}
    (output/'manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('status', 'admissions', 'med_any_coverage', 'micro_coverage', 'support_coverage', 'elapsed_seconds')}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('snapshot', 'supplement', 'state_dir', 'eos_dir', 'resp_micro_dir', 'output'):
        parser.add_argument('--'+key.replace('_', '-'), type=Path, required=True)
    args = parser.parse_args()
    build(args.snapshot, args.supplement, args.state_dir, args.eos_dir, args.resp_micro_dir, args.output)
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/compare_saved_labs.py · lines 1–54

<!-- REVIEW_FRAGMENT {"path":"patient_data/compare_saved_labs.py","part":1,"start_line":1,"end_line":54,"source_sha256":"6712e38059860dc39344883472abc808bbdd7e0bfeb18f00b4f347ca4c7faa20","fragment_sha256":"6712e38059860dc39344883472abc808bbdd7e0bfeb18f00b4f347ca4c7faa20","added_terminal_newline":false} -->
````python
"""将新检验事件与旧已用检验逐源记录核对，只输出聚合结果。"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from export_mimiciv import rows, sha256


def compare(old_path: Path, output: Path) -> dict[str, object]:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "preprocessed_and_checked":
        raise ValueError("预处理尚未校验完成")
    event_path = output / "events.csv.gz"
    if sha256(event_path) != manifest["output_sha256"][event_path.name]:
        raise ValueError("预处理事件文件变化")
    old = {r["labevent_id"]: r for r in rows(old_path)}
    remaining = set(old)
    added, kept = 0, 0
    for row in rows(event_path):
        if row["kind"] != "lab":
            continue
        previous = old.get(row["source_id"])
        if previous is None:
            added += 1
            continue
        if (row["hadm_id"] != previous["hadm_id"] or row["subject_id"] != previous["subject_id"]
                or float(row["value_raw"]) != float(previous["valuenum"])
                or row["value_unit"] != previous["valueuom"]
                or datetime.fromisoformat(row["charttime"]) != datetime.fromisoformat(previous["charttime"])):
            raise ValueError("新旧同一原始记录的患者／住院／数值／单位／时间不一致")
        remaining.discard(row["source_id"])
        kept += 1
    excluded = {r["source_id"]: r["reason"] for r in rows(output / "excluded_records.csv.gz") if r["source_table"] == "labevents"}
    if remaining - set(excluded):
        raise ValueError("旧检验有未说明原因的遗漏")
    result = {"status": "source_identity_checked", "old_lab_records": len(old), "old_records_retained": kept,
              "newly_added_records": added, "old_records_excluded": len(remaining),
              "exclusion_reasons": Counter(excluded[source] for source in remaining),
              "raw_values_units_times_and_members_unchanged": True,
              "old_source_sha256": sha256(old_path), "new_events_sha256": sha256(event_path)}
    (output / "source_comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-labs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.old_labs, args.output)))
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/core_lab_value_audit.py · lines 1–171

<!-- REVIEW_FRAGMENT {"path":"patient_data/core_lab_value_audit.py","part":1,"start_line":1,"end_line":171,"source_sha256":"ae8fbf0d57f6d3621e772b6f2edd4203cd6d337cf640681a233d9a33790f23f7","fragment_sha256":"ae8fbf0d57f6d3621e772b6f2edd4203cd6d337cf640681a233d9a33790f23f7","added_terminal_newline":false} -->
````python
"""五核心检验原值与既有异常摘要：训练折无结局审计，不改变模型输入。"""
from collections import Counter,defaultdict
from datetime import datetime
import argparse
import gzip
import json
from pathlib import Path
import time
import numpy as np
from patient_data.export_mimiciv import rows,sha256
from patient_data.preprocess_events import LABS
from patient_data.lab_recovery_features import AXIS_CONCEPT,abnormal_severity
from patient_data.v6_lab_state import summarize

ROOT=Path('training_runs/v6_core_lab_value_audit_2026-10-14')
PROTOCOL=Path('docs/V6_CORE_LAB_VALUE_AUDIT_PROTOCOL_2026-10-14.md')
SNAP=Path('patient_data/mimiciv_copd/snapshot_2026-09-27')
STATE=Path('patient_data/processed/v6_clinical_state_2026-10-09_r2')
EVENTS=Path('patient_data/processed/mimiciv_copd_events_clinical_reviewed_2026-09-29')
AXES=('arterial_low_ph','arterial_high_pco2','hemoglobin_low','albumin_low','creatinine_high')

def accepted(row,boundary,unit,arterial_ids=None):
    try:
        chart,stored=(datetime.fromisoformat(row[k]) for k in ('charttime','storetime'))
        values=tuple(float(row[k]) for k in ('valuenum','ref_range_lower','ref_range_upper'))
        value,lower,upper=values
        if not boundary[0]<=chart<=stored<=boundary[1]:return None,'not_hospital_visible'
        if not all(np.isfinite(values)) or value<=0 or lower>=upper:return None,'invalid_numeric_reference'
    except (ValueError,TypeError,KeyError):return None,'invalid_time_numeric_reference'
    if row['valueuom'].strip()!=unit:return None,'unit_mismatch'
    if arterial_ids is not None and row['labevent_id'] not in arterial_ids:return None,'not_old_verified_arterial'
    return (chart,value,lower,upper),None

def readout(samples,end):
    """原异常摘要与物理值首末并列；不把引用/同刻行数当独立复测。"""
    if not samples:return np.zeros(9),np.zeros(2),np.zeros(2),dict(observed=False)
    grouped=defaultdict(list)
    for day,value,lower,upper,severity in samples:grouped[day].append((value,lower,upper,severity))
    ordered=sorted(grouped);physical=[];positions=[]
    for day in (ordered[0],ordered[-1]):
        physical.append(float(np.median([x[0] for x in grouped[day]])))
        positions.append(float(np.median([(x[0]-x[1])/(x[2]-x[1]) for x in grouped[day]])))
    nine,_=summarize([(x[0],x[4]) for x in samples],end)
    return nine,np.asarray(physical),np.asarray(positions),dict(observed=True,
        timestamps=len(ordered),raw_rows=len(samples),same_time_multi_rows=sum(len(v)>1 for v in grouped.values()),
        entirely_zero_direction=all(x[4]==0 for x in samples),
        original_value_varies=len({x[1] for x in samples})>1,
        first_last_value_diff=physical[0]!=physical[1],latest_age=end-ordered[-1])

def derive():
    start=time.perf_counter();sm=json.loads((SNAP/'metadata/manifest.json').read_text())
    st=json.loads((STATE/'manifest.json').read_text());em=json.loads((EVENTS/'manifest.json').read_text())
    if sm['status']!='validated' or em['status']!='preprocessed_and_checked':raise ValueError('unvalidated source')
    if sha256(EVENTS/'manifest.json')!=st['processed_manifest_sha256'] or sha256(EVENTS/'events.csv.gz')!=st['source_hashes']['processed_events']:
        raise ValueError('arterial event source differs')
    if sha256(STATE/'clinical_state.npz')!=st['output_sha256']['clinical_state.npz']:raise ValueError('state differs')
    sources={}
    for name in ('cohort/admissions','raw/labevents'):
        p=SNAP/(name+'.csv.gz');h=sha256(p)
        if h!=sm['sources'][name]['sha256'] or h!=st['source_hashes'][name]:raise ValueError('source differs')
        sources[str(p)]=h
    for p in (Path(__file__),PROTOCOL,Path('training/tests/test_v6_core_lab_value_audit.py'),
              SNAP/'metadata/manifest.json',STATE/'manifest.json',STATE/'clinical_state.npz',
              EVENTS/'manifest.json',EVENTS/'events.csv.gz',Path('patient_data/preprocess_events.py'),
              Path('patient_data/lab_recovery_features.py'),Path('patient_data/v6_lab_state.py')):sources[str(p)]=sha256(p)
    with np.load(STATE/'clinical_state.npz') as z:
        state={k:z[k] for k in ('hadm_ids','subject_ids','patient_splits','axis_names','lab_axis_raw')}
    indices=np.flatnonzero(state['patient_splits']=='train');hadms=state['hadm_ids'][indices];subjects=state['subject_ids'][indices]
    lookup={str(h):i for i,h in enumerate(hadms)};bounds={}
    for row in rows(SNAP/'cohort/admissions.csv.gz'):
        if row['hadm_id'] not in lookup:continue
        i=lookup[row['hadm_id']]
        if int(row['subject_id'])!=subjects[i] or row['discharge_landmark_eligible']!='t':raise ValueError('training membership differs')
        bounds[row['hadm_id']]=tuple(datetime.fromisoformat(row[k]) for k in ('admittime','dischtime'))
    if len(bounds)!=13592:raise ValueError('training cohort differs')
    arterial=set()
    for row in rows(EVENTS/'events.csv.gz'):
        if row['hadm_id'] in lookup and row['kind']=='lab' and row['concept'] in ('ph:arterial','pco2:arterial'):
            arterial.add(row['source_id'])
    item_axes={}
    for j,axis in enumerate(AXES):
        concept=AXIS_CONCEPT[axis].split(':')[0]
        item,unit=next((item,unit) for item,(c,unit) in LABS.items() if c==concept)
        item_axes[item]=(j,axis,unit)
    observations=defaultdict(list);evidence=[];reasons=defaultdict(Counter);total=0
    for row in rows(SNAP/'raw/labevents.csv.gz'):
        total+=1;h=row['hadm_id'];item=row['itemid']
        if h not in lookup or item not in item_axes:continue
        j,axis,unit=item_axes[item]
        value,reason=accepted(row,bounds[h],unit,arterial if axis.startswith('arterial_') else None)
        reasons[axis]['raw_rows']+=1
        if reason:reasons[axis][reason]+=1;continue
        chart,physical,lower,upper=value;day=(chart-bounds[h][0]).total_seconds()/86400
        severity=abnormal_severity(axis,physical,lower,upper,severity_cap=None)
        observations[h,j].append((day,physical,lower,upper,severity));reasons[axis]['accepted_rows']+=1
        evidence.append(dict(hadm_id=h,labevent_id=row['labevent_id'],specimen_id=row['specimen_id'],
            axis=axis,charttime=row['charttime'],storetime=row['storetime'],unit=unit,
            value=physical,lower=lower,upper=upper,severity=severity))
    if total!=sm['sources']['raw/labevents']['rows']:raise ValueError('lab count differs')
    nine=np.zeros((len(indices),5,9));physical=np.zeros((len(indices),5,2));position=np.zeros_like(physical)
    observed=np.zeros((len(indices),5),bool);stats={};max_error=0.
    for j,axis in enumerate(AXES):
        detail=[];signatures=defaultdict(list);old_j=list(state['axis_names']).index(axis)
        for i,h in enumerate(hadms):
            key=str(h);length=(bounds[key][1]-bounds[key][0]).total_seconds()/86400
            nine[i,j],physical[i,j],position[i,j],info=readout(observations.get((key,j),[]),length)
            observed[i,j]=info['observed']
            if info['observed']:detail.append(info);signatures[tuple(nine[i,j])].append(i)
        error=float(np.abs(nine[:,j]-state['lab_axis_raw'][indices,old_j]).max());max_error=max(max_error,error)
        if error>1e-12:raise ValueError('old axis replay differs: '+axis)
        shared=[ix for ix in signatures.values() if len(set(subjects[ix]))>1 and len(set(physical[ix,j,1]))>1]
        ages=np.asarray([x['latest_age'] for x in detail]);use=observed[:,j]
        stats[axis]=dict(coverage=int(use.sum()),patients=len(np.unique(subjects[use])),
            repeated_timestamps_admissions=sum(x['timestamps']>=2 for x in detail),
            all_zero_direction_admissions=sum(x['entirely_zero_direction'] for x in detail),
            all_zero_with_original_value_variation=sum(x['entirely_zero_direction'] and x['original_value_varies'] for x in detail),
            first_last_physical_difference=sum(x['first_last_value_diff'] for x in detail),
            same_time_multirow_admissions=sum(x['same_time_multi_rows']>0 for x in detail),
            cross_patient_exact_nine_groups_with_different_latest_value=len(shared),
            discharges_in_those_groups=sum(len(ix) for ix in shared),
            latest_age_median_days=float(np.median(ages)) if len(ages) else None,
            latest_physical_quantiles=np.quantile(physical[use,j,1],[.1,.5,.9]).tolist() if use.any() else [],
            original_axis_replay_error=error)
    arrays=dict(hadm_ids=hadms,subject_ids=subjects,axis_names=np.asarray(AXES),
        original_nine=nine,physical_first_latest=physical,reference_position_first_latest=position,observed=observed)
    report=dict(status='train_only_core_lab_value_audit_not_model_interface',source_sha256=sources,
        training_admissions=len(indices),training_patients=len(np.unique(subjects)),
        assigned_lab_source_rows_scanned=total,axis_statistics=stats,acceptance_audit={k:dict(v) for k,v in reasons.items()},
        maximum_original_axis_replay_error=max_error,new_fits=0,new_vectors=0,no_outcomes_read=True,
        actual_seconds=time.perf_counter()-start,
        limits=['zero direction is not globally normal clinical state',
                'same one-axis signature does not imply same full model patient input',
                'same-time physical median is an audit readout, all source rows preserved',
                'latest observed is not necessarily discharge measurement; no prognostic gain evaluated'])
    return report,arrays,evidence

def run():
    if ROOT.exists():raise FileExistsError('audit exists')
    ROOT.mkdir()
    try:
        r,a,e=derive();np.savez_compressed(ROOT/'training_audit.npz',**a)
        with gzip.open(ROOT/'local_evidence.jsonl.gz','wt',encoding='utf-8') as f:
            for row in e:f.write(json.dumps(row)+'\n')
        r['output_sha256']={p:sha256(ROOT/p) for p in ('training_audit.npz','local_evidence.jsonl.gz')}
        (ROOT/'audit.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        (ROOT/'state.json').write_text('{"status":"audit_pending_independent_replay","new_fits":0}\n')
        print(json.dumps(dict(status=r['status'],stats=r['axis_statistics'],max_error=r['maximum_original_axis_replay_error'],seconds=r['actual_seconds'])),flush=True)
    except Exception as error:
        (ROOT/'failure.json').write_text(json.dumps(dict(error=repr(error)))+'\n');raise

def verify():
    if (ROOT/'verification.json').exists():raise FileExistsError('already independently verified')
    old=json.loads((ROOT/'audit.json').read_text());new,a,e=derive()
    for k,h in old['output_sha256'].items():
        if sha256(ROOT/k)!=h:raise ValueError('saved output changed')
    seconds=new.pop('actual_seconds');old.pop('actual_seconds');old.pop('output_sha256')
    if old!=new:raise ValueError('source/statistics differ')
    with np.load(ROOT/'training_audit.npz') as z:
        for k,v in a.items():np.testing.assert_array_equal(z[k],v)
    with gzip.open(ROOT/'local_evidence.jsonl.gz','rt',encoding='utf-8') as f:
        if [json.loads(line) for line in f]!=e:raise ValueError('source rows differ')
    proof=dict(status='one_independent_full_source_array_statistic_replay',maximum_error=0.,
        original_axis_replay_error=new['maximum_original_axis_replay_error'],actual_rebuild_seconds=seconds,
        audit_sha256=sha256(ROOT/'audit.json'),new_fits=0)
    (ROOT/'verification.json').write_text(json.dumps(proof,indent=2)+'\n')
    (ROOT/'state.json').write_text('{"status":"core_value_audit_independently_verified","new_fits":0,"goal_achieved":false}\n')
    print(json.dumps(proof),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('run','verify'),required=True)
    run() if p.parse_args().mode=='run' else verify()
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/cxr_report_selection.py · lines 1–146

<!-- REVIEW_FRAGMENT {"path":"patient_data/cxr_report_selection.py","part":1,"start_line":1,"end_line":146,"source_sha256":"33017e630e5eeaab6f3445bdb223e4d426c5d8dc0382c7e2c5f47bff2169b0c2","fragment_sha256":"33017e630e5eeaab6f3445bdb223e4d426c5d8dc0382c7e2c5f47bff2169b0c2","added_terminal_newline":false} -->
````python
"""胸片报告识别、预测时点核验和章节定位；不预测影像病理标签。"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from patient_data.export_mimiciv import rows, sha256

HEADER = re.compile(r'(?im)^\s*(REASON FOR (?:EXAMINATION|EXAM)|CLINICAL HISTORY|EXAMINATION|EXAM|TECHNIQUE|INDICATION|HISTORY|COMPARISON|FINDINGS|IMPRESSION|CONCLUSION|RECOMMENDATION(?:\(S\)|S)?|NOTIFICATION)\s*:\s*')
NON_CXR = re.compile(r'\b(CT|CTA|MDCT|HRCT|COMPUTED TOMOGRAPHY|MRI|MR|ULTRASOUND|SONOGRAM|ANGIOGRAM|BIOPSY|FLUOROSCOPY)\b', re.I)
CXR = re.compile(r'\bCHEST\b', re.I)
VIEW = re.compile(r'RADIOGRAPH|\bX[ -]?RAY\b|\bPORTABLE\b|\bPORT\.?\b|\b(?:AP|PA|LAT|LATERAL|FRONTAL|VIEW|DX)\b|LINE PLACEMENT', re.I)


def identify(text):
    """检查名称优先；不因HISTORY/COMPARISON中提到胸片或CT判当前检查。"""
    headings = list(HEADER.finditer(text))
    for i,h in enumerate(headings):
        if h.group(1).upper()=='TECHNIQUE':
            end=headings[i+1].start() if i+1<len(headings) else len(text)
            technique=' '.join(text[h.end():end].split())
            if NON_CXR.search(technique): return 'non_cxr_technique', technique
    for h in headings:
        if h.group(1).upper() in ('EXAMINATION', 'EXAM'):
            tail = text[h.end():].splitlines()
            title = next((line.strip() for line in tail if line.strip()), '')
            if NON_CXR.search(title): return 'non_cxr_exam', title
            if CXR.search(title) and VIEW.search(title): return 'cxr_exam', title
            return 'unrecognized_exam', title
    stop = min((h.start() for h in headings if h.group(1).upper() not in ('TECHNIQUE',)), default=len(text))
    for line in text[:stop].splitlines()[:12]:
        title = line.strip()
        if NON_CXR.search(title): return 'non_cxr_title', title
        if CXR.search(title) and VIEW.search(title): return 'cxr_title', title
    for h in headings:
        if h.group(1).upper() == 'TECHNIQUE':
            end = next((x.start() for x in headings if x.start() > h.start()), len(text))
            title = ' '.join(text[h.end():end].split())
            if NON_CXR.search(title): return 'non_cxr_technique', title
            if CXR.search(title) and VIEW.search(title): return 'cxr_technique', title
    return 'unrecognized', ''


def report_section(text):
    headings = list(HEADER.finditer(text))
    for names, kind in ((('IMPRESSION', 'CONCLUSION'), 'impression'), (('FINDINGS',), 'findings')):
        for i, h in enumerate(headings):
            if h.group(1).upper() in names:
                start = h.end(); end = headings[i+1].start() if i+1 < len(headings) else len(text)
                value = text[start:end].strip()
                if value:
                    start += len(text[start:end]) - len(text[start:end].lstrip())
                    end = start + len(value)
                    return value, kind, start, end
    # 没有可靠章节时保留为缺失候选，不将病史/检查指征当影像结论。
    return '', 'no_findings_or_impression', 0, 0


def usable(row, admission):
    try:
        chart = datetime.fromisoformat(row['charttime']); store = datetime.fromisoformat(row['storetime'])
        start = datetime.fromisoformat(admission['admittime']); end = datetime.fromisoformat(admission['dischtime'])
    except (ValueError, KeyError): return False, 'invalid_time'
    if row['subject_id'] != admission['subject_id']: return False, 'subject_mismatch'
    if not start <= chart <= store <= end: return False, 'not_available_during_admission'
    return True, ''


def choose(reports):
    """最新有效报告，平局以存档时点/note_id确定；无报告保持缺失。"""
    return max(reports, key=lambda r: (r['charttime'], r['storetime'], r['note_id'])) if reports else None


def select(snapshot, state_dir, output):
    if output.exists(): raise FileExistsError(output)
    manifest_path = snapshot / 'metadata/manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    state_manifest = json.loads((state_dir/'manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'validated' or state_manifest['status'] != 'derived_and_roundtrip_checked_not_prediction_validated':
        raise ValueError('unvalidated source')
    for name in ('cohort/admissions', 'raw/radiology'):
        if sha256(snapshot/(name+'.csv.gz')) != manifest['sources'][name]['sha256']:
            raise ValueError('raw source changed')
    arrays_path = state_dir/'clinical_state.npz'
    if sha256(arrays_path) != state_manifest['output_sha256']['clinical_state.npz']:
        raise ValueError('clinical members changed')
    with np.load(arrays_path, allow_pickle=False) as f:
        identities = {key:f[key] for key in ('hadm_ids', 'subject_ids', 'patient_splits')}
    admission = {r['hadm_id']:r for r in rows(snapshot/'cohort/admissions.csv.gz') if r['discharge_landmark_eligible'] == 't'}
    if set(admission) != set(map(str, identities['hadm_ids'])): raise ValueError('eligible members differ')
    chosen, audit = {}, Counter()
    for r in rows(snapshot/'raw/radiology.csv.gz'):
        audit['raw_reports'] += 1
        if r['hadm_id'] not in admission:
            audit['not_eligible_discharge'] += 1; continue
        category, title = identify(r['text']); audit['type_'+category] += 1
        if not category.startswith('cxr_'): continue
        okay, reason = usable(r, admission[r['hadm_id']])
        if not okay:
            audit[reason] += 1; continue
        selected, section, start, end = report_section(r['text'])
        if not selected:
            audit['no_findings_or_impression'] += 1; continue
        if r['text'][start:end] != selected or not 0 <= start <= end <= len(r['text']):
            raise ValueError('section provenance mismatch')
        record = {k:r[k] for k in ('hadm_id', 'subject_id', 'note_id', 'charttime', 'storetime')}
        record.update(text=selected, section=section, start_char=start, end_char=end, examination=title,
                      original_sha256=hashlib.sha256(r['text'].encode()).hexdigest(),
                      section_sha256=hashlib.sha256(selected.encode()).hexdigest())
        previous = chosen.get(r['hadm_id'])
        chosen[r['hadm_id']] = choose([record] + ([previous] if previous else []))
        audit['qualified_reports'] += 1
    if audit['raw_reports'] != manifest['sources']['raw/radiology']['rows']: raise ValueError('radiology count changed')
    output.mkdir(parents=True)
    records = []
    for hadm, subject, split in zip(identities['hadm_ids'], identities['subject_ids'], identities['patient_splits'], strict=True):
        r = chosen.get(str(hadm))
        if r:
            if r['subject_id'] != str(subject): raise ValueError('subject alignment failed')
            r['patient_split'] = str(split)
            r['days_before_discharge'] = (datetime.fromisoformat(admission[str(hadm)]['dischtime'])-datetime.fromisoformat(r['charttime'])).total_seconds()/86400
            records.append(r); audit['selected_'+r['section']] += 1
    destination = output/'selected_reports.jsonl'
    with destination.open('x', encoding='utf-8') as f:
        for r in records: f.write(json.dumps(r, ensure_ascii=False)+'\n')
    np.savez_compressed(output/'members.npz', **identities)
    result = {'status':'selected_temporally_checked_not_labeled', 'admissions':len(admission), 'reports_selected':len(records),
              'patients_with_report':len({r['subject_id'] for r in records}), 'audit':dict(audit),
              'parent_state_manifest_sha256':sha256(state_dir/'manifest.json'), 'snapshot_manifest_sha256':sha256(manifest_path),
              'raw_radiology_sha256':manifest['sources']['raw/radiology']['sha256'], 'code_sha256':sha256(Path(__file__)),
              'output_sha256':{name:sha256(output/name) for name in ('selected_reports.jsonl','members.npz')},
              'policy':'one latest chest-radiograph report with admission <= charttime <= storetime <= discharge; Impression preferred, Findings fallback',
              'limits':['Conservative title matcher and section availability may omit real CXRs', 'Reports only, no image pixels or CT morphology']}
    (output/'manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',type=Path,default=Path('patient_data/mimiciv_copd/snapshot_2026-09-27'))
    p.add_argument('--state-dir',type=Path,default=Path('patient_data/processed/v6_clinical_state_2026-10-09_r2'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); select(args.snapshot,args.state_dir,args.output)
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/discharge_regimen_audit.py · lines 1–233

<!-- REVIEW_FRAGMENT {"path":"patient_data/discharge_regimen_audit.py","part":1,"start_line":1,"end_line":233,"source_sha256":"21921a64bab7a982c6f1c2ead25b700fab8e02bd6c023058e6e0532104b5fa00","fragment_sha256":"21921a64bab7a982c6f1c2ead25b700fab8e02bd6c023058e6e0532104b5fa00","added_terminal_newline":false} -->
````python
"""已保存出院药物章来源库存；词面候选不是处方事实，不读结局。"""
from collections import Counter, defaultdict
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import time
import numpy as np
from patient_data.export_mimiciv import sha256
from training.v5.clinical_concepts import medication_classes

ROOT=Path('training_runs/v6_discharge_regimen_audit_2026-10-14')
PROTOCOL=Path('docs/V6_DISCHARGE_REGIMEN_AUDIT_PROTOCOL_2026-10-14.md')
STATE=Path('patient_data/processed/v6_clinical_state_2026-10-09_r2')
TASKS=Path('patient_data/processed/v6_bonsai_tasks_2026-10-10_validated')
SCREEN=Path('patient_data/processed/v6_rule_sentence_screen_2026-10-10_r4')
ENCODED=Path('patient_data/processed/v6_screen_candidate_bert_2026-10-10')
ALIGNED=Path('patient_data/processed/v6_saved_sections_aligned_2026-10-11')
COMPACT=Path('patient_data/processed/v6_compact_multisource_2026-10-09')
CLASSES=('inhaled_steroid','laba','lama')
FORM=re.compile(r'\b(?:inhal\w*|puffs?|nebul\w*|neb|diskus|ellipta|respimat|HFA|aerosol|MDI|DPI|respules?)\b',re.I)
ORAL=re.compile(r'\b(?:PO|by mouth|orally)\b',re.I)
FLAGS={
    'stop_hold_words':r'\b(?:stop\w*|discontinu\w*|hold|held|do not take)\b',
    'conditional_words':r'\b(?:if|unless|as needed|PRN)\b',
    'history_words':r'\b(?:previous\w*|prior|formerly|used to)\b',
    'dose_string_present':r'\b\d+(?:\.\d+)?\s*(?:mg|mcg|ug|puffs?|mL)\b',
    'frequency_string_present':r'\b(?:daily|BID|TID|QID|q\s*\d+\s*h|twice|once|three times|four times)\b',
}


def lexical(quote):
    # 此处name是保存的原句，只能称词面候选；不输出prescribed/stopped事实值。
    route=('PO',) if ORAL.search(quote) and not FORM.search(quote) else ()
    groups=medication_classes(quote,product_descriptions=(quote,),routes=route,
                              mapping_version='clinical_reviewed_2026_09')
    return dict(classes=sorted(groups&set(CLASSES)),inhaled_form_word=bool(FORM.search(quote)),
                explicit_oral_word=bool(ORAL.search(quote)),
                flags=[k for k,p in FLAGS.items() if re.search(p,quote,re.I)],clinically_verified=False)


def span(text,a,b,quote):
    if not 0<=a<=b<=len(text) or text[a:b]!=quote:raise ValueError('invalid saved source offset/quote')


def check_maps(source,task):
    spans=[]
    for m in task['source_maps']:
        if m['role']!='discharge_medications':continue
        a,b=m['original_start'],m['original_end'];ia,ib=m['input_start'],m['input_end']
        if not 0<=ia<=ib<=len(task['text']):raise ValueError('task input bounds invalid')
        span(source['text'],a,b,task['text'][ia:ib])
        if not any(s['start_char']<=a<=b<=s['end_char'] for s in source['sections'] if s['role']=='discharge_medications'):
            raise ValueError('map not within medication section')
        spans.append((a,b))
    return spans


def covered(section,ranges):
    a,b=section['start_char'],section['end_char'];rr=sorted((x,y) for x,y in ranges if a<=x<=y<=b)
    right=a
    for x,y in rr:
        if x!=right or not x<y:raise ValueError('medication map gap/overlap')
        right=y
    if right!=b:raise ValueError('medication section not fully task-covered')


def load_npz(path,keys):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in keys}


def derive():
    started=time.perf_counter();sources={}
    def protect(p,expected=None):
        h=sha256(p)
        if expected is not None and h!=expected:raise ValueError('source hash changed: '+str(p))
        sources[str(p)]=h;return h
    manifests={}
    for directory in (STATE,TASKS,SCREEN,ENCODED,ALIGNED,COMPACT):
        protect(directory/'manifest.json');manifests[str(directory)]=json.loads((directory/'manifest.json').read_text())
    st,tm,rm,em,am,cm=(manifests[str(p)] for p in (STATE,TASKS,SCREEN,ENCODED,ALIGNED,COMPACT))
    if tm['parent_state_manifest_sha256']!=sources[str(STATE/'manifest.json')] or cm['parent_state_manifest_sha256']!=sources[str(STATE/'manifest.json')]:
        raise ValueError('parent member source differs')
    if tm['status']!='llm_tasks_source_checked' or rm['status']!='completed_rule_candidates_not_semantic_gold' or em['status']!='candidate_bert_complete':
        raise ValueError('source unfinished')
    for directory,names,manifest in (
        (STATE,('clinical_state.npz',),st),(TASKS,('sources.jsonl','tasks.jsonl','members.npz'),tm),
        (SCREEN,('selected_sentences.jsonl.gz',),rm),(ENCODED,('chunks.jsonl.gz',),em),
        (ALIGNED,('alignment.npz',),am),(COMPACT,('compact_context.npz',),cm)):
        for name in names:protect(directory/name,manifest['output_sha256'][name])
    protect(ENCODED/'independent_encoding_verification.json')
    ep=json.loads((ENCODED/'independent_encoding_verification.json').read_text())
    if ep['manifest_sha256']!=sources[str(ENCODED/'manifest.json')]:raise ValueError('old encoding proof differs')
    state=load_npz(STATE/'clinical_state.npz',('hadm_ids','subject_ids','patient_splits'))
    ti=np.flatnonzero(state['patient_splits']=='train');n=len(ti)
    if len(state['hadm_ids'])!=19401 or n!=13592:raise ValueError('cohort drift')
    people={str(h):str(s) for h,s in zip(state['hadm_ids'][ti],state['subject_ids'][ti])};lookup={str(h):i for i,h in enumerate(state['hadm_ids'][ti])}
    for directory,filename in ((TASKS,'members.npz'),(COMPACT,'compact_context.npz'),(ALIGNED,'alignment.npz')):
        a=load_npz(directory/filename,tuple(state))
        for k,v in state.items():np.testing.assert_array_equal(a[k],v)
    compact=load_npz(COMPACT/'compact_context.npz',('med_features','med_group_names'))
    med_indices=[list(compact['med_group_names']).index(x) for x in ('inhaled_corticosteroid','long_beta_agonist','long_muscarinic_antagonist')]
    old=(compact['med_features'][ti][:,med_indices,:2]>0).any(2)
    summaries={};sections={}
    with (TASKS/'sources.jsonl').open(encoding='utf-8') as f:
        for r in map(json.loads,f):
            if r['channel']!='discharge' or r['hadm_id'] not in people:continue
            if r['subject_id']!=people[r['hadm_id']] or r['note_id'] in summaries:raise ValueError('source subject/duplicate differs')
            if hashlib.sha256(r['text'].encode()).hexdigest()!=r['original_sha256']:raise ValueError('saved note SHA differs')
            ss=[s for s in r['sections'] if s['role']=='discharge_medications']
            for s in ss:span(r['text'],s['start_char'],s['end_char'],s['text'])
            summaries[r['note_id']]=r;sections[r['note_id']]=ss
    taskranges=defaultdict(list);tasks_count=0
    with (TASKS/'tasks.jsonl').open(encoding='utf-8') as f:
        for task in map(json.loads,f):
            if task['route']!='discharge_all' or task['note_id'] not in summaries:continue
            source=summaries[task['note_id']]
            for k in ('hadm_id','subject_id','original_sha256'):
                if task[k]!=source[k]:raise ValueError('task source identity differs')
            ranges=check_maps(source,task);taskranges[task['note_id']].extend(ranges);tasks_count+=bool(ranges)
    for note,ss in sections.items():
        for s in ss:covered(s,taskranges[note])
    encoded_units=set()
    with gzip.open(ENCODED/'chunks.jsonl.gz','rt',encoding='utf-8') as f:
        for r in map(json.loads,f):
            if r['patient_split']!='train':continue
            if r['hadm_id'] not in people or r['subject_id']!=people[r['hadm_id']]:raise ValueError('encoded membership differs')
            encoded_units.update((r['note_id'],int(a),int(b)) for a,b in r['source_units'])
    evidence=[];counts=Counter();named=np.zeros((n,3),bool);with_form=np.zeros_like(named);available=np.zeros(n,bool)
    for note,ss in sections.items():
        if ss:available[lookup[summaries[note]['hadm_id']]]=True
    with gzip.open(SCREEN/'selected_sentences.jsonl.gz','rt',encoding='utf-8') as f:
        for r in map(json.loads,f):
            if r['kind']!='discharge' or r['patient_split']!='train' or r['section']!='discharge_medications':continue
            if r['hadm_id'] not in people or r['subject_id']!=people[r['hadm_id']] or r['note_id'] not in summaries:raise ValueError('sentence member differs')
            source=summaries[r['note_id']];a,b=r['start_char'],r['end_char'];span(source['text'],a,b,r['quote'])
            if r['original_sha256']!=source['original_sha256'] or not any(s['start_char']<=a<=b<=s['end_char'] for s in sections[r['note_id']]):
                raise ValueError('sentence section/identity differs')
            counts['saved_selected_medication_units']+=1;lex=lexical(r['quote'])
            if not lex['classes']:continue
            i=lookup[r['hadm_id']];encoded=(r['note_id'],a,b) in encoded_units
            counts['maintenance_mapping_candidate_units']+=1;counts['candidate_units_in_old_encoding_groups']+=encoded
            counts['candidate_units_with_inhaled_form_words']+=lex['inhaled_form_word']
            counts.update(lex['flags'])
            for group in lex['classes']:
                j=CLASSES.index(group);named[i,j]=True;with_form[i,j]|=lex['inhaled_form_word']
            evidence.append(dict(**{k:r[k] for k in ('hadm_id','subject_id','note_id','unit_key','start_char','end_char','quote','original_sha256')},
                lexical=lex,in_old_encoding_group=encoded))
    aligned=load_npz(ALIGNED/'alignment.npz',('refs',));med_role=am['roles'].index('discharge_medications')
    apositions={int(p) for p,f,_ in aligned['refs'] if int(f)==med_role and state['patient_splits'][int(p)]=='train'}
    prior={}
    for folder in ('v6_bonsai_discharge_pilot_2026-10-10_r3','v6_short_topics_results_96_2026-10-10_r2'):
        p=Path('patient_data/processed')/folder;protect(p/'status.json');protect(p/'contract.json')
        status=json.loads((p/'status.json').read_text());contract=json.loads((p/'contract.json').read_text())
        prior[folder]=dict(status=status['status'],total=status['total'],successful=status['successful'],limit=contract['limit'])
    pilot=Path('patient_data/processed/v6_short_topic_audit_96_2026-10-10_r2/audit.json');protect(pilot)
    audit=json.loads(pilot.read_text());prior['96_note_postcheck']=dict(notes=audit['notes'],tasks=audit['tasks'],
        accepted_after_postcheck=audit['accepted_after_postcheck'],maintenance_field_counts={k:audit['by_field'].get(k,0) for k in ('inhaled_ics','inhaled_laba','inhaled_lama')})
    cxr=Path('training_runs/v6_cxr_trajectory_seed42_2026-10-10_r2');protect(cxr/'config.json');protect(cxr/'contract.json')
    cfg=json.loads((cxr/'config.json').read_text());cc=json.loads((cxr/'contract.json').read_text())
    if cfg['facts_dir'] is not None or cc['available_channels']!=['cxr']:raise ValueError('prior full facts changed; inspect before repeating')
    byclass={group:dict(candidate_admissions=int(named[:,j].sum()),candidate_patients=len(np.unique(state['subject_ids'][ti][named[:,j]])),
        with_inhaled_form_word_admissions=int(with_form[:,j].sum()),
        old_compact_class_zero_despite_candidate_admissions=int((named[:,j]&~old[:,j]).sum())) for j,group in enumerate(CLASSES)}
    # 只有候选库存，绝不把这些数组当已验证的处方标签。
    arrays=dict(hadm_ids=state['hadm_ids'][ti],subject_ids=state['subject_ids'][ti],section_available=available,
                mapping_mentions=named,inhaled_form_word_mentions=with_form,old_compact_presence=old)
    reference_contract=Path('training_runs/v6_core_lab_cox_2026-10-14/contract.json');protect(reference_contract)
    reference=json.loads(reference_contract.read_text())
    teacher=Path('training_runs/v6_module_surgery_2026-10-12_r3/seed42/control/best.pt')
    protect(teacher,reference['parent_teacher_sha256'])
    for p in (Path(__file__),PROTOCOL,Path('training/tests/test_discharge_regimen_audit.py'),
        Path('training/v5/clinical_concepts.py'),Path('training/v6/clinical_facts_context.py'),
        Path('patient_data/clinical_facts/catalog.py'),Path('patient_data/clinical_facts/features.py')):protect(p)
    report=dict(status='train_only_saved_discharge_medication_source_inventory_not_fact_interface',
        source_sha256=sources,training_admissions=n,training_patients=len(np.unique(state['subject_ids'][ti])),
        saved_training_summary_notes=len(summaries),medication_section_notes=sum(bool(s) for s in sections.values()),
        medication_section_admissions=int(available.sum()),medication_section_patients=len(np.unique(state['subject_ids'][ti][available])),
        medication_sections=sum(len(s) for s in sections.values()),medication_tasks_with_maps=tasks_count,
        aligned_A_training_medication_admissions=len(apositions),counts=dict(counts),by_class=byclass,
        candidate_any_maintenance_admissions=int(named.any(1).sum()),
        old_fact_channel_experiment=dict(facts_dir=cfg['facts_dir'],available_channels=cc['available_channels']),
        inherited_frozen_reference=dict(teacher_sha256=reference['parent_teacher_sha256'],
            base_sha256=reference['parent_base_sha256'],data_provenance=reference['data_provenance'],
            label_array_sha256={k:reference['array_sha256'][k] for k in ('times','events')}),
        existing_pilots=prior,new_fits=0,new_vectors=0,new_llm_calls=0,no_outcomes_read=True,
        raw_discharge_read=False,source_maps_fully_covered=True,actual_seconds=time.perf_counter()-started,
        limits=['drug mapping words/form words are not confirmed prescribed/stopped facts or clinical gold',
            'old compact zero includes unobserved source; discordance is not proof of missing administration',
            'source_units identifies encoding-group membership, not per-chunk token inclusion or BERT semantics',
            'FINAL is retrospective; no real-time availability established',
            'B has no independent medication role but course may share information'])
    return report,arrays,evidence


def write(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def run():
    if ROOT.exists():raise FileExistsError('audit exists; do not repeat')
    ROOT.mkdir();r,a,e=derive();np.savez_compressed(ROOT/'audit_arrays.npz',**a)
    with gzip.open(ROOT/'local_evidence.jsonl.gz','wt',encoding='utf-8') as f:
        for row in e:f.write(json.dumps(row)+'\n')
    r['output_sha256']={k:sha256(ROOT/k) for k in ('audit_arrays.npz','local_evidence.jsonl.gz')}
    write(ROOT/'audit.json',r);write(ROOT/'state.json',dict(status='pending_independent_rebuild',new_fits=0))
    print(json.dumps({k:v for k,v in r.items() if k not in ('source_sha256','output_sha256')}),flush=True)


def verify():
    if (ROOT/'verification.json').exists():raise FileExistsError('already verified')
    old=json.loads((ROOT/'audit.json').read_text());new,a,e=derive();seconds=new.pop('actual_seconds');old.pop('actual_seconds')
    for k,h in old.pop('output_sha256').items():
        if sha256(ROOT/k)!=h:raise ValueError('audit artifact changed')
    if old!=new:raise ValueError('source/statistics differ')
    with np.load(ROOT/'audit_arrays.npz') as z:
        for k,v in a.items():np.testing.assert_array_equal(z[k],v)
    with gzip.open(ROOT/'local_evidence.jsonl.gz','rt',encoding='utf-8') as f:
        saved=list(map(json.loads,f))
    if saved!=e:raise ValueError('evidence differs')
    write(ROOT/'verification.json',dict(status='one_independent_saved_source_map_inventory_rebuild',maximum_error=0.,
        audit_sha256=sha256(ROOT/'audit.json'),actual_seconds=seconds,new_fits=0,new_vectors=0,raw_discharge_read=False))
    write(ROOT/'state.json',dict(status='source_inventory_independently_verified',new_fits=0,goal_achieved=False))
    print(json.dumps(dict(error=0.,seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('run','verify'),required=True);mode=p.parse_args().mode
    try:run() if mode=='run' else verify()
    except Exception as error:
        if ROOT.exists():write(ROOT/f'failure_{mode}.json',dict(error=repr(error)))
        raise
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/discharge_regimen_entries.py · lines 1–112

<!-- REVIEW_FRAGMENT {"path":"patient_data/discharge_regimen_entries.py","part":1,"start_line":1,"end_line":112,"source_sha256":"952bf2867cc23e0f23ffb224404da965c3078c115aec56611cbaa161ad425e10","fragment_sha256":"952bf2867cc23e0f23ffb224404da965c3078c115aec56611cbaa161ad425e10","added_terminal_newline":false} -->
````python
"""保存药物候选的窄条目读取；文书列示候选不等实际给药/专家gold。"""
import re
from patient_data.medication_identity import ALIASES as OLD_ALIASES
from patient_data.v6_expanded_candidates import ALIASES as ADDED_ALIASES, identity
from patient_data.compact_multisource import INHALED_ROUTES
from training.v5.clinical_concepts import medication_classes, _MED_PATTERNS, _BRANDS

CLASSES={'inhaled_steroid','laba','lama'}
MAINTENANCE=re.compile('|'.join(_MED_PATTERNS[k] for k in sorted(CLASSES)),re.I)
ALL_MOLECULES=re.compile('|'.join(_MED_PATTERNS.values()),re.I)
ALIASES=sorted(set(_BRANDS)|{a for aa in (*OLD_ALIASES.values(),*ADDED_ALIASES.values()) for a in aa},key=lambda s:(-len(s),s))
BRAND=re.compile(r'\b(?:'+'|'.join(map(re.escape,ALIASES))+r')\b',re.I)
BOUNDARY=re.compile(r'(?m)^[ \t]*(?:\d+[.)][ \t]+|[-*][ \t]+)(?=[A-Za-z])')
PREFIX=re.compile(r'^\s*(?:\d+[.)]\s*|[-*]\s*)?(?:(?P<action>discontinued|discontinue|stopped|stop|held|hold|continue)\b[\s:-]*)?',re.I)
NAME_END=re.compile(r'\d|\b(?:Sig|take|inhale|inhalation|oral|nasal|IH|INH|NEB|PO|daily|BID|TID|QID|QHS|puffs?|capsules?|tablets?)\b',re.I)
ROUTE=re.compile(r'\b(?:ORAL INHALATION|INHALATION|IH|INH|NEB|PO|NG|GT|PEG|OGT|nasal|topical|ophthalmic|otic)\b',re.I)
FORM=re.compile(r'\b(?:inhal\w*|nebul\w*|aerosol|diskus|ellipta|respimat|HFA|MDI|DPI|respules?|puffs?|handihaler)\b',re.I)
CONDITION=re.compile(r'\b(?:if|unless|until|only when)\b',re.I)
AS_NEEDED=re.compile(r'\b(?:PRN|as needed)\b',re.I)
STOP=re.compile(r'\b(?:stop\w*|discontinu\w*|hold|held|do not take)\b',re.I)
NONLISTING=re.compile(r'\b(?:previously|formerly|history of|used to|instead of home|formulary substitute)\b',re.I)
DOSE=re.compile(r'\b\d+(?:\.\d+)?(?:\s*[-/]\s*\d+(?:\.\d+)?)?\s*(?:mg|mcg|ug|puffs?|mL)\b',re.I)
FREQUENCY=re.compile(r'\b(?:daily|BID|TID|QID|QHS|q\s*\d+\s*h|twice|once|three times|four times)\b',re.I)


def intervals(text):
    starts={0,len(text)}
    starts.update(m.start() for m in BOUNDARY.finditer(text))
    output=[]
    ordered=sorted(starts)
    for a,b in zip(ordered,ordered[1:]):
        while a<b and text[a].isspace():a+=1
        while b>a and text[b-1].isspace():b-=1
        if a<b:output.append((a,b))
    return output


def read_entry(text):
    """只分析一个条目；返回精确证据位置与保守未知原因。"""
    base=dict(status='unresolved',classes=[],reason=None,clinically_verified=False,
              product_name=None,canonical_identity=None,product_span=None,route_spans=[],
              as_needed=False,dose_strings=[],frequency_strings=[])
    prefix=PREFIX.match(text);start=prefix.end();action=prefix.group('action')
    tail=text[start:];name_end=NAME_END.search(tail)
    end=start+(name_end.start() if name_end else len(tail))
    # 分号后另一药物不是本产品名或途径说明。
    semi=text.find(';',start)
    if semi>=0:end=min(end,semi)
    while end>start and text[end-1].isspace():end-=1
    while start<end and text[start].isspace():start+=1
    name=text[start:end]
    match=MAINTENANCE.match(name) or BRAND.match(name)
    if not match:return dict(base,reason='no_anchored_product_name')
    key=identity(name);ingredients={m.group().lower() for m in MAINTENANCE.finditer(name)}
    brands=[m.group().lower() for m in BRAND.finditer(name)]
    # and/分号是独立产品连接，不能按固定复方优先把整串吞为一个药。
    if re.search(r'\band\b|;',name,re.I):return dict(base,reason='multiple_products_in_header')
    branded_keys={identity(b) or 'brand:'+b for b in brands}
    if len(branded_keys)>1 and key is None:return dict(base,reason='multiple_product_identities')
    if key is not None:
        if any(w not in key.split('_') for w in ingredients):return dict(base,reason='extra_ingredient_in_header')
        if any((identity(b) is not None and identity(b)!=key) for b in brands):return dict(base,reason='extra_product_in_header')
    elif len(ingredients)>1 and not re.search(r'[-/]',name):
        return dict(base,reason='unbound_multiple_ingredients')
    scope_end=text.find(';',end)
    if scope_end<0:scope_end=len(text)
    scope=text[start:scope_end]
    # Sig里又明确出现独立药名时，不能将混杂途径套到本药。
    extra={m.group().lower() for m in ALL_MOLECULES.finditer(text[end:scope_end])}
    allowed=ingredients|set((key or '').split('_'))
    if extra-allowed:return dict(base,product_name=name,product_span=[start,end],canonical_identity=key,reason='other_drug_in_instruction_scope')
    routes=[m for m in ROUTE.finditer(text,end,scope_end)]
    route_values={m.group().upper() for m in routes}
    inh=route_values&INHALED_ROUTES
    nonpulm=route_values- INHALED_ROUTES
    if inh and nonpulm:return dict(base,product_name=name,product_span=[start,end],canonical_identity=key,reason='conflicting_local_route')
    normalized=tuple(sorted(route_values))
    # 与既有compact匹配方式一致：明确IH/INH/NEB补正规化描述，不改原句证据。
    descriptions=(scope,)+(('inhalation',) if inh else ())
    groups=medication_classes(name,product_descriptions=descriptions,routes=normalized,
                              mapping_version='clinical_reviewed_2026_09')&CLASSES
    if not groups and key is not None:
        groups=medication_classes(key.replace('_',' '),product_descriptions=descriptions,routes=normalized,
                                  mapping_version='clinical_reviewed_2026_09')&CLASSES
    if not groups:return dict(base,product_name=name,product_span=[start,end],canonical_identity=key,
                              reason='nonpulmonary_or_unresolved_form')
    result=dict(base,classes=sorted(groups),product_name=name,product_span=[start,end],canonical_identity=key,
                route_spans=[[m.start(),m.end(),m.group()] for m in routes],as_needed=bool(AS_NEEDED.search(scope)),
                dose_strings=[m.group() for m in DOSE.finditer(text,end,scope_end)],
                frequency_strings=[m.group() for m in FREQUENCY.finditer(text,end,scope_end)])
    if CONDITION.search(text[:scope_end]) or NONLISTING.search(text[:scope_end]):
        return dict(result,status='conditional_unknown',reason='conditional_or_historical_scope')
    if action and action.lower()!='continue':return dict(result,status='explicit_stop',reason='direct_stop_prefix')
    if STOP.search(text[end:scope_end]):return dict(result,status='conditional_unknown',reason='later_action_scope_unresolved')
    return dict(result,status='listed',reason='anchored_discharge_list_entry')


def read_unit(text):
    result=[]
    for a,b in intervals(text):
        item=read_entry(text[a:b]);item['entry_span']=[a,b];item['entry_quote']=text[a:b]
        result.append(item)
    return result


def masked(text):
    """只显示固定药物/剂型/状态词，所有数字与其他词遮蔽。"""
    allowed=set(re.findall(r'[A-Za-z]+',' '.join(ALIASES)+' '+' '.join(_MED_PATTERNS.values())))
    allowed|={'sig','ih','inh','neb','po','daily','bid','tid','qid','mg','mcg','puff','puffs','capsule','capsules',
        'inhalation','diskus','hfa','handihaler','stop','stopped','hold','held','discontinued','continue','if','unless',
        'prn','as','needed','nasal','oral','topical','bromide','device','with','propionate','take','and'}
    return re.sub(r'[A-Za-z]+|\d+',lambda m:m.group().lower() if m.group().lower() in allowed else ('N' if m.group().isdigit() else 'X'),text)
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/discharge_regimen_entries_checked.py · lines 1–34

<!-- REVIEW_FRAGMENT {"path":"patient_data/discharge_regimen_entries_checked.py","part":1,"start_line":1,"end_line":34,"source_sha256":"6c5c4713ebad5bd9d8269322d745da9039d656b10b042d85995f4a712f8719cf","fragment_sha256":"6c5c4713ebad5bd9d8269322d745da9039d656b10b042d85995f4a712f8719cf","added_terminal_newline":false} -->
````python
"""r3限定补丁：规范成分剂型门禁、复方身份优先和原动作子类。"""
from patient_data import discharge_regimen_entries as original
from patient_data.v6_expanded_medication_fix import identity


def read_entry(text):
    result=original.read_entry(text)
    action=original.PREFIX.match(text).group('action')
    result['prefix_action']=action.lower() if action else None
    name=result['product_name']
    if not name:return result
    key=identity(name)
    result['canonical_identity']=key
    # 盐名/EC等修饰词不能绕过裸成分制剂门禁；明确品牌或本条目IH仍保留。
    if key in {'budesonide','fluticasone','mometasone'} and result['classes']:
        a,b=result['product_span'];end=text.find(';',b)
        scope=text[a:end if end>=0 else len(text)]
        routes=tuple(sorted({v.upper() for _,_,v in result['route_spans']}))
        descriptions=(scope,)+(('inhalation',) if set(routes)&original.INHALED_ROUTES else ())
        brand=next((v for v in original._BRANDS if v in name.lower()
                    and 'inhaled_steroid' in original._BRANDS[v]),None)
        groups=original.medication_classes(brand or key,product_descriptions=descriptions,
            routes=routes,mapping_version='clinical_reviewed_2026_09')&original.CLASSES
        if not groups:
            result.update(status='unresolved',classes=[],reason='canonical_form_unresolved')
    return result


def read_unit(text):
    result=[]
    for a,b in original.intervals(text):
        item=read_entry(text[a:b]);item['entry_span']=[a,b];item['entry_quote']=text[a:b]
        result.append(item)
    return result
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/discharge_regimen_entry_audit.py · lines 1–145

<!-- REVIEW_FRAGMENT {"path":"patient_data/discharge_regimen_entry_audit.py","part":1,"start_line":1,"end_line":145,"source_sha256":"b577ca70fc3f365064532b1be6a3a2f7b76166bce7def486157035861946c649","fragment_sha256":"b577ca70fc3f365064532b1be6a3a2f7b76166bce7def486157035861946c649","added_terminal_newline":false} -->
````python
"""同保存训练候选条目核验；不重抽全文、不编码、不读结局或训练。"""
from collections import Counter,defaultdict
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from patient_data.export_mimiciv import sha256
from patient_data.discharge_regimen_entries import read_unit, masked

ROOT=Path('training_runs/v6_discharge_regimen_entry_2026-10-14_r2')
PARENT=Path('training_runs/v6_discharge_regimen_audit_2026-10-14')
PROTOCOL=Path('docs/V6_DISCHARGE_REGIMEN_ENTRY_PROTOCOL_2026-10-14.md')
CLASSES=('inhaled_steroid','laba','lama')


def digest(text):return hashlib.sha256(text.encode()).hexdigest()


def inherited_hash(mapping,path):
    matches=[value for key,value in mapping.items() if Path(key).resolve()==path.resolve()]
    if len(matches)!=1:raise ValueError('missing or ambiguous inherited source key')
    return matches[0]


def derive():
    started=time.perf_counter();old=json.loads((PARENT/'audit.json').read_text())
    proof=json.loads((PARENT/'verification.json').read_text());sources={}
    def protect(p,expected=None):
        h=sha256(p)
        if expected is not None and h!=expected:raise ValueError('changed source: '+str(p))
        sources[str(p)]=h;return h
    if proof['maximum_error']!=0 or proof['audit_sha256']!=protect(PARENT/'audit.json'):
        raise ValueError('parent proof stale')
    protect(PARENT/'verification.json')
    for k,h in old['output_sha256'].items():protect(PARENT/k,h)
    for p in (Path('training/v5/clinical_concepts.py'),Path('patient_data/discharge_regimen_audit.py')):
        protect(p,inherited_hash(old['source_sha256'],p))
    teacher=Path('training_runs/v6_module_surgery_2026-10-12_r3/seed42/control/best.pt')
    protect(teacher,old['inherited_frozen_reference']['teacher_sha256'])
    with np.load(PARENT/'audit_arrays.npz') as z:
        metadata={k:z[k] for k in ('hadm_ids','subject_ids')}
    lookup={str(h):i for i,h in enumerate(metadata['hadm_ids'])};subjects=metadata['subject_ids']
    records=[];counts=Counter();by_status=defaultdict(set);per_class=defaultdict(lambda:defaultdict(set))
    product_keys=set();bucket=defaultdict(list);unit_count=0;multi_units=0
    with gzip.open(PARENT/'local_evidence.jsonl.gz','rt',encoding='utf-8') as f:
        for row in map(json.loads,f):
            unit_count+=1;h=row['hadm_id'];i=lookup[h]
            if row['subject_id']!=str(subjects[i]) or row['end_char']-row['start_char']!=len(row['quote']):
                raise ValueError('parent identity/offset mismatch')
            entries=read_unit(row['quote']);multi_units+=len(entries)>1
            for entry in entries:
                a,b=entry['entry_span']
                if not 0<=a<b<=len(row['quote']) or row['quote'][a:b]!=entry['entry_quote']:
                    raise ValueError('derived entry source mismatch')
                product=entry['product_span']
                if product is not None and entry['entry_quote'][slice(*product)]!=entry['product_name']:
                    raise ValueError('product source mismatch')
                for ra,rb,rv in entry['route_spans']:
                    if not 0<=ra<rb<=len(entry['entry_quote']) or entry['entry_quote'][ra:rb]!=rv:
                        raise ValueError('route source mismatch')
                key=digest(row['unit_key']+':'+str(a)+':'+str(b))
                result=dict(**{k:row[k] for k in ('hadm_id','subject_id','note_id','unit_key','original_sha256')},
                    entry_key=key,source_start=row['start_char']+a,source_end=row['start_char']+b,
                    quote=entry['entry_quote'],reading={k:v for k,v in entry.items() if k not in ('entry_span','entry_quote')},
                    parent_unit_source_start=row['start_char'],parent_unit_source_end=row['end_char'])
                records.append(result);counts['status:'+entry['status']]+=1;counts['reason:'+entry['reason']]+=1
                by_status[entry['status']].add(i)
                if entry['canonical_identity']:product_keys.add(entry['canonical_identity'])
                for group in entry['classes']:per_class[group][entry['status']].add(i)
                tags=[entry['status']]
                if entry['status']=='listed':tags+=entry['classes']
                if any(x[2].upper()=='IH' for x in entry['route_spans']):tags.append('explicit_IH')
                if len(entries)>1:tags.append('multi_entry_parent')
                if entry['product_name'] and entry['canonical_identity'] is None:tags.append('identity_unknown')
                for tag in tags:bucket[tag].append(result)
    if unit_count!=old['counts']['maintenance_mapping_candidate_units']:raise ValueError('parent candidates differ')
    samples=[];chosen=set()
    for tag,items in sorted(bucket.items()):
        for r in sorted(items,key=lambda r:r['entry_key'])[:3]:
            marker=(tag,r['entry_key'])
            if marker in chosen:continue
            chosen.add(marker);v=r['reading']
            samples.append(dict(bucket=tag,entry_key=r['entry_key'],masked_quote=masked(r['quote']),
                                status=v['status'],reason=v['reason'],classes=v['classes']))
    for p in (Path(__file__),PROTOCOL,Path('patient_data/discharge_regimen_entries.py'),
        Path('training/tests/test_discharge_regimen_entries.py'),Path('patient_data/medication_identity.py'),
        Path('patient_data/v6_expanded_candidates.py'),Path('patient_data/compact_multisource.py')):protect(p)
    cls={group:{s:dict(admissions=len(ii),patients=len(np.unique(subjects[list(ii)]))) for s,ii in statuses.items()}
         for group,statuses in per_class.items()}
    union=set().union(*(v.get('listed',set()) for v in per_class.values())) if per_class else set()
    report=dict(status='train_only_saved_regimen_entry_candidate_audit',source_sha256=sources,
        inherited_original_source_hashes=old['source_sha256'],inherited_frozen_reference=old['inherited_frozen_reference'],
        training_admissions=len(lookup),training_patients=len(np.unique(subjects)),parent_candidate_units=unit_count,
        derived_entries=len(records),multiple_entry_parent_units=multi_units,counts=dict(counts),
        status_admissions={k:len(v) for k,v in by_status.items()},by_class=cls,
        listed_maintenance_admissions=len(union),listed_maintenance_patients=len(np.unique(subjects[list(union)])),
        known_canonical_products=len(product_keys),masked_sample_records=len(samples),
        related_tests=17,new_fits=0,new_vectors=0,new_llm_calls=0,no_outcomes_read=True,
        raw_discharge_read=False,whole_inventory_rebuilt=False,actual_seconds=time.perf_counter()-started,
        quality='candidate_noise_unquantified_limited_masked_readout_pending',
        limits=['listed is document-list candidate, not confirmed actual prescribing/administration or adherence',
                'source offsets and synthetic boundary tests are not expert clinical accuracy',
                'PRN is retained as as-needed, not stopped',
                'dose/frequency strings are evidence, not standardized total-dose predictors',
                'full original source hashes inherited from verified parent, not re-read this run'])
    return report,records,samples


def write(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def run():
    if ROOT.exists():raise FileExistsError('batch exists')
    ROOT.mkdir();r,e,s=derive()
    with gzip.open(ROOT/'entry_evidence.jsonl.gz','wt',encoding='utf-8') as f:
        for row in e:f.write(json.dumps(row)+'\n')
    write(ROOT/'masked_sample.json',s)
    r['output_sha256']={k:sha256(ROOT/k) for k in ('entry_evidence.jsonl.gz','masked_sample.json')}
    write(ROOT/'audit.json',r);write(ROOT/'state.json',dict(status='pending_independent_rebuild',new_fits=0))
    print(json.dumps({k:v for k,v in r.items() if k not in ('source_sha256','inherited_original_source_hashes','inherited_frozen_reference','output_sha256')}),flush=True)


def verify():
    if (ROOT/'verification.json').exists():raise FileExistsError('already verified')
    old=json.loads((ROOT/'audit.json').read_text());new,e,s=derive();seconds=new.pop('actual_seconds');old.pop('actual_seconds')
    for k,h in old.pop('output_sha256').items():
        if sha256(ROOT/k)!=h:raise ValueError('saved output changed')
    if old!=new:raise ValueError('source/statistics differ')
    with gzip.open(ROOT/'entry_evidence.jsonl.gz','rt',encoding='utf-8') as f:saved=list(map(json.loads,f))
    if saved!=e or json.loads((ROOT/'masked_sample.json').read_text())!=s:raise ValueError('entry/sample differs')
    write(ROOT/'verification.json',dict(status='one_independent_parent_evidence_entry_scope_statistic_rebuild',
        maximum_error=0.,actual_seconds=seconds,audit_sha256=sha256(ROOT/'audit.json'),new_fits=0,raw_discharge_read=False))
    write(ROOT/'state.json',dict(status='entry_candidates_independently_verified_readout_pending',new_fits=0,goal_achieved=False))
    print(json.dumps(dict(error=0.,seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',required=True,choices=('run','verify'));mode=p.parse_args().mode
    try:run() if mode=='run' else verify()
    except Exception as error:
        if ROOT.exists():write(ROOT/f'failure_{mode}.json',dict(error=repr(error)))
        raise
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/discharge_regimen_entry_checked_audit.py · lines 1–58

<!-- REVIEW_FRAGMENT {"path":"patient_data/discharge_regimen_entry_checked_audit.py","part":1,"start_line":1,"end_line":58,"source_sha256":"e844971beaa7f17c530992b347a734ec5e96dd3975e6f56ecd708baf0ee21ff6","fragment_sha256":"e844971beaa7f17c530992b347a734ec5e96dd3975e6f56ecd708baf0ee21ff6","added_terminal_newline":false} -->
````python
"""r3审计适配器：不改r2文件，复用父引用校验，仅替换限定条目reader。"""
from collections import Counter
import argparse
import gzip
import json
from pathlib import Path
from patient_data import discharge_regimen_entry_audit as audit
from patient_data.discharge_regimen_entries_checked import read_unit
from patient_data.export_mimiciv import sha256

R2=audit.ROOT
ROOT=Path('training_runs/v6_discharge_regimen_entry_2026-10-14_r3')
PROTOCOL=Path('docs/V6_DISCHARGE_REGIMEN_ENTRY_R3_PROTOCOL_2026-10-14.md')
original_derive=audit.derive


def derive():
    prior=json.loads((R2/'audit.json').read_text())
    proof=json.loads((R2/'verification.json').read_text())
    if proof['maximum_error']!=0 or proof['audit_sha256']!=sha256(R2/'audit.json'):
        raise ValueError('r2 proof stale')
    for file,h in prior['output_sha256'].items():
        if sha256(R2/file)!=h:raise ValueError('r2 evidence changed')
    for file,h in prior['source_sha256'].items():
        if sha256(Path(file))!=h:raise ValueError('r2 captured source changed')
    report,entries,samples=original_derive()
    with gzip.open(R2/'entry_evidence.jsonl.gz','rt',encoding='utf-8') as stream:
        old={r['entry_key']:r for r in map(json.loads,stream)}
    differences=Counter();actions=Counter()
    if set(old)!={r['entry_key'] for r in entries}:raise ValueError('candidate membership changed')
    for record in entries:
        previous=old[record['entry_key']]
        if any(record[k]!=previous[k] for k in ('quote','source_start','source_end','original_sha256')):
            raise ValueError('original quote/offset changed')
        reading=record['reading'];actions[str(reading['prefix_action'])]+=1
        for k in ('status','classes','canonical_identity'):
            differences[k]+=reading[k]!=previous['reading'][k]
    report.update(rule_version='r3_canonical_form_combo_identity_original_action',related_tests=21,
        differences_from_verified_r2=dict(differences),prefix_actions=dict(actions))
    for p in (Path(__file__),Path('patient_data/discharge_regimen_entries_checked.py'),
              Path('training/tests/test_discharge_regimen_entries_checked.py'),
              Path('patient_data/v6_expanded_medication_fix.py'),R2/'audit.json',R2/'verification.json',
              R2/'entry_evidence.jsonl.gz'):
        report['source_sha256'][str(p)]=sha256(p)
    return report,entries,samples


# 显式仅进程内适配；所有受保护r2源文件保持不变。
audit.ROOT=ROOT;audit.PROTOCOL=PROTOCOL;audit.read_unit=read_unit;audit.derive=derive


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',required=True,choices=('run','verify'))
    mode=parser.parse_args().mode
    try:audit.run() if mode=='run' else audit.verify()
    except Exception as error:
        if ROOT.exists():audit.write(ROOT/f'failure_{mode}.json',dict(error=repr(error)))
        raise
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/discharge_regimen_interface.py · lines 1–202

<!-- REVIEW_FRAGMENT {"path":"patient_data/discharge_regimen_interface.py","part":1,"start_line":1,"end_line":202,"source_sha256":"0e282c7f2c609c9885abef4b4c2fe83ee92199ca6d6ac8516cf39a642c36d824","fragment_sha256":"0e282c7f2c609c9885abef4b4c2fe83ee92199ca6d6ac8516cf39a642c36d824","added_terminal_newline":false} -->
````python
"""冻结r3出院方案读取，全队列类别集合与引用；不读标签/原raw或训练。"""
import argparse
from collections import Counter,defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from patient_data.discharge_regimen_audit import STATE,TASKS,SCREEN,CLASSES,lexical,span,check_maps,covered
from patient_data.discharge_regimen_entries_checked import read_unit
from patient_data.export_mimiciv import sha256

ROOT=Path('patient_data/processed/v6_discharge_regimen_interface_2026-10-14')
PARENT=Path('training_runs/v6_discharge_regimen_entry_2026-10-14_r3')
INVENTORY=Path('training_runs/v6_discharge_regimen_audit_2026-10-14')
PROTOCOL=Path('docs/V6_DISCHARGE_REGIMEN_INTERFACE_PROTOCOL_2026-10-14.md')
ROLES=('listed','held','stopped','conditional_unknown','unresolved')


def digest(text):return hashlib.sha256(text.encode()).hexdigest()


def role(reading):
    status=reading['status']
    if status=='explicit_stop':
        action=reading['prefix_action']
        if action in ('held','hold'):return 'held'
        if action in ('stop','stopped','discontinue','discontinued'):return 'stopped'
        raise ValueError('unknown explicit action')
    if status not in ROLES:raise ValueError('unrecognized frozen status')
    return status


def member_key(record,lookup):
    value=record['reading'];name=value['canonical_identity']
    # 未知文本只作证据去重键，不进入训练产品身份词表。
    product=('canonical',name) if name else ('unknown',' '.join((value['product_name'] or record['quote']).lower().split()))
    return (lookup[record['hadm_id']],product,tuple(value['classes']),role(value),bool(value['as_needed']))


def collect(records,lookup):
    bag={}
    for record in records:
        key=member_key(record,lookup)
        if key not in bag:
            bag[key]=dict(position=key[0],canonical_identity=record['reading']['canonical_identity'],
                product_key=key[1],classes=list(key[2]),role=key[3],as_needed=key[4],evidence=[])
        if record not in bag[key]['evidence']:bag[key]['evidence'].append(record)
    output=[]
    for key in sorted(bag):
        value=bag[key];value['evidence'].sort(key=lambda r:r['entry_key']);output.append(value)
    return output


def encode(members,base,available,candidates):
    n=len(base['hadm_ids']);positions=np.asarray([r['position'] for r in members],np.int64)
    vocab=sorted({r['canonical_identity'] for r in members if r['canonical_identity'] is not None
                  and base['patient_splits'][r['position']]=='train'})
    names=['UNKNOWN',*vocab];out=dict(base)
    out.update(section_available=np.asarray(available,bool),candidate_present=np.asarray(candidates,bool),
        positions=positions,products=np.asarray([names.index(r['canonical_identity']) if r['canonical_identity'] in names else 0 for r in members],np.int64),
        roles=np.asarray([ROLES.index(r['role']) for r in members],np.int64),
        class_memberships=np.asarray([[c in r['classes'] for c in CLASSES] for r in members],bool).reshape(-1,3),
        as_needed=np.asarray([r['as_needed'] for r in members],bool))
    # 无引用数/检测次数，也不把产品编号当有序连续值。
    features=np.zeros((n,20+len(vocab)),np.float32)
    for i,r in enumerate(members):
        p=r['position'];k=ROLES.index(r['role']);features[p,15+k]=1
        for c in r['classes']:features[p,CLASSES.index(c)*5+k]=1
        product=out['products'][i]
        if product:features[p,19+product]=1
    if np.any(candidates&~available):raise ValueError('candidate without saved chapter')
    out['features']=features
    return out,dict(product_vocabulary=names,roles=list(ROLES),classes=list(CLASSES),
        feature_names=[f'{c}:{r}' for c in CLASSES for r in ROLES]+[f'role:{r}' for r in ROLES]+[f'product:{v}' for v in vocab],
        numeric_scaling='not_applicable_binary_features_no_sd_amplification',product_vocabulary_training_only=True)


def derive():
    started=time.perf_counter();sources={}
    def protect(path,expected=None):
        result=sha256(path)
        if expected is not None and result!=expected:raise ValueError('protected source changed: '+str(path))
        sources[str(path)]=result;return result
    parent=json.loads((PARENT/'audit.json').read_text());proof=json.loads((PARENT/'verification.json').read_text())
    if proof['maximum_error']!=0 or proof['audit_sha256']!=protect(PARENT/'audit.json'):raise ValueError('r3 proof stale')
    protect(PARENT/'verification.json')
    for path,h in parent['source_sha256'].items():protect(Path(path),h)
    for file,h in parent['output_sha256'].items():protect(PARENT/file,h)
    inherited=parent['inherited_original_source_hashes']
    for directory,names in ((STATE,('manifest.json','clinical_state.npz')),
        (TASKS,('manifest.json','sources.jsonl','tasks.jsonl','members.npz')),
        (SCREEN,('manifest.json','selected_sentences.jsonl.gz'))):
        for name in names:
            path=directory/name
            keys=[h for k,h in inherited.items() if Path(k).resolve()==path.resolve()]
            if len(keys)!=1:raise ValueError('missing/ambiguous inherited file')
            protect(path,keys[0])
    with np.load(STATE/'clinical_state.npz') as z:base={k:z[k] for k in ('hadm_ids','subject_ids','patient_splits')}
    with np.load(TASKS/'members.npz') as z:
        for k,value in base.items():np.testing.assert_array_equal(value,z[k])
    if len(base['hadm_ids'])!=19401 or len(set(base['subject_ids']))!=5351 or (base['patient_splits']=='train').sum()!=13592:
        raise ValueError('cohort drift')
    lookup={str(h):i for i,h in enumerate(base['hadm_ids'])}
    if len(lookup)!=len(base['hadm_ids']):raise ValueError('duplicate admissions')
    summaries={};sections={};available=np.zeros(len(lookup),bool);candidates=np.zeros_like(available)
    with (TASKS/'sources.jsonl').open(encoding='utf-8') as stream:
        for r in map(json.loads,stream):
            if r['channel']!='discharge':continue
            if r['hadm_id'] not in lookup:raise ValueError('source outside fixed cohort')
            p=lookup[r['hadm_id']]
            if r['subject_id']!=str(base['subject_ids'][p]) or r['note_id'] in summaries:raise ValueError('source identity/duplicate')
            if digest(r['text'])!=r['original_sha256']:raise ValueError('saved note hash differs')
            ss=[s for s in r['sections'] if s['role']=='discharge_medications']
            for s in ss:span(r['text'],s['start_char'],s['end_char'],s['text'])
            summaries[r['note_id']]=r;sections[r['note_id']]=ss;available[p]|=bool(ss)
    ranges=defaultdict(list)
    with (TASKS/'tasks.jsonl').open(encoding='utf-8') as stream:
        for task in map(json.loads,stream):
            if task['route']!='discharge_all' or task['note_id'] not in summaries:continue
            source=summaries[task['note_id']]
            if any(task[k]!=source[k] for k in ('hadm_id','subject_id','original_sha256')):raise ValueError('task source identity')
            ranges[task['note_id']].extend(check_maps(source,task))
    for note,ss in sections.items():
        for s in ss:covered(s,ranges[note])
    records=[];train_units=[]
    with gzip.open(SCREEN/'selected_sentences.jsonl.gz','rt',encoding='utf-8') as stream:
        for r in map(json.loads,stream):
            if r['kind']!='discharge' or r['section']!='discharge_medications':continue
            if r['hadm_id'] not in lookup or r['note_id'] not in summaries:raise ValueError('candidate outside source')
            p=lookup[r['hadm_id']];source=summaries[r['note_id']]
            if r['subject_id']!=str(base['subject_ids'][p]) or r['patient_split']!=base['patient_splits'][p]:raise ValueError('candidate patient/fold mismatch')
            a,b=r['start_char'],r['end_char'];span(source['text'],a,b,r['quote'])
            if r['original_sha256']!=source['original_sha256'] or not any(s['start_char']<=a<b<=s['end_char'] for s in sections[r['note_id']]):raise ValueError('candidate chapter/hash mismatch')
            if not lexical(r['quote'])['classes']:continue
            candidates[p]=True
            if base['patient_splits'][p]=='train':train_units.append(r['unit_key'])
            for entry in read_unit(r['quote']):
                x,y=entry['entry_span'];quote=entry['entry_quote'];span(r['quote'],x,y,quote)
                if entry['product_span'] is not None:span(quote,*entry['product_span'],entry['product_name'])
                for ra,rb,rv in entry['route_spans']:span(quote,ra,rb,rv)
                records.append(dict(**{k:r[k] for k in ('hadm_id','subject_id','note_id','unit_key','original_sha256')},
                    entry_key=digest(r['unit_key']+':'+str(x)+':'+str(y)),source_start=a+x,source_end=a+y,quote=quote,
                    reading={k:v for k,v in entry.items() if k not in ('entry_span','entry_quote')},
                    parent_unit_source_start=a,parent_unit_source_end=b))
    if len({r['entry_key'] for r in records})!=len(records):raise ValueError('duplicate source entry key')
    with gzip.open(PARENT/'entry_evidence.jsonl.gz','rt',encoding='utf-8') as stream:old=list(map(json.loads,stream))
    train_records=[r for r in records if base['patient_splits'][lookup[r['hadm_id']]]=='train']
    if sorted(train_records,key=lambda r:r['entry_key'])!=sorted(old,key=lambda r:r['entry_key']) or len(train_units)!=9369:
        raise ValueError('training frozen entries differ')
    members=collect(records,lookup);arrays,metadata=encode(members,base,available,candidates)
    for path in (Path(__file__),PROTOCOL,Path('training/tests/test_discharge_regimen_interface.py')):protect(path)
    ti=base['patient_splits']=='train';training=[r for r in members if ti[r['position']]];counts=Counter(r['role'] for r in training)
    report=dict(status='fixed_rule_full_cohort_regimen_interface_not_model_input',source_sha256=sources,
        inherited_frozen_reference=parent['inherited_frozen_reference'],metadata=metadata,
        total_admissions=len(lookup),training_admissions=int(ti.sum()),training_member_count=len(training),
        training_source_available=int((available&ti).sum()),training_candidate_available=int((candidates&ti).sum()),
        training_role_members=dict(counts),training_entries_match_frozen_r3=True,training_entry_count=len(train_records),
        training_members_with_multiple_references=sum(len(r['evidence'])>1 for r in training),
        training_role_admissions={k:len({r['position'] for r in training if r['role']==k}) for k in ROLES},
        original_candidate_rule_unchanged=True,raw_discharge_read=False,outcomes_read=False,
        new_fits=0,new_vectors=0,actual_seconds=time.perf_counter()-started)
    return report,arrays,members


def write(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def run():
    if ROOT.exists():raise FileExistsError('interface exists')
    ROOT.mkdir();report,arrays,members=derive();np.savez_compressed(ROOT/'interface.npz',**arrays)
    with gzip.open(ROOT/'members.jsonl.gz','wt',encoding='utf-8') as stream:
        for r in members:stream.write(json.dumps(r)+'\n')
    report['output_sha256']={f:sha256(ROOT/f) for f in ('interface.npz','members.jsonl.gz')}
    write(ROOT/'manifest.json',report);write(ROOT/'state.json',dict(status='pending_independent_rebuild',new_fits=0))
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_sha256','inherited_frozen_reference','output_sha256','metadata')}),flush=True)


def verify():
    if (ROOT/'verification.json').exists():raise FileExistsError('already verified')
    old=json.loads((ROOT/'manifest.json').read_text());new,arrays,members=derive();seconds=new.pop('actual_seconds');old.pop('actual_seconds')
    for file,h in old.pop('output_sha256').items():
        if sha256(ROOT/file)!=h:raise ValueError('output changed')
    if old!=new:raise ValueError('source/metadata/statistics differ')
    with np.load(ROOT/'interface.npz') as z:
        for k,value in arrays.items():np.testing.assert_array_equal(z[k],value)
    with gzip.open(ROOT/'members.jsonl.gz','rt',encoding='utf-8') as stream:saved=list(map(json.loads,stream))
    # JSON将tuple转list，统一到落盘表示后逐引用比较。
    if saved!=json.loads(json.dumps(members)):raise ValueError('members/references differ')
    write(ROOT/'verification.json',dict(status='one_independent_saved_source_full_interface_rebuild',maximum_error=0.,
        manifest_sha256=sha256(ROOT/'manifest.json'),actual_seconds=seconds,new_fits=0))
    write(ROOT/'state.json',dict(status='full_cohort_interface_verified_not_network_consumed',new_fits=0,goal_achieved=False))
    print(json.dumps(dict(error=0.,seconds=seconds)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',required=True,choices=('run','verify'));mode=parser.parse_args().mode
    try:run() if mode=='run' else verify()
    except Exception as error:
        if ROOT.exists():write(ROOT/f'failure_{mode}.json',dict(error=repr(error)))
        raise
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/discharge_resp_fact_audit.py · lines 1–142

<!-- REVIEW_FRAGMENT {"path":"patient_data/discharge_resp_fact_audit.py","part":1,"start_line":1,"end_line":142,"source_sha256":"41ce7690aa89c2ef41123ec7dcbbd904f5d478c36c8de6d7d6ff3c8724d19ecb","fragment_sha256":"41ce7690aa89c2ef41123ec7dcbbd904f5d478c36c8de6d7d6ff3c8724d19ecb","added_terminal_newline":false} -->
````python
"""已有安全原句的训练折事实审计；不训练/编码，不向日志输出病历。"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from patient_data.discharge_resp_rules import extract
from patient_data.export_mimiciv import rows, sha256
from patient_data.text_time_coverage import SNAP, SCREEN, STATE, checked_sources
from training.v6.encode_screen_candidates import ROUTES
from training.v6.train import write_json

ROOT=Path('patient_data/processed/v6_discharge_resp_fact_audit_2026-10-13')
LOCAL=Path('patient_data/processed/v6_local_text_interface_2026-10-13_r2')
PROTOCOL=Path('docs/V6_DISCHARGE_RESP_FACT_AUDIT_PROTOCOL_2026-10-13.md')

def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()

def token_bounds(offsets, start, end):
    selected=[i for i,(a,b) in enumerate(offsets) if a<end and b>start]
    if not selected or selected!=list(range(selected[0],selected[-1]+1)):
        raise ValueError('fact token span invalid')
    return [selected[0],selected[-1]+1]

def derive():
    from transformers import AutoTokenizer
    source,_=checked_sources()
    m=json.loads((LOCAL/'manifest.json').read_text());proof=json.loads((LOCAL/'source_verification.json').read_text())
    if proof['manifest_sha256']!=sha256(LOCAL/'manifest.json') or proof['source_and_arrays_max_error']!=0:
        raise ValueError('old token source proof differs')
    for p,h in m['source_sha256'].items():
        if sha256(Path(p))!=h:raise ValueError('protected old source changed: '+p)
    for name,h in m['output_sha256'].items():
        if sha256(LOCAL/name)!=h:raise ValueError('old token interface output changed')
    with np.load(STATE/'clinical_state.npz') as z:
        members={str(h):str(s) for h,s,f in zip(z['hadm_ids'],z['subject_ids'],z['patient_splits']) if f=='train'}
    if len(members)!=13592:raise ValueError('training membership changed')
    with gzip.open(LOCAL/'index.jsonl.gz','rt',encoding='utf-8') as f:
        index=[r for r in map(json.loads,f) if r['patient_split']=='train']
    if len(index)!=6613:raise ValueError('locked safe candidate set changed')
    with gzip.open(LOCAL/'local_tasks.jsonl.gz','rt',encoding='utf-8') as f:tasks=list(map(json.loads,f))
    wanted={(r['note_id'],r['start_char'],r['end_char']) for r in index}
    units={}
    with gzip.open(SCREEN/'selected_sentences.jsonl.gz','rt',encoding='utf-8') as f:
        for r in map(json.loads,f):
            key=(r['note_id'],r['start_char'],r['end_char'])
            if key in wanted:
                if key in units:raise ValueError('duplicate source unit')
                units[key]=r
    if set(units)!=wanted:raise ValueError('saved source unit missing')
    wanted_notes={r['note_id'] for r in index}
    notes={r['note_id']:r for r in rows(SNAP/'raw/discharge.csv.gz') if r['hadm_id'] in members and r['note_id'] in wanted_notes}
    tokenizer=AutoTokenizer.from_pretrained('training/model_cache/Bio_ClinicalBERT',local_files_only=True,use_fast=True)
    if not tokenizer.is_fast:raise ValueError('offset tokenizer required')
    counters=Counter();accepted=[];rejections=[];sets=defaultdict(set);examples=[];role_admissions=defaultdict(set)
    for old in index:
        key=(old['note_id'],old['start_char'],old['end_char']);unit=units[key];original=notes[old['note_id']]
        h=old['hadm_id'];s=old['subject_id'];text=unit['quote'];start,end=key[1:]
        if h not in members or members[h]!=s or unit['hadm_id']!=h or unit['subject_id']!=s or unit['patient_split']!='train':
            raise ValueError('fact fold/subject differs')
        if unit['unit_key']!=old['unit_key'] or unit['section']!=old['section']:
            raise ValueError('fact unit identity/section differs')
        if not 0<=start<end<=len(original['text']) or original['text'][start:end]!=text or digest(text)!=old['quote_sha256']:
            raise ValueError('raw quote offset/hash differs')
        if digest(original['text'])!=old['original_sha256'] or unit['original_sha256']!=old['original_sha256']:
            raise ValueError('original document differs')
        encoded=tokenizer(text,add_special_tokens=False,truncation=False,return_offsets_mapping=True)
        prefix=tokenizer('Clinical topic: '+ROUTES[old['route']]+'\n',add_special_tokens=False)['input_ids']
        local=tasks[old['task_start']:old['task_end']];right=0;rebuilt_ids=[]
        for task in local:
            a,b=task['token_start'],task['token_end']
            if task['sentence_row']!=old['sentence_row'] or a!=right or task['input_ids']!=[tokenizer.cls_token_id,*prefix,*encoded['input_ids'][a:b],tokenizer.sep_token_id]:
                raise ValueError('saved actual local token IDs differ')
            rebuilt_ids.extend(encoded['input_ids'][a:b]);right=b
        if rebuilt_ids!=encoded['input_ids'] or right!=old['local_token_count']:raise ValueError('local token tail missing')
        facts,rejected=extract(text,unit['section']);counters['checked_safe_units']+=1
        for r in rejected:
            counters['rejected:'+r['reason']]+=1
        base={k:old[k] for k in ('hadm_id','subject_id','note_id','section','unit_key','sentence_row','quote_sha256','original_sha256','containing_rows_and_tokens')}
        base.update(patient_split='train',source_start_char=start,source_end_char=end,quote=text)
        if rejected:rejections.append(dict(**base,rejections=rejected))
        for fact in facts:
            a,b=fact['start'],fact['end'];token=token_bounds(encoded['offset_mapping'],a,b)
            record=dict(**base,**fact,evidence_start_char=start+a,evidence_end_char=start+b,
                evidence_sha256=digest(text[a:b]),local_token_span=token,
                measurement_time='unknown',clinically_verified=False)
            accepted.append(record);counters['concept:'+fact['concept']]+=1;counters['role:'+fact['role']]+=1
            counters['polarity:'+fact['polarity']]+=1;role_admissions[fact['role']].add(h)
            sets[(h,fact['concept'])].add((fact['polarity'],fact['role'],fact['value'],fact['unit'],fact['exertional']))
    # 固定hash分层回查材料只存本地，不输出原句/标识到日志。
    groups=defaultdict(list)
    for r in accepted:groups[(r['concept'],r['role'],r['polarity'])].append(r)
    for key,items in sorted(groups.items()):
        examples.extend(sorted(items,key=lambda r:digest(r['unit_key']+str(r['start'])))[:3])
    explicit=[r for r in accepted if r['role'] in ('explicit_discharge','discharge_section')]
    repeated=defaultdict(set)
    for r in explicit:repeated[r['quote_sha256']].add(r['subject_id'])
    counters['explicit_discharge_facts']=len(explicit)
    counters['explicit_shared_quote_facts']=sum(len(repeated[r['quote_sha256']])>1 for r in explicit)
    for p in (LOCAL/'manifest.json',LOCAL/'source_verification.json',LOCAL/'index.jsonl.gz',LOCAL/'local_tasks.jsonl.gz',
              Path(__file__),Path('patient_data/discharge_resp_rules.py'),PROTOCOL,Path('training/tests/test_v6_discharge_resp_rules.py')):
        source[str(p)]=sha256(p)
    report=dict(status='training_only_lexical_fact_audit_not_gold',source_sha256=source,counters=dict(counters),
        training_admissions=13592,safe_candidate_units=len(index),safe_candidate_admissions=len({r['hadm_id'] for r in index}),
        fact_admissions=len({r['hadm_id'] for r in accepted}),fact_patients=len({r['subject_id'] for r in accepted}),
        explicit_discharge_admissions=len({r['hadm_id'] for r in explicit}),explicit_discharge_patients=len({r['subject_id'] for r in explicit}),
        role_admissions={k:len(v) for k,v in role_admissions.items()},
        conflicting_same_admission_concepts=sum(len(v)>1 for v in sets.values()),review_sample_records=len(examples),
        new_fits=0,new_vectors=0,limits=['No clinical gold or precision/recall estimate','Unknown dates remain unknown; FINAL retrospective',
        'Source candidate membership finite, missed fact not normal','Sets not recovery sequence; discordant evidence not majority vote'])
    return report,accepted,rejections,examples

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('build','verify'),required=True)
    parser.add_argument('--tag',choices=('a','b'),default='a');args=parser.parse_args();started=time.perf_counter()
    report,facts,rejected,samples=derive();payload={'facts.jsonl.gz':facts,'rejected.jsonl.gz':rejected,'review_sample.jsonl.gz':samples}
    if args.mode=='build':
        if ROOT.exists():raise FileExistsError('audit exists, inspect rather than overwrite')
        ROOT.mkdir(parents=True)
        for name,records in payload.items():
            with gzip.open(ROOT/name,'xt',encoding='utf-8') as f:
                for r in records:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        write_json(ROOT/'audit.json',report)
        write_json(ROOT/'state.json',dict(status='built_pending_two_source_replays',new_fits=0,new_vectors=0))
    else:
        if json.loads((ROOT/'audit.json').read_text())!=report:raise ValueError('full source report reconstruction differs')
        for name,records in payload.items():
            with gzip.open(ROOT/name,'rt',encoding='utf-8') as f:
                if list(map(json.loads,f))!=records:raise ValueError('full evidence/token records differ')
        write_json(ROOT/f'verification_{args.tag}.json',dict(status='independent_full_source_replay_verified',maximum_error=0,
            audit_sha256=sha256(ROOT/'audit.json'),output_sha256={n:sha256(ROOT/n) for n in payload},new_fits=0,new_vectors=0))
        if all((ROOT/f'verification_{tag}.json').exists() for tag in ('a','b')):
            write_json(ROOT/'state.json',dict(status='two_independent_full_source_replays_verified',new_fits=0,new_vectors=0,goal_achieved=False))
    print(json.dumps(dict(counters=report['counters'],fact_admissions=report['fact_admissions'],
        explicit_discharge_admissions=report['explicit_discharge_admissions'],explicit_discharge_patients=report['explicit_discharge_patients'],
        role_admissions=report['role_admissions'],seconds=time.perf_counter()-started)),flush=True)

if __name__=='__main__':main()
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/discharge_resp_fact_audit_r2.py · lines 1–151

<!-- REVIEW_FRAGMENT {"path":"patient_data/discharge_resp_fact_audit_r2.py","part":1,"start_line":1,"end_line":151,"source_sha256":"be395eed42daa6033125894026e722a893ed5655f9c693cad9495640c876f24c","fragment_sha256":"be395eed42daa6033125894026e722a893ed5655f9c693cad9495640c876f24c","added_terminal_newline":false} -->
````python
"""独立r2限定语义修复读出；同原句同来源，原r1不改，不训练。"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from patient_data.discharge_resp_rules_r2 import extract
from patient_data.export_mimiciv import rows, sha256
from patient_data.text_time_coverage import SNAP, SCREEN, STATE, checked_sources
from training.v6.encode_screen_candidates import ROUTES
from training.v6.train import write_json

ROOT=Path('patient_data/processed/v6_discharge_resp_fact_audit_2026-10-13_r2_final')
LOCAL=Path('patient_data/processed/v6_local_text_interface_2026-10-13_r2')
PROTOCOL=Path('docs/V6_DISCHARGE_RESP_R2_PROTOCOL_2026-10-13.md')

def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()

def token_bounds(offsets, start, end):
    selected=[i for i,(a,b) in enumerate(offsets) if a<end and b>start]
    if not selected or selected!=list(range(selected[0],selected[-1]+1)):
        raise ValueError('fact token span invalid')
    return [selected[0],selected[-1]+1]

def derive():
    from transformers import AutoTokenizer
    source,_=checked_sources()
    parent=Path('patient_data/processed/v6_discharge_resp_fact_audit_2026-10-13')
    parent_report=json.loads((parent/'audit.json').read_text())
    for tag in ('a','b'):
        p=json.loads((parent/f'verification_{tag}.json').read_text())
        if p['audit_sha256']!=sha256(parent/'audit.json') or p['maximum_error']!=0:raise ValueError('r1 source proof changed')
    for path,h in parent_report['source_sha256'].items():
        if sha256(Path(path))!=h:raise ValueError('r1 protected source changed: '+path)
    for p in (parent/'audit.json',parent/'verification_a.json',parent/'verification_b.json',parent/'quality_readout.json',parent/'quality_flagged_local.json'):
        source[str(p)]=sha256(p)
    m=json.loads((LOCAL/'manifest.json').read_text());proof=json.loads((LOCAL/'source_verification.json').read_text())
    if proof['manifest_sha256']!=sha256(LOCAL/'manifest.json') or proof['source_and_arrays_max_error']!=0:
        raise ValueError('old token source proof differs')
    for p,h in m['source_sha256'].items():
        if sha256(Path(p))!=h:raise ValueError('protected old source changed: '+p)
    for name,h in m['output_sha256'].items():
        if sha256(LOCAL/name)!=h:raise ValueError('old token interface output changed')
    with np.load(STATE/'clinical_state.npz') as z:
        members={str(h):str(s) for h,s,f in zip(z['hadm_ids'],z['subject_ids'],z['patient_splits']) if f=='train'}
    if len(members)!=13592:raise ValueError('training membership changed')
    with gzip.open(LOCAL/'index.jsonl.gz','rt',encoding='utf-8') as f:
        index=[r for r in map(json.loads,f) if r['patient_split']=='train']
    if len(index)!=6613:raise ValueError('locked safe candidate set changed')
    with gzip.open(LOCAL/'local_tasks.jsonl.gz','rt',encoding='utf-8') as f:tasks=list(map(json.loads,f))
    wanted={(r['note_id'],r['start_char'],r['end_char']) for r in index}
    units={}
    with gzip.open(SCREEN/'selected_sentences.jsonl.gz','rt',encoding='utf-8') as f:
        for r in map(json.loads,f):
            key=(r['note_id'],r['start_char'],r['end_char'])
            if key in wanted:
                if key in units:raise ValueError('duplicate source unit')
                units[key]=r
    if set(units)!=wanted:raise ValueError('saved source unit missing')
    wanted_notes={r['note_id'] for r in index}
    notes={r['note_id']:r for r in rows(SNAP/'raw/discharge.csv.gz') if r['hadm_id'] in members and r['note_id'] in wanted_notes}
    tokenizer=AutoTokenizer.from_pretrained('training/model_cache/Bio_ClinicalBERT',local_files_only=True,use_fast=True)
    if not tokenizer.is_fast:raise ValueError('offset tokenizer required')
    counters=Counter();accepted=[];rejections=[];sets=defaultdict(set);examples=[];role_admissions=defaultdict(set)
    for old in index:
        key=(old['note_id'],old['start_char'],old['end_char']);unit=units[key];original=notes[old['note_id']]
        h=old['hadm_id'];s=old['subject_id'];text=unit['quote'];start,end=key[1:]
        if h not in members or members[h]!=s or unit['hadm_id']!=h or unit['subject_id']!=s or unit['patient_split']!='train':
            raise ValueError('fact fold/subject differs')
        if unit['unit_key']!=old['unit_key'] or unit['section']!=old['section']:
            raise ValueError('fact unit identity/section differs')
        if not 0<=start<end<=len(original['text']) or original['text'][start:end]!=text or digest(text)!=old['quote_sha256']:
            raise ValueError('raw quote offset/hash differs')
        if digest(original['text'])!=old['original_sha256'] or unit['original_sha256']!=old['original_sha256']:
            raise ValueError('original document differs')
        encoded=tokenizer(text,add_special_tokens=False,truncation=False,return_offsets_mapping=True)
        prefix=tokenizer('Clinical topic: '+ROUTES[old['route']]+'\n',add_special_tokens=False)['input_ids']
        local=tasks[old['task_start']:old['task_end']];right=0;rebuilt_ids=[]
        for task in local:
            a,b=task['token_start'],task['token_end']
            if task['sentence_row']!=old['sentence_row'] or a!=right or task['input_ids']!=[tokenizer.cls_token_id,*prefix,*encoded['input_ids'][a:b],tokenizer.sep_token_id]:
                raise ValueError('saved actual local token IDs differ')
            rebuilt_ids.extend(encoded['input_ids'][a:b]);right=b
        if rebuilt_ids!=encoded['input_ids'] or right!=old['local_token_count']:raise ValueError('local token tail missing')
        facts,rejected=extract(text,unit['section']);counters['checked_safe_units']+=1
        for r in rejected:
            counters['rejected:'+r['reason']]+=1
        base={k:old[k] for k in ('hadm_id','subject_id','note_id','section','unit_key','sentence_row','quote_sha256','original_sha256','containing_rows_and_tokens')}
        base.update(patient_split='train',source_start_char=start,source_end_char=end,quote=text)
        if rejected:rejections.append(dict(**base,rejections=rejected))
        for fact in facts:
            a,b=fact['start'],fact['end'];token=token_bounds(encoded['offset_mapping'],a,b)
            record=dict(**base,**fact,evidence_start_char=start+a,evidence_end_char=start+b,
                evidence_sha256=digest(text[a:b]),local_token_span=token,
                measurement_time='unknown',clinically_verified=False)
            accepted.append(record);counters['concept:'+fact['concept']]+=1;counters['role:'+fact['role']]+=1
            counters['polarity:'+fact['polarity']]+=1;role_admissions[fact['role']].add(h)
            sets[(h,fact['concept'])].add((fact['polarity'],fact['role'],fact['value'],fact['unit'],fact['activity_condition']))
    # 固定hash分层回查材料只存本地，不输出原句/标识到日志。
    groups=defaultdict(list)
    for r in accepted:groups[(r['concept'],r['role'],r['polarity'])].append(r)
    for key,items in sorted(groups.items()):
        examples.extend(sorted(items,key=lambda r:digest(r['unit_key']+str(r['start'])))[:3])
    explicit=[r for r in accepted if r['role'] in ('explicit_discharge','discharge_section')]
    repeated=defaultdict(set)
    for r in explicit:repeated[r['quote_sha256']].add(r['subject_id'])
    counters['explicit_discharge_facts']=len(explicit)
    counters['explicit_shared_quote_facts']=sum(len(repeated[r['quote_sha256']])>1 for r in explicit)
    for p in (LOCAL/'manifest.json',LOCAL/'source_verification.json',LOCAL/'index.jsonl.gz',LOCAL/'local_tasks.jsonl.gz',
              Path(__file__),Path('patient_data/discharge_resp_rules.py'),Path('patient_data/discharge_resp_rules_r2.py'),PROTOCOL,Path('training/tests/test_v6_discharge_resp_rules.py'),Path('training/tests/test_v6_discharge_resp_rules_r2.py')):
        source[str(p)]=sha256(p)
    report=dict(status='training_only_lexical_fact_audit_not_gold',source_sha256=source,counters=dict(counters),
        training_admissions=13592,safe_candidate_units=len(index),safe_candidate_admissions=len({r['hadm_id'] for r in index}),
        fact_admissions=len({r['hadm_id'] for r in accepted}),fact_patients=len({r['subject_id'] for r in accepted}),
        explicit_discharge_admissions=len({r['hadm_id'] for r in explicit}),explicit_discharge_patients=len({r['subject_id'] for r in explicit}),
        role_admissions={k:len(v) for k,v in role_admissions.items()},
        conflicting_same_admission_concepts=sum(len(v)>1 for v in sets.values()),review_sample_records=len(examples),
        new_fits=0,new_vectors=0,limits=['No clinical gold or precision/recall estimate','Unknown dates remain unknown; FINAL retrospective',
        'Source candidate membership finite, missed fact not normal','Sets not recovery sequence; discordant evidence not majority vote'])
    return report,accepted,rejections,examples

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('build','verify'),required=True)
    parser.add_argument('--tag',choices=('a','b'),default='a');args=parser.parse_args();started=time.perf_counter()
    report,facts,rejected,samples=derive();payload={'facts.jsonl.gz':facts,'rejected.jsonl.gz':rejected,'review_sample.jsonl.gz':samples}
    if args.mode=='build':
        if ROOT.exists():raise FileExistsError('audit exists, inspect rather than overwrite')
        ROOT.mkdir(parents=True)
        for name,records in payload.items():
            with gzip.open(ROOT/name,'xt',encoding='utf-8') as f:
                for r in records:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        write_json(ROOT/'audit.json',report)
        write_json(ROOT/'state.json',dict(status='built_pending_two_source_replays',new_fits=0,new_vectors=0))
    else:
        if json.loads((ROOT/'audit.json').read_text())!=report:raise ValueError('full source report reconstruction differs')
        for name,records in payload.items():
            with gzip.open(ROOT/name,'rt',encoding='utf-8') as f:
                if list(map(json.loads,f))!=records:raise ValueError('full evidence/token records differ')
        write_json(ROOT/f'verification_{args.tag}.json',dict(status='independent_full_source_replay_verified',maximum_error=0,
            audit_sha256=sha256(ROOT/'audit.json'),output_sha256={n:sha256(ROOT/n) for n in payload},new_fits=0,new_vectors=0))
        if all((ROOT/f'verification_{tag}.json').exists() for tag in ('a','b')):
            write_json(ROOT/'state.json',dict(status='two_independent_full_source_replays_verified',new_fits=0,new_vectors=0,goal_achieved=False))
    print(json.dumps(dict(counters=report['counters'],fact_admissions=report['fact_admissions'],
        explicit_discharge_admissions=report['explicit_discharge_admissions'],explicit_discharge_patients=report['explicit_discharge_patients'],
        role_admissions=report['role_admissions'],seconds=time.perf_counter()-started)),flush=True)

if __name__=='__main__':main()
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/discharge_resp_fact_review.py · lines 1–45

<!-- REVIEW_FRAGMENT {"path":"patient_data/discharge_resp_fact_review.py","part":1,"start_line":1,"end_line":45,"source_sha256":"aabf8d8bd41bc1506ff2c324edc8fc49323e61630c23221fbe81606f529dd46b","fragment_sha256":"aabf8d8bd41bc1506ff2c324edc8fc49323e61630c23221fbe81606f529dd46b","added_terminal_newline":false} -->
````python
"""有限审计样本去标识回查：只展示白名单临床词，非人工gold。"""
from collections import Counter
import gzip
import json
from pathlib import Path
import re
from patient_data.discharge_resp_fact_audit import ROOT
from patient_data.export_mimiciv import sha256
from training.v6.train import write_json

WORDS=set('at on upon by prior to time of discharge discharged discharging she he they patient was is were are had has have required requires requiring oxygen o2 supplemental nc nasal cannula liters litre l min room air weaned wean off from successfully unable failed could not able still dyspnea sob doe shortness breath with exertion exertional ambulation walking activity improved resolved subsided remained remains continued continue home no without denies denied any residual significant new ongoing but however yet if should will plan baseline history previously normal stable comfortable breathing tolerating and due needed need needs'.split())

def clinical_mask(text):
    return ' '.join(token.lower() if token.lower() in WORDS else '[x]' for token in re.findall(r'[A-Za-z]+|\d+(?:\.\d+)?',text))

def main():
    with gzip.open(ROOT/'review_sample.jsonl.gz','rt',encoding='utf-8') as f:sample=list(map(json.loads,f))
    summaries=[]
    for i,r in enumerate(sample):
        summaries.append(dict(sample_number=i,concept=r['concept'],role=r['role'],polarity=r['polarity'],value=r['value'],unit=r['unit'],
            clause_mask=clinical_mask(r['quote'][r['clause_start']:r['clause_end']])))
    with gzip.open(ROOT/'facts.jsonl.gz','rt',encoding='utf-8') as f:facts=list(map(json.loads,f))
    issues=Counter();flagged=[]
    for r in facts:
        clause=r['quote'][r['clause_start']:r['clause_end']];reasons=[]
        if r['concept']=='oxygen_failed_wean' and not re.search(r'\b(?:oxygen|O2|room air|NC|nasal cannula)\b',clause,re.I):
            reasons.append('failed_wean_no_explicit_oxygen_object')
        if r['concept']=='dyspnea' and r['polarity']=='affirmed' and re.search(r'\b(?:improv\w*|resolv\w*)\b',clause,re.I):
            reasons.append('dyspnea_mention_with_improvement_scope_unresolved')
        if r['concept'].startswith('oxygen_weaned') and re.search(r'\b(?:attempt\w*|try|trying|instruct\w*|goal)\b',clause,re.I):
            reasons.append('weaning_attempt_or_instruction_not_success_proof')
        if r['concept']=='oxygen_flow_litres_notation' and re.search(r'\brest\b',clause,re.I) and re.search(r'\b(?:ambulat\w*|exertion|activity|walking)\b',clause,re.I):
            reasons.append('mixed_rest_activity_flow_scope_requires_binding')
        for reason in reasons:issues[reason]+=1
        if reasons:
            flagged.append(dict(sentence_row=r['sentence_row'],unit_key=r['unit_key'],start=r['start'],end=r['end'],issues=reasons))
    write_json(ROOT/'quality_readout.json',dict(status='not_ready_for_model_interface',all_lexical_records=len(facts),
        fixed_stratified_sample_records=len(sample),masked_review_only=True,issue_counts=dict(issues),new_fits=0,new_vectors=0,
        clinical_precision_recall=None,source_audit_sha256=sha256(ROOT/'audit.json'),review_code_sha256=sha256(Path(__file__)),
        limits=['Issue flags are review targets, not exhaustive semantic error counts','Masked limited review is not clinical gold labeling',
        'All first-version lexical records retained for audit, not approved current-state facts']))
    write_json(ROOT/'quality_flagged_local.json',flagged)
    print(json.dumps(dict(samples=summaries,role_counts=dict(Counter(r['role'] for r in sample)),issue_counts=dict(issues)),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/discharge_resp_r2_readout.py · lines 1–39

<!-- REVIEW_FRAGMENT {"path":"patient_data/discharge_resp_r2_readout.py","part":1,"start_line":1,"end_line":39,"source_sha256":"72f703c07f342a01c99a4b190332d56ac6a5ea8a06b0cc91c64f856320e331c0","fragment_sha256":"72f703c07f342a01c99a4b190332d56ac6a5ea8a06b0cc91c64f856320e331c0","added_terminal_newline":false} -->
````python
"""r2限定修复去向及有限掩码回查，未建立临床gold。"""
from collections import Counter,defaultdict
import gzip
import json
from pathlib import Path
from patient_data.discharge_resp_fact_audit_r2 import ROOT
from patient_data.discharge_resp_fact_review import clinical_mask
from patient_data.export_mimiciv import sha256
from training.v6.train import write_json

PARENT=Path('patient_data/processed/v6_discharge_resp_fact_audit_2026-10-13')

def main():
    with gzip.open(ROOT/'facts.jsonl.gz','rt',encoding='utf-8') as f:facts=list(map(json.loads,f))
    old_flags=json.loads((PARENT/'quality_flagged_local.json').read_text())
    flags={};outcomes=defaultdict(Counter)
    for r in facts:flags[(r['unit_key'],r['r1_start'],r['r1_end'])]=r
    for old in old_flags:
        r=flags.get((old['unit_key'],old['start'],old['end']))
        disposition='isolated' if r is None else ('rebound_improvement' if r['concept']=='dyspnea_improved' else
            'retained_symptom_local_scope' if r['concept']=='dyspnea' else r['activity_condition'] if r['concept']=='oxygen_flow_litres_notation' else 'retained')
        for issue in old['issues']:outcomes[issue][disposition]+=1
    with gzip.open(ROOT/'review_sample.jsonl.gz','rt',encoding='utf-8') as f:sample=list(map(json.loads,f))
    summaries=[]
    for i,r in enumerate(sample):
        summaries.append(dict(sample_number=i,concept=r['concept'],role=r['role'],polarity=r['polarity'],value=r['value'],unit=r['unit'],
            activity_condition=r['activity_condition'],clause_mask=clinical_mask(r['quote'][r['clause_start']:r['clause_end']])))
    result=dict(status='limited_semantic_revision_source_verified_not_clinical_gold',
        original_flag_dispositions={k:dict(v) for k,v in outcomes.items()},
        activity_conditions=dict(Counter(r['activity_condition'] for r in facts if r['value'] is not None)),
        records=len(facts),masked_review_samples=len(sample),source_sha256={str(p):sha256(p) for p in
            (ROOT/'audit.json',ROOT/'facts.jsonl.gz',ROOT/'review_sample.jsonl.gz',PARENT/'quality_flagged_local.json',Path(__file__))},
        clinical_precision_recall=None,new_fits=0,new_vectors=0,
        limits=['Fixed clinical-vocabulary masked review not full gold annotation','Isolated targets not true-negative medical diagnoses',
            'Local binding conservatively handles limited surface grammar; no exhaustive accuracy claim'])
    write_json(ROOT/'repair_readout.json',result)
    print(json.dumps(dict(summary=result,samples=summaries),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/discharge_resp_rules.py · lines 1–93

<!-- REVIEW_FRAGMENT {"path":"patient_data/discharge_resp_rules.py","part":1,"start_line":1,"end_line":93,"source_sha256":"c1022e35032129fa93522dfe25f747f2772d3ba4671faf277feea5e5b5c285e4","fragment_sha256":"c1022e35032129fa93522dfe25f747f2772d3ba4671faf277feea5e5b5c285e4","added_terminal_newline":false} -->
````python
"""有限明确呼吸事实规则；未知时间不自动升格为出院状态。"""
import re

def rx(pattern):
    return re.compile(pattern, re.I)

DISCHARGE = rx(r'\b(?:(?:at|on|upon|by|before|prior to|time of)\s+(?:the\s+)?discharge|discharged|discharging)\b')
FORBIDDEN = (
    ('conditional_or_plan', rx(r'\b(?:if|unless|should|will|would|plan\w*|recommend\w*|advis\w*|scheduled|consider\w*)\b')),
    ('post_discharge', rx(r'\b(?:after|following)\s+(?:the\s+)?discharge\b|\bpost[- ]discharge\b')),
    ('historical_or_baseline', rx(r'\b(?:history of|previous\w*|prior (?!to discharge)|baseline|at home|at admission|on admission|presented with)\b')),
    ('other_experiencer', rx(r'\b(?:mother|father|wife|husband|daughter|son|family member)\b')),
    ('uncertain', rx(r'\b(?:possibly|possible|may|might|suspected|rule out)\b')),
)
CHANGE = rx(r'\b(?:improv\w*|resolv\w*|wean\w*|persist\w*|still|remained|requiring)\b')
NEG = rx(r'\b(?:no|not|never|without|denies|denied|negative for|absence of|unable to|failed to|could not)\b')
DYSPNEA = r'(?:dyspn[eo]a|shortness of breath|SOB|DOE)'
OXYGEN = r'(?:oxygen|O2)'
FLOW = rx(r'\b(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>L(?:/min)?|lit(?:er|re)s?)\s*(?:(?:of|via|by|on)\s+)?(?:NC\b|nasal cannula\b|oxygen\b|O2\b)')
RULES = (
    ('oxygen_failed_wean', rx(r'\b(?:unable|failed|could not|not able)\b[^;.!\n]{0,35}\bwean\w*\b'), False),
    ('oxygen_weaned_room_air', rx(r'\bwean\w*\b[^;.!\n]{0,55}?\bto\s+room air\b'), True),
    ('oxygen_weaned_off', rx(r'\bwean\w*\s+(?:successfully\s+)?off\s+(?:of\s+)?(?:supplemental\s+)?'+OXYGEN+r'\b'), True),
    ('oxygen_required', rx(r'\b(?:requiring|requires?|required|remains?|remained|discharged|sent home|continues? to require)\b[^;.!\n]{0,30}?\b'+OXYGEN+r'\b'), True),
    ('room_air_status', rx(r'\b(?:on|breathing|tolerating|maintained on)\s+room air\b'), True),
    ('dyspnea', rx(r'\b'+DYSPNEA+r'\b'), False),
)

def clauses(text):
    """保留真实字符偏移；小数点不分句，不跨换行借时间或否定。"""
    cursor = 0
    for separator in re.finditer(r'[;!\n]+|(?<!\d)\.(?!\d)', text):
        if cursor < separator.start():
            yield cursor, separator.start(), text[cursor:separator.start()]
        cursor = separator.end()
    if cursor < len(text):
        yield cursor, len(text), text[cursor:]

def temporal_role(text, section):
    if DISCHARGE.search(text):
        return 'explicit_discharge'
    if section in ('discharge_condition', 'discharge_physical_exam'):
        return 'discharge_section'
    if section == 'hospital_course' and CHANGE.search(text):
        return 'inpatient_change'
    return 'unknown'

def extract(text, section):
    """返回事实及隔离理由，未命中不表示正常。范围/比较语义不自动当单值。"""
    if section not in ('hospital_course', 'discharge_condition', 'discharge_physical_exam',
                       'discharge_instructions', 'transitional_issues', 'followup_instructions'):
        return [], [{'reason': 'section_not_current', 'start': 0, 'end': len(text)}]
    facts, rejected = [], []
    for offset, end, clause in clauses(text):
        reasons = [name for name, pattern in FORBIDDEN if pattern.search(clause)]
        if reasons:
            rejected.extend(dict(reason=r, start=offset, end=end) for r in reasons)
            continue
        role = temporal_role(clause, section)
        for concept, pattern, reject_negative in RULES:
            for match in pattern.finditer(clause):
                prior = re.split(r'\b(?:but|however|yet)\b', clause[max(0, match.start()-45):match.start()], flags=re.I)[-1]
                if reject_negative and NEG.search(prior):
                    rejected.append(dict(reason='negated_'+concept, start=offset+match.start(), end=offset+match.end()))
                    continue
                polarity, name = 'affirmed', concept
                if concept == 'dyspnea':
                    direct_neg = rx(r'\b(?:no|without|denies|denied|negative for|absence of)\s+(?:(?:any|significant|new|residual|further|ongoing|complaints of|episodes of)\s+){0,3}$')
                    if direct_neg.search(prior):
                        polarity = 'negated'
                    elif NEG.search(prior):
                        rejected.append(dict(reason='ambiguous_dyspnea_negation_scope', start=offset+match.start(), end=offset+match.end()))
                        continue
                    elif rx(r'^\s+(?:had\s+)?(?:improved|resolved|subsided)\b').search(clause[match.end():]):
                        name = 'dyspnea_improved'
                facts.append(dict(concept=name, polarity=polarity, role=role, value=None, unit=None,
                    exertional=bool(rx(r'\b(?:exertion|exertional|ambulat\w*|walking|activity|DOE)\b').search(clause)),
                    start=offset+match.start(), end=offset+match.end(), clause_start=offset, clause_end=end))
        for match in FLOW.finditer(clause):
            prior = clause[max(0, match.start()-45):match.start()]
            if rx(r'\d\s*(?:[-–]|to)\s*$|[<>]=?\s*$|\b(?:at least|at most|more than|less than)\s*$').search(prior):
                rejected.append(dict(reason='flow_range_not_scalar', start=offset+match.start(), end=offset+match.end()))
                continue
            if NEG.search(prior):
                rejected.append(dict(reason='negated_oxygen_flow', start=offset+match.start(), end=offset+match.end()))
                continue
            if rx(r'\b(?:wean\w*|reduced|decreased)\b[^;.!]{0,30}\bfrom\s*$').search(prior):
                rejected.append(dict(reason='transition_start_not_latest_flow', start=offset+match.start(), end=offset+match.end()))
                continue
            facts.append(dict(concept='oxygen_flow_litres_notation', polarity='affirmed', role=role,
                value=float(match['value']), unit='L/min' if '/min' in match['unit'].lower() else 'L_oxygen_notation', exertional=bool(rx(r'\b(?:exertion|ambulat\w*|walking|activity)\b').search(clause)),
                start=offset+match.start(), end=offset+match.end(), clause_start=offset, clause_end=end))
    return facts, rejected
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/discharge_resp_rules_r2.py · lines 1–80

<!-- REVIEW_FRAGMENT {"path":"patient_data/discharge_resp_rules_r2.py","part":1,"start_line":1,"end_line":80,"source_sha256":"085f28fc372604ff359bb622615eb875ab146f6e22aad695aefb814c7ef8ce7e","fragment_sha256":"085f28fc372604ff359bb622615eb875ab146f6e22aad695aefb814c7ef8ce7e","added_terminal_newline":false} -->
````python
"""r1不变；限定氧对象/改善/动作完成/局部活动条件语义修复。"""
import re
from patient_data.discharge_resp_rules import extract as original_extract, rx, DYSPNEA, OXYGEN, FLOW

TERMINATOR=rx(r'\b(?:but|however|yet)\b')
ATTEMPT=rx(r'\b(?:attempt\w*|try|trying|instruct\w*|goal)\b')
FAILED_OBJECT=rx(r'\b(?:unable|failed|could not|not able)(?:\s+to)?(?:\s+(?:be|successfully))?\s+wean(?:ed)?\s+(?:(?:off|from|to)\s+)?(?:supplemental\s+)?(?:oxygen|O2|room air|NC|nasal cannula)\b')
ADVERB=r'(?:(?:significantly|markedly|much|somewhat|greatly|substantially|partially)\s+)?'
IMPROVED_AFTER=rx(r'^\s+(?:(?:is|was|has|had)\s+)?(?:been\s+)?'+ADVERB+r'(?:improved|resolved|subsided)\b')
IMPROVED_BEFORE=rx(r'\b(?:improved|improving|resolved)\s+(?:(?:significant|residual)\s+)?$')
IMPROVEMENT=rx(r'\b(?:improv\w*|resolv\w*|subsided)\b')
NEED_BRIDGE=r'(?:(?:on|with|home|supplemental|additional|continuous|some|low|flow|of|via|nasal|cannula|NC|L|min|liters?|litres?)\s+|\d+(?:\.\d+)?\s*(?:L(?:/min)?\s*)?){0,12}'
NEED_OBJECT=rx(r'\b(?:requiring|requires?|required|remains?|remained|discharged|sent home|continues? to require)\s+'+NEED_BRIDGE+OXYGEN+r'\b')
REST=rx(r'\b(?:at\s+rest|resting|rest)\b')
ACTIVITY=rx(r'\b(?:exertion|exertional|ambulat\w*|walking|activity|DOE)\b')

def local_scope(clause,start,end):
    """否定/改善的转折边界；保留原始坐标，不跨前一转折。"""
    left=0;right=len(clause)
    for m in TERMINATOR.finditer(clause):
        if m.end()<=start:left=m.end()
        elif m.start()>=end:right=m.start();break
    return left,right

def activity_binding(clause,match):
    """数值条件在相邻氧量/协调边界内绑定，双条件无法分清则显式ambiguous。"""
    flows=list(FLOW.finditer(clause));previous=0;following=len(clause)
    if len(flows)==1 and REST.search(clause) and ACTIVITY.search(clause):
        return 'ambiguous'
    for f in flows:
        if f.end()<=match.start():previous=f.end()
        elif f.start()>=match.end():following=f.start();break
    before=re.split(r'\b(?:and|but|however|yet)\b|,',clause[previous:match.start()],flags=re.I)[-1]
    after=re.split(r'\b(?:and|but|however|yet)\b|,',clause[match.end():following],flags=re.I)[0]
    context=before+' '+after;rest=bool(REST.search(context));active=bool(ACTIVITY.search(context))
    return 'ambiguous' if rest and active else 'rest' if rest else 'exertion' if active else 'unknown'

def extract(text,section):
    candidates,rejected=original_extract(text,section);accepted=[]
    for raw in candidates:
        r=dict(raw);concept=r['concept'];start,end=r['start'],r['end']
        clause=text[r['clause_start']:r['clause_end']];a=start-r['clause_start'];b=end-r['clause_start']
        lo,hi=local_scope(clause,a,b);left=clause[lo:a];right=clause[b:hi];scope=clause[lo:hi]
        reason=None
        if concept=='oxygen_failed_wean':
            match=FAILED_OBJECT.match(clause,a)
            if match is None:reason='wean_failure_object_unbound'
            else:r['end']=r['clause_start']+match.end()
        elif concept.startswith('oxygen_weaned'):
            if ATTEMPT.search(scope):reason='wean_attempt_instruction_or_goal'
            elif not rx(r'^weaned\b').match(text[start:end]):reason='wean_completion_not_explicit'
        elif concept=='oxygen_required':
            match=NEED_OBJECT.match(clause,a)
            if match is None or match.end()!=b:reason='oxygen_need_object_unbound'
        elif concept in ('dyspnea','dyspnea_improved') and r['polarity']=='affirmed':
            # 原r1证据可能含改善后缀；先只找真正症状位置，再绑定修饰。
            symptom=rx(r'\b'+DYSPNEA+r'\b').match(clause,a)
            if symptom is None:raise ValueError('old dyspnea span does not start with symptom')
            b=symptom.end();right=clause[b:hi]
            post=IMPROVED_AFTER.match(right);pre=IMPROVED_BEFORE.search(left)
            if post or pre:
                r['concept']='dyspnea_improved'
                if post:r['end']=r['clause_start']+b+post.end()
                if pre:r['start']=r['clause_start']+lo+pre.start()
            elif IMPROVEMENT.search(scope):reason='dyspnea_modifier_scope_unbound'
        if ATTEMPT.search(scope) and concept in ('room_air_status','oxygen_required','oxygen_flow_litres_notation'):
            reason=reason or 'instruction_or_attempt_context'
        if reason:
            rejected.append(dict(reason=reason,start=start,end=end));continue
        if concept=='oxygen_flow_litres_notation':
            match=FLOW.match(clause,a)
            if match is None:raise ValueError('old flow span mismatch')
            condition=activity_binding(clause,match)
        else:
            active=bool(ACTIVITY.search(scope));rest=bool(REST.search(scope))
            condition='ambiguous' if active and rest else 'exertion' if active else 'rest' if rest else 'unknown'
        r['activity_condition']=condition;r['exertional']=condition=='exertion'
        r['r1_start']=start;r['r1_end']=end
        accepted.append(r)
    return accepted,rejected
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/disposition_audit.py · lines 1–79

<!-- REVIEW_FRAGMENT {"path":"patient_data/disposition_audit.py","part":1,"start_line":1,"end_line":79,"source_sha256":"3517ea03c9e09300aae2141c9784d787a3ceda3c0c6c5e837b7b2d0965cc1f03","fragment_sha256":"3517ea03c9e09300aae2141c9784d787a3ceda3c0c6c5e837b7b2d0965cc1f03","added_terminal_newline":false} -->
````python
"""训练折原去向与医院资格来源QC；不读取下游结局、不改队列。"""
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import numpy as np
from patient_data.export_mimiciv import rows,sha256

PARENT=Path('training_runs/v6_admission_context_2026-10-14')
ROOT=Path('training_runs/v6_disposition_audit_2026-10-14')
PROTOCOL=Path('docs/V6_DISPOSITION_PROTOCOL_2026-10-14.md')
SQL=Path('cohort_pipeline/sql/08c_copd_coded_admissions_mimiciv.sql')

def eligibility(row):
    try:
        return int(row['hospital_expire_flag'])==0 and datetime.fromisoformat(row['dischtime'])>datetime.fromisoformat(row['admittime'])
    except (KeyError,ValueError,TypeError):
        raise ValueError('unreadable hospital qualification') from None

def time_role(value,row):
    if value in ('',r'\N',None):return 'unrecorded'
    value=datetime.fromisoformat(value)
    a,b=(datetime.fromisoformat(row[k]) for k in ('admittime','dischtime'))
    return 'before_admission' if value<a else 'within_hospital' if value<=b else 'after_discharge'

def derive():
    m=json.loads((PARENT/'interface_manifest.json').read_text());proof=json.loads((PARENT/'verification.json').read_text())
    if m['sha256']!=sha256(PARENT/'interface.npz') or proof['maximum_error']!=0 or proof['screening_sha256']!=sha256(PARENT/'screening.json'):
        raise ValueError('parent proof changed')
    for p,h in m['sources_sha256'].items():
        if sha256(Path(p))!=h:raise ValueError('source changed')
    with np.load(PARENT/'interface.npz') as z:
        members={str(h):int(s) for h,s,f in zip(z['hadm_ids'],z['subject_ids'],z['patient_splits']) if f=='train'}
    paths={key:next(Path(p) for p in m['sources_sha256'] if p.endswith(suffix)) for key,suffix in
           (('cohort','/cohort/admissions.csv.gz'),('raw','/raw/admissions.csv.gz'))}
    data={}
    for name,path in paths.items():
        found={}
        for row in rows(path):
            h=row['hadm_id']
            if h not in members:continue
            if h in found or int(row['subject_id'])!=members[h]:raise ValueError('training identity conflict')
            found[h]=row
        if set(found)!=set(members):raise ValueError('training source missing')
        data[name]=found
    targets=[]; conflicts=Counter();categories=Counter()
    for h,cohort in data['cohort'].items():
        raw=data['raw'][h];categories[cohort['discharge_location']]+=1
        for key in ('subject_id','admittime','dischtime','discharge_location','hospital_expire_flag'):
            if cohort[key]!=raw[key]:conflicts[key]+=1
        if cohort['discharge_landmark_eligible']!='t' or not eligibility(raw):conflicts['eligibility']+=1
        if cohort['discharge_location']=='DIED':
            targets.append(dict(hadm_id=h,subject_id=members[h],
                cohort_eligible=cohort['discharge_landmark_eligible']=='t',
                hospital_expire_flag=int(raw['hospital_expire_flag']),
                positive_hospital_duration=datetime.fromisoformat(raw['dischtime'])>datetime.fromisoformat(raw['admittime']),
                deathtime_role=time_role(raw.get('deathtime'),raw)))
    if len(targets)!=2 or len(members)!=13592:raise ValueError('prior training audit differs')
    result=dict(status='train_only_hospital_source_qc',training_admissions=len(members),
        source_conflicts=dict(conflicts),died_records=len(targets),
        died_expire_flags=dict(Counter(str(r['hospital_expire_flag']) for r in targets)),
        died_deathtime_roles=dict(Counter(r['deathtime_role'] for r in targets)),
        all_died_match_original_eligibility=all(r['cohort_eligible'] and r['hospital_expire_flag']==0 and r['positive_hospital_duration'] for r in targets),
        raw_categories=dict(categories),safe_to_research_non_death_categories=not bool(conflicts),
        died_new_input_policy='unusable_source_mask_false_original_risk_unchanged_not_removed',
        no_downstream_outcomes=True,no_new_fits=True,no_new_vectors=True,
        source_sha256={**{str(p):sha256(p) for p in paths.values()},str(SQL):sha256(SQL),str(PROTOCOL):sha256(PROTOCOL),
            str(PARENT/'interface.npz'):sha256(PARENT/'interface.npz'),str(Path(__file__)):sha256(Path(__file__))})
    return result,targets

def run():
    if ROOT.exists():raise FileExistsError('audit exists; do not overwrite')
    result,targets=derive();ROOT.mkdir()
    (ROOT/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    # 身份只在本地私有QC底账；不打印原标识或日期。
    (ROOT/'local_qc_records.json').write_text(json.dumps(targets,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','source_conflicts','died_records','died_expire_flags','died_deathtime_roles','all_died_match_original_eligibility','safe_to_research_non_death_categories')}),flush=True)

if __name__=='__main__':run()
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/eos_med_context.py · lines 1–243

<!-- REVIEW_FRAGMENT {"path":"patient_data/eos_med_context.py","part":1,"start_line":1,"end_line":243,"source_sha256":"58803c3ddac466af653225e53f6c80564e3da42835b3aa2a1d26602873da602f","fragment_sha256":"58803c3ddac466af653225e53f6c80564e3da42835b3aa2a1d26602873da602f","added_terminal_newline":false} -->
````python
"""源可用性与EOS/已记录激素时间上下文；不训练，不覆盖原状态。"""
from __future__ import annotations
import argparse
import gzip
from collections import defaultdict, Counter
from datetime import datetime
import json
import math
from pathlib import Path
import time
import numpy as np
from patient_data.export_mimiciv import rows, sha256
from patient_data.preprocess_events import ADMINISTERED, NON_SYSTEMIC_ROUTES
from training.v5.numeric_preprocessing import robust_parameters
from training.v5.clinical_concepts import medication_classes

EOS_ITEMS={'51199':'absolute','52073':'absolute','51200':'percent'}
EOS_NAMES=('absolute','percent')
VALUE_NAMES=('first','minimum','maximum','latest','first_before_recorded_steroid','latest_age_log_days')
RELATIONS=('before_recorded_steroid','after_recorded_steroid','same_time_order_unknown','steroid_timing_unknown')
# 途径是上游判定，不作为网络输入。未知不猜测为全身，也不称未给药。
SYSTEMIC_ROUTES={'PO','ORAL','IV','IV DRIP','IV PUSH','IM','INTRAMUSCULAR','INTRAVENOUS','NG','NGT','OG','PO/NG','PO/NGT','PO/OG','G TUBE','GT','PEG','J TUBE','JT','PER G TUBE','PER J TUBE','ENTERAL'}


def finite(value):
    try:
        v=float(value)
        return v if math.isfinite(v) else None
    except (ValueError,TypeError): return None


def canonical_eos(value, unit, kind):
    v=finite(value)
    if v is None or v<0: return None
    unit=unit.strip().replace('µ','u').replace('μ','u')
    if kind=='absolute':
        if unit=='K/uL': return v*1000
        if unit in ('#/uL','cells/uL'): return v
        return None
    if kind=='percent' and unit=='%' and v<=100: return v
    return None


def relation(draw, confirmed, ambiguous=()):
    if confirmed is None: return 3
    if draw==confirmed: return 2
    if draw>confirmed: return 1
    if any(t<=draw for t in ambiguous): return 3
    return 0


def summarize_eos(samples, discharge, confirmed, ambiguous=()):
    values=np.zeros(6,dtype=np.float64); masks=np.zeros(6,dtype=bool); context=np.zeros(8)
    if not samples: return values,masks,context
    by_time=defaultdict(list)
    for t,v in samples:
        if t>discharge or not math.isfinite(v) or v<0: raise ValueError('invalid EOS observation')
        by_time[t].append(v)
    ordered=[(t,float(np.median(v))) for t,v in sorted(by_time.items())]
    first,latest=ordered[0],ordered[-1]
    values[:4]=[first[1],min(v for _,v in ordered),max(v for _,v in ordered),latest[1]]
    masks[:4]=True
    pre=[(t,v) for t,v in ordered if relation(t,confirmed,ambiguous)==0]
    if pre: values[4]=pre[0][1]; masks[4]=True
    values[5]=math.log1p((discharge-latest[0]).total_seconds()/86400); masks[5]=True
    context[relation(first[0],confirmed,ambiguous)]=1
    context[4+relation(latest[0],confirmed,ambiguous)]=1
    return values,masks,context


def build(raw_root, state_dir, processed, output):
    if output.exists(): raise FileExistsError(output)
    started=time.perf_counter()
    sm=json.loads((raw_root/'metadata/manifest.json').read_text(encoding='utf-8'))
    st=json.loads((state_dir/'manifest.json').read_text(encoding='utf-8'))
    pm=json.loads((processed/'manifest.json').read_text(encoding='utf-8'))
    if sm['status']!='validated' or st['status']!='derived_and_roundtrip_checked_not_prediction_validated' or pm['medication_mapping']!='clinical_reviewed_2026_09':
        raise ValueError('unvalidated source/version')
    if sha256(state_dir/'clinical_state.npz')!=st['output_sha256']['clinical_state.npz']:
        raise ValueError('state changed')
    inputs={}
    for name in ('raw/labevents','raw/emar','raw/emar_detail','raw/prescriptions','cohort/admissions','cohort/patient_splits'):
        digest=sha256(raw_root/(name+'.csv.gz'))
        if digest!=sm['sources'][name]['sha256']: raise ValueError('raw source changed')
        inputs[name]=digest
    event_file=processed/'events.csv.gz'
    if sha256(event_file)!=pm['output_sha256']['events.csv.gz']: raise ValueError('accepted medication source changed')
    inputs['accepted_events']=sha256(event_file)
    with np.load(state_dir/'clinical_state.npz',allow_pickle=False) as archive:
        state={k:archive[k] for k in archive.files}
    hadms=[str(h) for h in state['hadm_ids']]; selected=set(hadms)
    admissions={r['hadm_id']:(datetime.fromisoformat(r['admittime']),datetime.fromisoformat(r['dischtime']))
                for r in rows(raw_root/'cohort/admissions.csv.gz') if r['hadm_id'] in selected}
    if len(hadms)!=19401 or len(admissions)!=19401: raise ValueError('member drift')
    accepted={}
    for r in rows(event_file):
        if r['kind']=='medication' and r['concept']=='systemic_steroid':
            key=r['source_id']
            if key in accepted: raise ValueError('duplicate accepted steroid source')
            accepted[key]=(r['hadm_id'],datetime.fromisoformat(r['charttime']))
    raw_any, available_any=set(),set()
    steroid_records={}
    dose_info=defaultdict(list); routes=defaultdict(set)
    audit=Counter(); ambiguous=defaultdict(list); confirmed=defaultdict(list)
    raw_count=0
    for r in rows(raw_root/'raw/emar.csv.gz'):
        raw_count+=1
        h=r['hadm_id']
        if h not in selected: continue
        raw_any.add(h)
        start,end=admissions[h]
        try:
            t,s=datetime.fromisoformat(r['charttime']),datetime.fromisoformat(r['storetime'])
            if start<=t<=s<=end: available_any.add(h)
        except ValueError: pass
        key=r['emar_id']+':'+r['emar_seq']
        if key in accepted:
            if key in steroid_records: raise ValueError('duplicate eMAR key')
            steroid_records[key]=r
    if set(steroid_records)!=set(accepted): raise ValueError('missing steroid source')
    if raw_count!=sm['sources']['raw/emar']['rows']: raise ValueError('raw eMAR count changed')
    for r in rows(raw_root/'raw/emar_detail.csv.gz'):
        key=r['emar_id']+':'+r['emar_seq']
        if key not in accepted: continue
        dose_info[key].append((finite(r['dose_given']),r['dose_given_unit'],finite(r['product_amount_given'])))
        if r['route'] not in ('','\\N'): routes[key].add(r['route'].strip().upper())
    for key,(h,t) in accepted.items():
        r=steroid_records[key]
        if r['hadm_id']!=h or datetime.fromisoformat(r['charttime'])!=t or r['event_txt'] not in ADMINISTERED:
            raise ValueError('accepted steroid timing/status mismatch')
        start,end=admissions[h]
        stored=datetime.fromisoformat(r['storetime'])
        if not start<=t<=stored<=end: raise ValueError('steroid not available by discharge')
        if routes[key] and routes[key].issubset(NON_SYSTEMIC_ROUTES):
            audit['non_systemic_routes_excluded']+=1; continue
        doses=[v for v,unit,_ in dose_info[key] if v is not None and unit not in ('','\\N')]
        if doses and not any(v>0 for v in doses):
            audit['explicit_zero_or_negative_dose_excluded']+=1; continue
        positive=any(v>0 for v in doses)
        route_known=bool(routes[key]) and routes[key].issubset(SYSTEMIC_ROUTES)
        if positive and route_known:
            confirmed[h].append(t); audit['positive_dose_known_systemic_route_records']+=1
        else:
            ambiguous[h].append(t)
            audit['dose_or_route_not_confirmed_records']+=1
            for route in routes[key]: audit['ambiguous_route:'+route]+=1
    first={h:min(ts) for h,ts in confirmed.items()}
    prescriptions=set()
    for r in rows(raw_root/'raw/prescriptions.csv.gz'):
        h=r['hadm_id']
        if h not in selected or 'systemic_steroid' not in medication_classes(r['drug'],mapping_version='clinical_reviewed_2026_09'): continue
        try:
            t=datetime.fromisoformat(r['starttime'])
            if admissions[h][0]<=t<=admissions[h][1]: prescriptions.add(h)
        except ValueError: pass
    obs=defaultdict(list); eos_audit=defaultdict(Counter); count=0
    for r in rows(raw_root/'raw/labevents.csv.gz'):
        count+=1
        h,item=r['hadm_id'],r['itemid']
        if h not in selected or item not in EOS_ITEMS: continue
        kind=EOS_ITEMS[item]; a=eos_audit[kind]; a['candidate_rows']+=1
        try:
            t,s=datetime.fromisoformat(r['charttime']),datetime.fromisoformat(r['storetime'])
            if not admissions[h][0]<=t<=s<=admissions[h][1]:
                a['time_unavailable']+=1; continue
        except ValueError:
            a['invalid_timestamp']+=1; continue
        value=canonical_eos(r['valuenum'],r['valueuom'],kind)
        if value is None:
            a['invalid_value_or_unit']+=1; continue
        obs[h,kind].append((t,value)); a['accepted_rows']+=1
        a['unit:'+r['valueuom']]+=1
    if count!=sm['sources']['raw/labevents']['rows']: raise ValueError('raw lab count changed')
    values=np.zeros((len(hadms),2,6)); masks=np.zeros_like(values,dtype=bool); context=np.zeros((len(hadms),2,8))
    med_mask=np.zeros((len(hadms),7),dtype=bool); availability=np.zeros(len(hadms),dtype=np.int8)
    evidence=[]; precoverage=Counter(); timing=defaultdict(Counter)
    for i,h in enumerate(hadms):
        med_mask[i]=h in available_any
        target=bool(state['med_exposure'][i].any())
        if target and h not in available_any: raise ValueError('exposure without valid eMAR source')
        availability[i]=(3 if target else 2 if h in available_any else 1 if h in raw_any else 0)
        for k,kind in enumerate(EOS_NAMES):
            samples=obs.get((h,kind),[])
            values[i,k],masks[i,k],context[i,k]=summarize_eos(samples,admissions[h][1],first.get(h),ambiguous.get(h,[]))
            if masks[i,k,0]:
                eos_audit[kind]['observed_admissions']+=1
                eos_audit[kind]['admissions_with_distinct_repeat_times']+=int(len({t for t,_ in samples})>=2)
                eos_audit[kind]['same_timestamp_rows_collapsed']+=len(samples)-len({t for t,_ in samples})
                timing[kind][RELATIONS[int(context[i,k,:4].argmax())]]+=1
            precoverage[kind]+=int(masks[i,k,4])
        evidence.append({'hadm_id':h,'raw_emar_present':int(h in raw_any),'available_emar_present':int(h in available_any),
                         'target_exposure_present':int(target),'first_positive_systemic_steroid_charttime':first[h].isoformat() if h in first else '',
                         'ambiguous_recorded_steroid_present':int(bool(ambiguous.get(h))),
                         'steroid_prescription_present':int(h in prescriptions),'prescription_only':int(h in prescriptions and h not in first and not ambiguous.get(h))})
    scaled=values.copy(); norm={}
    transformed=values.copy(); transformed[:,:,:5]=np.log1p(transformed[:,:,:5])
    train=state['patient_splits']=='train'
    for k,kind in enumerate(EOS_NAMES):
        for f,name in enumerate(VALUE_NAMES):
            valid=train&masks[:,k,f]
            if not valid.any():
                masks[:,k,f]=False; scaled[:,k,f]=0; norm[kind+':'+name]={'method':'no_training_observation'}; continue
            p=robust_parameters(transformed[valid,k,f]); norm[kind+':'+name]=p
            scaled[:,k,f]=np.where(masks[:,k,f],np.clip((transformed[:,k,f]-p['median'])/p['scale'],-8,8),0)
    features=np.concatenate((scaled,masks.astype(float),context),axis=2).astype(np.float32)
    for name,digest in inputs.items():
        path=event_file if name=='accepted_events' else raw_root/(name+'.csv.gz')
        if sha256(path)!=digest: raise ValueError('source changed during build')
    output.mkdir(parents=True)
    np.savez_compressed(output/'eos_med_context.npz',hadm_ids=state['hadm_ids'],subject_ids=state['subject_ids'],patient_splits=state['patient_splits'],
                        eos_values_raw=values,eos_values_scaled=scaled,eos_value_masks=masks,eos_relation_flags=context,eos_features=features,
                        eos_names=np.asarray(EOS_NAMES),med_exposure=state['med_exposure'],med_source_mask=med_mask,med_availability_state=availability)
    with gzip.open(output/'steroid_source_audit.csv.gz','wt',encoding='utf-8',newline='') as out:
        import csv
        writer=csv.DictWriter(out,fieldnames=list(evidence[0])); writer.writeheader(); writer.writerows(evidence)
    (output/'normalization.json').write_text(json.dumps(norm,indent=2),encoding='utf-8')
    with np.load(output/'eos_med_context.npz',allow_pickle=False) as archive:
        restored={k:archive[k] for k in archive.files}
    if not np.array_equal(restored['eos_features'],features) or not np.isfinite(features).all(): raise ValueError('roundtrip failed')
    report={'status':'eos_med_context_validated_not_trained','admissions':len(hadms),'input_sha256':inputs,
            'parent_clinical_state_manifest_sha256':sha256(state_dir/'manifest.json'),'code_sha256':sha256(Path(__file__)),
            'output_sha256':{n:sha256(output/n) for n in ('eos_med_context.npz','steroid_source_audit.csv.gz','normalization.json')},
            'med_availability_codes':{'0':'no raw eMAR','1':'raw eMAR but none available by discharge','2':'available eMAR without retained seven-class exposure','3':'retained target exposure'},
            'med_availability_counts':{str(i):int((availability==i).sum()) for i in range(4)},
            'eos_audit':{k:dict(v) for k,v in eos_audit.items()},'first_eos_relation_counts':{k:dict(v) for k,v in timing.items()},
            'eos_observed_patients':{kind:len(set(state['subject_ids'][masks[:,k,0]].tolist())) for k,kind in enumerate(EOS_NAMES)},
            'pre_recorded_steroid_value_coverage':dict(precoverage),'steroid_audit':dict(audit),
            'positive_systemic_steroid_admissions':len(first),'only_prescription_admissions':sum(r['prescription_only'] for r in evidence),
            'eos_feature_shape':list(features.shape),'value_names':VALUE_NAMES,'relation_names':RELATIONS,'elapsed_seconds':round(time.perf_counter()-started,2),
            'limitations':['eMAR present is partial source visibility, not proof of complete medication observation',
                           'Before recorded steroid does not prove no steroid in ED, outside hospital or before admission',
                           'First/last relation concerns confirmed positive dose and known systemic route; uncertain earlier records block pre designation',
                           'Prescriptions never establish actual administration; EOS zero is a valid observed value',
                           'EOS is not a monotonic recovery severity axis; values retained separately, no claim of treatment efficacy']}
    (output/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':report['status'],'availability':report['med_availability_counts'],'seconds':report['elapsed_seconds']}),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('raw_root','state_dir','processed','output'): parser.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    args=parser.parse_args(); build(args.raw_root,args.state_dir,args.processed,args.output)
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/expanded_abnormality_features.py · lines 1–175

<!-- REVIEW_FRAGMENT {"path":"patient_data/expanded_abnormality_features.py","part":1,"start_line":1,"end_line":175,"source_sha256":"7eaf8ed5ca73debfb99abd6c1dddbb97831d48dbad93177061e534c4895befba","fragment_sha256":"7eaf8ed5ca73debfb99abd6c1dddbb97831d48dbad93177061e534c4895befba","added_terminal_newline":false} -->
````python
"""从出院前真实检验构建方向明确的异常轨迹；不输入原始事件序列。"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from patient_data.lab_recovery_features import AXES as ORIGINAL_AXES
from patient_data.lab_recovery_features import FEATURE_NAMES as ORIGINAL_NAMES
from patient_data.lab_recovery_features import SUMMARY_NAMES, summarize_axis


# 同一项目的双向异常共用原始观测；不把相关项目（如 Hb/Hct）都加进来。
ADDED_AXES: dict[str, tuple[int, str, str]] = {
    "wbc_low": (51301, "low", "K/uL"),
    "sodium_low": (50983, "low", "mEq/L"),
    "sodium_high": (50983, "high", "mEq/L"),
    "potassium_low": (50971, "low", "mEq/L"),
    "potassium_high": (50971, "high", "mEq/L"),
    "urea_nitrogen_high": (51006, "high", "mg/dL"),
    "glucose_low": (50931, "low", "mg/dL"),
    "glucose_high": (50931, "high", "mg/dL"),
    "platelet_low": (51265, "low", "K/uL"),
    "magnesium_low": (50960, "low", "mg/dL"),
    "lactate_high": (50813, "high", "mmol/L"),
    "crp_high": (50889, "high", "mg/L"),
    "ntprobnp_high": (50963, "high", "pg/mL"),
}
AXES = ORIGINAL_AXES + tuple(ADDED_AXES)
FEATURE_NAMES = ORIGINAL_NAMES + tuple(
    f"{axis}__{name}" for axis in ADDED_AXES for name in SUMMARY_NAMES
)


def _rows(path: Path):
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(value: str) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def directional_severity(value: float, lower: float, upper: float, direction: str,
                         *, severity_cap: float | None = 5.0) -> float:
    if direction not in ("low", "high") or not all(map(math.isfinite, (value, lower, upper))) or lower >= upper:
        raise ValueError("invalid result/reference range")
    offset = lower - value if direction == "low" else value - upper
    departure = max(0.0, offset / (upper - lower))
    return min(severity_cap, departure) if severity_cap is not None else departure


def build(original: Path, raw: Path, output: Path,
          *, severity_cap: float | None = 5.0) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    old_manifest = json.loads((original / "manifest.json").read_text(encoding="utf-8"))
    if old_manifest.get("severity_cap", 5.0) != severity_cap:
        raise ValueError("original and added abnormality axes must use the same severity cap")
    raw_manifest = json.loads((raw / "metadata/manifest.json").read_text(encoding="utf-8"))
    old_file = original / "lab_recovery.npz"
    raw_file = raw / "raw/labevents.csv.gz"
    if (old_manifest["status"] != "derived_not_yet_validated_by_prediction"
            or raw_manifest["status"] != "validated"
            or _sha(old_file) != old_manifest["output_sha256"]
            or _sha(raw_file) != raw_manifest["sources"]["raw/labevents"]["sha256"]):
        raise ValueError("unvalidated abnormality source")
    with np.load(old_file, allow_pickle=False) as archive:
        hadms = archive["hadm_ids"]
        old_features = archive["features"]
        if tuple(archive["feature_names"].tolist()) != ORIGINAL_NAMES:
            raise ValueError("original abnormality vocabulary changed")
    if old_features.shape != (len(hadms), len(ORIGINAL_NAMES)) or len(set(hadms.tolist())) != len(hadms):
        raise ValueError("original abnormality shape/identity changed")
    selected_hadms = {str(hadm) for hadm in hadms}
    landmarks = {}
    for row in _rows(raw / "cohort/admissions.csv.gz"):
        if row["hadm_id"] in selected_hadms:
            landmarks[row["hadm_id"]] = (datetime.fromisoformat(row["admittime"]),
                                           datetime.fromisoformat(row["dischtime"]))
    if set(landmarks) != selected_hadms:
        raise ValueError("abnormality admissions differ from raw cohort")
    by_item: dict[int, list[tuple[str, str, str]]] = defaultdict(list)
    for axis, (itemid, direction, unit) in ADDED_AXES.items():
        by_item[itemid].append((axis, direction, unit))
    observations: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    audit: dict[str, Counter[str]] = defaultdict(Counter)
    for row in _rows(raw_file):
        itemid = int(row["itemid"])
        if itemid not in by_item or row["hadm_id"] not in landmarks:
            continue
        admission, discharge = landmarks[row["hadm_id"]]
        chart = datetime.fromisoformat(row["charttime"])
        stored = datetime.fromisoformat(row["storetime"]) if row["storetime"] not in ("", "\\N") else None
        value, lower, upper = (_number(row[key]) for key in ("valuenum", "ref_range_lower", "ref_range_upper"))
        for axis, direction, unit in by_item[itemid]:
            counts = audit[axis]
            counts["raw_rows"] += 1
            if not admission <= chart <= discharge or stored is None or not chart <= stored <= discharge:
                counts["not_available_by_discharge"] += 1
                continue
            if value is None or lower is None or upper is None or lower >= upper:
                counts["invalid_numeric_or_reference"] += 1
                continue
            if row["valueuom"] != unit:
                counts["unit_mismatch"] += 1
                continue
            day = (chart - admission).total_seconds() / 86400.0
            observations[(row["hadm_id"], axis)].append(
                (day, directional_severity(value, lower, upper, direction,
                                           severity_cap=severity_cap)))
            counts["accepted_rows"] += 1
    added = np.zeros((len(hadms), len(ADDED_AXES) * len(SUMMARY_NAMES)), dtype=np.float32)
    for index, hadm in enumerate(hadms):
        for axis_index, axis in enumerate(ADDED_AXES):
            summary = summarize_axis(observations.get((str(hadm), axis), []))
            lo = axis_index * len(SUMMARY_NAMES)
            added[index, lo:lo + len(SUMMARY_NAMES)] = summary
            audit[axis]["observed_admissions"] += bool(summary[0])
            audit[axis]["abnormal_admissions"] += summary[2] > 0
            audit[axis]["observed_recovery_admissions"] += summary[4] > 0
    features = np.concatenate((old_features, added), axis=1)
    if not np.isfinite(features).all() or (features < 0).any():
        raise ValueError("invalid derived abnormality values")
    output.mkdir(parents=True)
    np.savez_compressed(output / "abnormality.npz", hadm_ids=hadms, features=features,
                        feature_names=np.asarray(FEATURE_NAMES))
    report = {
        "status": "derived_not_yet_validated_by_prediction", "admissions": len(hadms),
        "severity_cap": severity_cap,
        "axes": AXES, "feature_count": len(FEATURE_NAMES),
        "added_axis_audit": {axis: dict(audit[axis]) for axis in ADDED_AXES},
        "input_sha256": {"original_abnormality": _sha(old_file), "raw_labevents": _sha(raw_file),
                         "processed_arrays": old_manifest["input_sha256"]["processed_arrays"]},
        "output_sha256": _sha(output / "abnormality.npz"),
        "limitations": ["Absent test is not normal", "Recovery requires later measured result",
                        "Reference-range departure is not COPD-specific or a treatment effect",
                        "Only results stored by discharge are eligible"],
    }
    (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True, type=Path)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--uncapped", action="store_true", help="retain raw reference-range departure before log1p")
    args = parser.parse_args()
    result = build(args.original, args.raw, args.output,
                   severity_cap=None if args.uncapped else 5.0)
    print(json.dumps({"admissions": result["admissions"], "feature_count": result["feature_count"],
                      "added_axis_audit": result["added_axis_audit"]}, ensure_ascii=False))
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/export_mimiciv.py · lines 1–206

<!-- REVIEW_FRAGMENT {"path":"patient_data/export_mimiciv.py","part":1,"start_line":1,"end_line":206,"source_sha256":"a3824c1f12bfe3da98b8931734dcf70547f8eb18e73299969c8fca518cf489cb","fragment_sha256":"a3824c1f12bfe3da98b8931734dcf70547f8eb18e73299969c8fca518cf489cb","added_terminal_newline":false} -->
````python
"""从 PostgreSQL 同一只读快照导出既定 COPD 住院的原始底账。"""

from __future__ import annotations

import argparse
import csv
import getpass
import gzip
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


COHORT = "project_aecopd.copd_coded_admissions_v1_gate"
LABELS = "project_aecopd.copd_qualified_readmission_labels_v1"
NULL = "\\N"


@dataclass(frozen=True)
class Source:
    name: str
    table: str
    select: str
    primary_key: tuple[str, ...] = ()


def sources() -> tuple[Source, ...]:
    """源表不作数值清洗、特征合并或文本编码。"""
    selected = f"SELECT hadm_id FROM {COHORT}"
    patients = f"SELECT DISTINCT subject_id FROM {COHORT}"
    result = [
        Source("cohort/admissions", COHORT, f"SELECT * FROM {COHORT} ORDER BY subject_id,admittime,hadm_id", ("hadm_id",)),
        Source("raw/patients", "mimiciv_hosp.patients", f"SELECT * FROM mimiciv_hosp.patients WHERE subject_id IN ({patients}) ORDER BY subject_id", ("subject_id",)),
    ]
    for table, key in (
        ("admissions", ("hadm_id",)), ("labevents", ("labevent_id",)),
        ("diagnoses_icd", ()), ("emar", ("emar_id", "emar_seq")),
        ("prescriptions", ()), ("pharmacy", ()),
    ):
        result.append(Source(f"raw/{table}", f"mimiciv_hosp.{table}",
                             f"SELECT * FROM mimiciv_hosp.{table} WHERE hadm_id IN ({selected})", key))
    result.append(Source("raw/emar_detail", "mimiciv_hosp.emar_detail",
                         f"SELECT d.* FROM mimiciv_hosp.emar_detail d WHERE EXISTS (SELECT 1 FROM mimiciv_hosp.emar e WHERE e.emar_id=d.emar_id AND e.emar_seq=d.emar_seq AND e.hadm_id IN ({selected}))"))
    for table in ("discharge", "radiology"):
        result.append(Source(f"raw/{table}", f"mimiciv_note.{table}",
                             f"SELECT * FROM mimiciv_note.{table} WHERE hadm_id IN ({selected})", ("note_id",)))
    result.append(Source("raw/labevents_unassigned", "mimiciv_hosp.labevents",
                         f"SELECT x.* FROM mimiciv_hosp.labevents x WHERE x.hadm_id IS NULL AND x.subject_id IN ({patients}) AND EXISTS (SELECT 1 FROM {COHORT} c WHERE c.subject_id=x.subject_id AND x.charttime BETWEEN c.admittime AND c.dischtime)", ("labevent_id",)))
    for table in ("d_labitems", "d_icd_diagnoses"):
        result.append(Source(f"raw/{table}", f"mimiciv_hosp.{table}",
                             f"SELECT * FROM mimiciv_hosp.{table}"))
    result.append(Source("outcomes/readmission_death_labels", LABELS,
                         f"SELECT * FROM {LABELS} ORDER BY subject_id,admittime,hadm_id", ("hadm_id",)))
    tables = sorted({s.table for s in result})
    conditions = " OR ".join(f"(table_schema='{t.split('.')[0]}' AND table_name='{t.split('.')[1]}')" for t in tables)
    result.append(Source("metadata/columns", "information_schema.columns",
                         "SELECT table_schema,table_name,column_name,ordinal_position,data_type,udt_name,is_nullable FROM information_schema.columns WHERE " + conditions + " ORDER BY table_schema,table_name,ordinal_position"))
    return tuple(result)


def rows(path: Path) -> Iterator[dict[str, str]]:
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def patient_split(subject: str) -> str:
    bucket = int(hashlib.md5(f"aecopd-model-v1:{subject}".encode("ascii")).hexdigest()[:8], 16) % 100
    return "train" if bucket < 70 else "validation" if bucket < 85 else "test"


def validate(root: Path) -> dict[str, object]:
    """独立读取落盘文件检查成员、主键和关联；不输出患者正文。"""
    cohort = {r["hadm_id"]: r for r in rows(root / "cohort/admissions.csv.gz")}
    eligible = {h for h, r in cohort.items() if r["discharge_landmark_eligible"] == "t"}
    subjects = {r["subject_id"] for r in cohort.values()}
    if (len(cohort), len(subjects), len(eligible)) != (19774, 5390, 19401):
        raise ValueError("既定队列数量不一致，停止导出验证")
    actual_patients = {r["subject_id"] for r in rows(root / "raw/patients.csv.gz")}
    actual_admissions = {r["hadm_id"] for r in rows(root / "raw/admissions.csv.gz")}
    actual_labels = {r["hadm_id"] for r in rows(root / "outcomes/readmission_death_labels.csv.gz")}
    if actual_patients != subjects or actual_admissions != set(cohort) or actual_labels != eligible:
        raise ValueError("患者、住院或结局成员不匹配")
    emar_keys = {(r["emar_id"], r["emar_seq"]) for r in rows(root / "raw/emar.csv.gz")}
    counts: dict[str, int] = {}
    for source in sources():
        seen: set[tuple[str, ...]] = set()
        count = 0
        for row in rows(root / (source.name + ".csv.gz")):
            count += 1
            if source.primary_key:
                key = tuple(row[k] for k in source.primary_key)
                if key in seen or NULL in key:
                    raise ValueError(f"{source.name}: 原始记录主键重复或为空")
                seen.add(key)
            if source.name == "raw/emar_detail" and (row["emar_id"], row["emar_seq"]) not in emar_keys:
                raise ValueError("给药详情缺少原始给药关联")
            if source.name.startswith("raw/") and "subject_id" in row:
                if row["subject_id"] not in subjects:
                    raise ValueError(f"{source.name}: 患者超出范围")
                hadm = row.get("hadm_id", NULL)
                if hadm != NULL and (hadm not in cohort or cohort[hadm]["subject_id"] != row["subject_id"]):
                    raise ValueError(f"{source.name}: 住院与患者关联不一致")
        counts[source.name] = count
    split_path = root / "cohort/patient_splits.csv.gz"
    with gzip.open(split_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("subject_id", "patient_split", "has_eligible_discharge"))
        eligible_subjects = {cohort[h]["subject_id"] for h in eligible}
        for subject in sorted(subjects, key=int):
            writer.writerow((subject, patient_split(subject), subject in eligible_subjects))
    return {"status": "validated", "patients": len(subjects), "coded_admissions": len(cohort),
            "eligible_discharges": len(eligible), "eligible_patients": len(eligible_subjects),
            "non_landmark_admissions": len(cohort) - len(eligible), "rows": counts}


def export(root: Path, psql: Path, password: str) -> dict[str, object]:
    if root.exists():
        raise FileExistsError("拒绝覆盖已有患者数据目录")
    if not psql.is_file():
        raise FileNotFoundError("找不到 PostgreSQL 导出客户端")
    root.mkdir(parents=True)
    (root / "metadata").mkdir()
    (root / "metadata/status.json").write_text(json.dumps({"status": "exporting"}), encoding="utf-8")
    env = os.environ.copy()
    env["PGPASSWORD"] = password
    env["PGOPTIONS"] = "-c default_transaction_read_only=on -c statement_timeout=600000"
    command = [str(psql), "-X", "-q", "-w", "-h", "localhost", "-U", "postgres",
               "-d", "mimiciv", "-v", "ON_ERROR_STOP=1"]
    keeper = subprocess.Popen(command + ["-A", "-t"], stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, env=env)
    assert keeper.stdin is not None and keeper.stdout is not None
    keeper.stdin.write("BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY;\nSELECT pg_export_snapshot();\n")
    keeper.stdin.flush()
    snapshot = keeper.stdout.readline().strip()
    if not snapshot or any(c not in "0123456789ABCDEFabcdef-" for c in snapshot):
        keeper.terminate()
        raise RuntimeError("无法建立一致的只读导出快照")
    manifest: dict[str, object] = {"status": "exporting", "database": "mimiciv", "cohort_source": COHORT,
                                  "snapshot": snapshot, "utc_started": datetime.now(timezone.utc).isoformat(),
                                  "format": "UTF-8 PostgreSQL COPY CSV gzip", "null_token": NULL,
                                  "sources": {}}
    try:
        for source in sources():
            target = root / (source.name + ".csv.gz")
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(target.suffix + ".partial")
            copy_sql = f"COPY ({source.select}) TO STDOUT WITH (FORMAT CSV, HEADER TRUE, NULL '\\N', ENCODING 'UTF8');"
            sql = f"BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY; SET TRANSACTION SNAPSHOT '{snapshot}'; {copy_sql} COMMIT;"
            # 持久化查询不包含导出结束后失效的事务快照 ID。
            (root / "metadata" / (source.name.replace("/", "_") + ".sql")).write_text(copy_sql, encoding="utf-8")
            with gzip.open(partial, "wb", compresslevel=3) as destination:
                process = subprocess.Popen(command + ["-c", sql], stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE, env=env)
                assert process.stdout is not None
                for block in iter(lambda: process.stdout.read(1024 * 1024), b""):
                    destination.write(block)
                error = process.stderr.read() if process.stderr is not None else b""
                if process.wait():
                    raise RuntimeError(f"{source.name} 导出失败: {error.decode('utf-8', errors='replace')}")
            partial.replace(target)
            manifest["sources"][source.name] = {"source_table": source.table, "sha256": sha256(target),
                                                  "compressed_bytes": target.stat().st_size}
            print(json.dumps({"exported": source.name, "compressed_bytes": target.stat().st_size}), flush=True)
        checked = validate(root)
        (root / "metadata/validation.json").write_text(json.dumps(checked, indent=2), encoding="utf-8")
        for name, count in checked["rows"].items():
            manifest["sources"][name]["rows"] = count
        manifest["sources"]["cohort/patient_splits"] = {"rows": checked["patients"], "sha256": sha256(root / "cohort/patient_splits.csv.gz")}
        manifest.update(status="validated", utc_finished=datetime.now(timezone.utc).isoformat())
        (root / "metadata/manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (root / "metadata/status.json").write_text(json.dumps({"status": "validated"}), encoding="utf-8")
        return checked
    except Exception:
        (root / "metadata/status.json").write_text(json.dumps({"status": "failed_preserved_for_audit"}), encoding="utf-8")
        raise
    finally:
        keeper.communicate("ROLLBACK;\n", timeout=20)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--psql", type=Path, default=Path("C:/Program Files/PostgreSQL/18/bin/psql.exe"))
    args = parser.parse_args()
    password = os.environ.get("PGPASSWORD") or getpass.getpass("PostgreSQL password: ")
    print(json.dumps(export(args.output, args.psql, password), ensure_ascii=False))


if __name__ == "__main__":
    main()
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/extend_lab_candidate_audit.py · lines 1–68

<!-- REVIEW_FRAGMENT {"path":"patient_data/extend_lab_candidate_audit.py","part":1,"start_line":1,"end_line":68,"source_sha256":"26d9f00363e840a581c6cb64b6d7149dad6be234b1b9201fb37447763f94f290","fragment_sha256":"26d9f00363e840a581c6cb64b6d7149dad6be234b1b9201fb37447763f94f290","added_terminal_newline":false} -->
````python
"""在不覆盖原审计的前提下，补充原始检验临床候选和其重复线索。"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from patient_data.audit_feature_candidates import REVIEW_LABS
from patient_data.export_mimiciv import rows


def run(audit_dir: Path, raw_root: Path) -> Path:
    source = audit_dir / "all_raw_lab_items.csv"
    output = audit_dir / "all_raw_lab_items_reviewed.csv"
    if output.exists():
        raise FileExistsError(output)
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or ())
        catalog = list(reader)
    available = {row["hadm_id"]: (row["admittime"], row["dischtime"])
                 for row in rows(raw_root / "cohort/admissions.csv.gz")
                 if row["discharge_landmark_eligible"] == "t"}
    seen_time: set[tuple[str, str, str]] = set()
    seen_source: set[str] = set()
    collisions: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows(raw_root / "raw/labevents.csv.gz"):
        itemid = row["itemid"]
        boundary = available.get(row["hadm_id"])
        if itemid not in REVIEW_LABS or boundary is None:
            continue
        if not boundary[0] <= row["charttime"] <= row["storetime"] <= boundary[1]:
            continue
        time_key = (row["hadm_id"], itemid, row["charttime"])
        collisions[itemid][0] += time_key in seen_time
        collisions[itemid][1] += row["labevent_id"] in seen_source
        seen_time.add(time_key)
        seen_source.add(row["labevent_id"])
    for row in catalog:
        itemid = row["itemid"]
        if itemid in REVIEW_LABS:
            row["review_status"] = "clinical_candidate_not_in_input"
            row["review_reason"] = REVIEW_LABS[itemid]
            row["same_timestamp_collisions_not_proven_duplicates"] = str(collisions[itemid][0])
            row["repeated_source_ids"] = str(collisions[itemid][1])
        elif row["review_status"] == "unreviewed":
            # 未审计不应写成零重复。
            row["same_timestamp_collisions_not_proven_duplicates"] = ""
            row["repeated_source_ids"] = ""
        else:
            # 当前入模概念的重复信息应读取 current_event_concepts.csv。
            row["same_timestamp_collisions_not_proven_duplicates"] = ""
            row["repeated_source_ids"] = ""
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(catalog)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    args = parser.parse_args()
    print(run(args.audit_dir, args.raw_root))
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/lab_recovery_features.py · lines 1–182

<!-- REVIEW_FRAGMENT {"path":"patient_data/lab_recovery_features.py","part":1,"start_line":1,"end_line":182,"source_sha256":"8c930369ea3022214495ec9fe386e74375b5d8f5aa530faacecfe89f0dfa5435","fragment_sha256":"8c930369ea3022214495ec9fe386e74375b5d8f5aa530faacecfe89f0dfa5435","added_terminal_newline":false} -->
````python
"""从已接受检验中提取低容量、方向明确的异常及复查恢复表示。"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


# 只把动脉 pH / pCO2 用于 COPD 高碳酸血症线索；不把静脉氧分压当 PaO2。
# PaO2 的给氧状态不可用，因此本轮不设单项氧合“恢复”阈值。
AXIS_CONCEPT = {
    "arterial_low_ph": "ph:arterial",
    "arterial_high_pco2": "pco2:arterial",
    "bicarbonate_low": "bicarbonate",
    "bicarbonate_high": "bicarbonate",
    "wbc_high": "wbc",
    "albumin_low": "albumin",
    "hemoglobin_low": "hemoglobin",
    "creatinine_high": "creatinine",
}
PAIRED_AXIS = "arterial_hypercapnic_acidosis"
AXES = (*AXIS_CONCEPT, PAIRED_AXIS)
SUMMARY_NAMES = ("observed", "first_severity", "peak_severity", "last_severity",
                 "post_peak_recovery", "observed_span_log_days")
FEATURE_NAMES = tuple(f"{axis}__{name}" for axis in AXES for name in SUMMARY_NAMES)


def abnormal_severity(axis: str, value: float, lower: float, upper: float,
                      *, severity_cap: float | None = 5.0) -> float:
    """按项目参考范围构建无量纲偏离；旧输入默认截断于 5。"""
    if not all(map(math.isfinite, (value, lower, upper))) or not lower < upper:
        raise ValueError("invalid laboratory result or reference interval")
    if axis == "arterial_low_ph":
        severity = (7.35 - value) / .10
    elif axis == "arterial_high_pco2":
        severity = (value - 45.0) / 15.0
    elif axis in AXIS_CONCEPT:
        width = upper - lower
        severity = ((lower - value) if axis.endswith("_low") else (value - upper)) / width
    else:
        raise ValueError(f"unknown abnormality axis: {axis}")
    if severity_cap is not None and severity_cap <= 0:
        raise ValueError("severity cap must be positive")
    return float(min(severity_cap, max(0.0, severity)) if severity_cap is not None
                 else max(0.0, severity))


def summarize_axis(samples: list[tuple[float, float]]) -> tuple[float, ...]:
    """只在峰值后有更晚复查时给恢复分；缺测和同刻记录不能冒充正常。"""
    if not samples:
        return (0., 0., 0., 0., 0., 0.)
    by_time: dict[float, list[float]] = defaultdict(list)
    for day, severity in samples:
        if not math.isfinite(day) or not math.isfinite(severity) or day < 0 or severity < 0:
            raise ValueError("invalid sample")
        by_time[day].append(severity)
    ordered = [(day, float(np.median(values))) for day, values in sorted(by_time.items())]
    peak_index = max(range(len(ordered)), key=lambda i: ordered[i][1])
    peak = ordered[peak_index][1]
    last = ordered[-1][1]
    recovery = max(0., (peak - last) / peak) if peak > 0 and peak_index < len(ordered) - 1 else 0.
    return (1., math.log1p(ordered[0][1]), math.log1p(peak), math.log1p(last),
            float(recovery), math.log1p(ordered[-1][0] - ordered[0][0]))


def _rows(path: Path):
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(processed: Path, raw: Path, output: Path,
          *, severity_cap: float | None = 5.0) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    processed_manifest = json.loads((processed / "manifest.json").read_text(encoding="utf-8"))
    raw_manifest = json.loads((raw / "metadata/manifest.json").read_text(encoding="utf-8"))
    if processed_manifest["status"] != "preprocessed_and_checked" or raw_manifest["status"] != "validated":
        raise ValueError("unvalidated laboratory sources")
    for name in ("events.csv.gz", "event_arrays.npz"):
        if _sha(processed / name) != processed_manifest["output_sha256"][name]:
            raise ValueError(f"processed laboratory source changed: {name}")
    if _sha(raw / "raw/labevents.csv.gz") != raw_manifest["sources"]["raw/labevents"]["sha256"]:
        raise ValueError("raw labevents source changed")
    with np.load(processed / "event_arrays.npz", allow_pickle=False) as archive:
        hadm_ids = archive["hadm_ids"]
    accepted = {}
    for row in _rows(processed / "events.csv.gz"):
        if row["kind"] == "lab":
            key = row["source_id"]
            if key in accepted:
                raise ValueError("repeated accepted laboratory source")
            accepted[key] = (row["hadm_id"], row["concept"], row["specimen_id"],
                             float(row["elapsed_days"]), float(row["value_raw"]))
    samples: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    specimens: dict[tuple[str, str], dict[str, tuple[float, float]]] = defaultdict(dict)
    matched = 0
    for row in _rows(raw / "raw/labevents.csv.gz"):
        record = accepted.get(row["labevent_id"])
        if record is None:
            continue
        hadm, concept, specimen_id, day, value = record
        if hadm != row["hadm_id"] or specimen_id != row["specimen_id"] or value != float(row["valuenum"]):
            raise ValueError("accepted/raw laboratory row mismatch")
        matched += 1
        lower, upper = float(row["ref_range_lower"]), float(row["ref_range_upper"])
        for axis, expected in AXIS_CONCEPT.items():
            if concept == expected:
                samples[(hadm, axis)].append((day, abnormal_severity(
                    axis, value, lower, upper, severity_cap=severity_cap)))
        if concept in ("ph:arterial", "pco2:arterial") and specimen_id not in ("", "\\N"):
            slot = specimens[(hadm, specimen_id)]
            if concept in slot:
                raise ValueError("multiple accepted arterial values in one specimen")
            slot[concept] = (day, abnormal_severity(
                "arterial_low_ph" if concept == "ph:arterial" else "arterial_high_pco2",
                value, lower, upper, severity_cap=severity_cap))
    if matched != len(accepted):
        raise ValueError("not all accepted laboratory sources were found")
    paired = 0
    for (hadm, _), slot in specimens.items():
        if len(slot) != 2:
            continue
        ph_day, ph_severity = slot["ph:arterial"]
        co2_day, co2_severity = slot["pco2:arterial"]
        if abs(ph_day - co2_day) > 1 / 24:
            continue
        samples[(hadm, PAIRED_AXIS)].append((max(ph_day, co2_day), min(ph_severity, co2_severity)))
        paired += 1
    output.mkdir(parents=True)
    features = np.zeros((len(hadm_ids), len(FEATURE_NAMES)), dtype=np.float32)
    coverage: Counter[str] = Counter()
    recovery: Counter[str] = Counter()
    for index, hadm in enumerate(hadm_ids):
        for axis_index, axis in enumerate(AXES):
            summary = summarize_axis(samples.get((str(hadm), axis), []))
            features[index, axis_index * len(SUMMARY_NAMES):(axis_index + 1) * len(SUMMARY_NAMES)] = summary
            coverage[axis] += bool(summary[0])
            recovery[axis] += summary[4] > 0
    np.savez_compressed(output / "lab_recovery.npz", hadm_ids=hadm_ids,
                        features=features, feature_names=np.asarray(FEATURE_NAMES))
    report = {"status": "derived_not_yet_validated_by_prediction", "admissions": len(hadm_ids),
              "severity_cap": severity_cap,
              "accepted_lab_events": matched, "paired_arterial_specimens": paired,
              "feature_count": len(FEATURE_NAMES), "coverage_admissions": dict(coverage),
              "post_peak_recovery_admissions": dict(recovery),
              "input_sha256": {"processed_events": _sha(processed / "events.csv.gz"),
                               "processed_arrays": _sha(processed / "event_arrays.npz"),
                               "raw_labevents": _sha(raw / "raw/labevents.csv.gz")},
              "output_sha256": _sha(output / "lab_recovery.npz"),
              "limitations": ["Post-peak recovery requires a later observed measurement; no observation is not normal.",
                              "Arterial pH and pCO2 are not interchangeable with venous samples.",
                              "PaO2 is excluded because supplemental oxygen/FiO2 is not available in this input.",
                              "Laboratory reference range abnormality is not a diagnosis or treatment effect."]}
    (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", required=True, type=Path)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--uncapped", action="store_true", help="retain raw reference-range departure before log1p")
    args = parser.parse_args()
    print(json.dumps(build(args.processed, args.raw, args.output,
                           severity_cap=None if args.uncapped else 5.0), ensure_ascii=False))
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/local_text_interface.py · lines 1–188

<!-- REVIEW_FRAGMENT {"path":"patient_data/local_text_interface.py","part":1,"start_line":1,"end_line":188,"source_sha256":"48b0b08eeaa3a868e185bf81d155169d54e4bacc86326894219125ff25b59a30","fragment_sha256":"48b0b08eeaa3a868e185bf81d155169d54e4bacc86326894219125ff25b59a30","added_terminal_newline":false} -->
````python
"""同安全原句的已有包含块/局部冻结编码接口，不使用高频轨迹。"""
from collections import Counter,defaultdict
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from patient_data.export_mimiciv import rows,sha256
from patient_data.text_time_coverage import SCREEN,ENCODED,STATE,time_flags,checked_sources
from training.v6.encode_screen_candidates import selected_groups,ROUTES
from training.v6.train import write_json

ROOT=Path('patient_data/processed/v6_local_text_interface_2026-10-13_r2')
PROTOCOL=Path('docs/V6_LOCAL_TEXT_INTERFACE_PROTOCOL_2026-10-13.md')
MODEL=Path('training/model_cache/Bio_ClinicalBERT')
SECTIONS=('hospital_course','discharge_condition','discharge_physical_exam','discharge_instructions','transitional_issues','followup_instructions')
FLAG_KEYS=('explicit_discharge','change_or_residual','plan','historical')


def overlaps(a,b,chunks):
    if not 0<=a<b:raise ValueError('invalid sentence token span')
    result=[]
    for row,start,end in chunks:
        n=max(0,min(b,end)-max(a,start))
        if n:result.append((row,n))
    if sum(n for _,n in result)!=b-a:raise ValueError('sentence tokens missing or multiply covered')
    return result


def index_mapping(mapped):
    """落盘索引与源重建均使用JSON原生list，防止tuple/list假差异。"""
    return [[int(row),int(tokens)] for row,tokens in mapped]


def token_span(offsets,start,end):
    if not 0<=start<end:raise ValueError('invalid joined sentence character span')
    use=[i for i,(a,b) in enumerate(offsets) if a<end and b>start]
    if not use or use!=list(range(use[0],use[-1]+1)):raise ValueError('sentence token positions not contiguous')
    return use[0],use[-1]+1


def weighted_vector(vectors,weights):
    if len(vectors)!=len(weights) or not len(weights) or np.any(np.asarray(weights)<=0):raise ValueError('invalid weights')
    result=np.average(np.asarray(vectors,np.float32),axis=0,weights=np.asarray(weights,np.float32)).astype(np.float32)
    norm=np.linalg.norm(result)
    if not np.isfinite(result).all() or norm<=1e-12:raise ValueError('invalid vector')
    return result/norm


def derive():
    from transformers import AutoTokenizer
    source,_=checked_sources();groups,counts,source_hash=selected_groups(SCREEN)
    m=json.loads((ENCODED/'manifest.json').read_text());proof=json.loads((ENCODED/'independent_encoding_verification.json').read_text())
    if source_hash!=m['source_manifest_sha256'] or proof['manifest_sha256']!=sha256(ENCODED/'manifest.json'):raise ValueError('existing encoding proof differs')
    for name,h in m['model_files'].items():
        if sha256(MODEL/name)!=h:raise ValueError('model asset changed')
    with np.load(STATE/'clinical_state.npz') as z:members={k:z[k] for k in ('hadm_ids','subject_ids','patient_splits')}
    positions={str(h):i for i,h in enumerate(members['hadm_ids'])}
    chunks=defaultdict(list)
    with gzip.open(ENCODED/'chunks.jsonl.gz','rt',encoding='utf-8') as f:
        for j,r in enumerate(map(json.loads,f)):
            if r['embedding_row']!=j:raise ValueError('existing rows reordered')
            chunks[(r['hadm_id'],r['subject_id'],r['patient_split'],r['field'],r['note_id'])].append(r)
    if set(chunks)!=set(groups):raise ValueError('source group coverage differs')
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,use_fast=True)
    if not tokenizer.is_fast:raise ValueError('fast tokenizer original offsets required')
    existing=np.load(ENCODED/'vectors.float16.npy',mmap_mode='r');index=[];tasks=[];contained=[];metadata=[];sentence_positions=[];stats=Counter()
    for key,units in sorted(groups.items()):
        units=sorted(units,key=lambda r:r['start_char']);wanted=[r for r in units if time_flags(r['quote'],r['section'])['priority_candidate']]
        if not wanted:continue
        h,s,fold,route,note=key;pos=positions[h]
        if str(members['subject_ids'][pos])!=s or members['patient_splits'][pos]!=fold:raise ValueError('subject/fold differs')
        content='\n'.join(r['quote'] for r in units)
        tokenized=tokenizer(content,add_special_tokens=False,return_offsets_mapping=True,truncation=False)
        ids=tokenized['input_ids'];offsets=tokenized['offset_mapping'];prefix=tokenizer('Clinical topic: '+ROUTES[route]+'\n',add_special_tokens=False)['input_ids']
        budget=384-len(prefix)-2;expected_units=[[r['start_char'],r['end_char']] for r in units];right=0;spans=[]
        for r in sorted(chunks[key],key=lambda x:x['token_start']):
            a,b=r['token_start'],r['token_end']
            if a!=right or not a<b<=len(ids) or r['source_units']!=expected_units or r['input_ids']!=[tokenizer.cls_token_id,*prefix,*ids[a:b],tokenizer.sep_token_id]:
                raise ValueError('actual saved block tokens/source differ')
            spans.append((r['embedding_row'],a,b));right=b;stats['source_blocks_token_verified']+=1
        if right!=len(ids):raise ValueError('missing tail tokens')
        cursor=0;wanted_keys={r['unit_key'] for r in wanted}
        for unit in units:
            start,end=cursor,cursor+len(unit['quote']);cursor=end+1
            if unit['unit_key'] not in wanted_keys:continue
            a,b=token_span(offsets,start,end);mapped=overlaps(a,b,spans)
            # 单句再tokenize是局部上下文，不偷用整组hidden；原句内容不改写。
            local_ids=tokenizer(unit['quote'],add_special_tokens=False,truncation=False)['input_ids']
            if not local_ids:raise ValueError('empty local sentence')
            row=len(index);first_task=len(tasks)
            for offset in range(0,len(local_ids),budget):
                end_token=min(offset+budget,len(local_ids))
                tasks.append(dict(sentence_row=row,token_start=offset,token_end=end_token,content_tokens=end_token-offset,
                    input_ids=[tokenizer.cls_token_id,*prefix,*local_ids[offset:end_token],tokenizer.sep_token_id]))
            flags=time_flags(unit['quote'],unit['section']);meta=np.zeros(10,np.float32);meta[SECTIONS.index(unit['section'])]=1
            meta[6:]=[flags[k] for k in FLAG_KEYS];metadata.append(meta);sentence_positions.append(pos)
            record={k:unit[k] for k in ('hadm_id','subject_id','patient_split','note_id','original_sha256','section','start_char','end_char','unit_key')}
            record.update(sentence_row=row,route=route,joined_token_start=a,joined_token_end=b,containing_rows_and_tokens=index_mapping(mapped),
                local_token_count=len(local_ids),task_start=first_task,task_end=len(tasks),quote_sha256=hashlib.sha256(unit['quote'].encode()).hexdigest(),metadata=meta.tolist())
            index.append(record);contained.append(weighted_vector(existing[[j for j,_ in mapped]],[n for _,n in mapped]));stats['sentences:'+fold]+=1
            if fold=='train':stats['training_local_tokens']+=len(local_ids);stats['training_cross_block_sentences']+=int(len(mapped)>1)
    if stats['sentences:train']!=6613:raise ValueError('locked training candidate membership differs')
    arr=dict(**members,sentence_positions=np.asarray(sentence_positions,np.int64),metadata=np.asarray(metadata,np.float32),
        containing_vectors=np.asarray(contained,np.float16))
    source.update({str(p):sha256(p) for p in (Path(__file__),Path('training/v6/encode_screen_candidates.py'),Path('patient_data/text_time_coverage.py'),
        Path('training/tests/test_v6_local_text_interface.py'),PROTOCOL,Path('docs/V6_DATA_SCOPE_DECISION_2026-10-13.md'))})
    report=dict(source_sha256=source,existing_candidate_encoding_manifest_sha256=sha256(ENCODED/'manifest.json'),existing_source_counts=dict(counts),
        encoder_files=m['model_files'],statistics=dict(stats),sentences=len(index),local_tasks=len(tasks),sections=list(SECTIONS),flag_keys=list(FLAG_KEYS),
        training_unique_admissions=len({r['hadm_id'] for r in index if r['patient_split']=='train'}),
        limits=['No high-frequency trajectories used','Same exact safe sentence membership in both views','No outcome used for selection',
            'Containing vector retains other block content, not exact local sentence representation','Lexical time metadata not clinical assertion',
            'Retrospective FINAL policy, not prospective availability'])
    return arr,index,tasks,report


def build():
    if ROOT.exists():raise FileExistsError('new interface directory exists; inspect')
    arr,index,tasks,report=derive();ROOT.mkdir(parents=True)
    np.savez_compressed(ROOT/'interface.npz',**arr)
    for name,values in [('index.jsonl.gz',index),('local_tasks.jsonl.gz',tasks)]:
        with gzip.open(ROOT/name,'xt',encoding='utf-8') as f:
            for r in values:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    report.update(status='source_token_interface_built_local_encoding_pending',output_sha256={n:sha256(ROOT/n) for n in ('interface.npz','index.jsonl.gz','local_tasks.jsonl.gz')})
    write_json(ROOT/'manifest.json',report);write_json(ROOT/'state.json',dict(status='local_encoding_pending',new_model_fits=0))
    print(json.dumps({k:report[k] for k in ('status','statistics','sentences','local_tasks','training_unique_admissions')}),flush=True)


def verify():
    arr,index,tasks,report=derive();m=json.loads((ROOT/'manifest.json').read_text())
    for k,v in report.items():
        if m[k]!=v:raise ValueError('source interface metadata differs: '+k)
    for name,h in m['output_sha256'].items():
        if sha256(ROOT/name)!=h:raise ValueError('interface output changed')
    with np.load(ROOT/'interface.npz') as z:
        for k,v in arr.items():np.testing.assert_array_equal(z[k],v)
    for name,values in [('index.jsonl.gz',index),('local_tasks.jsonl.gz',tasks)]:
        with gzip.open(ROOT/name,'rt',encoding='utf-8') as f:
            if list(map(json.loads,f))!=values:raise ValueError('source token index differs')
    proof=dict(status='independent_source_token_interface_verified',source_and_arrays_max_error=0,manifest_sha256=sha256(ROOT/'manifest.json'),
        sentences=len(index),tasks=len(tasks),no_model_fits=True)
    write_json(ROOT/'source_verification.json',proof);print(json.dumps(proof),flush=True)


def encode():
    import torch
    from transformers import AutoModel,AutoTokenizer
    if not torch.cuda.is_available():raise RuntimeError('actual GPU required')
    if (ROOT/'local_vectors.float16.npy').exists():raise FileExistsError('local vectors exist; inspect before any recovery')
    m=json.loads((ROOT/'manifest.json').read_text());proof=json.loads((ROOT/'source_verification.json').read_text())
    if proof['manifest_sha256']!=sha256(ROOT/'manifest.json'):raise ValueError('source token verification stale')
    for p,h in m['source_sha256'].items():
        if sha256(Path(p))!=h:raise ValueError('protected interface source changed')
    for name,h in m['encoder_files'].items():
        if sha256(MODEL/name)!=h:raise ValueError('encoder asset changed')
    with gzip.open(ROOT/'local_tasks.jsonl.gz','rt',encoding='utf-8') as f:tasks=list(map(json.loads,f))
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);model=AutoModel.from_pretrained(MODEL,local_files_only=True).cuda().eval().requires_grad_(False)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sums=np.zeros((m['sentences'],768),np.float32);weight=np.zeros(m['sentences'],np.float32);start=time.perf_counter()
    for offset in range(0,len(tasks),64):
        current=tasks[offset:offset+64];encoded={k:v.cuda() for k,v in tokenizer.pad([{'input_ids':r['input_ids']} for r in current],padding=True,return_tensors='pt').items()}
        with torch.inference_mode():
            hidden=model(**encoded).last_hidden_state.float();mask=encoded['attention_mask'].bool().clone();lengths=mask.sum(1)
            mask[:,0]=False;mask.scatter_(1,(lengths-1)[:,None],False)
            vectors=torch.nn.functional.normalize((hidden*mask[:,:,None]).sum(1)/mask.sum(1)[:,None],dim=1).cpu().numpy()
        for r,v in zip(current,vectors):sums[r['sentence_row']]+=v*r['content_tokens'];weight[r['sentence_row']]+=r['content_tokens']
        if offset%1024==0:write_json(ROOT/'encoding_progress.json',dict(completed_tasks=offset+len(current),tasks=len(tasks),seconds=time.perf_counter()-start))
    if not np.all(weight>0):raise ValueError('sentence encoding lost')
    sums/=weight[:,None];sums/=np.linalg.norm(sums,axis=1)[:,None]
    if not np.isfinite(sums).all():raise ValueError('nonfinite local vectors')
    np.save(ROOT/'local_vectors.float16.npy',sums.astype(np.float16),allow_pickle=False)
    write_json(ROOT/'encoding.json',dict(status='local_frozen_vectors_complete',rows=m['sentences'],width=768,tasks=len(tasks),
        seconds=time.perf_counter()-start,source_manifest_sha256=sha256(ROOT/'manifest.json'),output_sha256=sha256(ROOT/'local_vectors.float16.npy'),
        GPU=torch.cuda.get_device_name(0),code_sha256=sha256(Path(__file__)),encoder_files=m['encoder_files'],batch_size=64,model_fitted=False,
        pooling='same Clinical topic prefix; content mean minus CLS/SEP/PAD; per-task L2; content-token weighted sentence aggregate then L2'))
    write_json(ROOT/'state.json',dict(status='local_encoding_complete_pending_GPU_replay',new_model_fits=0))
    print(json.dumps(dict(status='local_vectors_complete',rows=m['sentences'],seconds=time.perf_counter()-start)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('build','verify','encode'),required=True);a=p.parse_args()
    try:globals()[a.mode]()
    except Exception as e:
        Path('work/v6_local_text_interface_2026-10-13').mkdir(parents=True,exist_ok=True)
        write_json(Path('work/v6_local_text_interface_2026-10-13')/f'failure_{a.mode}.json',dict(error=repr(e),new_model_fits=0))
        raise
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/medication_identity.py · lines 1–174

<!-- REVIEW_FRAGMENT {"path":"patient_data/medication_identity.py","part":1,"start_line":1,"end_line":174,"source_sha256":"e98c409646975f9fcc06b01db898af112013b8507da390bd4dbef2856c5e7ec3","fragment_sha256":"e98c409646975f9fcc06b01db898af112013b8507da390bd4dbef2856c5e7ec3","added_terminal_newline":false} -->
````python
"""保留原药效映射的独立药物身份/实际剂量接口；不覆盖历史派生数据。"""
from collections import Counter, defaultdict
from datetime import datetime
import argparse
import json
from pathlib import Path
import re
import time
import numpy as np
from patient_data.export_mimiciv import rows, sha256
from patient_data.preprocess_events import ADMINISTERED
from patient_data.eos_med_context import finite
from patient_data.compact_multisource import INHALED_ROUTES, ORDER_SYSTEMIC_ROUTES, order_available
from patient_data.v6_detail_extension import MOLECULES, molecule_matches
from training.v5.numeric_preprocessing import robust_parameters

NAMES = MOLECULES + ('salmeterol', 'formoterol', 'tiotropium', 'fluticasone', 'budesonide',
                    'albuterol_ipratropium', 'fluticasone_salmeterol', 'budesonide_formoterol',
                    'umeclidinium_vilanterol')
UNITS = ('unknown', 'mg', 'puff', 'neb', 'inh', 'ml', 'mixed')
CHANNELS = ('executed_records', 'executed_days', 'maximum_executions_per_day',
            'mean_executions_per_recorded_day', 'complete_cumulative_given_dose',
            'maximum_given_dose', 'planned_doses_per_24h')
ALIASES = {'albuterol_ipratropium': ('duoneb', 'combivent'),
           'fluticasone_salmeterol': ('advair',), 'budesonide_formoterol': ('symbicort',),
           'umeclidinium_vilanterol': ('anoro',), 'tiotropium': ('spiriva',),
           'fluticasone': ('flovent',), 'budesonide': ('pulmicort',),
           'salmeterol': ('serevent',), 'formoterol': ('foradil', 'perforomist')}
INHALED = {'albuterol','ipratropium',*NAMES[12:]}


def identity(name):
    s = str(name).lower()
    for combo, constituents in (('albuterol_ipratropium', ('albuterol','ipratropium')),
                                ('fluticasone_salmeterol', ('fluticasone','salmeterol')),
                                ('budesonide_formoterol', ('budesonide','formoterol')),
                                ('umeclidinium_vilanterol', ('umeclidinium','vilanterol'))):
        if all(c in s for c in constituents) or any(a in s for a in ALIASES[combo]): return combo
    hits = list(molecule_matches(s))
    hits += [n for n in NAMES[12:17] if re.search(r'\b'+n+r'\b', s) or any(a in s for a in ALIASES[n])]
    return hits[0] if len(hits) == 1 else None


def component_dose(detail, drug):
    """返回数值、原单位类别、质量原因；不使用dose_due或浓度猜测。"""
    components = {}
    for r in detail:
        ordinal = r.get('parent_field_ordinal', '')
        if ordinal in ('', r'\N'): continue
        product = r.get('product_description','')
        if product not in ('',r'\N') and identity(product) != drug: return None, 'unknown', 'product_identity_ambiguous'
        value = finite(r.get('dose_given')); unit = r.get('dose_given_unit','').strip().lower()
        factors = {'mg':1., 'mcg':.001, 'ug':.001, 'g':1000., 'gm':1000.}
        aliases = {'puff':'puff','puffs':'puff','neb':'neb','nebs':'neb','inh':'inh','ml':'ml'}
        if value is None or value < 0 or unit not in factors and unit not in aliases:
            return None, 'unknown', 'unknown_given_dose_or_unit'
        pair = (value*factors[unit], 'mg') if unit in factors else (value, aliases[unit])
        if ordinal in components and components[ordinal] != pair: return None, 'unknown', 'ordinal_conflict'
        components[ordinal] = pair
    if not components: return None, 'unknown', 'no_scanned_component'
    units = {u for _,u in components.values()}
    if len(units) != 1: return None, 'mixed', 'component_units_mixed'
    return sum(v for v,_ in components.values()), units.pop(), 'verified_given_components'


def build(snapshot, state_dir, output):
    if output.exists(): raise FileExistsError(output)
    started=time.perf_counter(); sm=json.loads((snapshot/'metadata/manifest.json').read_text())
    st=json.loads((state_dir/'manifest.json').read_text())
    if sm['status']!='validated' or sha256(state_dir/'clinical_state.npz')!=st['output_sha256']['clinical_state.npz']:
        raise ValueError('unvalidated parent')
    with np.load(state_dir/'clinical_state.npz') as z: state={k:z[k] for k in ('hadm_ids','subject_ids','patient_splits')}
    ids=list(map(str,state['hadm_ids'])); ix={h:i for i,h in enumerate(ids)}
    admissions={r['hadm_id']:(datetime.fromisoformat(r['admittime']),datetime.fromisoformat(r['dischtime']))
                for r in rows(snapshot/'cohort/admissions.csv.gz') if r['hadm_id'] in ix}
    hashes={}
    for name in ('cohort/admissions','raw/emar','raw/emar_detail','raw/pharmacy'):
        hashes[name]=sha256(snapshot/(name+'.csv.gz'))
        if hashes[name]!=sm['sources'][name]['sha256']: raise ValueError('raw changed '+name)
    orders=defaultdict(list); routes={}; planned=set(); executed=set(); raw_names=defaultdict(Counter); audit=Counter()
    for r in rows(snapshot/'raw/pharmacy.csv.gz'):
        h=r['hadm_id']
        if h not in ix or not order_available(r,admissions[h])[0]: continue
        planned.add(h); routes[h,r['pharmacy_id']]=(r['route'].strip().upper(),r['starttime'],r['stoptime'])
        drug=identity(r['medication'])
        if not drug: continue
        raw_names[drug][r['medication']]+=1
        allowed=INHALED_ROUTES if drug in INHALED else ORDER_SYSTEMIC_ROUTES
        if r['route'].strip().upper() not in allowed: continue
        f=finite(r['doses_per_24_hrs'])
        if f is not None and 0<f<=48: orders[h,drug].append(f)
    parents={}
    for r in rows(snapshot/'raw/emar.csv.gz'):
        h=r['hadm_id']
        if h not in ix: continue
        try: t,s=datetime.fromisoformat(r['charttime']),datetime.fromisoformat(r['storetime'])
        except ValueError: continue
        if not admissions[h][0]<=t<=s<=admissions[h][1]: continue
        executed.add(h); drug=identity(r['medication'])
        if not drug or r['event_txt'] not in ADMINISTERED: continue
        raw_names[drug][r['medication']]+=1
        key=r['emar_id'],r['emar_seq']
        if key in parents: raise ValueError('duplicate eMAR parent')
        parents[key]=(h,t,drug,r['pharmacy_id'])
    details=defaultdict(list)
    for r in rows(snapshot/'raw/emar_detail.csv.gz'):
        key=r['emar_id'],r['emar_seq']
        if key in parents: details[key].append(r)
    observations=defaultdict(list)
    for key,(h,t,drug,pid) in parents.items():
        ds=details[key]; route={r['route'].strip().upper() for r in ds if r['route'] not in ('',r'\N')}
        if not route and (h,pid) in routes:
            u,first,last=routes[h,pid]
            if datetime.fromisoformat(first)<=t and (last in ('',r'\N') or t<=datetime.fromisoformat(last)):
                route={u}; audit['route_from_linked_visible_order']+=1
        allowed=INHALED_ROUTES if drug in INHALED else ORDER_SYSTEMIC_ROUTES
        if not route or not route.issubset(allowed): audit['unverified_route']+=1; continue
        if any(r['complete_dose_not_given'].strip().upper() in ('Y','YES','1','TRUE') for r in ds):
            audit['incomplete_execution']+=1; continue
        dose,unit,reason=component_dose(ds,drug); audit[reason]+=1
        if dose==0: audit['zero_dose_execution_excluded']+=1; continue
        observations[h,drug].append((t,dose,unit))
    values=np.zeros((len(ids),len(NAMES),7),np.float32); masks=np.zeros_like(values,bool)
    units=np.zeros((len(ids),len(NAMES)),np.int64); source=np.zeros((len(ids),len(NAMES),2),bool)
    present=np.zeros((len(ids),len(NAMES)),bool); coverage={}
    for h,i in ix.items():
        for j,drug in enumerate(NAMES):
            source[i,j]=h in executed,h in planned
            if h in executed: masks[i,j,:4]=True
            obs=observations.get((h,drug),[]); order=orders.get((h,drug),[])
            present[i,j]=bool(obs or order)
            if obs:
                days=Counter(t.date() for t,_,_ in obs)
                values[i,j,:4]=len(obs),len(days),max(days.values()),len(obs)/len(days)
                known={u for _,d,u in obs if d is not None}
                units[i,j]=UNITS.index(next(iter(known))) if len(known)==1 else UNITS.index('mixed') if known else 0
                if all(d is not None for _,d,_ in obs) and len(known)==1:
                    doses=[d for _,d,_ in obs]; values[i,j,4:6]=sum(doses),max(doses); masks[i,j,4:6]=True
            if order: values[i,j,6]=np.median(order); masks[i,j,6]=True
    train=state['patient_splits']=='train'; scaled=np.zeros_like(values); norm={}
    for j,drug in enumerate(NAMES):
        for k in range(7):
            buckets=range(len(UNITS)) if k in (4,5) else (-1,)
            for u in buckets:
                valid=masks[:,j,k] & (units[:,j]==u if u>=0 else True); fitted=valid & train
                if not fitted.any(): masks[valid,j,k]=False; continue
                x=np.log1p(values[:,j,k]); p=robust_parameters(x[fitted])
                scaled[valid,j,k]=np.clip((x[valid]-p['median'])/p['scale'],-8,8)
                norm[f'{drug}/{CHANNELS[k]}/{UNITS[u] if u>=0 else "count"}']=p
        coverage[drug]={'execution_admissions':int((values[:,j,0]>0).sum()),
                        'complete_dose_admissions':int(masks[:,j,4].sum()),'present_admissions':int(present[:,j].sum()),
                        'units':dict(Counter(UNITS[u] for u in units[present[:,j],j]))}
    unit_onehot=np.eye(len(UNITS),dtype=np.float32)[units]
    features=np.concatenate((scaled,masks.astype(np.float32),unit_onehot,source.astype(np.float32)),2)
    # 缺乏本药内容的token彻底屏蔽，来源不等同本药存在。
    features*=present[:,:,None]
    output.mkdir(parents=True)
    np.savez_compressed(output/'medication_identity.npz',**state,features=features,present=present,
                        values_raw=values,value_masks=masks,dose_unit_id=units,drug_names=np.asarray(NAMES))
    (output/'normalization.json').write_text(json.dumps(norm,indent=2),encoding='utf-8')
    (output/'raw_drug_names.json').write_text(json.dumps({k:dict(v) for k,v in raw_names.items()},indent=2),encoding='utf-8')
    manifest=dict(status='validated_medication_identity',parent_state_manifest_sha256=sha256(state_dir/'manifest.json'),
        sources_sha256=hashes,output_sha256=sha256(output/'medication_identity.npz'),
        names=NAMES,units=UNITS,channels=CHANNELS,feature_width=23,coverage=coverage,audit=dict(audit),
        elapsed_seconds=time.perf_counter()-started,limits=['Dose absence is unknown, not zero.',
        'Actual frequency per recorded day is not prescription frequency or complete outside exposure.',
        'Raw name aliases are audited; canonical identity preserves explicit fixed combinations.'])
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(dict(status=manifest['status'],seconds=manifest['elapsed_seconds'],coverage=coverage)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--state-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();build(a.snapshot,a.state_dir,a.output)
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/oxygen_device.py · lines 1–185

<!-- REVIEW_FRAGMENT {"path":"patient_data/oxygen_device.py","part":1,"start_line":1,"end_line":185,"source_sha256":"68515ea9852a7b474476c9a88dba60dd4970e2009d8ef286c1971429c78ab6d6","fragment_sha256":"68515ea9852a7b474476c9a88dba60dd4970e2009d8ef286c1971429c78ab6d6","added_terminal_newline":false} -->
````python
"""完整设备同刻集合/同stay真实转换；不推断氧浓度、撤机或因果恢复。"""
from collections import Counter,defaultdict
from datetime import datetime
import argparse
import json
import math
from pathlib import Path
import numpy as np
from patient_data.export_mimiciv import rows,sha256
from patient_data.respiratory_transition import SNAP,SOURCE,STATE,icu_contains,normalize
from training.v6.train import write_json

ROOT=Path('patient_data/processed/v6_oxygen_device_2026-10-13')
PROTOCOL=Path('docs/V6_OXYGEN_DEVICE_PROTOCOL_2026-10-13.md')
LABELS=('Endotracheal tube','Nasal cannula','Face tent','High flow neb','High flow nasal cannula',
        'Aerosol-cool','Bipap mask','Non-rebreather','Venti mask','Oxymizer','CPAP mask','None',
        'Tracheostomy tube','Medium conc mask','Trach mask','Other','T-piece','Ultrasonic neb')
LOOKUP={v.casefold():i for i,v in enumerate(LABELS)}
PROC=('224385','227194','225468','225477','225792','225794')
PROC_NAMES=('Intubation','Extubation','Unplanned Extubation (patient-initiated)',
            'Unplanned Extubation (non-patient initiated)','Invasive Ventilation','Non-invasive Ventilation')
TIMES=('log_recorded_time_count','log_first_admission_offset_days','log_latest_age_days',
       'log_first_latest_span_days','log_latest_identical_set_record_span_days','log_set_change_count')
AVAIL=[90,91,92,93,114]
TIMING=AVAIL+list(range(94,98))+list(range(100,104))
DIM=115


def label(value):
    if value.strip() in ('',r'\N'):return None
    key=value.strip().casefold()
    if key not in LOOKUP:raise ValueError('unknown device label: '+value)
    return LOOKUP[key]


def compress(records):
    """先集合后压缩；末时间保留。同一stay，同一时刻没有人工先后关系。"""
    grouped=defaultdict(set)
    for t,k in records:grouped[t].add(k)
    result=[]
    for t,values in sorted(grouped.items()):
        values=frozenset(values)
        if result and result[-1][2]==values:result[-1]=(result[-1][0],t,values)
        else:result.append((t,t,values))
    return result,grouped


def available(start,end,t,stored,stop=None):
    return start<=t<=stored<=end and (stop is None or t<=stop<=end)


def summarize(by_stay,start,end,procedures=()):
    hot=np.zeros((5,len(LABELS)));flags=np.zeros(4);raw=np.zeros(6);mask=np.zeros(6,bool);proc=np.zeros(9)
    sets=[];runs={};transitions=0;observed_times=set()
    for stay,records in by_stay.items():
        compressed,grouped=compress(records);runs[stay]=compressed
        sets.extend((t,stay,v) for t,v in grouped.items());observed_times.update(grouped)
        for _,values in grouped.items():hot[2,list(values)]=1
        for a,b in zip(compressed,compressed[1:]):
            hot[3,list(b[2]-a[2])]=1;hot[4,list(a[2]-b[2])]=1;transitions+=1
    if sets:
        first=min(t for t,_,_ in sets);last=max(t for t,_,_ in sets)
        for t,_,values in sets:
            if t==first:hot[0,list(values)]=1
            if t==last:hot[1,list(values)]=1
        # 同刻跨stay末记录不能任意选一个，用最短的已知末连续段跨度。
        tail=[(last-runs[s][-1][0]).total_seconds()/86400 for t,s,_ in sets if t==last]
        raw[:]=[math.log1p(len(observed_times)),math.log1p((first-start).total_seconds()/86400),
            math.log1p((end-last).total_seconds()/86400),math.log1p((last-first).total_seconds()/86400),
            math.log1p(min(tail)),math.log1p(transitions)]
        mask[:]=True;flags[:]=[1,len(observed_times)>1,any(len(v)>1 for _,_,v in sets),len(by_stay)>1]
        if not start<=first<=last<=end:raise ValueError('device time unavailable')
    for stay,item,a,b in procedures:
        proc[PROC.index(item)]=1;proc[8]=1
        if sets and item in ('225792','225794') and any(t==last and s==stay and a<=t<=b for t,s,_ in sets):
            proc[6+('225792','225794').index(item)]=1
    return hot,flags,raw,mask,proc


def derive():
    sm=json.loads((SOURCE/'metadata/manifest.json').read_text());parent=json.loads((SNAP/'metadata/manifest.json').read_text())
    state=json.loads((STATE/'manifest.json').read_text())
    if sm['status']!='validated_restricted_source_export_not_model_validated' or parent['status']!='validated':raise ValueError('source not validated')
    if sha256(SNAP/'metadata/manifest.json')!=sm['parent_snapshot_manifest_sha256'] or sha256(SNAP/'cohort/admissions.csv.gz')!=sm['parent_cohort_sha256']:
        raise ValueError('parent lineage differs')
    names=('respiratory_chartevents','icustays','d_items','respiratory_procedureevents')
    for name in names:
        if sha256(SOURCE/'raw'/f'{name}.csv.gz')!=sm['sources'][name]['sha256']:raise ValueError('source changed: '+name)
    if sha256(STATE/'clinical_state.npz')!=state['output_sha256']['clinical_state.npz']:raise ValueError('state changed')
    dictionary={r['itemid']:r for r in rows(SOURCE/'raw/d_items.csv.gz')}
    if dictionary['226732']['label']!='O2 Delivery Device(s)' or dictionary['226732']['linksto']!='chartevents':raise ValueError('device dictionary differs')
    for item,name in zip(PROC,PROC_NAMES):
        if dictionary[item]['label']!=name or dictionary[item]['linksto']!='procedureevents':raise ValueError('procedure dictionary differs')
    with np.load(STATE/'clinical_state.npz') as z:hadms=z['hadm_ids'];people=z['subject_ids'];splits=z['patient_splits']
    by_hadm={str(h):i for i,h in enumerate(hadms)};cohort={r['hadm_id']:r for r in rows(SNAP/'cohort/admissions.csv.gz')}
    stays={r['stay_id']:r for r in rows(SOURCE/'raw/icustays.csv.gz')}
    observations=defaultdict(lambda:defaultdict(list));procedures=defaultdict(list);audit=Counter();counts=Counter();seen=set()
    def bounds(r,t,stored,stop=None):
        h=r['hadm_id'];c=cohort[h];stay=stays[r['stay_id']]
        if c['subject_id']!=r['subject_id'] or stay['subject_id']!=r['subject_id'] or stay['hadm_id']!=h:raise ValueError('source subject/hospital/stay mismatch')
        start,end=datetime.fromisoformat(c['admittime']),datetime.fromisoformat(c['dischtime'])
        if not available(start,end,t,stored,stop):raise ValueError('source not available by discharge')
        contains=icu_contains(stay,t)
        if contains is None:audit['unknown_ICU_boundary_hospital_available']+=1
        elif not contains:audit['outside_ICU_boundary_hospital_available']+=1
    for r in rows(SOURCE/'raw/respiratory_chartevents.csv.gz'):
        if r['itemid']!='226732':continue
        if r['hadm_id'] not in by_hadm:audit['raw_ineligible_device_rows_retained']+=1;continue
        if r['selection_role']!='state_record':raise ValueError('device states not fully retained')
        t=datetime.fromisoformat(r['charttime']);stored=datetime.fromisoformat(r['storetime']);bounds(r,t,stored)
        k=label(r['value'])
        if k is None:audit['empty_device_label']+=1;continue
        key=(r['hadm_id'],r['stay_id'],t,stored,k)
        if key in seen:audit['duplicate_device_rows']+=1;continue
        seen.add(key);observations[r['hadm_id']][r['stay_id']].append((t,k));audit['accepted_device_rows']+=1
        if splits[by_hadm[r['hadm_id']]]=='train':counts[LABELS[k]]+=1
    seen=set()
    for r in rows(SOURCE/'raw/respiratory_procedureevents.csv.gz'):
        if r['itemid'] not in PROC:raise ValueError('unexpected procedure item')
        if r['hadm_id'] not in by_hadm:audit['raw_ineligible_procedure_rows_retained']+=1;continue
        if r['statusdescription'] not in ('FinishedRunning','Stopped'):audit['procedure_status_excluded:'+r['statusdescription']]+=1;continue
        a,b,s=(datetime.fromisoformat(r[k]) for k in ('starttime','endtime','storetime'))
        hospital=cohort[r['hadm_id']];start,end=(datetime.fromisoformat(hospital[k]) for k in ('admittime','dischtime'))
        if not available(start,end,a,s,b):audit['unavailable_or_malformed_procedure_excluded']+=1;continue
        bounds(r,a,s,b)
        key=(r['hadm_id'],r['stay_id'],r['itemid'],r['orderid'],a,b)
        if key in seen:audit['duplicate_procedures']+=1;continue
        seen.add(key);procedures[r['hadm_id']].append((r['stay_id'],r['itemid'],a,b));audit['accepted_procedures:'+r['itemid']]+=1
    n=len(hadms);hot=np.zeros((n,5,len(LABELS)));flags=np.zeros((n,4));raw=np.zeros((n,6));mask=np.zeros((n,6),bool);proc=np.zeros((n,9))
    compressed_counts=[];multi_times=0
    for h,i in by_hadm.items():
        r=cohort[h];a,b=(datetime.fromisoformat(r[k]) for k in ('admittime','dischtime'))
        hot[i],flags[i],raw[i],mask[i],proc[i]=summarize(observations[h],a,b,procedures[h])
        if splits[i]=='train':
            for records in observations[h].values():
                runs,grouped=compress(records);compressed_counts.append(len(runs));multi_times+=sum(len(v)>1 for v in grouped.values())
    scaled,parameters=normalize(raw,mask,splits=='train')
    features=np.concatenate((hot.reshape(n,5*len(LABELS)),flags,scaled,mask,proc),1).astype(np.float32)
    availability=np.zeros_like(features);availability[:,AVAIL]=features[:,AVAIL]
    timing=np.zeros_like(features);timing[:,TIMING]=features[:,TIMING]
    arr=dict(hadm_ids=hadms,subject_ids=people,patient_splits=splits,features=features,availability=availability,timing=timing,
        observed=flags[:,0].astype(bool),device_multihot=hot,device_flags=flags,time_raw=raw,time_masks=mask,procedure_evidence=proc)
    training=splits=='train';observed=training&arr['observed']
    report=dict(source_manifest_sha256=sha256(SOURCE/'metadata/manifest.json'),parent_state_manifest_sha256=sha256(STATE/'manifest.json'),
        source_sha256={name:sm['sources'][name]['sha256'] for name in names},dictionary={k:dictionary[k] for k in ('226732',*PROC)},
        normalization=parameters,audit=dict(audit),training_device_label_counts=dict(counts),labels=list(LABELS),time_channels=list(TIMES),features=DIM,
        training_observed=int(observed.sum()),training_procedure_observed=int(proc[training,8].sum()),
        training_multilabel_timestamps=int(multi_times),compressed_state_count_quantiles=np.quantile(compressed_counts,[0,.1,.5,.9,1]).tolist(),
        training_latest_age_days_quantiles=np.quantile(np.expm1(raw[observed,2]),[0,.1,.5,.9,1]).tolist(),
        training_latest_procedure_match_counts=proc[training,6:8].sum(0).astype(int).tolist(),
        training_ever_device_admissions=hot[training,2].sum(0).astype(int).tolist(),
        limitations=['device None not verified room air','multi-labels can be valid simultaneous devices',
            'only within-stay transitions','no interpolation or causal recovery','latest ICU record not discharge state',
            'absence of procedure evidence is not clinical non-use','new branch gated by device observation'])
    return arr,report


def build():
    if ROOT.exists():raise FileExistsError('new derived directory exists; inspect')
    arr,report=derive();json.dumps(report,allow_nan=False);ROOT.mkdir(parents=True)
    np.savez_compressed(ROOT/'oxygen_device.npz',**arr)
    report.update(status='derived_roundtrip_pending_source_rebuild',output_sha256=sha256(ROOT/'oxygen_device.npz'),protocol_sha256=sha256(PROTOCOL),
        code_sha256={p:sha256(Path(p)) for p in ('patient_data/oxygen_device.py','patient_data/respiratory_transition.py',
            'patient_data/resp_micro_context.py','training/v5/numeric_preprocessing.py','training/tests/test_v6_oxygen_device.py')})
    with np.load(ROOT/'oxygen_device.npz') as z:
        for k,v in arr.items():np.testing.assert_array_equal(v,z[k])
    write_json(ROOT/'manifest.json',report);print(json.dumps(report),flush=True)


def verify():
    arr,report=derive();m=json.loads((ROOT/'manifest.json').read_text())
    for k,v in report.items():
        if m[k]!=v:raise ValueError('source metadata differs: '+k)
    for p,h in m['code_sha256'].items():
        if sha256(Path(p))!=h:raise ValueError('source code differs')
    if sha256(PROTOCOL)!=m['protocol_sha256'] or sha256(ROOT/'oxygen_device.npz')!=m['output_sha256']:raise ValueError('protocol/data differs')
    with np.load(ROOT/'oxygen_device.npz') as z:
        for k,v in arr.items():np.testing.assert_array_equal(v,z[k])
    proof=dict(status='independent_source_rebuild_verified',maximum_array_error=0.,arrays=len(arr),output_sha256=m['output_sha256'])
    write_json(ROOT/'independent_verification.json',proof);print(json.dumps(proof),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('build','verify'),required=True);a=p.parse_args()
    build() if a.mode=='build' else verify()
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/pft_evidence.py · lines 1–174

<!-- REVIEW_FRAGMENT {"path":"patient_data/pft_evidence.py","part":1,"start_line":1,"end_line":174,"source_sha256":"4a986c1f71259849e951f75a8d98b6761e11c847ebb3f86882f736d62ff3e3ce","fragment_sha256":"4a986c1f71259849e951f75a8d98b6761e11c847ebb3f86882f736d62ff3e3ce","added_terminal_newline":false} -->
````python
"""仅训练文本肺功能原源绑定和跨住院重复审计，不提供临床gold。"""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import re

from patient_data.export_mimiciv import rows, sha256
from patient_data.pft_inventory import ROOT as INVENTORY, NAME, HISTORY, PLAN, PRE, POST, sha256_text
from patient_data.text_time_coverage import SCREEN, SNAP, checked_sources
from training.v6.encode_screen_candidates import selected_groups
from training.v6.train import write_json

ROOT = Path('patient_data/processed/v6_pft_evidence_2026-10-13')
PROTOCOL = Path('docs/V6_PFT_EVIDENCE_PROTOCOL_2026-10-13.md')
NUM = r'(?:\d+(?:\.\d+)?|\.\d+)'
FIRST = re.compile(r'^\s*(?:(?:was|is|of)\b\s*|[:=]\s*)?([<>]=?\s*)?('+NUM+r')(?![\d.])', re.I)
UNIT = re.compile(r'^\s*(%|ml\b|l\b|lit(?:er|re)s?\b)', re.I)
PRED = r'(?:of\s+)?(?:predicted|pred\.?)(?!\w)'
AFTER_PRED = re.compile(r'^\s*'+PRED, re.I)
ANNOTATION = re.compile(r'^\s*\(\s*('+NUM+r')\s*%\s*('+PRED+r')?\s*\)', re.I)
DATE = re.compile(r'\b(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}(?:/\d{2,4})?)\b')
YEAR = re.compile(r'\b(?:19|20|21)\d{2}\b')
REDACTED = re.compile(r'_{3,}|\[\*\*.*?\*\*\]')


def bindings(text):
    """边界/明确词面量纲绑定；不推断生理值、列顺序或检测时间。"""
    result = []; names = list(NAME.finditer(text))
    for i, name in enumerate(names):
        end = names[i+1].start() if i+1 < len(names) else len(text)
        tail = text[name.end():end]; first = FIRST.match(tail)
        metric = re.sub(r'\s+', '', name.group()).upper()
        row = dict(metric=metric, metric_start=name.start(), segment_end=end,
                   status='no_adjacent_numeric', quantities=[])
        if first is None:
            result.append(row); continue
        a = name.end()+first.start(2); b = name.end()+first.end(2); after = text[b:end]
        qualifier = (first.group(1) or '').strip(); unit = UNIT.match(after)
        range_match = re.match(r'^\s*[-–]\s*'+NUM, after)
        date_number = bool(DATE.match(text[a:end])) or bool(YEAR.fullmatch(first.group(2)))
        multi = bool(re.match(r'^\s+'+NUM, after))
        reason = ('date_like_number' if date_number else 'range_ambiguous' if range_match else
                  'bound_not_scalar' if qualifier else 'unlabeled_numeric_columns' if multi else None)
        literal = unit.group(1) if unit else None; kind = 'unit_unknown'
        unit_end = b+unit.end() if unit else b
        predicted = bool(AFTER_PRED.match(text[unit_end:end]))
        if literal == '%':
            kind = ('ratio_percent_predicted' if predicted else 'ratio_percent') if metric == 'FEV1/FVC' else (
                    'percent_predicted' if predicted else 'percent_unspecified')
        elif literal:
            kind = 'volume_ml' if literal.lower() == 'ml' else 'volume_l'
            if metric not in ('FEV1', 'FVC'):
                reason = reason or 'volume_unit_metric_mismatch'
        item = dict(value=float(first.group(2)), kind=kind, unit_literal=literal, qualifier=qualifier,
                    value_start=a, value_end=b, evidence_start=name.start(), evidence_end=unit_end,
                    evidence_sha256=sha256_text(text[name.start():unit_end]))
        row['quantities'].append(item)
        if not reason and literal and literal != '%':
            annotation = ANNOTATION.match(text[unit_end:end])
            if annotation:
                pa = unit_end+annotation.start(1); pb = unit_end+annotation.end(1)
                row['quantities'].append(dict(value=float(annotation.group(1)),
                    kind='percent_predicted' if annotation.group(2) else 'percent_unspecified', unit_literal='%',
                    qualifier='', value_start=pa, value_end=pb, evidence_start=unit_end,
                    evidence_end=unit_end+annotation.end(), evidence_sha256=sha256_text(text[unit_end:unit_end+annotation.end()])))
        ready = reason is None and kind in ('volume_l', 'volume_ml', 'percent_predicted', 'ratio_percent')
        row.update(status=reason or ('explicit_quantity_lexical' if ready else 'quantity_semantics_unknown'),
                   scalar_binding_eligible=ready)
        result.append(row)
    return result


def repeat_stats(records, key):
    groups = defaultdict(set)
    for row in records:
        groups[(row['subject_id'], row[key])].add(row['hadm_id'])
    repeated = {k: v for k, v in groups.items() if len(v) > 1}
    return dict(cross_admission_groups=len(repeated), patients=len({k[0] for k in repeated}),
                admissions=len(set().union(*repeated.values())) if repeated else 0)


def derive():
    original_report = json.loads((INVENTORY/'inventory.json').read_text())
    for tag in ('a','b'):
        proof = json.loads((INVENTORY/f'verification_{tag}.json').read_text())
        if proof['maximum_error'] or proof['inventory_sha256'] != sha256(INVENTORY/'inventory.json') or proof['records_sha256'] != sha256(INVENTORY/'records.jsonl.gz'):
            raise ValueError('original PFT inventory proof differs')
    for p, digest in original_report['source_sha256'].items():
        if sha256(Path(p)) != digest:
            raise ValueError('original inventory source changed: '+p)
    sources, _ = checked_sources(); groups, quarantine, _ = selected_groups(SCREEN)
    units = [u for k, g in groups.items() if k[2] == 'train' for u in g if 'pulmonary_function' in u['topics']]
    needed = {u['note_id'] for u in units}
    raw = {r['note_id']: r for r in rows(SNAP/'raw/discharge.csv.gz') if r['note_id'] in needed}
    with (SCREEN/'documents.jsonl').open(encoding='utf-8') as f:
        docs = {r['note_id']: r for r in map(json.loads,f) if r['note_id'] in needed}
    status = Counter(); quantity = Counter(); flags_count = Counter(); all_records = []
    eligible_hadms = set(); eligible_patients = set(); metadata = Counter()
    for u in sorted(units,key=lambda r:(r['note_id'],r['start_char'])):
        original = raw[u['note_id']]; text = original['text']; quote = u['quote']
        if text[u['start_char']:u['end_char']] != quote or sha256_text(text) != u['original_sha256']:
            raise ValueError('PFT original source differs')
        sec = docs[u['note_id']]['section_spans'][u['section_ordinal']]
        if not 0 <= sec['start_char'] <= u['start_char'] < u['end_char'] <= sec['end_char'] <= len(text):
            raise ValueError('PFT section membership differs')
        facts = bindings(quote)
        if not facts:
            metadata['topic_without_target_metric_name'] += 1; continue
        for fact in facts:
            status[fact['status']] += 1
            for q in fact['quantities']:
                if float(quote[q['value_start']:q['value_end']]) != q['value']:
                    raise ValueError('numeric value source span differs')
                if sha256_text(quote[q['evidence_start']:q['evidence_end']]) != q['evidence_sha256']:
                    raise ValueError('quantity evidence source span differs')
                quantity[(fact['metric']+'/'+q['kind'])] += 1
            if fact.get('scalar_binding_eligible',False):
                eligible_hadms.add(u['hadm_id']); eligible_patients.add(u['subject_id'])
        section = text[sec['start_char']:sec['end_char']]
        flags = dict(historical_section=u['section']=='past_medical_history', history_words=bool(HISTORY.search(quote)),
                     plan_words=bool(PLAN.search(quote)), pre_bronchodilator_words=bool(PRE.search(quote)),
                     post_bronchodilator_words=bool(POST.search(quote)), same_unit_date_hint=bool(DATE.search(quote)),
                     same_unit_year_hint=bool(YEAR.search(quote)), same_unit_redaction=bool(REDACTED.search(quote)),
                     same_section_date_hint=bool(DATE.search(section)), same_section_redaction=bool(REDACTED.search(section)))
        flags_count.update({k:int(v) for k,v in flags.items()})
        signature = [(f['metric'],f['status'],[(q['value'],q['kind'],q['qualifier']) for q in f['quantities']]) for f in facts if f['quantities']]
        row = dict(note_id=u['note_id'],hadm_id=u['hadm_id'],subject_id=u['subject_id'],section=u['section'],
                   start_char=u['start_char'],end_char=u['end_char'],original_sha256=u['original_sha256'],
                   quote_sha256=sha256_text(quote),quantities_signature=sha256_text(json.dumps(signature,sort_keys=True)),
                   section_sha256=sha256_text(section),flags=flags,bindings=facts,measurement_time='unknown',
                   charttime=original['charttime'],storetime=original['storetime'])
        all_records.append(row)
    numeric = [r for r in all_records if any(f['quantities'] for f in r['bindings'])]
    sources.update({str(p):sha256(p) for p in (PROTOCOL,Path(__file__),Path('training/tests/test_v6_pft_evidence.py'),
                    INVENTORY/'inventory.json',INVENTORY/'records.jsonl.gz',INVENTORY/'verification_a.json',INVENTORY/'verification_b.json')})
    report = dict(status='training_evidence_binding_audit_not_clinical_gold',units=len(units),
                  units_with_target_metric=len(all_records),units_with_numeric_candidate=len(numeric),
                  binding_status=dict(status),metric_quantity_counts=dict(quantity),time_role_flags=dict(flags_count),
                  extra_counts=dict(metadata),explicit_scalar_binding_admissions=len(eligible_hadms),
                  explicit_scalar_binding_patients=len(eligible_patients),
                  same_patient_exact_quote_repeat=repeat_stats(numeric,'quote_sha256'),
                  same_patient_numeric_signature_repeat=repeat_stats(numeric,'quantities_signature'),
                  source_sha256=sources,quarantine=quarantine,new_model_fits=0,new_vectors=0,
                  high_frequency_used=False,outcomes_used=False,nontraining_facts_built=False,
                  limits=['explicit quantity binding not valid measurement gold', 'all PFT measurement times unknown',
                          'same section date hints are not assigned to measurements', 'repeats not proof of copy or new measurement',
                          'r4 topic coverage not exhaustive original text', 'FINAL retrospective availability unchanged'])
    return report,all_records


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('build','verify'),required=True)
    parser.add_argument('--tag',choices=('a','b'),default='a');args=parser.parse_args()
    report,records=derive()
    if args.mode=='build':
        if ROOT.exists():raise FileExistsError('audit exists; inspect rather than overwrite')
        ROOT.mkdir();write_json(ROOT/'audit.json',report)
        with gzip.open(ROOT/'records.jsonl.gz','wt',encoding='utf-8') as f:
            for r in records:f.write(json.dumps(r,sort_keys=True)+'\n')
        write_json(ROOT/'state.json',dict(status='built_pending_two_independent_replays',new_model_fits=0))
    else:
        saved=json.loads((ROOT/'audit.json').read_text())
        with gzip.open(ROOT/'records.jsonl.gz','rt',encoding='utf-8') as f:prior=[json.loads(line) for line in f]
        if saved!=report or prior!=records:raise ValueError('independent PFT evidence replay differs')
        write_json(ROOT/f'verification_{args.tag}.json',dict(maximum_error=0,records=len(records),
                   audit_sha256=sha256(ROOT/'audit.json'),records_sha256=sha256(ROOT/'records.jsonl.gz')))
        if all((ROOT/f'verification_{tag}.json').exists() for tag in ('a','b')):
            write_json(ROOT/'state.json',dict(status='two_independent_source_replays_verified',new_model_fits=0))
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_sha256','quarantine')}))


if __name__=='__main__':main()
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/pft_evidence_report.py · lines 1–51

<!-- REVIEW_FRAGMENT {"path":"patient_data/pft_evidence_report.py","part":1,"start_line":1,"end_line":51,"source_sha256":"939a0330e9615b48aa9a70b271fab3517732d71d9c30d03f5f0a087f1d77aa9d","fragment_sha256":"939a0330e9615b48aa9a70b271fab3517732d71d9c30d03f5f0a087f1d77aa9d","added_terminal_newline":false} -->
````python
"""源重建合格后的肺功能证据聚合，不读取结局或改变候选规则。"""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
from patient_data.pft_evidence import ROOT, repeat_stats
from patient_data.export_mimiciv import sha256
from training.v6.train import write_json


def readout():
    for tag in ('a','b'):
        p=json.loads((ROOT/f'verification_{tag}.json').read_text())
        if p['maximum_error'] or p['audit_sha256']!=sha256(ROOT/'audit.json') or p['records_sha256']!=sha256(ROOT/'records.jsonl.gz'):
            raise ValueError('PFT evidence is not independently source-verified')
    with gzip.open(ROOT/'records.jsonl.gz','rt',encoding='utf-8') as f:records=[json.loads(line) for line in f]
    eligible=[r for r in records if any(b.get('scalar_binding_eligible',False) for b in r['bindings'])]
    flags=Counter(); types=Counter(); annotation_units=0; ambiguous=Counter()
    supported={'volume_l','volume_ml','percent_predicted','ratio_percent'}
    for r in records:
        annotation_units+=any(len(b['quantities'])>1 for b in r['bindings'])
    for r in eligible:
        flags.update({k:int(v) for k,v in r['flags'].items()})
        for b in r['bindings']:
            if not b.get('scalar_binding_eligible',False):continue
            for q in b['quantities']:
                # 显式体积后的未知括号百分比只保留审计，不纳入明确标量计数。
                target=types if q['kind'] in supported else ambiguous
                target[b['metric']+'/'+q['kind']]+=1
    return dict(eligible_units=len(eligible),eligible_time_role_flags=dict(flags),
                explicit_quantity_counts=dict(types),retained_unknown_secondary_counts=dict(ambiguous),
                annotated_numeric_units=annotation_units,
                eligible_exact_quote_repeat=repeat_stats(eligible,'quote_sha256'),
                audit_sha256=sha256(ROOT/'audit.json'),records_sha256=sha256(ROOT/'records.jsonl.gz'),
                report_code_sha256=sha256(Path(__file__)),new_model_fits=0,new_vectors=0)


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('build','verify'),required=True);args=p.parse_args()
    result=readout();path=ROOT/'binding_readout.json'
    if args.mode=='build':
        if path.exists():raise FileExistsError('binding readout already exists')
        write_json(path,result)
    else:
        if json.loads(path.read_text())!=result:raise ValueError('binding aggregate replay differs')
        write_json(ROOT/'binding_readout_verification.json',dict(maximum_error=0,report_sha256=sha256(path)))
    print(json.dumps(result))


if __name__=='__main__':main()
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/pft_fact_ledger.py · lines 1–115

<!-- REVIEW_FRAGMENT {"path":"patient_data/pft_fact_ledger.py","part":1,"start_line":1,"end_line":115,"source_sha256":"ac803b3362d77de95ef761bb928cb0ddfa06f7de995e44d9fa3c730ddc06fb50","fragment_sha256":"ac803b3362d77de95ef761bb928cb0ddfa06f7de995e44d9fa3c730ddc06fb50","added_terminal_newline":false} -->
````python
"""已核验肺功能记录→六概念训练集合，保留全部证据，不平均/选最新。"""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
from patient_data.export_mimiciv import sha256
from patient_data.pft_evidence import ROOT as EVIDENCE
from patient_data.pft_table_binding import ROOT as TABLE
from training.v6.train import write_json

ROOT=Path('patient_data/processed/v6_pft_fact_ledger_2026-10-13_r2')
PROTOCOL=Path('docs/V6_PFT_LEDGER_PROTOCOL_2026-10-13.md')
CONCEPTS=('fev1_volume_l','fvc_volume_l','fev1_percent_predicted','fvc_percent_predicted','fev1_fvc_ratio','dlco_percent_predicted')


def canonical(metric,kind,value):
    if kind in ('volume_l','volume_ml') and metric in ('FEV1','FVC'):
        return metric.lower()+'_volume_l',value/(1000 if kind=='volume_ml' else 1),'L'
    if kind=='percent_predicted' and metric in ('FEV1','FVC','DLCO'):
        return metric.lower()+'_percent_predicted',value,'%predicted'
    if kind=='ratio_percent' and metric=='FEV1/FVC':
        return 'fev1_fvc_ratio',value/100,'ratio'
    return None


def add_fact(bag,hadm,subject,metric,kind,value,reference):
    item=canonical(metric,kind,value)
    if item is None:return False
    concept,converted,unit=item;key=(hadm,concept,converted)
    if key not in bag:
        bag[key]=dict(hadm_id=hadm,subject_id=subject,concept=concept,value=converted,unit=unit,
                      measurement_time='unknown',current_discharge_measurement='unknown',evidence=[])
    if bag[key]['subject_id']!=subject:raise ValueError('patient membership differs')
    if reference not in bag[key]['evidence']:bag[key]['evidence'].append(reference)
    return True


def checked_parent(directory):
    audit=json.loads((directory/'audit.json').read_text())
    for tag in ('a','b'):
        p=json.loads((directory/f'verification_{tag}.json').read_text())
        if p['maximum_error'] or p['audit_sha256']!=sha256(directory/'audit.json') or p['records_sha256']!=sha256(directory/'records.jsonl.gz'):
            raise ValueError('parent proof differs')
    for name,digest in audit['source_sha256'].items():
        if sha256(Path(name))!=digest:raise ValueError('protected parent source differs: '+name)
    with gzip.open(directory/'records.jsonl.gz','rt',encoding='utf-8') as f:records=[json.loads(line) for line in f]
    return audit,records


def derive():
    bag={};counts=Counter();sources={}
    for parent in (EVIDENCE,TABLE):
        audit,records=checked_parent(parent);sources.update(audit['source_sha256'])
        for r in records:
            if parent==EVIDENCE:
                use=[(b['metric'],q) for b in r['bindings'] if b.get('scalar_binding_eligible',False) for q in b['quantities']]
                for metric,q in use:
                    reference=dict(parent='lexical',note_id=r['note_id'],original_sha256=r['original_sha256'],
                        quote_start=r['start_char'],quote_end=r['end_char'],quote_sha256=r['quote_sha256'],
                        value_start=r['start_char']+q['value_start'],value_end=r['start_char']+q['value_end'],
                        evidence_start=r['start_char']+q['evidence_start'],evidence_end=r['start_char']+q['evidence_end'],
                        evidence_sha256=q['evidence_sha256'],flags=r['flags'])
                    counts['source_quantities_seen']+=1
                    if add_fact(bag,r['hadm_id'],r['subject_id'],metric,q['kind'],q['value'],reference):counts['accepted_source_quantities']+=1
                    else:counts['unsupported_quantity_preserved_only_in_parent']+=1
            else:
                for q in r['quantities']:
                    counts['source_quantities_seen']+=1
                    if q['role']=='predicted':
                        counts['predicted_absolute_quantity_not_actual']+=1;continue
                    reference=dict(parent='table',note_id=r['note_id'],original_sha256=r['original_sha256'],
                        row_start=r['row_start'],row_end=r['row_end'],row_sha256=r['row_sha256'],
                        header_start=r['header_start'],header_end=r['header_end'],header_sha256=r['header_sha256'],
                        value_start=q['start_char'],value_end=q['end_char'],flags=r['flags'])
                    if add_fact(bag,r['hadm_id'],r['subject_id'],r['metric'],q['kind'],q['value'],reference):counts['accepted_source_quantities']+=1
                    else:counts['unsupported_quantity_preserved_only_in_parent']+=1
        for name in ('audit.json','records.jsonl.gz','verification_a.json','verification_b.json'):
            sources[str(parent/name)]=sha256(parent/name)
    facts=[bag[key] for key in sorted(bag)]
    for r in facts:r['evidence'].sort(key=lambda e:(e['note_id'],e['value_start'],e['parent']))
    per_concept=Counter(r['concept'] for r in facts)
    for p in (PROTOCOL,Path(__file__),Path('training/tests/test_v6_pft_fact_ledger.py')):sources[str(p)]=sha256(p)
    report=dict(status='training_background_PFT_set_not_model_or_gold',concepts=list(CONCEPTS),
        unique_members=len(facts),admissions=len({r['hadm_id'] for r in facts}),patients=len({r['subject_id'] for r in facts}),
        members_per_concept=dict(per_concept),assembly_counts=dict(counts),
        members_with_multiple_evidence=sum(len(r['evidence'])>1 for r in facts),source_sha256=sources,
        outcomes_used=False,high_frequency_used=False,new_model_fits=0,new_vectors=0,normalization_fitted=False,
        limits=['train only; not full cohort model input','all measurement times unknown',
                'no cross-date mean/latest/recovery','evidence multiplicity not count of tests',
                'canonical unit conversion not medical measurement validation'])
    return report,facts


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('build','verify'),required=True);p.add_argument('--tag',choices=('a','b'),default='a');args=p.parse_args()
    report,records=derive()
    if args.mode=='build':
        if ROOT.exists():raise FileExistsError('ledger exists; inspect')
        ROOT.mkdir();write_json(ROOT/'manifest.json',report)
        with gzip.open(ROOT/'facts.jsonl.gz','wt',encoding='utf-8') as f:
            for r in records:f.write(json.dumps(r,sort_keys=True)+'\n')
        write_json(ROOT/'state.json',dict(status='built_pending_parent_record_replays',new_model_fits=0))
    else:
        saved=json.loads((ROOT/'manifest.json').read_text())
        with gzip.open(ROOT/'facts.jsonl.gz','rt',encoding='utf-8') as f:prior=[json.loads(s) for s in f]
        if saved!=report or prior!=records:raise ValueError('independent fact ledger replay differs')
        write_json(ROOT/f'verification_{args.tag}.json',dict(maximum_error=0,members=len(records),
                   manifest_sha256=sha256(ROOT/'manifest.json'),facts_sha256=sha256(ROOT/'facts.jsonl.gz'),parent_records_rebuilt=True))
        if all((ROOT/f'verification_{t}.json').exists() for t in ('a','b')):
            write_json(ROOT/'state.json',dict(status='two_independent_parent_record_replays_verified',new_model_fits=0))
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'}))


if __name__=='__main__':main()
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/pft_inventory.py · lines 1–135

<!-- REVIEW_FRAGMENT {"path":"patient_data/pft_inventory.py","part":1,"start_line":1,"end_line":135,"source_sha256":"3f667a82ea92da79955014cceef98c04815523c8f4b55a576353f3141a383d04","fragment_sha256":"3f667a82ea92da79955014cceef98c04815523c8f4b55a576353f3141a383d04","added_terminal_newline":false} -->
````python
"""训练折肺功能数值词面库存，不是事实gold或可直接入网特征。"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re

from patient_data.export_mimiciv import rows, sha256
from patient_data.text_time_coverage import SCREEN, SNAP, checked_sources
from training.v6.encode_screen_candidates import selected_groups
from training.v6.train import write_json

ROOT = Path('patient_data/processed/v6_pft_inventory_2026-10-13')
PROTOCOL = Path('docs/V6_PFT_INVENTORY_PROTOCOL_2026-10-13.md')
NAME = re.compile(r'\b(?:FEV\s*1\s*/\s*FVC|FEV\s*1|FVC|DLCO)\b', re.I)
VALUE = re.compile(r'^\s*(?:[:=]|(?:was|is|of)\b)?\s*([<>]=?\s*)?(\d+(?:\.\d+)?|\.\d+)(?![\d.])', re.I)
UNIT = re.compile(r'^\s*(%|m[lL]\b|[lL]\b|lit(?:er|re)s?\b)', re.I)
PREDICTED = re.compile(r'\b(?:predicted|pred\.?\b)', re.I)
DATE = re.compile(r'\b(?:\d{1,2}/\d{1,2}(?:/\d{2,4})?|\d{4}-\d{2}-\d{2})\b')
HISTORY = re.compile(r'\b(?:history|prior|previous\w*|baseline|last PFT|old)\b', re.I)
PLAN = re.compile(r'\b(?:plan\w*|recommend\w*|will|should|schedule\w*)\b', re.I)
POST = re.compile(r'\bpost[- ]?(?:bronchodilator|BD)\b', re.I)
PRE = re.compile(r'\bpre[- ]?(?:bronchodilator|BD)\b', re.I)


def candidates(text):
    """只取指标后首数值；保留未知单位，禁止跨下个指标借限定。"""
    names = list(NAME.finditer(text))
    result = []
    for i, match in enumerate(names):
        end = names[i + 1].start() if i + 1 < len(names) else len(text)
        tail = text[match.end():end]
        value = VALUE.match(tail)
        if not value:
            continue
        after = tail[value.end():]
        unit = UNIT.match(after)
        literal = unit.group(1) if unit else None
        metric = re.sub(r'\s+', '', match.group()).upper()
        ranged = bool(re.match(r'^\s*[-–]\s*\d', after))
        # 同指标50字符内线索，只标记，不证明预计值或测量有效性。
        predicted = bool(PREDICTED.search(after[:50]))
        role = ('ratio_percent' if metric == 'FEV1/FVC' else
                'percent_predicted_lexical' if predicted else 'percent_unspecified') if literal == '%' else (
                'volume_ml' if literal and literal.lower() == 'ml' else
                'volume_l' if literal else 'unit_unknown')
        result.append(dict(metric=metric, value=float(value.group(2)),
                           qualifier=(value.group(1) or '').strip(), unit_literal=literal,
                           quantity_role=role, range_ambiguous=ranged,
                           metric_start=match.start(), value_start=match.end()+value.start(2),
                           value_end=match.end()+value.end(2)))
    return result


def derive():
    sources, _ = checked_sources()
    groups, quarantine, manifest_hash = selected_groups(SCREEN)
    units = [u for key, group in groups.items() if key[2] == 'train'
             for u in group if 'pulmonary_function' in u['topics']]
    needed = {u['note_id'] for u in units}
    original = {r['note_id']: r for r in rows(SNAP/'raw/discharge.csv.gz') if r['note_id'] in needed}
    stats = Counter(units=len(units)); section_counts = Counter(); metric_counts = Counter(); roles = Counter()
    mentions = set(); numeric_admissions = set(); numeric_patients = set(); all_notes = set(); records = []
    for u in sorted(units, key=lambda u: (u['note_id'], u['start_char'])):
        raw = original[u['note_id']]
        if raw['text'][u['start_char']:u['end_char']] != u['quote']:
            raise ValueError('original quote differs')
        if sha256_text(raw['text']) != u['original_sha256']:
            raise ValueError('original document hash differs')
        mentions.add(u['hadm_id']); all_notes.add(u['note_id']); section_counts[u['section']] += 1
        values = candidates(u['quote'])
        stats['units_with_numeric_candidate'] += bool(values)
        stats['units_without_adjacent_numeric_candidate'] += not bool(values)
        for v in values:
            metric_counts[v['metric']] += 1; roles[v['quantity_role']] += 1
            stats['range_ambiguous'] += v['range_ambiguous']
        if not values:
            continue
        numeric_admissions.add(u['hadm_id']); numeric_patients.add(u['subject_id'])
        flags = dict(historical_words=bool(HISTORY.search(u['quote'])), plan_words=bool(PLAN.search(u['quote'])),
                     pre_bronchodilator_words=bool(PRE.search(u['quote'])), post_bronchodilator_words=bool(POST.search(u['quote'])))
        stats.update({k: int(v) for k, v in flags.items()})
        dates = DATE.findall(u['quote']); stats['numeric_units_with_date_words'] += bool(dates)
        records.append(dict(note_id=u['note_id'], hadm_id=u['hadm_id'], subject_id=u['subject_id'],
                            section=u['section'], start_char=u['start_char'], end_char=u['end_char'],
                            quote_sha256=sha256_text(u['quote']), original_sha256=u['original_sha256'],
                            flags=flags, date_words=dates, candidates=values))
    sources[str(PROTOCOL)] = sha256(PROTOCOL); sources[str(Path(__file__))] = sha256(Path(__file__))
    sources['training/tests/test_v6_pft_inventory.py'] = sha256(Path('training/tests/test_v6_pft_inventory.py'))
    report = dict(status='training_lexical_inventory_not_validated_measurements', stats=dict(stats),
                  sections=dict(section_counts), metrics=dict(metric_counts), quantity_roles=dict(roles),
                  mentioned_admissions=len(mentions), mentioned_notes=len(all_notes),
                  numeric_candidate_admissions=len(numeric_admissions), numeric_candidate_patients=len(numeric_patients),
                  selected_manifest_sha256=manifest_hash, source_sha256=sources, future_quarantine=quarantine,
                  outcome_labels_used=False, new_model_fits=0, new_vectors=0, high_frequency_used=False,
                  limitations=['selected r4 candidates not exhaustive original text', 'date and role words are not temporal gold',
                               'adjacent number is not verified measurement; missing units stay unknown',
                               'FINAL retrospective notes not guaranteed real-time discharge availability'])
    return report, records


def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--mode', choices=('build', 'verify'), required=True)
    parser.add_argument('--tag', choices=('a', 'b'), default='a'); args = parser.parse_args()
    report, records = derive()
    if args.mode == 'build':
        if ROOT.exists():
            raise FileExistsError('inventory exists; inspect rather than overwrite')
        ROOT.mkdir(); write_json(ROOT/'inventory.json', report)
        with gzip.open(ROOT/'records.jsonl.gz', 'wt', encoding='utf-8') as f:
            for r in records:
                f.write(json.dumps(r, sort_keys=True)+'\n')
        write_json(ROOT/'state.json', dict(status='inventory_pending_independent_verification', new_model_fits=0))
    else:
        saved = json.loads((ROOT/'inventory.json').read_text())
        with gzip.open(ROOT/'records.jsonl.gz', 'rt', encoding='utf-8') as f:
            prior = [json.loads(line) for line in f]
        if saved != report or prior != records:
            raise ValueError('independent PFT inventory differs')
        write_json(ROOT/f'verification_{args.tag}.json', dict(maximum_error=0, records=len(records),
                   inventory_sha256=sha256(ROOT/'inventory.json'), records_sha256=sha256(ROOT/'records.jsonl.gz')))
        if all((ROOT/f'verification_{tag}.json').exists() for tag in ('a', 'b')):
            write_json(ROOT/'state.json', dict(status='inventory_two_independent_processes_verified', new_model_fits=0))
    # 只输出聚合，患者标识／日期原词／原文均留本地。
    print(json.dumps({k: v for k, v in report.items() if k not in ('source_sha256', 'future_quarantine')}))


if __name__ == '__main__':
    main()
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/pft_layout_probe.py · lines 1–58

<!-- REVIEW_FRAGMENT {"path":"patient_data/pft_layout_probe.py","part":1,"start_line":1,"end_line":58,"source_sha256":"e2e98b72fea1cc3f0fe46b5aed15b0438bb7db6aae39f3b29d72ccc8f6bdcf77","fragment_sha256":"e2e98b72fea1cc3f0fe46b5aed15b0438bb7db6aae39f3b29d72ccc8f6bdcf77","added_terminal_newline":false} -->
````python
"""训练PFT未标列候选的表头可行性，只输出去内容化布局聚合。"""
from collections import Counter
import gzip
import json
import re
from patient_data.export_mimiciv import rows, sha256
from patient_data.pft_evidence import ROOT
from patient_data.pft_inventory import sha256_text
from patient_data.text_time_coverage import SCREEN, SNAP

WORDS = re.compile(r'[A-Za-z]+|%|/|\d+(?:\.\d+)?')
WHITELIST = {'pred','predicted','actual','observed','measured','best','ref','reference','lln','uln','units',
             'pre','post','bd','bronchodilator','pct','percent','l','ml','liters','litres','normal',
             'fev','fvc','dlco','ratio'}


def main():
    for tag in ('a','b'):
        p=json.loads((ROOT/f'verification_{tag}.json').read_text())
        if p['maximum_error'] or p['records_sha256']!=sha256(ROOT/'records.jsonl.gz'):
            raise ValueError('PFT evidence not source-verified')
    with gzip.open(ROOT/'records.jsonl.gz','rt',encoding='utf-8') as f:
        records=[r for r in map(json.loads,f) if any(b['status']=='unlabeled_numeric_columns' for b in r['bindings'])]
    need={r['note_id'] for r in records}
    raw={r['note_id']:r['text'] for r in rows(SNAP/'raw/discharge.csv.gz') if r['note_id'] in need}
    with (SCREEN/'documents.jsonl').open(encoding='utf-8') as f:
        docs={r['note_id']:r for r in map(json.loads,f) if r['note_id'] in need}
    headers=Counter(); counts=Counter(); row_shapes=Counter()
    for r in records:
        text=raw[r['note_id']]
        if sha256_text(text)!=r['original_sha256'] or sha256_text(text[r['start_char']:r['end_char']])!=r['quote_sha256']:
            raise ValueError('original probe source differs')
        sec=next(s for s in docs[r['note_id']]['section_spans'] if s['start_char']<=r['start_char']<s['end_char'])
        for b in r['bindings']:
            if b['status']!='unlabeled_numeric_columns':continue
            counts['ambiguous_metric_segments']+=1; at=r['start_char']+b['metric_start']
            line_start=text.rfind('\n',sec['start_char'],at)+1
            line_start=max(line_start,sec['start_char'])
            line_end=text.find('\n',at,sec['end_char'])
            if line_end<0:line_end=sec['end_char']
            line=text[line_start:line_end]
            # 一切数字和非指标词去内容化；不存在病例原文/标识输出。
            shape=re.sub(r'\d+(?:\.\d+)?','N',line)
            shape=re.sub(r'\b[A-Za-z]+\b',lambda m:m.group().upper() if m.group().lower() in WHITELIST else 'W',shape)
            row_shapes[re.sub(r'\s+',' ',shape).strip()[:100]]+=1
            candidates=[]
            for previous in text[sec['start_char']:line_start].splitlines()[-12:]:
                toks=WORDS.findall(previous)
                if not toks or any(t[0].isdigit() for t in toks):continue
                if all(t.lower() in WHITELIST or t in ('%','/') for t in toks):
                    roles=sum(t.lower() in ('pred','predicted','actual','observed','measured','best') for t in toks)
                    if roles>=2:candidates.append(' '.join(t.upper() for t in toks))
            if candidates:
                counts['preceding_header_lexical_candidate']+=1;headers[candidates[-1]]+=1
    print(json.dumps(dict(counts=counts,header_shapes=headers,row_shapes=row_shapes)))


if __name__=='__main__':main()
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/pft_set_interface.py · lines 1–164

<!-- REVIEW_FRAGMENT {"path":"patient_data/pft_set_interface.py","part":1,"start_line":1,"end_line":164,"source_sha256":"bbbfbf5547ee22c11abfcae19ba861bd81d71e0c85aced3358b7438fbaa0c0ae","fragment_sha256":"bbbfbf5547ee22c11abfcae19ba861bd81d71e0c85aced3358b7438fbaa0c0ae","added_terminal_newline":false} -->
````python
"""锁定六概念PFT全原队列集合，保守未来计划隔离和训练尺度。"""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import re
import numpy as np

from patient_data.export_mimiciv import rows,sha256
from patient_data.pft_fact_ledger import CONCEPTS,canonical,ROOT as LEDGER
from patient_data.pft_evidence import bindings,HISTORY
from patient_data.pft_inventory import sha256_text
from patient_data.pft_table_binding import line_ranges,row_values,find_header
from patient_data.text_time_coverage import SCREEN,SNAP,STATE,checked_sources
from training.v6.encode_screen_candidates import selected_groups
from training.v5.numeric_preprocessing import robust_parameters
from training.v6.train import write_json

ROOT=Path('patient_data/processed/v6_pft_set_interface_2026-10-13')
PROTOCOL=Path('docs/V6_PFT_SET_INTERFACE_PROTOCOL_2026-10-13.md')
POST_DISCHARGE=re.compile(r'\b(?:(?:after|following)\s+(?:the\s+)?discharge|post[- ]?discharge|after\s+leaving\s+(?:the\s+)?hospital)\b',re.I)
PFT_NAME=r'(?:PFTs?|spirometr\w*|pulmonary\s+function|FEV\s*1|FVC)'
FUTURE_PFT=re.compile(r'\b(?:will|plan\w*|schedul\w*|recommend\w*|should|to undergo)\b.{0,70}\b'+PFT_NAME+r'\b|\b'+PFT_NAME+r'\b.{0,40}\b(?:planned|scheduled)\b',re.I|re.S)


def temporal_reason(text):
    return 'explicit_post_discharge_words' if POST_DISCHARGE.search(text) else 'future_PFT_plan_words' if FUTURE_PFT.search(text) else None


def normalized(raw,concepts,positions,folds):
    values=np.zeros(len(raw),np.float32);scales={}
    for j,name in enumerate(CONCEPTS):
        mask=(concepts==j)&(folds[positions]=='train')
        if not mask.any():raise ValueError('concept has no training observation: '+name)
        scale=robust_parameters(raw[mask]);scales[name]=scale
        use=concepts==j;values[use]=np.clip((raw[use]-scale['median'])/scale['scale'],-8,8)
    if not np.isfinite(values).all():raise ValueError('PFT standardized values nonfinite')
    return values,scales


def derive():
    sources,_=checked_sources()
    lm=json.loads((LEDGER/'manifest.json').read_text())
    for tag in ('a','b'):
        p=json.loads((LEDGER/f'verification_{tag}.json').read_text())
        if p['maximum_error'] or p['manifest_sha256']!=sha256(LEDGER/'manifest.json') or p['facts_sha256']!=sha256(LEDGER/'facts.jsonl.gz'):
            raise ValueError('formal training PFT ledger proof differs')
    for p,h in lm['source_sha256'].items():
        if sha256(Path(p))!=h:raise ValueError('PFT source changed: '+p)
    with np.load(STATE/'clinical_state.npz') as z:
        base={k:z[k] for k in ('hadm_ids','subject_ids','patient_splits')}
    if len(base['hadm_ids'])!=19401:raise ValueError('original prediction cohort changed')
    lookup={int(h):i for i,h in enumerate(base['hadm_ids'])}
    groups,quarantine,_=selected_groups(SCREEN)
    units=[u for g in groups.values() for u in g if 'pulmonary_function' in u['topics']]
    need={u['note_id'] for u in units};raw={r['note_id']:r for r in rows(SNAP/'raw/discharge.csv.gz') if r['note_id'] in need}
    with (SCREEN/'documents.jsonl').open(encoding='utf-8') as f:docs={r['note_id']:r for r in map(json.loads,f) if r['note_id'] in need}
    bag={};train_counts=Counter();banned_spans={};line_cache={}
    for u in units:
        h=int(u['hadm_id']);pos=lookup[h]
        if int(u['subject_id'])!=int(base['subject_ids'][pos]) or u['patient_split']!=base['patient_splits'][pos]:raise ValueError('PFT member/fold differs')
        text=raw[u['note_id']]['text'];quote=u['quote']
        if sha256_text(text)!=u['original_sha256'] or text[u['start_char']:u['end_char']]!=quote:raise ValueError('PFT quote source differs')
        reason=temporal_reason(quote)
        if reason:
            banned_spans.setdefault(u['note_id'],[]).append((u['start_char'],u['end_char']))
            if u['patient_split']=='train':train_counts[reason]+=1
    def add(u,metric,kind,value,reference):
        item=canonical(metric,kind,value)
        if item is None:return
        name,v,unit=item
        if not np.isfinite(v):raise ValueError('canonical PFT value nonfinite')
        pos=lookup[int(u['hadm_id'])];key=(pos,CONCEPTS.index(name),v)
        if key not in bag:bag[key]=dict(position=pos,concept=name,value=v,unit=unit,evidence=[])
        if reference not in bag[key]['evidence']:bag[key]['evidence'].append(reference)
    for u in sorted(units,key=lambda r:(r['note_id'],r['start_char'])):
        quote=u['quote'];reason=temporal_reason(quote)
        if reason:continue
        text=raw[u['note_id']]['text'];s=docs[u['note_id']]['section_spans'][u['section_ordinal']]
        if not 0<=s['start_char']<=u['start_char']<u['end_char']<=s['end_char']<=len(text):raise ValueError('PFT section source differs')
        flags=[int(u['section']=='past_medical_history'),int(bool(HISTORY.search(quote))),1]
        for b in bindings(quote):
            if b.get('scalar_binding_eligible',False):
                for q in b['quantities']:
                    a=u['start_char']+q['value_start'];z=u['start_char']+q['value_end']
                    ea=u['start_char']+q['evidence_start'];ez=u['start_char']+q['evidence_end']
                    if float(text[a:z])!=q['value'] or sha256_text(text[ea:ez])!=q['evidence_sha256']:raise ValueError('PFT numeric/quantity span differs')
                    ref=dict(parent='lexical',note_id=u['note_id'],original_sha256=u['original_sha256'],quote_start=u['start_char'],quote_end=u['end_char'],
                             quote_sha256=sha256_text(quote),value_start=a,value_end=z,evidence_start=ea,evidence_end=ez,
                             evidence_sha256=q['evidence_sha256'],metadata=flags)
                    add(u,b['metric'],q['kind'],q['value'],ref)
            key=(u['note_id'],u['section_ordinal'])
            if key not in line_cache:line_cache[key]=line_ranges(text,s['start_char'],s['end_char'])
            ranges=line_cache[key];lines=[r[2] for r in ranges];at=u['start_char']+b['metric_start']
            index=next(i for i,(a,z,_) in enumerate(ranges) if a<=at<z);parsed=row_values(lines[index])
            if parsed is None or parsed['metric']!=b['metric']:continue
            header=find_header(lines,index)
            if header is None:continue
            hi,roles=header;rs,re_,line=ranges[index];hs,he,hline=ranges[hi]
            if parsed['trailing_percent'] and roles[-1]!='percent_predicted':continue
            if any(a<re_ and z>hs for a,z in banned_spans.get(u['note_id'],[])):continue
            for role,q in zip(roles,parsed['cells'],strict=True):
                if role=='predicted':continue
                kind=('ratio_percent_predicted' if parsed['metric']=='FEV1/FVC' else 'percent_predicted') if role=='percent_predicted' else (
                    'volume_'+parsed['unit_literal'].lower() if parsed['unit_literal'] else 'unit_unknown')
                a=rs+q['start'];z=rs+q['end']
                if float(text[a:z])!=q['value']:raise ValueError('PFT table cell differs')
                ref=dict(parent='table',note_id=u['note_id'],original_sha256=u['original_sha256'],row_start=rs,row_end=rs+len(line),
                         row_sha256=sha256_text(line),header_start=hs,header_end=hs+len(hline),header_sha256=sha256_text(hline),
                         value_start=a,value_end=z,metadata=flags)
                add(u,parsed['metric'],kind,q['value'],ref)
    facts=[bag[k] for k in sorted(bag)];positions=np.asarray([r['position'] for r in facts],np.int64)
    concepts=np.asarray([CONCEPTS.index(r['concept']) for r in facts],np.int64);raw_values=np.asarray([r['value'] for r in facts],np.float64)
    metadata=np.asarray([np.max([e['metadata'] for e in r['evidence']],axis=0) for r in facts],np.float32)
    values,scales=normalized(raw_values,concepts,positions,base['patient_splits'])
    aggregate=np.zeros((len(base['hadm_ids']),9),np.float32)
    aggregate[positions,concepts]=1
    for i in range(3):np.maximum.at(aggregate[:,6+i],positions,metadata[:,i])
    observed=aggregate[:,:6].any(axis=1)
    arrays=dict(**base,positions=positions,concepts=concepts,raw_values=raw_values,values=values,metadata=metadata,aggregate=aggregate,observed=observed)
    train=base['patient_splits']=='train';selected=train[positions]
    if train.sum()!=13592:raise ValueError('original training cohort changed')
    for r in facts:r['evidence'].sort(key=lambda e:(e['note_id'],e['value_start'],e['parent']))
    for p in (PROTOCOL,Path(__file__),Path('training/tests/test_v6_pft_set_interface.py'),LEDGER/'manifest.json',LEDGER/'facts.jsonl.gz',
              Path('patient_data/pft_table_binding.py'),Path('patient_data/pft_evidence.py'),Path('patient_data/pft_fact_ledger.py'),Path('training/v5/numeric_preprocessing.py')):
        sources[str(p)]=sha256(p)
    report=dict(status='all_original_members_fixed_PFT_rules_training_only_scaling',source_sha256=sources,scales=scales,
        training=dict(admissions=int((train&observed).sum()),patients=len(np.unique(base['subject_ids'][train&observed])),
                      members=int(selected.sum()),temporal_exclusions=dict(train_counts),
                      per_concept={name:int((selected&(concepts==j)).sum()) for j,name in enumerate(CONCEPTS)}),
        cohort=19401,quarantine=quarantine,nontraining_outcomes_used=False,new_model_fits=0,new_vectors=0,high_frequency_used=False,
        limits=['word-rule PFT not medical gold','measurement dates unknown','FINAL retrospective not real-time','no mean/latest/recovery','implicit future information may remain'])
    return arrays,facts,report


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('build','verify'),required=True);p.add_argument('--tag',choices=('a','b'),default='a');args=p.parse_args()
    arrays,facts,report=derive()
    if args.mode=='build':
        if ROOT.exists():raise FileExistsError('PFT interface exists; inspect')
        ROOT.mkdir();np.savez_compressed(ROOT/'interface.npz',**arrays)
        with gzip.open(ROOT/'facts.jsonl.gz','wt',encoding='utf-8') as f:
            for r in facts:f.write(json.dumps(r,sort_keys=True)+'\n')
        report['output_sha256']={name:sha256(ROOT/name) for name in ('interface.npz','facts.jsonl.gz')}
        write_json(ROOT/'manifest.json',report);write_json(ROOT/'state.json',dict(status='built_pending_source_replay',new_model_fits=0))
    else:
        saved=json.loads((ROOT/'manifest.json').read_text());report['output_sha256']=saved['output_sha256']
        if saved!=report:raise ValueError('PFT interface manifest differs')
        for name,h in saved['output_sha256'].items():
            if sha256(ROOT/name)!=h:raise ValueError('PFT interface output differs')
        with np.load(ROOT/'interface.npz') as z:
            if set(z.files)!=set(arrays):raise ValueError('PFT interface keys differ')
            for key,value in arrays.items():np.testing.assert_array_equal(value,z[key])
        with gzip.open(ROOT/'facts.jsonl.gz','rt',encoding='utf-8') as f:prior=[json.loads(s) for s in f]
        if prior!=facts:raise ValueError('PFT interface references differ')
        write_json(ROOT/f'verification_{args.tag}.json',dict(maximum_error=0,manifest_sha256=sha256(ROOT/'manifest.json'),
                  original_source_and_arrays_rebuilt=True,training_scales_rebuilt=True))
        if all((ROOT/f'verification_{t}.json').exists() for t in ('a','b')):
            write_json(ROOT/'state.json',dict(status='two_independent_full_source_replays_verified',new_model_fits=0))
    print(json.dumps(dict(status=report['status'],training=report['training'],cohort=report['cohort'])))


if __name__=='__main__':main()
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/pft_table_binding.py · lines 1–138

<!-- REVIEW_FRAGMENT {"path":"patient_data/pft_table_binding.py","part":1,"start_line":1,"end_line":138,"source_sha256":"c6bb645eef6f5424fd4009042f5b8f1b4db5c5d18b1735d41ff73488663a4c6b","fragment_sha256":"c6bb645eef6f5424fd4009042f5b8f1b4db5c5d18b1735d41ff73488663a4c6b","added_terminal_newline":false} -->
````python
"""同章节明确表头三列PFT绑定，只做训练原源审计。"""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import re

from patient_data.export_mimiciv import rows, sha256
from patient_data.pft_evidence import ROOT as EVIDENCE
from patient_data.pft_inventory import sha256_text
from patient_data.text_time_coverage import SNAP, SCREEN
from training.v6.train import write_json

ROOT=Path('patient_data/processed/v6_pft_table_2026-10-13')
PROTOCOL=Path('docs/V6_PFT_TABLE_PROTOCOL_2026-10-13.md')
HEADER=re.compile(r'%\s*(?:predicted|pred)\b|(?:actual|observed|measured|predicted|pred)\b',re.I)
NUM=r'(?:\d+(?:\.\d+)?|\.\d+)'
ROW=re.compile(r'^\s*(FEV\s*1\s*/\s*FVC|FEV\s*1|FVC|DLCO|TLC|RV)(?:\s*\((L|mL)\))?\s+('+NUM+r')\s+('+NUM+r')\s+('+NUM+r')\s*%?\s*$',re.I)


def header_roles(line):
    matches=list(HEADER.finditer(line))
    if re.sub(r'\s','',HEADER.sub('',line)) or len(matches)!=3:return None
    roles=tuple('percent_predicted' if m.group().startswith('%') else
                'predicted' if m.group().lower() in ('pred','predicted') else 'actual' for m in matches)
    return roles if set(roles)=={'actual','predicted','percent_predicted'} else None


def row_values(line):
    match=ROW.fullmatch(line)
    if match is None:return None
    metric=re.sub(r'\s+','',match.group(1)).upper()
    if match.group(2) and metric not in ('FEV1','FVC'):return None
    return dict(metric=metric,unit_literal=match.group(2),trailing_percent=line.rstrip().endswith('%'),cells=[dict(value=float(match.group(i)),
                start=match.start(i),end=match.end(i)) for i in (3,4,5)])


def find_header(lines,index):
    """不能越过叙述/另一表/不完整行，把任意前表头当当前证据。"""
    for j in range(index-1,max(-1,index-13),-1):
        if not lines[j].strip():continue
        roles=header_roles(lines[j])
        if roles:return j,roles
        if row_values(lines[j]) is None:return None
    return None


def line_ranges(text,start,end):
    result=[];at=start
    for line in text[start:end].splitlines(keepends=True):
        result.append((at,at+len(line),line.rstrip('\r\n')));at+=len(line)
    if at!=end:raise ValueError('section line coverage differs')
    return result


def derive():
    audit=json.loads((EVIDENCE/'audit.json').read_text())
    for tag in ('a','b'):
        p=json.loads((EVIDENCE/f'verification_{tag}.json').read_text())
        if p['maximum_error'] or p['audit_sha256']!=sha256(EVIDENCE/'audit.json') or p['records_sha256']!=sha256(EVIDENCE/'records.jsonl.gz'):
            raise ValueError('PFT evidence proof differs')
    for p,h in audit['source_sha256'].items():
        if sha256(Path(p))!=h:raise ValueError('protected source changed: '+p)
    with gzip.open(EVIDENCE/'records.jsonl.gz','rt',encoding='utf-8') as f:members=[json.loads(s) for s in f]
    need={r['note_id'] for r in members}
    raw={r['note_id']:r['text'] for r in rows(SNAP/'raw/discharge.csv.gz') if r['note_id'] in need}
    with (SCREEN/'documents.jsonl').open(encoding='utf-8') as f:docs={r['note_id']:r for r in map(json.loads,f) if r['note_id'] in need}
    records=[];seen=set();counts=Counter();metrics=Counter();roles_count=Counter();status=Counter();new_admissions=set();all_admissions=set();patients=set()
    prior_explicit={r['hadm_id'] for r in members if any(b.get('scalar_binding_eligible',False) for b in r['bindings'])}
    for r in members:
        text=raw[r['note_id']]
        if sha256_text(text)!=r['original_sha256'] or sha256_text(text[r['start_char']:r['end_char']])!=r['quote_sha256']:
            raise ValueError('original table member differs')
        section=next(s for s in docs[r['note_id']]['section_spans'] if s['start_char']<=r['start_char']<s['end_char'])
        ranges=line_ranges(text,section['start_char'],section['end_char']);lines=[p[2] for p in ranges]
        for b in r['bindings']:
            at=r['start_char']+b['metric_start'];index=next(i for i,(a,z,_) in enumerate(ranges) if a<=at<z)
            parsed=row_values(lines[index])
            if not parsed or parsed['metric']!=b['metric']:
                counts['not_complete_supported_three_column_row']+=1;continue
            found=find_header(lines,index)
            if found is None:
                counts['no_safe_contiguous_explicit_header']+=1;continue
            hi,roles=found;start,end,line=ranges[index];hs,he,header=ranges[hi]
            if parsed['trailing_percent'] and roles[-1]!='percent_predicted':
                counts['percent_marker_column_conflict']+=1;continue
            key=(r['note_id'],start)
            if key in seen:continue
            seen.add(key);qs=[]
            for role,cell in zip(roles,parsed['cells'],strict=True):
                a=start+cell['start'];z=start+cell['end']
                if float(text[a:z])!=cell['value']:raise ValueError('table numeric span differs')
                kind=('ratio_percent_predicted' if parsed['metric']=='FEV1/FVC' else 'percent_predicted') if role=='percent_predicted' else (
                    'volume_'+parsed['unit_literal'].lower() if parsed['unit_literal'] and parsed['metric'] in ('FEV1','FVC') else 'unit_unknown')
                qs.append(dict(role=role,value=cell['value'],kind=kind,start_char=a,end_char=z,
                               unit_literal='%' if role=='percent_predicted' else parsed['unit_literal']))
                roles_count[role+'/'+kind]+=1
            records.append(dict(note_id=r['note_id'],hadm_id=r['hadm_id'],subject_id=r['subject_id'],section=r['section'],
                header_start=hs,header_end=hs+len(header),header_sha256=sha256_text(header),
                row_start=start,row_end=start+len(line),row_sha256=sha256_text(line),original_sha256=r['original_sha256'],
                column_roles=list(roles),metric=parsed['metric'],quantities=qs,measurement_time='unknown',
                previous_binding_status=b['status'],flags=r['flags']))
            metrics[parsed['metric']]+=1;status[b['status']]+=1;all_admissions.add(r['hadm_id']);patients.add(r['subject_id'])
            if r['hadm_id'] not in prior_explicit:new_admissions.add(r['hadm_id'])
    sources={**audit['source_sha256']}
    for p in (PROTOCOL,Path(__file__),Path('training/tests/test_v6_pft_table_binding.py'),EVIDENCE/'audit.json',EVIDENCE/'records.jsonl.gz'):
        sources[str(p)]=sha256(p)
    report=dict(status='training_explicit_table_binding_not_clinical_gold',eligible_table_rows=len(records),
                table_admissions=len(all_admissions),table_patients=len(patients),additional_admissions_with_table_evidence=len(new_admissions),
                row_metrics=dict(metrics),column_quantity_roles=dict(roles_count),previous_binding_status=dict(status),rejection_counts=dict(counts),
                source_sha256=sources,new_model_fits=0,new_vectors=0,high_frequency_used=False,outcomes_used=False,
                limits=['unknown absolute units stay unknown','column words not measurement quality gold',
                        'all measurement dates unknown','ratio percent-predicted not actual ratio','no numeric size column inference'])
    return report,records


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('build','verify'),required=True);p.add_argument('--tag',choices=('a','b'),default='a');args=p.parse_args()
    report,records=derive()
    if args.mode=='build':
        if ROOT.exists():raise FileExistsError('table audit exists; inspect')
        ROOT.mkdir();write_json(ROOT/'audit.json',report)
        with gzip.open(ROOT/'records.jsonl.gz','wt',encoding='utf-8') as f:
            for r in records:f.write(json.dumps(r,sort_keys=True)+'\n')
        write_json(ROOT/'state.json',dict(status='built_pending_two_source_replays',new_model_fits=0))
    else:
        saved=json.loads((ROOT/'audit.json').read_text())
        with gzip.open(ROOT/'records.jsonl.gz','rt',encoding='utf-8') as f:prior=[json.loads(s) for s in f]
        if saved!=report or prior!=records:raise ValueError('independent PFT table binding differs')
        write_json(ROOT/f'verification_{args.tag}.json',dict(maximum_error=0,rows=len(records),
                   audit_sha256=sha256(ROOT/'audit.json'),records_sha256=sha256(ROOT/'records.jsonl.gz')))
        if all((ROOT/f'verification_{t}.json').exists() for t in ('a','b')):
            write_json(ROOT/'state.json',dict(status='two_independent_source_replays_verified',new_model_fits=0))
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'}))


if __name__=='__main__':main()
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/preprocess_events.py · lines 1–473

<!-- REVIEW_FRAGMENT {"path":"patient_data/preprocess_events.py","part":1,"start_line":1,"end_line":473,"source_sha256":"b390d7f1c205a28b0ed297708f5e52691a66927a9055e1a1d427f3caf0f95dfe","fragment_sha256":"b390d7f1c205a28b0ed297708f5e52691a66927a9055e1a1d427f3caf0f95dfe","added_terminal_newline":false} -->
````python
"""从已校验原始底账建立可追溯的静态输入与原始事件，不训练模型。"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

# 允许直接用 python patient_data/preprocess_events.py 启动。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from patient_data.export_mimiciv import NULL, patient_split, rows, sha256
from training.v5.clinical_concepts import MED_NAMES, diagnosis_groups, medication_classes
from training.v5.numeric_preprocessing import robust_parameters


LABS = {
    "50818": ("pco2", "mm Hg"), "50820": ("ph", "units"),
    "50821": ("po2", "mm Hg"), "50862": ("albumin", "g/dL"),
    "50882": ("bicarbonate", "mEq/L"), "50912": ("creatinine", "mg/dL"),
    "51222": ("hemoglobin", "g/dL"), "51301": ("wbc", "K/uL"),
}
BLOOD_GAS = {"50818", "50820", "50821"}
SPECIMENS = ("arterial", "venous", "mixed", "central_venous", "unknown", "conflict")
CHRONIC = ("heart_failure", "cardiac_other", "ckd", "diabetes", "malignancy", "chronic_respiratory")
ACUTE = ("acute_resp_failure", "pneumonia_or_aspiration", "sepsis", "aki")
ADMINISTERED = {"Administered", "Started", "Applied", "Delayed Administered"}
NON_SYSTEMIC_ROUTES = {"TP", "TOP", "TOPICAL", "OP", "OU", "OD", "OS", "OT", "AU", "AD", "AS", "NAS", "NASAL"}
EVENT_FIELDS = ("subject_id", "hadm_id", "patient_split", "kind", "concept", "specimen_type",
                "source_id", "specimen_id", "charttime", "storetime", "elapsed_days",
                "value_raw", "value_unit", "value_normalized", "numeric_mask", "clipped")
REJECT_FIELDS = ("source_table", "source_id", "reason")


@dataclass(frozen=True)
class Admission:
    subject: str
    hadm: str
    start: datetime
    end: datetime
    split: str
    strict: bool


@dataclass(frozen=True)
class Event:
    hadm: str
    kind: str
    concept: str
    specimen_type: str
    source_id: str
    specimen_id: str
    moment: datetime
    stored: datetime
    value: float | None
    unit: str


def csv_writer(path: Path, fields: tuple[str, ...]):
    handle = gzip.open(path, "wt", encoding="utf-8", newline="")
    writer = csv.writer(handle)
    writer.writerow(fields)
    return handle, writer


def parse_time(value: str) -> datetime | None:
    if value in (NULL, ""):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def temporal_reason(row: dict[str, str], admission: Admission) -> str | None:
    moment, stored = parse_time(row["charttime"]), parse_time(row["storetime"])
    if moment is None or stored is None:
        return "missing_or_invalid_timestamp"
    if not admission.start <= moment <= admission.end:
        return "outside_admission"
    if stored > admission.end:
        return "unavailable_at_discharge"
    if stored < moment:
        return "storetime_before_charttime"
    return None


def specimen_name(value: str) -> str:
    value = " ".join(value.strip().upper().rstrip(".").split())
    return {"ART": "arterial", "VEN": "venous", "MIX": "mixed",
            "CENTRAL VENOUS": "central_venous"}.get(value, "unknown")


def specimen_type(candidates: list[tuple[str, datetime, datetime]], discharge: datetime) -> str:
    # 不能从出院之后记录的标本类型回填出院输入。
    available = {name for name, moment, stored in candidates if moment <= stored <= discharge}
    if len(available) > 1:
        return "conflict"
    return next(iter(available), "unknown")


def accepted_lab(row: dict[str, str], admission: Admission,
                 specimen: str) -> tuple[Event | None, str | None]:
    reason = temporal_reason(row, admission)
    if reason:
        return None, reason
    name, unit = LABS[row["itemid"]]
    if row["valueuom"].strip() != unit:
        return None, "unit_mismatch_or_missing"
    try:
        value = float(row["valuenum"])
    except ValueError:
        return None, "missing_or_nonnumeric_value"
    if not math.isfinite(value) or value <= 0:
        return None, "nonfinite_or_nonpositive_value"
    concept = f"{name}:{specimen}" if row["itemid"] in BLOOD_GAS else name
    return Event(admission.hadm, "lab", concept, specimen, row["labevent_id"],
                 row["specimen_id"], parse_time(row["charttime"]), parse_time(row["storetime"]), value, unit), None


def fit_lab_parameters(events: list[Event], admissions: dict[str, Admission]) -> dict[str, dict[str, object]]:
    observed: dict[str, list[float]] = defaultdict(list)
    concepts = {e.concept for e in events if e.kind == "lab"}
    for event in events:
        if event.kind == "lab" and admissions[event.hadm].split == "train":
            observed[event.concept].append(event.value)
    result = {}
    for concept in sorted(concepts):
        if observed[concept]:
            result[concept] = robust_parameters(np.asarray(observed[concept]))
        else:
            # 保留观测，但无训练支持时不把验证／测试的分布用于拟合。
            result[concept] = {"median": 0.0, "scale": 1.0, "method": "no_training_support",
                               "training_observations": 0, "iqr": 0.0}
    return result


def normalized(value: float, parameters: dict[str, object]) -> tuple[float, int, int]:
    if parameters["method"] == "no_training_support":
        return 0.0, 0, 0
    z = (value - float(parameters["median"])) / float(parameters["scale"])
    return float(np.clip(z, -8.0, 8.0)), 1, int(abs(z) > 8.0)


def verify_sources(root: Path) -> dict[str, object]:
    manifest = json.loads((root / "metadata/manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "validated":
        raise ValueError("原始底账尚未通过校验")
    used = ("cohort/admissions", "cohort/patient_splits", "raw/patients", "raw/labevents",
            "raw/diagnoses_icd", "raw/emar", "raw/emar_detail", "outcomes/readmission_death_labels")
    for name in used:
        if sha256(root / (name + ".csv.gz")) != manifest["sources"][name]["sha256"]:
            raise ValueError(f"原始底账文件变化：{name}")
    return {name: manifest["sources"][name]["sha256"] for name in used}


def static_features(root: Path, admissions: dict[str, Admission],
                    diagnoses: dict[str, set[str]], output: Path) -> dict[str, object]:
    patients = {r["subject_id"]: r for r in rows(root / "raw/patients.csv.gz")}
    grouped: dict[str, list[Admission]] = defaultdict(list)
    for admission in admissions.values():
        grouped[admission.subject].append(admission)
    names = ("age", "female", "prior_completed_copd_admissions", "prior_completed_strict_admissions",
             "current_strict_code", *(f"current_code:{c}" for c in CHRONIC + ACUTE),
             *(f"prior_code:{c}" for c in CHRONIC))
    continuous = {"age", "prior_completed_copd_admissions", "prior_completed_strict_admissions"}
    records: dict[str, tuple[list[float], list[int]]] = {}
    for patient_admissions in grouped.values():
        patient_admissions.sort(key=lambda a: (a.start, int(a.hadm)))
        previous: list[Admission] = []
        for admission in patient_admissions:
            past = [p for p in previous if p.end <= admission.start]
            history = set().union(*(diagnoses[p.hadm] & set(CHRONIC) for p in past))
            patient = patients[admission.subject]
            age = float(patient["anchor_age"]) + admission.start.year - int(patient["anchor_year"])
            gender = patient["gender"]
            values = [age, float(gender == "F"), float(len(past)), float(sum(p.strict for p in past)),
                      float(admission.strict), *(float(c in diagnoses[admission.hadm]) for c in CHRONIC + ACUTE),
                      *(float(c in history) for c in CHRONIC)]
            masks = [1, int(gender in ("M", "F")), *([1] * (len(names) - 2))]
            records[admission.hadm] = values, masks
            previous.append(admission)
    parameters = {}
    for index, name in enumerate(names):
        if name in continuous:
            observed = [values[index] for hadm, (values, masks) in records.items()
                        if admissions[hadm].split == "train" and masks[index]]
            parameters[name] = robust_parameters(np.asarray(observed))
    handle, writer = csv_writer(output / "static_features.csv.gz",
                                ("subject_id", "hadm_id", "patient_split", "feature", "value_raw", "value_normalized", "mask", "clipped"))
    try:
        for hadm in sorted(records, key=int):
            admission = admissions[hadm]
            values, masks = records[hadm]
            for index, name in enumerate(names):
                z, clipped = values[index], 0
                if name in parameters:
                    z, _, clipped = normalized(values[index], parameters[name])
                writer.writerow((admission.subject, hadm, admission.split, name, values[index],
                                 z if masks[index] else 0.0, masks[index], clipped))
    finally:
        handle.close()
    return {"feature_names": names, "continuous_parameters": parameters,
            "history_scope": "earlier eligible COPD-coded admissions completed before current admission",
            "diagnosis_zero_means": "no recorded code, not proof of no disease"}


def preprocess(root: Path, output: Path, *, medication_mapping: str = "legacy") -> dict[str, object]:
    if medication_mapping not in ("legacy", "gold_review_v2", "clinical_reviewed_2026_09"):
        raise ValueError("unknown medication mapping version")
    if output.exists():
        raise FileExistsError("拒绝覆盖已有预处理目录")
    input_hashes = verify_sources(root)
    cohort = list(rows(root / "cohort/admissions.csv.gz"))
    eligible_rows = [r for r in cohort if r["discharge_landmark_eligible"] == "t"]
    labels = {r["hadm_id"]: r["subject_id"] for r in rows(root / "outcomes/readmission_death_labels.csv.gz")}
    split_records = {r["subject_id"]: r["patient_split"] for r in rows(root / "cohort/patient_splits.csv.gz")}
    admissions = {}
    for row in eligible_rows:
        subject, hadm = row["subject_id"], row["hadm_id"]
        if labels.get(hadm) != subject or split_records[subject] != patient_split(subject):
            raise ValueError("成员、标签或患者划分不一致")
        admissions[hadm] = Admission(subject, hadm, datetime.fromisoformat(row["admittime"]),
                                    datetime.fromisoformat(row["dischtime"]), split_records[subject], row["is_strict_aecopd"] == "t")
    if set(labels) != set(admissions) or len(admissions) != len(eligible_rows):
        raise ValueError("合格出院成员有重复或标签不完整")
    output.mkdir(parents=True)
    (output / "status.json").write_text('{"status":"processing"}', encoding="utf-8")
    try:
        return _build(root, output, admissions, input_hashes, medication_mapping)
    except Exception:
        (output / "status.json").write_text('{"status":"failed_preserved_for_audit"}', encoding="utf-8")
        raise


def _build(root: Path, output: Path, admissions: dict[str, Admission],
           input_hashes: dict[str, object], medication_mapping: str) -> dict[str, object]:
    specimen_records: dict[tuple[str, str], list[tuple[str, datetime, datetime]]] = defaultdict(list)
    selected_labs: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    for row in rows(root / "raw/labevents.csv.gz"):
        if row["hadm_id"] not in admissions:
            continue
        if row["itemid"] == "52033" and row["value"] not in (NULL, ""):
            moment, stored = parse_time(row["charttime"]), parse_time(row["storetime"])
            if moment is not None and stored is not None:
                specimen_records[(row["subject_id"], row["specimen_id"])].append((specimen_name(row["value"]), moment, stored))
        elif row["itemid"] in LABS:
            selected_labs.append(row)
    rejection_handle, rejection_writer = csv_writer(output / "excluded_records.csv.gz", REJECT_FIELDS)
    events: list[Event] = []
    for row in selected_labs:
        admission = admissions[row["hadm_id"]]
        specimen = specimen_type(specimen_records[(admission.subject, row["specimen_id"])], admission.end) if row["itemid"] in BLOOD_GAS else "not_applicable"
        event, reason = accepted_lab(row, admission, specimen)
        counts["selected_lab_records"] += 1
        if reason:
            counts[f"lab_excluded:{reason}"] += 1
            rejection_writer.writerow(("labevents", row["labevent_id"], reason))
        else:
            events.append(event)
    del selected_labs, specimen_records
    print(json.dumps({"stage": "labs_checked", "accepted": len(events), "counts": counts}), flush=True)
    medication_records: dict[tuple[str, str], tuple[dict[str, str], set[str]]] = {}
    for row in rows(root / "raw/emar.csv.gz"):
        if row["hadm_id"] not in admissions:
            continue
        counts["emar_records_reviewed"] += 1
        categories = (medication_classes(row["medication"], mapping_version=medication_mapping)
                      if row["medication"] != NULL else set())
        if not categories:
            counts["emar_outside_existing_categories"] += 1
            continue
        reason = temporal_reason(row, admissions[row["hadm_id"]])
        if row["event_txt"] not in ADMINISTERED:
            reason = "not_confirmed_administration"
        if reason:
            counts[f"emar_excluded:{reason}"] += 1
            rejection_writer.writerow(("emar", row["emar_id"] + ":" + row["emar_seq"], reason))
        else:
            medication_records[(row["emar_id"], row["emar_seq"])] = row, categories
    routes: dict[tuple[str, str], set[str]] = defaultdict(set)
    product_descriptions: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows(root / "raw/emar_detail.csv.gz"):
        key = row["emar_id"], row["emar_seq"]
        if key in medication_records and row["route"] not in (NULL, ""):
            routes[key].add(row["route"].strip().upper())
        if key in medication_records and row.get("product_description", NULL) not in (NULL, ""):
            product_descriptions[key].add(row["product_description"].strip())
    for key, (row, _candidate_categories) in medication_records.items():
        route_values = routes[key]
        # 途径仅用于排除明显不属于全身/吸入治疗的制剂，不入模。
        if route_values and route_values.issubset(NON_SYSTEMIC_ROUTES):
            counts["emar_excluded:non_systemic_route"] += 1
            rejection_writer.writerow(("emar", ":".join(key), "non_systemic_route"))
            continue
        categories = medication_classes(row["medication"],
                                        product_descriptions=tuple(product_descriptions[key]),
                                        routes=tuple(route_values), mapping_version=medication_mapping)
        if not categories:
            counts["emar_excluded:non_target_formulation"] += 1
            rejection_writer.writerow(("emar", ":".join(key), "non_target_formulation"))
            continue
        if not route_values:
            counts["accepted_emar_route_unknown"] += 1
        for category in sorted(categories):
            events.append(Event(row["hadm_id"], "medication", category, "not_applicable", ":".join(key), NULL,
                                parse_time(row["charttime"]), parse_time(row["storetime"]), None, "not_applicable"))
    rejection_handle.close()
    del medication_records, routes, product_descriptions
    diagnoses: dict[str, set[str]] = defaultdict(set)
    for row in rows(root / "raw/diagnoses_icd.csv.gz"):
        if row["hadm_id"] in admissions:
            diagnoses[row["hadm_id"]].update(diagnosis_groups(row["icd_code"], int(row["icd_version"])))
    static = static_features(root, admissions, diagnoses, output)
    lab_parameters = fit_lab_parameters(events, admissions)
    ordered_admissions = sorted(admissions.values(), key=lambda a: (int(a.subject), a.start, int(a.hadm)))
    admission_index = {a.hadm: index for index, a in enumerate(ordered_admissions)}
    # 排序只是存储规则，张量中无行号位置、全局 gap 或距出院时间。
    events.sort(key=lambda e: (admission_index[e.hadm], e.moment, e.concept, e.source_id))
    concepts = tuple(sorted({e.concept for e in events}))
    concept_index = {name: index + 1 for index, name in enumerate(concepts)}
    n = len(events)
    values = np.zeros(n, dtype=np.float32)
    numeric_mask = np.zeros(n, dtype=bool)
    elapsed = np.zeros(n, dtype=np.float32)
    ids = np.zeros(n, dtype=np.int32)
    per_admission = np.zeros(len(admissions), dtype=np.int64)
    coverage: dict[str, set[str]] = defaultdict(set)
    concept_counts: Counter[str] = Counter()
    clipped: Counter[str] = Counter()
    unsupported: Counter[str] = Counter()
    handle, writer = csv_writer(output / "events.csv.gz", EVENT_FIELDS)
    seen: set[tuple[str, str]] = set()
    try:
        for index, event in enumerate(events):
            key = event.concept, event.source_id
            if key in seen:
                raise ValueError("同一概念原始记录重复；不自动删掉同时间戳不同标本")
            seen.add(key)
            admission = admissions[event.hadm]
            days = (event.moment - admission.start).total_seconds() / 86400
            z, mask, is_clipped = normalized(event.value, lab_parameters[event.concept]) if event.kind == "lab" else (0.0, 0, 0)
            values[index], numeric_mask[index] = z, bool(mask)
            elapsed[index], ids[index] = math.log1p(days), concept_index[event.concept]
            per_admission[admission_index[event.hadm]] += 1
            concept_counts[event.concept] += 1
            coverage[event.kind].add(event.hadm)
            clipped[event.concept] += is_clipped
            if event.kind == "lab" and not mask:
                unsupported[event.concept] += 1
            writer.writerow((admission.subject, event.hadm, admission.split, event.kind, event.concept,
                             event.specimen_type, event.source_id, event.specimen_id,
                             event.moment.isoformat(sep=" "), event.stored.isoformat(sep=" "), days,
                             event.value if event.value is not None else NULL, event.unit, z, mask, is_clipped))
    finally:
        handle.close()
    offsets = np.concatenate((np.array([0], dtype=np.int64), np.cumsum(per_admission)))
    np.savez_compressed(output / "event_arrays.npz", hadm_ids=np.array([int(a.hadm) for a in ordered_admissions]),
                        subject_ids=np.array([int(a.subject) for a in ordered_admissions]),
                        offsets=offsets, concept_ids=ids, values=values, numeric_mask=numeric_mask,
                        elapsed_log_days=elapsed)
    normalization = {"lab": lab_parameters, "static": static, "clip_range": [-8, 8],
                     "fit_scope": "training patients only; lab parameters weighted by observed events",
                     "unknown_specimen": "separate concept, never inferred arterial",
                     "empty_event_admission": "zero-length segment, not a zero-valued observation"}
    (output / "normalization.json").write_text(json.dumps(normalization, indent=2, ensure_ascii=False), encoding="utf-8")
    report = {"status": "preprocessed_and_checked", "patients": len({a.subject for a in admissions.values()}),
              "admissions": len(admissions), "events": n, "lab_events": sum(v for c, v in concept_counts.items() if c in lab_parameters),
              "medication_events": sum(v for c, v in concept_counts.items()
                                       if c in MED_NAMES or c == "tetracycline"),
              "admissions_with_labs": len(coverage["lab"]), "admissions_with_medications": len(coverage["medication"]),
              "admissions_without_events": int((per_admission == 0).sum()), "concept_counts": concept_counts,
              "clipped_events_by_concept": clipped, "lab_events_without_training_support": unsupported,
              "filter_counts": counts, "normalization": normalization,
              "text": "unchanged; no new extraction, encoding, or vector substitution",
              "labels": "unchanged and stored only in original outcomes directory",
              "model_integration": "not connected; new concept dictionary must be explicitly supported before training"}
    validate_outputs(output, admissions, concepts, lab_parameters)
    # 再次检查源文件，防止运行期间源被更换。
    if verify_sources(root) != input_hashes:
        raise ValueError("预处理期间原始底账变化")
    (output / "validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    artifacts = {path.name: sha256(path) for path in output.iterdir() if path.is_file() and path.name != "status.json"}
    manifest = {"status": report["status"], "raw_root": str(root.resolve()), "input_sha256": input_hashes,
                "medication_mapping": medication_mapping,
                "output_sha256": artifacts, "concepts": concepts, "code_sha256": sha256(Path(__file__)),
                "clinical_mapping_code_sha256": sha256(Path(__file__).resolve().parents[1] / "training/v5/clinical_concepts.py"),
                "scaling_code_sha256": sha256(Path(__file__).resolve().parents[1] / "training/v5/numeric_preprocessing.py")}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "status.json").write_text(json.dumps({"status": report["status"]}), encoding="utf-8")
    return {key: report[key] for key in ("status", "patients", "admissions", "events", "lab_events", "medication_events", "admissions_with_labs", "admissions_without_events")}


def validate_outputs(output: Path, admissions: dict[str, Admission], concepts: tuple[str, ...],
                     parameters: dict[str, dict[str, object]]) -> None:
    """从文件重新读回，验证时间、归属、尺度及 CSV／张量一致性。"""
    # NPZ 每次按键索引都会重新解压整列；一次载入，后续逐行查内存。
    with np.load(output / "event_arrays.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    offsets, hadms = arrays["offsets"], arrays["hadm_ids"]
    if len(hadms) != len(admissions) or set(map(str, hadms)) != set(admissions):
        raise ValueError("事件张量住院成员不一致")
    n = len(arrays["values"])
    if offsets[0] != 0 or offsets[-1] != n or np.any(np.diff(offsets) < 0):
        raise ValueError("事件偏移错误")
    for field in ("values", "elapsed_log_days"):
        if not np.isfinite(arrays[field]).all():
            raise ValueError("事件张量包含非有限值")
    train_counts: Counter[str] = Counter()
    observed_counts = np.zeros(len(hadms), dtype=np.int64)
    index_by_hadm = {str(h): i for i, h in enumerate(hadms)}
    count = 0
    for index, row in enumerate(rows(output / "events.csv.gz")):
        admission = admissions[row["hadm_id"]]
        if row["subject_id"] != admission.subject or row["patient_split"] != admission.split:
            raise ValueError("事件患者或划分错误")
        if temporal_reason(row, admission):
            raise ValueError("事件时间审查未通过")
        position = index_by_hadm[row["hadm_id"]]
        if not offsets[position] <= index < offsets[position + 1]:
            raise ValueError("CSV 与住院事件偏移不匹配")
        concept = row["concept"]
        if concepts[int(arrays["concept_ids"][index]) - 1] != concept:
            raise ValueError("事件概念映射错误")
        z = float(row["value_normalized"])
        if not np.isclose(z, arrays["values"][index], atol=1e-6):
            raise ValueError("归一化 CSV 与张量不一致")
        if row["kind"] == "lab":
            expected, mask, clipped = normalized(float(row["value_raw"]), parameters[concept])
            if not math.isclose(z, expected, abs_tol=1e-9) or int(row["numeric_mask"]) != mask or int(row["clipped"]) != clipped:
                raise ValueError("归一化值与训练参数不匹配")
            if admission.split == "train":
                train_counts[concept] += 1
        elif row["numeric_mask"] != "0":
            raise ValueError("药物不应伪造检验数值")
        if bool(int(row["numeric_mask"])) != bool(arrays["numeric_mask"][index]):
            raise ValueError("数值 mask 不匹配")
        expected_time = math.log1p(float(row["elapsed_days"]))
        if not np.isclose(expected_time, arrays["elapsed_log_days"][index], atol=1e-6):
            raise ValueError("相对时间不匹配")
        observed_counts[position] += 1
        count += 1
    if count != n or not np.array_equal(observed_counts, np.diff(offsets)):
        raise ValueError("落盘事件数量不一致")
    if any(train_counts[concept] != p["training_observations"] for concept, p in parameters.items()):
        raise ValueError("归一化拟合范围不等于训练事件")
    static_hadms = set()
    for row in rows(output / "static_features.csv.gz"):
        admission = admissions[row["hadm_id"]]
        if row["subject_id"] != admission.subject or row["patient_split"] != admission.split:
            raise ValueError("静态特征归属错误")
        if not math.isfinite(float(row["value_normalized"])):
            raise ValueError("静态特征非有限")
        static_hadms.add(row["hadm_id"])
    if static_hadms != set(admissions):
        raise ValueError("静态特征成员不完整")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--medication-mapping", choices=("legacy", "gold_review_v2", "clinical_reviewed_2026_09"), default="legacy")
    args = parser.parse_args()
    print(json.dumps(preprocess(args.raw_root, args.output, medication_mapping=args.medication_mapping), ensure_ascii=False))
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/resp_fact_set_interface.py · lines 1–186

<!-- REVIEW_FRAGMENT {"path":"patient_data/resp_fact_set_interface.py","part":1,"start_line":1,"end_line":186,"source_sha256":"382a75045f961878ce177080a2219d33b22c4713356ccd9b97a2b515bbb2bbd3","fragment_sha256":"382a75045f961878ce177080a2219d33b22c4713356ccd9b97a2b515bbb2bbd3","added_terminal_newline":false} -->
````python
"""冻结r2语义，全原队列typed候选事实集合，原模型与源不变。"""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import time
import numpy as np
from patient_data.discharge_resp_rules_r2 import extract
from patient_data.discharge_resp_fact_audit_r2 import LOCAL,ROOT as PARENT,digest,token_bounds
from patient_data.export_mimiciv import rows,sha256
from patient_data.text_time_coverage import SNAP,SCREEN,STATE
from training.v6.encode_screen_candidates import ROUTES as TEXT_ROUTES
from training.v5.numeric_preprocessing import robust_parameters
from training.v6.train import write_json

ROOT=Path('patient_data/processed/v6_resp_fact_set_interface_2026-10-13')
PROTOCOL=Path('docs/V6_RESP_FACT_SET_INTERFACE_PROTOCOL_2026-10-13.md')
CONCEPTS=('dyspnea','dyspnea_improved','oxygen_required','room_air_status','oxygen_weaned_room_air','oxygen_weaned_off','oxygen_failed_wean','oxygen_flow_litres_notation')
ROLES=('explicit_discharge','discharge_section','inpatient_change')
POLARITIES=('affirmed','negated')
UNITS=('not_numeric','L_oxygen_notation','L/min')
CONDITIONS=('not_applicable','unknown','rest','exertion')

def member_key(record,lookup):
    if record['role']=='unknown':return None,'unknown_time_role'
    if record['role'] not in ROLES:raise ValueError('unrecognized fixed role')
    numeric=record['value'] is not None
    if numeric and record['concept']!='oxygen_flow_litres_notation':raise ValueError('unexpected numeric concept')
    if numeric and record['activity_condition']=='ambiguous':return None,'ambiguous_flow_condition'
    unit=record['unit'] if numeric else 'not_numeric'
    condition=record['activity_condition'] if numeric else 'not_applicable'
    value=float(record['value']) if numeric else None
    if numeric and not np.isfinite(value):raise ValueError('nonfinite fact value')
    key=(lookup[int(record['hadm_id'])],CONCEPTS.index(record['concept']),ROLES.index(record['role']),
         POLARITIES.index(record['polarity']),UNITS.index(unit),CONDITIONS.index(condition),value)
    return key,None

def collect(records,lookup):
    bag={};excluded=[]
    for record in records:
        key,reason=member_key(record,lookup)
        if reason:
            excluded.append(dict(reason=reason,evidence=record));continue
        if key not in bag:
            bag[key]=dict(position=key[0],concept=CONCEPTS[key[1]],role=ROLES[key[2]],polarity=POLARITIES[key[3]],
                unit=UNITS[key[4]],condition=CONDITIONS[key[5]],value=key[6],evidence=[])
        if record not in bag[key]['evidence']:bag[key]['evidence'].append(record)
    facts=[bag[k] for k in sorted(bag)]
    for fact in facts:
        fact['evidence'].sort(key=lambda r:(r['note_id'],r['evidence_start_char'],r['evidence_end_char'],r['unit_key']))
    excluded.sort(key=lambda r:(r['evidence']['note_id'],r['evidence']['start'],r['reason']))
    return facts,excluded

def arrays_from(facts,base):
    n=len(facts);out=dict(base)
    out['positions']=np.asarray([r['position'] for r in facts],np.int64)
    for key,names in [('concepts',CONCEPTS),('roles',ROLES),('polarities',POLARITIES),('units',UNITS),('conditions',CONDITIONS)]:
        field={'concepts':'concept','roles':'role','polarities':'polarity','units':'unit','conditions':'condition'}[key]
        out[key]=np.asarray([names.index(r[field]) for r in facts],np.int64)
    out['numeric_present']=np.asarray([r['value'] is not None for r in facts],bool)
    out['raw_values']=np.asarray([r['value'] if r['value'] is not None else 0 for r in facts],np.float64)
    out['values']=np.zeros(n,np.float32);out['numeric_valid']=np.zeros(n,bool);scales={}
    for unit in ('L_oxygen_notation','L/min'):
        use=out['numeric_present']&(out['units']==UNITS.index(unit))
        training=use&(base['patient_splits'][out['positions']]=='train')
        if training.any():
            scale=robust_parameters(out['raw_values'][training]);scales[unit]=scale
            out['values'][use]=np.clip((out['raw_values'][use]-scale['median'])/scale['scale'],-8,8)
            out['numeric_valid'][use]=True
        else:scales[unit]={'method':'no_training_observation','training_observations':0}
    aggregate=np.zeros((len(base['hadm_ids']),len(ROLES)),np.float32)
    aggregate[out['positions'],out['roles']]=1
    out['aggregate']=aggregate;out['observed']=aggregate.any(axis=1)
    return out,scales

def source_records():
    from transformers import AutoTokenizer
    parent=json.loads((PARENT/'audit.json').read_text());source=dict(parent['source_sha256'])
    for p,h in source.items():
        if sha256(Path(p))!=h:raise ValueError('protected r2 source changed: '+p)
    for tag in ('a','b'):
        proof=json.loads((PARENT/f'verification_{tag}.json').read_text())
        if proof['maximum_error']!=0 or proof['audit_sha256']!=sha256(PARENT/'audit.json'):raise ValueError('formal r2 proof differs')
        for name,h in proof['output_sha256'].items():
            if sha256(PARENT/name)!=h:raise ValueError('formal r2 records changed')
    manifest=json.loads((LOCAL/'manifest.json').read_text())
    for name,h in manifest['output_sha256'].items():
        if sha256(LOCAL/name)!=h:raise ValueError('safe token interface changed')
    with np.load(STATE/'clinical_state.npz') as z:base={k:z[k] for k in ('hadm_ids','subject_ids','patient_splits')}
    if len(base['hadm_ids'])!=19401 or (base['patient_splits']=='train').sum()!=13592:raise ValueError('fixed cohort changed')
    lookup={int(h):i for i,h in enumerate(base['hadm_ids'])}
    with gzip.open(LOCAL/'index.jsonl.gz','rt',encoding='utf-8') as f:index=list(map(json.loads,f))
    if len(index)!=9339:raise ValueError('locked full safe index changed')
    with gzip.open(LOCAL/'local_tasks.jsonl.gz','rt',encoding='utf-8') as f:tasks=list(map(json.loads,f))
    wanted={(r['note_id'],r['start_char'],r['end_char']) for r in index};units={}
    with gzip.open(SCREEN/'selected_sentences.jsonl.gz','rt',encoding='utf-8') as f:
        for r in map(json.loads,f):
            key=(r['note_id'],r['start_char'],r['end_char'])
            if key in wanted:
                if key in units:raise ValueError('duplicate safe source unit')
                units[key]=r
    if set(units)!=wanted:raise ValueError('full source units missing')
    needed={r['note_id'] for r in index};raw={r['note_id']:r for r in rows(SNAP/'raw/discharge.csv.gz') if r['note_id'] in needed}
    raw_hash={note:digest(r['text']) for note,r in raw.items()}
    tokenizer=AutoTokenizer.from_pretrained('training/model_cache/Bio_ClinicalBERT',local_files_only=True,use_fast=True)
    if not tokenizer.is_fast:raise ValueError('original offset tokenizer needed')
    records=[];train_rejected=Counter()
    for old in index:
        key=(old['note_id'],old['start_char'],old['end_char']);unit=units[key];pos=lookup[int(old['hadm_id'])]
        if int(old['subject_id'])!=int(base['subject_ids'][pos]) or old['patient_split']!=base['patient_splits'][pos]:raise ValueError('full member/fold differs')
        if any(unit[k]!=old[k] for k in ('unit_key','section','hadm_id','subject_id','patient_split','original_sha256')):raise ValueError('unit lineage differs')
        text=unit['quote'];start,end=key[1:];original=raw[old['note_id']]['text']
        if raw[old['note_id']]['hadm_id']!=old['hadm_id'] or raw[old['note_id']]['subject_id']!=old['subject_id']:
            raise ValueError('original note admission/subject differs')
        if not 0<=start<end<=len(original) or original[start:end]!=text or digest(text)!=old['quote_sha256'] or raw_hash[old['note_id']]!=old['original_sha256']:
            raise ValueError('original full quote offset/hash differs')
        encoded=tokenizer(text,add_special_tokens=False,truncation=False,return_offsets_mapping=True)
        prefix=tokenizer('Clinical topic: '+TEXT_ROUTES[old['route']]+'\n',add_special_tokens=False)['input_ids'];right=0;ids=[]
        for task in tasks[old['task_start']:old['task_end']]:
            a,b=task['token_start'],task['token_end']
            if task['sentence_row']!=old['sentence_row'] or a!=right or task['input_ids']!=[tokenizer.cls_token_id,*prefix,*encoded['input_ids'][a:b],tokenizer.sep_token_id]:
                raise ValueError('actual saved full local task differs')
            ids.extend(encoded['input_ids'][a:b]);right=b
        if ids!=encoded['input_ids'] or right!=old['local_token_count']:raise ValueError('full token tail differs')
        facts,rejected=extract(text,old['section'])
        if old['patient_split']=='train':train_rejected.update(r['reason'] for r in rejected)
        envelope={k:old[k] for k in ('hadm_id','subject_id','note_id','section','unit_key','sentence_row','quote_sha256','original_sha256','containing_rows_and_tokens')}
        envelope.update(patient_split=old['patient_split'],source_start_char=start,source_end_char=end,quote=text)
        for fact in facts:
            a,b=fact['start'],fact['end']
            records.append(dict(**envelope,**fact,evidence_start_char=start+a,evidence_end_char=start+b,
                evidence_sha256=digest(text[a:b]),local_token_span=token_bounds(encoded['offset_mapping'],a,b),measurement_time='unknown',clinically_verified=False))
    with gzip.open(PARENT/'facts.jsonl.gz','rt',encoding='utf-8') as f:prior=list(map(json.loads,f))
    if [r for r in records if r['patient_split']=='train']!=prior:raise ValueError('training semantic records changed during full interface')
    for p in (PARENT/'audit.json',PARENT/'verification_a.json',PARENT/'verification_b.json',PARENT/'facts.jsonl.gz',
              PROTOCOL,Path(__file__),Path('training/tests/test_v6_resp_fact_set_interface.py'),Path('training/v5/numeric_preprocessing.py')):
        source[str(p)]=sha256(p)
    return base,lookup,records,source,train_rejected

def derive():
    base,lookup,records,source,rejected=source_records();facts,excluded=collect(records,lookup);arrays,scales=arrays_from(facts,base)
    train=base['patient_splits']=='train';selected=train[arrays['positions']]
    report=dict(status='fixed_r2_full_cohort_candidate_set_not_clinical_gold',source_sha256=source,scales=scales,cohort=19401,
        vocabularies=dict(concepts=CONCEPTS,roles=ROLES,polarities=POLARITIES,units=UNITS,conditions=CONDITIONS),
        training=dict(members=int(selected.sum()),admissions=int((train&arrays['observed']).sum()),
            patients=len(np.unique(base['subject_ids'][train&arrays['observed']])),
            role_admissions={name:int((train&arrays['aggregate'][:,j].astype(bool)).sum()) for j,name in enumerate(ROLES)},
            per_concept={name:int((selected&(arrays['concepts']==j)).sum()) for j,name in enumerate(CONCEPTS)},
            semantic_rejections=dict(rejected),interface_exclusions=dict(Counter(r['reason'] for r in excluded if r['evidence']['patient_split']=='train'))),
        training_parent_record_replay_error=0,nontraining_outcomes_used=False,high_frequency_used=False,new_fits=0,new_vectors=0,
        limits=['Candidate surface facts not medical gold','Unknown role and ambiguous numeric condition isolated','Inpatient change not discharge terminal state',
            'FINAL retrospective policy, not prospective availability','Reference repetition not measurement frequency','No whole-report encoding or model change'])
    return arrays,facts,excluded,report

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('build','verify'),required=True);parser.add_argument('--tag',choices=('a','b'),default='a')
    args=parser.parse_args();started=time.perf_counter();arrays,facts,excluded,report=derive()
    payload={'facts.jsonl.gz':facts,'excluded.jsonl.gz':excluded}
    if args.mode=='build':
        if ROOT.exists():raise FileExistsError('interface exists, inspect rather than overwrite')
        ROOT.mkdir();np.savez_compressed(ROOT/'interface.npz',**arrays)
        for name,records in payload.items():
            with gzip.open(ROOT/name,'xt',encoding='utf-8') as f:
                for r in records:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        report['output_sha256']={name:sha256(ROOT/name) for name in ('interface.npz',*payload)}
        write_json(ROOT/'manifest.json',report);write_json(ROOT/'state.json',dict(status='built_pending_two_full_source_replays',new_fits=0,new_vectors=0))
    else:
        saved=json.loads((ROOT/'manifest.json').read_text());report['output_sha256']=saved['output_sha256']
        # tuple词表只在JSON边界规范化，不改变源或数组。
        if saved!=json.loads(json.dumps(report)):raise ValueError('full interface manifest differs')
        for name,h in saved['output_sha256'].items():
            if sha256(ROOT/name)!=h:raise ValueError('full interface output changed')
        with np.load(ROOT/'interface.npz') as z:
            if set(z.files)!=set(arrays):raise ValueError('full interface keys differ')
            for key,value in arrays.items():np.testing.assert_array_equal(value,z[key])
        for name,records in payload.items():
            with gzip.open(ROOT/name,'rt',encoding='utf-8') as f:
                if list(map(json.loads,f))!=records:raise ValueError('set/evidence/isolation differs')
        write_json(ROOT/f'verification_{args.tag}.json',dict(maximum_error=0,manifest_sha256=sha256(ROOT/'manifest.json'),
            complete_original_source_and_arrays_rebuilt=True,training_scales_rebuilt=True,training_parent_record_replay_error=0,new_fits=0))
        if all((ROOT/f'verification_{tag}.json').exists() for tag in ('a','b')):
            write_json(ROOT/'state.json',dict(status='two_independent_full_source_replays_verified',new_fits=0,new_vectors=0,goal_achieved=False))
    print(json.dumps(dict(status=report['status'],training=report['training'],scales=report['scales'],seconds=time.perf_counter()-started)),flush=True)

if __name__=='__main__':main()
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/resp_micro_context.py · lines 1–290

<!-- REVIEW_FRAGMENT {"path":"patient_data/resp_micro_context.py","part":1,"start_line":1,"end_line":290,"source_sha256":"8eb5bdd9d468486cd2abb5529fe724a786ff1eb36664c95e1b0b83143fadf813","fragment_sha256":"8eb5bdd9d468486cd2abb5529fe724a786ff1eb36664c95e1b0b83143fadf813","added_terminal_newline":false} -->
````python
"""小型呼吸支持/微生物候选派生；只读来源、不训练、不解释为因果或定植。"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import json
import math
from pathlib import Path
import re
import time
import numpy as np
from patient_data.export_mimiciv import rows, sha256
from training.v5.numeric_preprocessing import robust_parameters

LOWER = {'SPUTUM','BRONCHOALVEOLAR LAVAGE','BRONCHIAL WASHINGS','MINI-BAL',
         'TRACHEAL ASPIRATE','BRONCHIAL BRUSH','BRONCHIAL BRUSH - PROTECTED'}
MICRO_NAMES = ('named_lower_airway_organism','pseudomonas_aeruginosa','haemophilus_influenzae',
               'streptococcus_pneumoniae','target_organism_resistance','respiratory_virus_positive')
NUMERIC_ITEMS = {'220210':'respiratory_rate','220277':'spo2','223834':'oxygen_flow',
                 '227287':'additional_oxygen_flow','227582':'bipap_oxygen_flow',
                 '223835':'fio2_percent','227579':'bipap_epap','227580':'bipap_ipap'}
NUMERIC_SUMMARY = ('first_selected','minimum_selected','maximum_selected','latest_selected','latest_age_log_days')
SUPPORT_NAMES = ('recorded_invasive_ventilation','recorded_noninvasive_ventilation',
                 'recorded_endotracheal_tube','recorded_bipap_mask','recorded_cpap_mask',
                 'recorded_high_flow_nasal_cannula','recorded_tracheostomy_tube',
                 'recorded_extubation','recorded_repeat_intubation')


def timestamp_interval(exact, date):
    """仅日期记录代表整日，不伪造午夜为精确发生时间。"""
    try:
        if exact not in ('',r'\N'):
            t=datetime.fromisoformat(exact); return t,t
        if date not in ('',r'\N'):
            t=datetime.fromisoformat(date).replace(hour=0,minute=0,second=0,microsecond=0)
            return t,t+timedelta(days=1)-timedelta(microseconds=1)
    except ValueError: pass
    return None


def available_micro(row, start, end):
    draw=timestamp_interval(row['charttime'],row['chartdate'])
    stored=timestamp_interval(row['storetime'],row['storedate'])
    if draw is None: return False,'draw_time_missing'
    if not start<=draw[0]<=draw[1]<=end: return False,'draw_outside_or_boundary_uncertain'
    if stored is None: return False,'result_time_missing'
    if not draw[1]<=stored[0]: return False,'draw_result_order_uncertain'
    if stored[1]>end: return False,'result_after_discharge_or_boundary_uncertain'
    return True,'available_by_discharge'


def micro_scope(row):
    spec=row['spec_type_desc'].strip().upper(); test=row['test_name'].strip().upper()
    if spec in LOWER: return 'lower'
    if ('RESPIRATORY VIRAL' in spec or 'INFLUENZA A/B' in spec or 'RESPIRATORY VIRAL' in test
            or ('INFLUENZA' in test and spec not in ('BLOOD CULTURE','URINE'))): return 'resp_viral'
    if spec=='PLEURAL FLUID': return 'pleural_not_lower_airway'
    return 'other'


def invalid_micro(row):
    org=row['org_name'].strip().upper(); comment=row['comments'].upper()
    if org in ('CANCELLED','CANCELED','PENDING','NEGATIVE'): return org!='NEGATIVE'
    return (bool(re.search(r'TEST CANCEL|TEST WAS CANCEL|NOT PERFORMED|SPECIMEN REJECT',comment))
            or ('INVALID' in comment and 'CONTAMIN' in comment))


def target_organism(org):
    name=org.strip().upper()
    # 精确属种/经核验别名，避免 NON-FERMENTER, NOT PSEUDOMONAS ... 误阳性。
    if name=='PSEUDOMONAS AERUGINOSA': return 1
    if name=='HAEMOPHILUS INFLUENZAE' or name.startswith('HAEMOPHILUS INFLUENZAE,'): return 2
    if name=='STREPTOCOCCUS PNEUMONIAE': return 3
    return None


def viral_positive(org, comment):
    virus=r'INFLUENZA|RESPIRATORY SYNCYTIAL|PARAINFLUENZA|ADENOVIRUS|RHINOVIRUS|METAPNEUMOVIRUS'
    org,comment=org.upper(),comment.upper()
    named=(org not in ('',r'\N','NEGATIVE') and bool(re.search(virus,org))
           and 'NEGATIVE' not in org and 'NOT DETECTED' not in org)
    explicit=bool(re.search(r'(?:POSITIVE FOR|(?<!NOT )DETECTED:)\s*(?:'+virus+r')',comment))
    return named or explicit


def union_hours(intervals):
    if not intervals: return 0.
    total=0.; left,right=sorted(intervals)[0]
    for start,end in sorted(intervals)[1:]:
        if end<start: raise ValueError('negative duration')
        if start<=right: right=max(right,end)
        else: total+=(right-left).total_seconds()/3600; left,right=start,end
    if right<left: raise ValueError('negative duration')
    return total+(right-left).total_seconds()/3600


def canonical_resp(item, value, unit):
    try: v=float(value)
    except (TypeError,ValueError): return None
    if not math.isfinite(v): return None
    if item=='220210': return v if unit=='insp/min' and 0<v<=100 else None
    if item=='220277': return v if unit=='%' and 0<v<=100 else None
    if item=='223835':
        if unit not in ('',r'\N','%'): return None
        if .20<=v<=1: return v*100
        return v if 20<=v<=100 else None
    if item in ('227579','227580'): return v if unit=='cmH2O' and 0<=v<=60 else None
    return v if unit=='L/min' and 0<=v<=100 else None


def build(snapshot, supplement, state_dir, output):
    if output.exists(): raise FileExistsError(output)
    started=time.perf_counter()
    sm=json.loads((supplement/'metadata/manifest.json').read_text(encoding='utf-8'))
    st=json.loads((state_dir/'manifest.json').read_text(encoding='utf-8'))
    parent=json.loads((snapshot/'metadata/manifest.json').read_text(encoding='utf-8'))
    if (sm['status']!='validated_restricted_source_export_not_model_validated'
            or parent['status']!='validated' or not sm['cohort_membership_exactly_matches_parent']
            or sha256(snapshot/'metadata/manifest.json')!=sm['parent_snapshot_manifest_sha256']
            or sha256(state_dir/'clinical_state.npz')!=st['output_sha256']['clinical_state.npz']):
        raise ValueError('source lineage mismatch')
    sources={}
    for name,meta in sm['sources'].items():
        path=supplement/'raw'/(name+'.csv.gz'); digest=sha256(path)
        if digest!=meta['sha256']: raise ValueError('source modified')
        sources[name]=digest
    if sha256(snapshot/'cohort/admissions.csv.gz')!=sm['parent_cohort_sha256']: raise ValueError('cohort changed')
    with np.load(state_dir/'clinical_state.npz',allow_pickle=False) as archive:
        state={k:archive[k] for k in archive.files}
    hadms=[str(h) for h in state['hadm_ids']]; positions={h:i for i,h in enumerate(hadms)}
    if len(hadms)!=19401 or len(positions)!=19401: raise ValueError('member drift')
    members={r['hadm_id']:r for r in rows(snapshot/'cohort/admissions.csv.gz')}
    admissions={h:(datetime.fromisoformat(members[h]['admittime']),datetime.fromisoformat(members[h]['dischtime'])) for h in hadms}
    for h,i in positions.items():
        if int(members[h]['subject_id'])!=state['subject_ids'][i]: raise ValueError('identity mismatch')
    def source_rows(name):
        count=0
        for r in rows(supplement/'raw'/(name+'.csv.gz')):
            count+=1
            if r['hadm_id'] not in members or r['subject_id']!=members[r['hadm_id']]['subject_id']: raise ValueError('raw association mismatch')
            if r['hadm_id'] in positions: yield r
        if count!=sm['sources'][name]['rows']: raise ValueError('source row count changed')
    n=len(hadms); audit=defaultdict(Counter)
    micro=np.zeros((n,6),dtype=np.float32); micro_mask=np.zeros_like(micro,dtype=bool)
    micro_raw=np.zeros(n,dtype=bool); lower_available=np.zeros(n,dtype=bool); viral_available=np.zeros(n,dtype=bool)
    unique_ids=set(); isolates=set(); available_rows=0; susceptible_isolates=set(); resistant_isolates=set()
    for r in source_rows('microbiologyevents'):
        if r['microevent_id'] in unique_ids: raise ValueError('duplicate microevent_id')
        unique_ids.add(r['microevent_id'])
        scope=micro_scope(r); audit['micro_specimens'][r['spec_type_desc']]+=1
        if scope not in ('lower','resp_viral'): audit['micro']['non_target_specimen_rows']+=1; continue
        i=positions[r['hadm_id']]; micro_raw[i]=True; audit['micro']['respiratory_candidate_rows']+=1
        ok,reason=available_micro(r,*admissions[r['hadm_id']]); audit['micro'][reason]+=1
        if not ok: continue
        if invalid_micro(r): audit['micro']['cancelled_or_invalid_result_rows']+=1; continue
        # 仅培养/明确呼吸病毒检查；涂片不能替代培养病原结果。
        if scope=='lower' and 'CULTURE' not in r['test_name'].upper():
            audit['micro']['nonculture_lower_airway_rows']+=1; continue
        org=r['org_name'].strip().upper(); available_rows+=1
        key=(r['hadm_id'],r['micro_specimen_id'],r['test_seq'],r['test_itemid'],org,r['isolate_num'])
        if key in isolates: audit['micro']['rows_collapsed_same_specimen_test_isolate']+=1
        isolates.add(key)
        if scope=='lower':
            lower_available[i]=True
            if org not in ('',r'\N','NEGATIVE','MIXED BACTERIAL FLORA'):
                micro[i,0]=1
            target=target_organism(org)
            if target is not None:
                micro[i,target]=1
                if r['interpretation'] in ('S','I','R'):
                    susceptible_isolates.add(key); micro_mask[i,4]=True
                    if r['interpretation']=='R': resistant_isolates.add(key); micro[i,4]=1
        else:
            viral_available[i]=True
            # 定向检查的明确阳性，不从泛化评论或孤立 POSITIVE 推断病毒。
            if viral_positive(org,r['comments']): micro[i,5]=1
    micro_mask[:,:4]=lower_available[:,None]; micro_mask[:,5]=viral_available
    micro_state=np.where(lower_available|viral_available,2,np.where(micro_raw,1,0)).astype(np.int8)
    icu=np.zeros(n,dtype=bool); stay_to_hadm={}
    for r in source_rows('icustays'):
        if r['stay_id'] in stay_to_hadm: raise ValueError('duplicate stay_id')
        stay_to_hadm[r['stay_id']]=r['hadm_id']; icu[positions[r['hadm_id']]]=True
    support=np.zeros((n,len(SUPPORT_NAMES)),dtype=np.float32); support_mask=np.zeros_like(support,dtype=bool)
    durations=np.zeros((n,2),dtype=np.float64); duration_mask=np.zeros_like(durations,dtype=bool)
    intervals=defaultdict(list); intubations=defaultdict(set); extubations=defaultdict(set); orders=set()
    for r in source_rows('respiratory_procedureevents'):
        h=r['hadm_id']; i=positions[h]
        if stay_to_hadm.get(r['stay_id'])!=h: raise ValueError('procedure/ICU stay mismatch')
        try: start,end,stored=(datetime.fromisoformat(r[k]) for k in ('starttime','endtime','storetime'))
        except ValueError: audit['procedure']['invalid_time_rows']+=1; continue
        if not admissions[h][0]<=start<=end<=admissions[h][1] or not start<=stored<=admissions[h][1]:
            audit['procedure']['unavailable_or_outside_rows']+=1; continue
        if r['statusdescription'] not in ('FinishedRunning','Stopped'):
            audit['procedure']['unconfirmed_status_rows']+=1; continue
        key=(h,r['stay_id'],r['itemid'],r['orderid'],start,end)
        if key in orders: audit['procedure']['duplicate_rows_collapsed']+=1; continue
        orders.add(key); audit['procedure']['accepted_rows']+=1
        if r['itemid'] in ('225792','225794'):
            j=0 if r['itemid']=='225792' else 1
            support[i,j]=1; support_mask[i,j]=True; intervals[h,j].append((start,end)); duration_mask[i,j]=True
        if r['itemid']=='224385': intubations[h].add(start)
        if r['itemid'] in ('225468','225477','227194'): extubations[h].add(start); support[i,7]=1; support_mask[i,7]=True
    for h,ts in intubations.items():
        i=positions[h]; support[i,8]=len(ts)>1; support_mask[i,8]=True
    for (h,j),spans in intervals.items(): durations[positions[h],j]=union_hours(spans)
    numeric=defaultdict(list); seen=set(); respiratory_rows=np.zeros(n,dtype=bool)
    devices={'ENDOTRACHEAL TUBE':2,'BIPAP MASK':3,'CPAP MASK':4,'HIGH FLOW NASAL CANNULA':5,'TRACHEOSTOMY TUBE':6}
    for r in source_rows('respiratory_chartevents'):
        h=r['hadm_id']; i=positions[h]
        if stay_to_hadm.get(r['stay_id'])!=h: raise ValueError('chart/ICU stay mismatch')
        t,s=(datetime.fromisoformat(r[k]) for k in ('charttime','storetime'))
        if not admissions[h][0]<=t<=s<=admissions[h][1]: raise ValueError('export time restriction failed')
        key=tuple(r[k] for k in ('hadm_id','stay_id','charttime','storetime','itemid','value','valuenum','valueuom'))
        if key in seen: audit['chartevents']['identical_rows_collapsed']+=1; continue
        seen.add(key); respiratory_rows[i]=True
        if r['selection_role']=='state_record':
            audit['device_values'][r['itemid']+':'+r['value']]+=1
            if r['itemid']=='226732':
                name=r['value'].strip().upper()
                if name not in ('',r'\N','OTHER'): support_mask[i,2:7]=True
                if name in devices: support[i,devices[name]]=1
            continue
        value=canonical_resp(r['itemid'],r['valuenum'],r['valueuom'])
        if value is None: audit['chartevents']['rejected_value_or_unit:'+r['itemid']]+=1; continue
        numeric[h,r['itemid']].append((t,value)); audit['chartevents']['accepted_numeric_anchors:'+r['itemid']]+=1
    raw=np.zeros((n,8,5),dtype=np.float64); masks=np.zeros_like(raw,dtype=bool)
    for (h,item),observations in numeric.items():
        k=list(NUMERIC_ITEMS).index(item); i=positions[h]; by_time=defaultdict(list)
        for t,v in observations: by_time[t].append(v)
        obs=[(t,float(np.median(v))) for t,v in sorted(by_time.items())]
        raw[i,k]=[obs[0][1],min(v for _,v in obs),max(v for _,v in obs),obs[-1][1],
                  math.log1p((admissions[h][1]-obs[-1][0]).total_seconds()/86400)]
        masks[i,k]=True
    scaled=np.zeros_like(raw); normalization={}; train=state['patient_splits']=='train'
    for k,name in enumerate(NUMERIC_ITEMS.values()):
        for f,label in enumerate(NUMERIC_SUMMARY):
            valid=train&masks[:,k,f]
            if not valid.any(): normalization[name+':'+label]={'method':'no_training_observation'}; masks[:,k,f]=False; continue
            p=robust_parameters(raw[valid,k,f]); normalization[name+':'+label]=p
            scaled[:,k,f]=np.where(masks[:,k,f],np.clip((raw[:,k,f]-p['median'])/p['scale'],-8,8),0)
    def coverage(present):
        return {'admissions':int(present.sum()),'patients':len(set(state['subject_ids'][present].tolist())),
                'missing_admissions':int((~present).sum())}
    def detection(values, masks):
        return {'positive_admissions':int((values>0).sum()),
                'positive_patients':len(set(state['subject_ids'][values>0].tolist())),
                'source_observed_admissions':int(masks.sum()),'unknown_admissions':int((~masks).sum())}
    report={'status':'resp_micro_context_validated_not_trained','admissions':n,
            'parent_supplement_manifest_sha256':sha256(supplement/'metadata/manifest.json'),
            'parent_state_manifest_sha256':sha256(state_dir/'manifest.json'),
            'source_sha256':sources,'code_sha256':sha256(Path(__file__)),'audit':{k:dict(v) for k,v in audit.items()},
            'coverage':{'ICU_stay':coverage(icu),'respiratory_chart_source':coverage(respiratory_rows),
                        'lower_airway_culture_available':coverage(lower_available),'respiratory_viral_test_available':coverage(viral_available)},
            'micro_coverage':{name:detection(micro[:,i],micro_mask[:,i]) for i,name in enumerate(MICRO_NAMES)},
            'numeric_coverage':{name:coverage(masks[:,i,0]) for i,name in enumerate(NUMERIC_ITEMS.values())},
            'support_coverage':{name:detection(support[:,i],support_mask[:,i]) for i,name in enumerate(SUPPORT_NAMES)},
            'micro_source_state_counts':{str(i):int((micro_state==i).sum()) for i in range(3)},
            'unique_available_specimen_test_isolates':len(isolates),'available_micro_rows':available_rows,
            'target_isolates_with_SIR':len(susceptible_isolates),'target_isolates_with_R':len(resistant_isolates),
            'micro_names':MICRO_NAMES,'resp_numeric_names':list(NUMERIC_ITEMS.values()),'resp_numeric_summary_names':NUMERIC_SUMMARY,
            'support_names':SUPPORT_NAMES,'duration_names':('recorded_invasive_hours','recorded_noninvasive_hours'),
            'limitations':['Positive culture does not establish infection versus colonization or causal prognosis',
                'Result time is the last known update; late final results may exclude earlier preliminary information',
                'No linked source does not prove no test/support outside ICU or outside this linked hospitalization',
                'Support zero means no positive evidence in the retained source, not proven clinical absence',
                'Last ICU observation is not the discharge state; numeric extrema concern valid selected anchors only',
                'Procedure durations are union of confirmed recorded intervals; do not infer indication or complete duration',
                'CPAP mask is not automatically acute COPD NIV; airway tube is not automatically invasive ventilation'],
            'elapsed_seconds':round(time.perf_counter()-started,2)}
    for name,digest in sources.items():
        if sha256(supplement/'raw'/(name+'.csv.gz'))!=digest: raise ValueError('source changed during build')
    output.mkdir(parents=True)
    np.savez_compressed(output/'resp_micro_context.npz',hadm_ids=state['hadm_ids'],subject_ids=state['subject_ids'],patient_splits=state['patient_splits'],
        micro_values=micro,micro_masks=micro_mask,micro_source_state=micro_state,icu_stay_present=icu,
        resp_numeric_raw=raw,resp_numeric_scaled=scaled,resp_numeric_masks=masks,
        support_values=support,support_masks=support_mask,recorded_duration_hours=durations,recorded_duration_masks=duration_mask)
    with np.load(output/'resp_micro_context.npz',allow_pickle=False) as archive:
        if len(archive['hadm_ids'])!=n or not np.isfinite(archive['resp_numeric_scaled']).all(): raise ValueError('roundtrip failed')
        if not np.array_equal(archive['micro_values'],micro): raise ValueError('micro roundtrip failed')
    (output/'normalization.json').write_text(json.dumps(normalization,indent=2),encoding='utf-8')
    report['output_sha256']={name:sha256(output/name) for name in ('resp_micro_context.npz','normalization.json')}
    (output/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('status','coverage','micro_source_state_counts','elapsed_seconds')},indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('snapshot','supplement','state_dir','output'): parser.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    args=parser.parse_args(); build(args.snapshot,args.supplement,args.state_dir,args.output)
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/respiratory_anchor_sampling_audit.py · lines 1–35

<!-- REVIEW_FRAGMENT {"path":"patient_data/respiratory_anchor_sampling_audit.py","part":1,"start_line":1,"end_line":35,"source_sha256":"472ddc33ad199ad4cd1f3e37888fbebbb3397ae393f6fa97436e428a9666e5f0","fragment_sha256":"472ddc33ad199ad4cd1f3e37888fbebbb3397ae393f6fa97436e428a9666e5f0","added_terminal_newline":false} -->
````python
"""原锚点采样密度聚合审计；只计训练患者，不改变模型输入或源码合同。"""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from patient_data.export_mimiciv import rows,sha256
from patient_data.respiratory_transition import ROOT,SOURCE,ITEMS,NAMES
from training.v6.train import write_json


def main():
    with np.load(ROOT/'respiratory_transition.npz') as z:
        train={str(h) for h,s in zip(z['hadm_ids'],z['patient_splits']) if s=='train'}
    counts={};selected=defaultdict(int)
    path=SOURCE/'raw/respiratory_chartevents.csv.gz'
    for row in rows(path):
        if row['hadm_id'] not in train or row['itemid'] not in ITEMS or row['selection_role']=='state_record':continue
        key=(row['hadm_id'],row['stay_id'],row['itemid']);number=int(row['available_numeric_rows'])
        if key in counts and counts[key]!=number:raise ValueError('inconsistent original candidate count')
        counts[key]=number;selected[key]+=1
    report={}
    for item,name in zip(ITEMS,NAMES):
        keys=[key for key in counts if key[2]==item];number=[counts[k] for k in keys]
        chosen=[selected[k] for k in keys]
        report[name]=dict(train_hospital_icu_item_groups=len(keys),original_candidate_rows=sum(number),exported_anchor_rows=sum(chosen),
            candidate_rows_per_ICU_item_quantiles=np.quantile(number,[0,.1,.5,.9,1]).tolist(),
            exported_anchor_rows_per_ICU_item_quantiles=np.quantile(chosen,[0,.1,.5,.9,1]).tolist())
    result=dict(status='training_anchor_sampling_audited',source_sha256=sha256(path),by_concept=report,
        limits=['candidate counts are SQL finite raw values before clinical cleaning',
            'underlying ICU observations may be dense; exported state anchors are sparse, not a full low-frequency native series',
            'unexported rows are not available to pair from this snapshot; no missing oxygen inferred as room air'])
    write_json(ROOT/'anchor_sampling_audit.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/respiratory_device_feasibility.py · lines 1–52

<!-- REVIEW_FRAGMENT {"path":"patient_data/respiratory_device_feasibility.py","part":1,"start_line":1,"end_line":52,"source_sha256":"93554ce9719a2e0ca94bb3b7116338cefadea64f56da055b611501ef1dc532b8","fragment_sha256":"93554ce9719a2e0ca94bb3b7116338cefadea64f56da055b611501ef1dc532b8","added_terminal_newline":false} -->
````python
"""下一数据路线只读可用性：全量保留设备状态vs选中数值锚点；不推断室内空气。"""
from collections import Counter,defaultdict
from datetime import datetime
import json
import numpy as np
from patient_data.export_mimiciv import rows,sha256
from patient_data.respiratory_transition import ROOT,SOURCE,ITEMS,canonical_resp
from training.v6.train import write_json


def main():
    with np.load(ROOT/'respiratory_transition.npz') as z:training={str(h) for h,s in zip(z['hadm_ids'],z['patient_splits']) if s=='train'}
    device=defaultdict(lambda:defaultdict(set));spo=defaultdict(list);counts=Counter();cross_stay=defaultdict(set)
    path=SOURCE/'raw/respiratory_chartevents.csv.gz'
    for row in rows(path):
        h=row['hadm_id']
        if h not in training:continue
        t=datetime.fromisoformat(row['charttime']);stay=row['stay_id']
        if row['itemid']=='226732':
            name=row['value'].strip();counts[name]+=1;device[h,stay][t].add(name)
        elif row['itemid']=='220277' and canonical_resp(row['itemid'],row['valuenum'],row['valueuom']) is not None:
            spo[h].append((t,stay))
        if row['itemid'] in ITEMS and row['selection_role']!='state_record':cross_stay[h,row['itemid'],t].add(stay)
    coverage=Counter();compressed=[];conflicts=0
    for key,states in device.items():
        ordered=sorted(states.items());changes=0;previous=None
        for _,names in ordered:
            if len(names)>1:conflicts+=1
            state=tuple(sorted(names))
            if state!=previous:changes+=1;previous=state
        compressed.append(changes)
    for h,records in spo.items():
        records=sorted(set(records))
        for name,target in [('first',records[0]),('latest',records[-1])]:
            t,stay=target;candidates=[(x,v) for x,v in device[h,stay].items() if abs((x-t).total_seconds())<=3600]
            if not candidates:coverage[name+':no_device_within1h']+=1;continue
            _,values=min(candidates,key=lambda r:(abs((r[0]-t).total_seconds()),r[0]))
            coverage[name+':paired']+=1
            for v in values:coverage[name+':'+v]+=1
    result=dict(status='read_only_training_device_feasibility_not_new_features',source_sha256=sha256(path),
        training_device_rows=sum(counts.values()),device_values=dict(counts),selected_spo2_pairing=dict(coverage),
        hospital_icu_device_groups=len(compressed),compressed_state_count_quantiles=np.quantile(compressed,[0,.1,.5,.9,1]).tolist(),
        simultaneous_device_conflict_timestamps=conflicts,
        numeric_same_hospital_item_time_multiple_stays=sum(len(v)>1 for v in cross_stay.values()),
        caveats=['None is explicit recorded device label, not proof of room air or FiO2=21%',
            'device transition is not automatically extubation, COPD NIV indication or treatment benefit',
            'conflicting same-time labels must not be arbitrarily sorted as transitions',
            'no new model fitting, labels or confirmation/test inspected'])
    write_json(ROOT/'device_feasibility.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/respiratory_transition.py · lines 1–194

<!-- REVIEW_FRAGMENT {"path":"patient_data/respiratory_transition.py","part":1,"start_line":1,"end_line":194,"source_sha256":"1133411673cc4e75d70d95335f22f9fe6cc8f32793c690f36a4ecf6541f10003","fragment_sha256":"1133411673cc4e75d70d95335f22f9fe6cc8f32793c690f36a4ecf6541f10003","added_terminal_newline":false} -->
````python
"""已保存的四呼吸项目锚点时序派生；不访问数据库/原文，不推断因果恢复。"""
from collections import Counter,defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
import argparse
import numpy as np
from patient_data.export_mimiciv import rows,sha256
from patient_data.resp_micro_context import canonical_resp
from training.v5.numeric_preprocessing import robust_parameters
from training.v6.train import write_json

SNAP=Path('patient_data/mimiciv_copd/snapshot_2026-09-27')
SOURCE=Path('patient_data/mimiciv_copd/supplement_resp_micro_2026-10-09')
STATE=Path('patient_data/processed/v6_clinical_state_2026-10-09_r2')
ROOT=Path('patient_data/processed/v6_respiratory_transition_2026-10-13')
PROTOCOL=Path('docs/V6_RESPIRATORY_TRANSITION_PROTOCOL_2026-10-13.md')
ITEMS=('220210','220277','223834','223835')
NAMES=('respiratory_rate','spo2','oxygen_flow','fio2_percent')
CHANNELS=('first_selected','directional_extreme_selected','latest_selected','latest_minus_first','latest_minus_extreme',
          'log_span_days','log_latest_age_days','log_extreme_to_latest_days','log_first_admission_offset_days')
FLAGS=('observed','different_time_retest','latest_at_directional_extreme')
PAIR_NAMES=('spo2_first_matched_oxygen_flow','spo2_latest_matched_oxygen_flow','matched_oxygen_flow_change',
            'spo2_first_matched_fio2_percent','spo2_latest_matched_fio2_percent','matched_fio2_change')


def icu_contains(stay,t):
    if any(stay[k] in ('',r'\N',r'\\N') for k in ('intime','outtime')):
        return None
    return datetime.fromisoformat(stay['intime'])<=t<=datetime.fromisoformat(stay['outtime'])


def collapse(observations):
    grouped=defaultdict(list)
    for t,v,stay in observations:grouped[t].append((v,stay))
    return [(t,float(np.median([v for v,_ in obs])), min(s for _,s in obs)) for t,obs in sorted(grouped.items())]


def summarize(obs,start,end,low=False):
    raw=np.zeros(9);mask=np.zeros(9,bool);flags=np.zeros(3)
    if not obs:return raw,mask,flags
    obs=collapse(obs);first,last=obs[0],obs[-1]
    worst=min(v for _,v,_ in obs) if low else max(v for _,v,_ in obs)
    peak=max(t for t,v,_ in obs if v==worst)
    repeated=len(obs)>1;after=peak<last[0]
    raw[:3]=[first[1],worst,last[1]];mask[:3]=True
    raw[3]=last[1]-first[1] if repeated else 0;mask[3]=repeated
    raw[4]=last[1]-worst if after else 0;mask[4]=after
    raw[5]=math.log1p((last[0]-first[0]).total_seconds()/86400);mask[5]=repeated
    raw[6]=math.log1p((end-last[0]).total_seconds()/86400);mask[6]=True
    raw[7]=math.log1p((last[0]-peak).total_seconds()/86400);mask[7]=after
    raw[8]=math.log1p((first[0]-start).total_seconds()/86400);mask[8]=True
    flags[:]=[1,repeated,peak==last[0]]
    if not np.isfinite(raw).all() or (raw[[5,6,7,8]]<0).any():raise ValueError('invalid observation time order')
    return raw,mask,flags


def nearest(target,obs):
    t,_,stay=target
    eligible=[o for o in obs if o[2]==stay and abs((o[0]-t).total_seconds())<=3600]
    return min(eligible,key=lambda o:(abs((o[0]-t).total_seconds()),o[0])) if eligible else None


def support_pairs(spo,flow,fio):
    raw=np.zeros(6);mask=np.zeros(6,bool)
    if not spo:return raw,mask
    spo=collapse(spo)
    for i,other in enumerate((flow,fio)):
        a,b=nearest(spo[0],collapse(other)),nearest(spo[-1],collapse(other))
        if a:raw[3*i]=a[1];mask[3*i]=True
        if b:raw[3*i+1]=b[1];mask[3*i+1]=True
        if a and b and spo[0][0]<spo[-1][0] and spo[0][2]==spo[-1][2]:
            raw[3*i+2]=b[1]-a[1];mask[3*i+2]=True
    return raw,mask


def normalize(raw,mask,train):
    out=np.zeros_like(raw);parameters={}
    for col in np.ndindex(raw.shape[1:]):
        key=(slice(None),)+col;valid=train&mask[key]
        if not valid.any():parameters[str(col)]={'method':'no_training_observation'};continue
        p=robust_parameters(raw[key][valid]);parameters[str(col)]=p
        out[key]=np.where(mask[key],np.clip((raw[key]-p['median'])/p['scale'],-8,8),0)
    return out,parameters


def derive():
    sm=json.loads((SOURCE/'metadata/manifest.json').read_text());st=json.loads((STATE/'manifest.json').read_text())
    parent=json.loads((SNAP/'metadata/manifest.json').read_text())
    if (sm['status']!='validated_restricted_source_export_not_model_validated' or parent['status']!='validated' or
        sha256(SNAP/'metadata/manifest.json')!=sm['parent_snapshot_manifest_sha256']):raise ValueError('parent lineage differs')
    for name in ('respiratory_chartevents','icustays','d_items'):
        if sha256(SOURCE/'raw'/f'{name}.csv.gz')!=sm['sources'][name]['sha256']:raise ValueError('source changed: '+name)
    if sha256(STATE/'clinical_state.npz')!=st['output_sha256']['clinical_state.npz']:raise ValueError('state differs')
    if sha256(SNAP/'cohort/admissions.csv.gz')!=sm['parent_cohort_sha256']:raise ValueError('cohort changed')
    with np.load(STATE/'clinical_state.npz') as z:hadms=z['hadm_ids'];people=z['subject_ids'];splits=z['patient_splits']
    cohort={r['hadm_id']:r for r in rows(SNAP/'cohort/admissions.csv.gz')};by_hadm={str(h):i for i,h in enumerate(hadms)}
    stays={r['stay_id']:r for r in rows(SOURCE/'raw/icustays.csv.gz')}
    obs=defaultdict(list);audit=Counter();units=Counter();roles=Counter();seen=set();stays_per=defaultdict(set)
    for r in rows(SOURCE/'raw/respiratory_chartevents.csv.gz'):
        if r['itemid'] not in ITEMS or r['selection_role']=='state_record':continue
        units[r['itemid']+':'+r['valueuom']]+=1
        if r['hadm_id'] not in by_hadm:audit['retained_only_raw_ineligible']+=1;continue
        row=cohort[r['hadm_id']];stay=stays[r['stay_id']]
        if row['subject_id']!=r['subject_id'] or stay['hadm_id']!=r['hadm_id'] or stay['subject_id']!=r['subject_id']:
            raise ValueError('subject/hospital/ICU mismatch')
        start,end=datetime.fromisoformat(row['admittime']),datetime.fromisoformat(row['dischtime'])
        t,s=datetime.fromisoformat(r['charttime']),datetime.fromisoformat(r['storetime'])
        if not start<=t<=s<=end:raise ValueError('unavailable observation after discharge')
        location=icu_contains(stay,t)
        if location is None:audit['unknown_icu_interval_but_hospital_time_available']+=1
        elif not location:audit['outside_icu_interval_but_within_hospital']+=1
        value=canonical_resp(r['itemid'],r['valuenum'],r['valueuom'])
        if value is None:
            audit['rejected:'+r['itemid']]+=1
            for role in r['selection_role'].split('|'):roles['rejected:'+r['itemid']+':'+role]+=1
            continue
        key=tuple(r[k] for k in ('hadm_id','stay_id','charttime','storetime','itemid','valuenum','valueuom'))
        if key in seen:audit['duplicate_source_rows']+=1;continue
        seen.add(key);obs[r['hadm_id'],r['itemid']].append((t,value,r['stay_id']));stays_per[r['hadm_id']].add(r['stay_id'])
        audit['accepted:'+r['itemid']]+=1
        if r['warning'] not in ('',r'\N','0','f'):audit['warning_retained:'+r['itemid']]+=1
    n=len(hadms);raw=np.zeros((n,4,9));masks=np.zeros_like(raw,bool);flags=np.zeros((n,4,3))
    paired=np.zeros((n,6));paired_mask=np.zeros_like(paired,bool);spans=defaultdict(list);ages=defaultdict(list);first_offsets=defaultdict(list)
    for h,i in by_hadm.items():
        r=cohort[h];start,end=datetime.fromisoformat(r['admittime']),datetime.fromisoformat(r['dischtime'])
        for j,item in enumerate(ITEMS):
            raw[i,j],masks[i,j],flags[i,j]=summarize(obs[h,item],start,end,low=item=='220277')
            if splits[i]=='train' and flags[i,j,0]:
                age=np.expm1(raw[i,j,6]);ages[item].append(age);spans[item].append(np.expm1(raw[i,j,5]));first_offsets[item].append(np.expm1(raw[i,j,8]))
        paired[i],paired_mask[i]=support_pairs(obs[h,'220277'],obs[h,'223834'],obs[h,'223835'])
    scaled,params=normalize(raw,masks,splits=='train');pair_scaled,pair_params=normalize(paired,paired_mask,splits=='train')
    feature=np.concatenate((np.concatenate((scaled,masks,flags),axis=2).reshape(n,84),pair_scaled,paired_mask),axis=1).astype(np.float32)
    availability=np.zeros_like(feature);timing=np.zeros_like(feature)
    for j in range(4):
        availability[:,j*21+18:j*21+20]=flags[:,j,:2]
        for col in (5,6,8):timing[:,j*21+col]=scaled[:,j,col];timing[:,j*21+9+col]=masks[:,j,col]
    availability[:,90:]=paired_mask;timing+=availability
    arr=dict(hadm_ids=hadms,subject_ids=people,patient_splits=splits,features=feature,availability=availability,timing=timing,
        observed=flags[:,:,0].any(1),numeric_raw=raw,numeric_masks=masks,numeric_flags=flags,support_pairs_raw=paired,support_pairs_masks=paired_mask)
    def quantile(v):return np.quantile(v,[0,.1,.5,.9,1]).tolist() if v else None
    summary={name:dict(training_observed=int(flags[splits=='train',j,0].sum()),training_retest=int(flags[splits=='train',j,1].sum()),
        latest_age_days_quantiles=quantile(ages[item]),selected_span_days_quantiles=quantile(spans[item]),first_admission_offset_days_quantiles=quantile(first_offsets[item]),
        training_latest_age_over2days=int(sum(v>2 for v in ages[item]))) for j,(item,name) in enumerate(zip(ITEMS,NAMES))}
    report=dict(source_manifest_sha256=sha256(SOURCE/'metadata/manifest.json'),parent_state_manifest_sha256=sha256(STATE/'manifest.json'),
        source_sha256={name:sm['sources'][name]['sha256'] for name in ('respiratory_chartevents','icustays','d_items')},
        normalization=params,pair_normalization=pair_params,audit=dict(audit),unit_counts=dict(units),rejected_selection_roles=dict(roles),
        coverage=summary,training_any_source=int(arr['observed'][splits=='train'].sum()),
        pairing_training_counts={name:int(paired_mask[splits=='train',j].sum()) for j,name in enumerate(PAIR_NAMES)},
        multiple_icu_stays_with_accepted_values=sum(len(v)>1 for v in stays_per.values()),
        names=list(NAMES),numeric_channels=list(CHANNELS),flags=list(FLAGS),pair_names=list(PAIR_NAMES),features=96,
        limitations=['selected anchors only; selection preceded cleaning','late ICU values are not discharge state',
            'latest_at_extreme does not mean pathologically abnormal','oxygen support missing is not room air',
            'paired support within60min does not establish causality or true recovery','availability can reflect measurement practice'])
    return arr,report


def build(*,finish_metadata=False):
    if ROOT.exists() and not finish_metadata:raise FileExistsError('derived directory exists; inspect')
    if finish_metadata and (not ROOT.exists() or {p.name for p in ROOT.iterdir()}!={'respiratory_transition.npz'}):
        raise ValueError('metadata recovery only permitted for exact preserved NPZ-only scene')
    arr,report=derive()
    if not finish_metadata:
        ROOT.mkdir(parents=True)
        np.savez_compressed(ROOT/'respiratory_transition.npz',**arr)
    else:
        with np.load(ROOT/'respiratory_transition.npz') as z:
            for key,v in arr.items():np.testing.assert_array_equal(z[key],v)
    report.update(status='derived_roundtrip_pending_independent_source_rebuild',output_sha256=sha256(ROOT/'respiratory_transition.npz'),
        protocol_sha256=sha256(PROTOCOL),code_sha256={p:sha256(Path(p)) for p in ('patient_data/respiratory_transition.py',
            'patient_data/resp_micro_context.py','training/v5/numeric_preprocessing.py','training/tests/test_v6_respiratory_transition.py')})
    with np.load(ROOT/'respiratory_transition.npz') as z:
        for key,v in arr.items():np.testing.assert_array_equal(z[key],v)
    write_json(ROOT/'manifest.json',report);print(json.dumps(dict(status=report['status'],coverage=report['coverage'],pairing=report['pairing_training_counts'],audit=report['audit'])),flush=True)


def verify():
    arr,report=derive();saved=json.loads((ROOT/'manifest.json').read_text())
    for key,v in report.items():
        if saved[key]!=v:raise ValueError('metadata/source reconstruction differs: '+key)
    for p,digest in saved['code_sha256'].items():
        if sha256(Path(p))!=digest:raise ValueError('derivation source differs')
    if sha256(PROTOCOL)!=saved['protocol_sha256'] or sha256(ROOT/'respiratory_transition.npz')!=saved['output_sha256']:
        raise ValueError('derived/protocol hash differs')
    with np.load(ROOT/'respiratory_transition.npz') as z:
        for key,v in arr.items():np.testing.assert_array_equal(z[key],v)
    proof=dict(status='independent_source_rebuild_verified',maximum_array_error=0.,arrays=len(arr),output_sha256=saved['output_sha256'])
    write_json(ROOT/'independent_verification.json',proof);print(json.dumps(proof),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('build','verify','finish-metadata'),required=True);a=p.parse_args()
    verify() if a.mode=='verify' else build(finish_metadata=a.mode=='finish-metadata')
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/review_cxr_samples.py · lines 1–48

<!-- REVIEW_FRAGMENT {"path":"patient_data/review_cxr_samples.py","part":1,"start_line":1,"end_line":48,"source_sha256":"3c3af50ed621262ca677d7896ff8b7df2223d5b5388781feabd53b377b09c537","fragment_sha256":"3c3af50ed621262ca677d7896ff8b7df2223d5b5388781feabd53b377b09c537","added_terminal_newline":false} -->
````python
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
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/same_specimen_audit.py · lines 1–239

<!-- REVIEW_FRAGMENT {"path":"patient_data/same_specimen_audit.py","part":1,"start_line":1,"end_line":239,"source_sha256":"e1fec3d4cd7e466ae38caa83488fcacea7ef3b85e18ccc39eff3e28191a79585","fragment_sha256":"e1fec3d4cd7e466ae38caa83488fcacea7ef3b85e18ccc39eff3e28191a79585","added_terminal_newline":false} -->
````python
"""训练折同标本血气信息审计；复用父接受行，不读取原raw检验或结局。"""
from collections import Counter, defaultdict
from datetime import datetime
import argparse
import gzip
import json
from pathlib import Path
import time
import numpy as np
from patient_data.export_mimiciv import rows, sha256
from patient_data.lab_recovery_features import abnormal_severity, PAIRED_AXIS
from patient_data.v6_lab_state import summarize
from patient_data.preprocess_events import LABS

ROOT=Path('training_runs/v6_same_specimen_audit_2026-10-14_r2')
PARENT=Path('training_runs/v6_core_lab_value_audit_2026-10-14')
STATE=Path('patient_data/processed/v6_clinical_state_2026-10-09_r2')
EVENTS=Path('patient_data/processed/mimiciv_copd_events_clinical_reviewed_2026-09-29')
SNAP=Path('patient_data/mimiciv_copd/snapshot_2026-09-27')
PROTOCOL=Path('docs/V6_SAME_SPECIMEN_AUDIT_PROTOCOL_2026-10-14.md')
AXES=('arterial_low_ph','arterial_high_pco2')
CONCEPTS=('ph:arterial','pco2:arterial')
# 复用真实既定字典，pH在本源明确使用units，不以无量纲推断为空。
UNITS=(LABS['50820'][1], LABS['50818'][1])


def bind(evidence, event, bounds, subject):
    """精确来源绑定，父接受规则与标本/单位/时间边界不可放宽。"""
    j=AXES.index(evidence['axis']);start,end=bounds
    if (event['kind']!='lab' or event['concept']!=CONCEPTS[j] or event['specimen_type']!='arterial'
            or event['patient_split']!='train' or event['subject_id']!=str(subject)
            or event['hadm_id']!=evidence['hadm_id'] or event['source_id']!=evidence['labevent_id']
            or event['specimen_id']!=evidence['specimen_id']
            or event['value_unit']!=UNITS[j] or evidence['unit']!=UNITS[j]):
        raise ValueError('source/specimen/type/unit mismatch')
    ec,es=(datetime.fromisoformat(event[k]) for k in ('charttime','storetime'))
    pc,ps=(datetime.fromisoformat(evidence[k]) for k in ('charttime','storetime'))
    value,lo,hi=(evidence[k] for k in ('value','lower','upper'))
    if (ec!=pc or es!=ps or not start<=pc<=ps<=end
            or float(event['value_raw'])!=value or not all(np.isfinite((value,lo,hi)))
            or value<=0 or not lo<hi):raise ValueError('numeric/time mismatch')
    day=(pc-start).total_seconds()/86400
    if abs(day-float(event['elapsed_days']))>1e-12:raise ValueError('elapsed time differs')
    severity=abnormal_severity(AXES[j],value,lo,hi,severity_cap=None)
    if abs(severity-evidence['severity'])>1e-12:raise ValueError('parent severity differs')
    return dict(evidence,day=day,subject_id=str(subject),concept=CONCEPTS[j],specimen_type='arterial')


def pair_slot(records):
    """同住院同标本集合；重复不平均，标本缺失不按时间补归属。"""
    if any(r['specimen_id'] in ('','\\N') for r in records):return None,'missing_specimen'
    if len({(r['hadm_id'],r['specimen_id']) for r in records})!=1:return None,'different_admission_or_specimen'
    slots={a:[r for r in records if r['axis']==a] for a in AXES}
    if any(len(v)>1 for v in slots.values()):return None,'duplicate_component'
    if any(len(v)==0 for v in slots.values()):return None,'missing_component'
    p,c=(slots[a][0] for a in AXES)
    if abs((datetime.fromisoformat(p['charttime'])-datetime.fromisoformat(c['charttime'])).total_seconds())>3600:
        return None,'chart_gap_over_one_hour'
    return dict(hadm_id=p['hadm_id'],specimen_id=p['specimen_id'],subject_id=p['subject_id'],
                values=[p['value'],c['value']],days=[p['day'],c['day']],
                day=max(p['day'],c['day']),severity=min(p['severity'],c['severity']),
                components=[p,c]),'paired'


def latest_sets(pairs,hadms):
    by=defaultdict(list)
    for p in pairs:by[p['hadm_id']].append(p)
    out=[]
    for h in hadms:
        pp=by[str(h)]
        if not pp:out.append(());continue
        last=max(p['day'] for p in pp)
        out.append(tuple(sorted({tuple(p['values']) for p in pp if p['day']==last})))
    return out


def collisions(raw,physical,subjects):
    groups=defaultdict(list)
    for i,v in enumerate(physical):
        if v:groups[tuple(raw[i].reshape(-1))].append(i)
    shared=[ix for ix in groups.values() if len({int(subjects[i]) for i in ix})>1
            and len({physical[i] for i in ix})>1]
    return dict(groups=len(shared),admissions=sum(len(ix) for ix in shared),
                patients=len({int(subjects[i]) for ix in shared for i in ix})),shared


def derive():
    started=time.perf_counter();pa=json.loads((PARENT/'audit.json').read_text())
    proof=json.loads((PARENT/'verification.json').read_text());sm=json.loads((STATE/'manifest.json').read_text())
    em=json.loads((EVENTS/'manifest.json').read_text())
    if (proof['maximum_error']!=0 or proof['audit_sha256']!=sha256(PARENT/'audit.json')
            or em['status']!='preprocessed_and_checked' or sm['processed_manifest_sha256']!=sha256(EVENTS/'manifest.json')):
        raise ValueError('parent lineage proof differs')
    for name,h in pa['output_sha256'].items():
        if sha256(PARENT/name)!=h:raise ValueError('parent accepted evidence changed')
    for p in ('patient_data/preprocess_events.py','patient_data/lab_recovery_features.py','patient_data/v6_lab_state.py'):
        if sha256(Path(p))!=pa['source_sha256'][p]:raise ValueError('inherited acceptance code changed')
    if sha256(STATE/'clinical_state.npz')!=sm['output_sha256']['clinical_state.npz']:raise ValueError('state changed')
    if sha256(EVENTS/'events.csv.gz')!=em['output_sha256']['events.csv.gz'] or em['output_sha256']['events.csv.gz']!=sm['source_hashes']['processed_events']:
        raise ValueError('accepted events changed')
    admission_path=SNAP/'cohort/admissions.csv.gz'
    if sha256(admission_path)!=sm['source_hashes']['cohort/admissions']:raise ValueError('cohort changed')
    with np.load(STATE/'clinical_state.npz',allow_pickle=False) as z:
        state={k:z[k] for k in ('hadm_ids','subject_ids','patient_splits','axis_names','lab_axis_raw')}
    if len(state['hadm_ids'])!=19401 or len(np.unique(state['subject_ids']))!=5351:raise ValueError('cohort drift')
    ti=np.flatnonzero(state['patient_splits']=='train');hadms=state['hadm_ids'][ti];subjects=state['subject_ids'][ti]
    if len(ti)!=13592:raise ValueError('train drift')
    lookup={str(h):i for i,h in enumerate(hadms)};bounds={}
    for r in rows(admission_path):
        h=r['hadm_id']
        if h not in lookup:continue
        if r['subject_id']!=str(subjects[lookup[h]]) or r['discharge_landmark_eligible']!='t':raise ValueError('train membership differs')
        bounds[h]=tuple(datetime.fromisoformat(r[k]) for k in ('admittime','dischtime'))
    if len(bounds)!=len(ti):raise ValueError('missing boundaries')
    selected={}
    with gzip.open(PARENT/'local_evidence.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            if r['axis'] not in AXES:continue
            if r['hadm_id'] not in lookup or r['labevent_id'] in selected:raise ValueError('parent accepted membership duplicate')
            selected[r['labevent_id']]=r
    slots=defaultdict(list);bound=[];bicarbonate=[];counts=Counter();seen=set()
    for r in rows(EVENTS/'events.csv.gz'):
        h=r['hadm_id']
        if h not in lookup or r['kind']!='lab':continue
        if r['concept']=='bicarbonate':bicarbonate.append(r);continue
        if r['concept'] not in CONCEPTS:continue
        counts['processed_train_arterial_rows']+=1
        if r['source_id'] not in selected:counts['not_parent_reference_validated']+=1;continue
        if r['source_id'] in seen:raise ValueError('accepted source duplicate')
        seen.add(r['source_id']);v=bind(selected[r['source_id']],r,bounds[h],subjects[lookup[h]])
        bound.append(v)
        if v['specimen_id'] in ('','\\N'):counts['accepted_missing_specimen_rows']+=1
        else:slots[h,v['specimen_id']].append(v)
    if seen!=set(selected):raise ValueError('parent gas source not bound completely')
    pairs=[];rejected=[];pc=Counter()
    for key,rr in sorted(slots.items()):
        p,why=pair_slot(rr);pc[why]+=1
        if p:pairs.append(p)
        else:rejected.append(dict(hadm_id=key[0],specimen_id=key[1],reason=why,components=rr))
    # 重建三个既有9通道，不从配对中把单轴缺测补成正常。
    samples=defaultdict(list)
    for r in bound:samples[r['hadm_id'],r['axis']].append((r['day'],r['severity']))
    for p in pairs:samples[p['hadm_id'],PAIRED_AXIS].append((p['day'],p['severity']))
    axis_names=(*AXES,PAIRED_AXIS);raw=np.zeros((len(ti),3,9));errors={}
    for j,axis in enumerate(axis_names):
        for i,h in enumerate(hadms):
            duration=(bounds[str(h)][1]-bounds[str(h)][0]).total_seconds()/86400
            raw[i,j],_=summarize(samples.get((str(h),axis),[]),duration)
        old=list(state['axis_names']).index(axis)
        error=float(np.abs(raw[:,j]-state['lab_axis_raw'][ti,old]).max());errors[axis]=error
        if error>1e-12:raise ValueError('existing nine-channel replay differs: '+axis)
    physical=latest_sets(pairs,hadms);joint_coll,jc=collisions(raw[:,2],physical,subjects)
    triple_coll,tc=collisions(raw,physical,subjects)
    groups=defaultdict(set)
    for p in pairs:groups[p['severity']].add(tuple(p['values']))
    charts=defaultdict(set);zero=defaultdict(set)
    for p in pairs:
        charts[p['hadm_id']].add(p['day'])
        if p['severity']==0:zero[p['hadm_id']].add(tuple(p['values']))
    pair_keys={(p['hadm_id'],p['specimen_id']) for p in pairs};bc=Counter()
    for r in bicarbonate:
        if r['specimen_id'] in ('','\\N'):bc['missing_specimen_rows']+=1;continue
        if (r['hadm_id'],r['specimen_id']) in pair_keys:
            bc['exact_same_specimen_rows']+=1
            bc['unit:'+r['value_unit']]+=1;bc['role:'+r['specimen_type']]+=1
    valid=raw[:,2,6]>0
    sourcepaths=(Path(__file__),PROTOCOL,Path('training/tests/test_same_specimen_audit.py'),
        PARENT/'audit.json',PARENT/'verification.json',PARENT/'local_evidence.jsonl.gz',PARENT/'training_audit.npz',
        STATE/'manifest.json',STATE/'clinical_state.npz',EVENTS/'manifest.json',EVENTS/'events.csv.gz',admission_path,
        Path('patient_data/v6_lab_state.py'),Path('patient_data/lab_recovery_features.py'),Path('patient_data/preprocess_events.py'))
    sources={str(p):sha256(p) for p in sourcepaths}
    report=dict(status='training_only_same_specimen_information_audit',source_sha256=sources,
        inherited_raw_lab_sha256=sm['source_hashes']['raw/labevents'],raw_lab_reparsed=False,
        training_admissions=len(ti),training_patients=len(np.unique(subjects)),gas_row_audit=dict(counts),
        accepted_parent_gas_rows=len(bound),specimen_audit=dict(pc),pairs=len(pairs),
        paired_admissions=int(valid.sum()),paired_patients=len(np.unique(subjects[valid])),
        admissions_with_multiple_pair_times=sum(len(v)>1 for v in charts.values()),
        same_pair_time_multiple_specimen_admissions=sum(len([p for p in pairs if p['hadm_id']==h])>len(v) for h,v in charts.items()),
        old_min_zero_pairs=sum(p['severity']==0 for p in pairs),
        old_min_zero_admissions=len(zero),zero_min_multiple_physical_pairs_admissions=sum(len(v)>1 for v in zero.values()),
        exact_min_groups_with_distinct_physical_pairs=sum(len(v)>1 for v in groups.values()),
        same_joint_nine_different_latest_physical_sets=joint_coll,
        same_three_axes_27_different_latest_physical_sets=triple_coll,
        original_axis_replay_errors=errors,bicarbonate_audit=dict(bc),
        bicarbonate_processed_train_rows=len(bicarbonate),bicarbonate_not_arterial_by_mapping=True,
        latest_pair_age_median_days=float(np.median(np.expm1(raw[valid,2,4]))) if valid.any() else None,
        latest_pair_multi_value_set_admissions=sum(len(v)>1 for v in physical),
        new_fits=0,new_vectors=0,no_outcomes_read=True,actual_seconds=time.perf_counter()-started,
        limitations=['exact signatures are not complete patient inputs or outcome information',
            'zero min is not globally normal; no new disease labels or effect estimate',
            'same specimen components remain distinct actual chart/store times',
            'latest sets are not confirmed discharge-day status; no physical pair averaging'])
    arrays=dict(hadm_ids=hadms,subject_ids=subjects,old_three_axis_raw=raw,
                pair_admission_indices=np.asarray([lookup[p['hadm_id']] for p in pairs],np.int64),
                pair_values=np.asarray([p['values'] for p in pairs],np.float64).reshape(-1,2),
                pair_chart_days=np.asarray([p['days'] for p in pairs],np.float64).reshape(-1,2),
                pair_joint_severity=np.asarray([p['severity'] for p in pairs],np.float64))
    evidence=dict(pairs=pairs,rejected=rejected,collision_indices=dict(joint=jc,three_axes=tc))
    return report,arrays,evidence


def run():
    if ROOT.exists():raise FileExistsError('audit exists; do not repeat')
    ROOT.mkdir()
    report,a,e=derive();np.savez_compressed(ROOT/'audit_arrays.npz',**a)
    with gzip.open(ROOT/'local_evidence.json.gz','wt',encoding='utf-8') as f:json.dump(e,f)
    report['output_sha256']={k:sha256(ROOT/k) for k in ('audit_arrays.npz','local_evidence.json.gz')}
    write(ROOT/'audit.json',report);write(ROOT/'state.json',dict(status='pending_independent_rebuild',new_fits=0))
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_sha256','output_sha256')}),flush=True)


def write(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def verify():
    if (ROOT/'verification.json').exists():raise FileExistsError('already verified')
    old=json.loads((ROOT/'audit.json').read_text());new,a,e=derive();seconds=new.pop('actual_seconds');old.pop('actual_seconds')
    for k,h in old.pop('output_sha256').items():
        if sha256(ROOT/k)!=h:raise ValueError('saved arrays/evidence changed')
    if old!=new:raise ValueError('statistics/source differ')
    with np.load(ROOT/'audit_arrays.npz') as z:
        for k,v in a.items():np.testing.assert_array_equal(z[k],v)
    with gzip.open(ROOT/'local_evidence.json.gz','rt',encoding='utf-8') as f:saved=json.load(f)
    # JSON converts tuples in collision lists, but all persisted evidence uses lists.
    if saved!=e:raise ValueError('local evidence differs')
    write(ROOT/'verification.json',dict(status='one_independent_accepted_source_parent_row_axis_array_statistic_rebuild',
        maximum_error=0.,audit_sha256=sha256(ROOT/'audit.json'),actual_seconds=seconds,
        raw_lab_reparsed=False,new_fits=0,new_vectors=0,no_outcomes_read=True))
    write(ROOT/'state.json',dict(status='training_audit_independently_verified',new_fits=0,goal_achieved=False))
    print(json.dumps(dict(error=0.,seconds=seconds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('run','verify'),required=True);mode=p.parse_args().mode
    try:run() if mode=='run' else verify()
    except Exception as error:
        if ROOT.exists():write(ROOT/f'failure_{mode}.json',dict(error=repr(error)))
        raise
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/supplemental_sources.py · lines 1–175

<!-- REVIEW_FRAGMENT {"path":"patient_data/supplemental_sources.py","part":1,"start_line":1,"end_line":175,"source_sha256":"eee0ba3235f144c68cc1667057e69c61b81d96f9cd3c7aaa96eaa722bba5eb5f","fragment_sha256":"eee0ba3235f144c68cc1667057e69c61b81d96f9cd3c7aaa96eaa722bba5eb5f","added_terminal_newline":false} -->
````python
"""只读补核/导出微生物与小型呼吸支持来源；不读取或持久化密码配置。"""
from __future__ import annotations
import argparse
import csv
import gzip
import io
import json
import os
from pathlib import Path
import subprocess
import time

from patient_data.export_mimiciv import rows, sha256

PSQL = Path('C:/Program Files/PostgreSQL/18/bin/psql.exe')
COHORT = 'project_aecopd.copd_coded_admissions_v1_gate'
STATE_ITEMS = (226732,223849,229314,223848)
NUMERIC_ITEMS = (223834,227582,227287,223835,227579,227580,220210,220277)
PROCEDURE_ITEMS = (224385,225468,225477,225792,225794,227194)


def pg_environment():
    env = os.environ.copy()
    if not env.get('PGPASSWORD'):
        raise RuntimeError('Configure credentials in process environment before invoking; no password saved here')
    env['PGOPTIONS'] = '-c default_transaction_read_only=on -c statement_timeout=600000'
    env['PGCONNECT_TIMEOUT'] = '10'
    return env


def command():
    return [str(PSQL), '-X', '-q', '-w', '-h', 'localhost', '-U', 'postgres', '-d', 'mimiciv', '-v', 'ON_ERROR_STOP=1']


def query(sql, snapshot=None):
    body = 'BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY; '
    if snapshot is not None:
        body += f"SET TRANSACTION SNAPSHOT '{snapshot}'; "
    body += f'COPY ({sql}) TO STDOUT WITH (FORMAT CSV, HEADER TRUE); ROLLBACK;'
    result = subprocess.run(command()+['-c', body], env=pg_environment(), capture_output=True, text=True, encoding='utf-8')
    if result.returncode:
        raise RuntimeError(result.stderr)
    return list(csv.DictReader(io.StringIO(result.stdout)))


def inventory():
    columns = query("SELECT table_schema,table_name,column_name,data_type FROM information_schema.columns WHERE "
                    "(table_schema='mimiciv_hosp' AND table_name='microbiologyevents') OR "
                    "(table_schema='mimiciv_icu' AND table_name IN ('icustays','chartevents','procedureevents','d_items')) "
                    "ORDER BY table_schema,table_name,ordinal_position")
    items = query("SELECT itemid,label,linksto,category,unitname,param_type FROM mimiciv_icu.d_items WHERE "
                  "itemid IN (226732,223849,229314,223848,223834,227582,227287,223835,220210,220277) "
                  "OR label ~* '(IPAP|EPAP|intubation|extubation|invasive ventilation|non.invasive ventilation)' ORDER BY itemid")
    indices = query("SELECT tablename,indexdef FROM pg_indexes WHERE schemaname='mimiciv_icu' AND tablename='chartevents'")
    return {'columns': columns, 'candidate_items_not_yet_approved': items, 'chartevents_indices': indices}


def stream_export(sql, path, snapshot):
    body = f"BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY; SET TRANSACTION SNAPSHOT '{snapshot}'; "
    body += f"COPY ({sql}) TO STDOUT WITH (FORMAT CSV,HEADER TRUE,NULL '\\N',ENCODING 'UTF8'); ROLLBACK;"
    process = subprocess.Popen(command()+['-c',body],env=pg_environment(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    with gzip.open(path,'wb',compresslevel=3) as destination:
        while block := process.stdout.read(1024*1024):
            destination.write(block)
    error=process.stderr.read().decode('utf-8',errors='replace')
    if process.wait():
        raise RuntimeError(error)
    count=sum(1 for _ in rows(path))
    return {'rows':count,'sha256':sha256(path),'compressed_bytes':path.stat().st_size}


def export(raw_root: Path, output: Path):
    if output.exists(): raise FileExistsError(output)
    start=time.perf_counter()
    raw_manifest=json.loads((raw_root/'metadata/manifest.json').read_text(encoding='utf-8'))
    if raw_manifest['status']!='validated': raise ValueError('raw snapshot unvalidated')
    source_file=raw_root/'cohort/admissions.csv.gz'
    if sha256(source_file)!=raw_manifest['sources']['cohort/admissions']['sha256']: raise ValueError('raw members changed')
    frozen={r['hadm_id']:r for r in rows(source_file)}
    keeper=subprocess.Popen(command()+['-A','-t'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                            text=True,env=pg_environment())
    keeper.stdin.write('BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY; SELECT pg_export_snapshot();\n')
    keeper.stdin.flush()
    snapshot=keeper.stdout.readline().strip()
    if not snapshot or any(c not in '0123456789ABCDEFabcdef-' for c in snapshot):
        keeper.terminate(); raise RuntimeError('failed to establish read-only snapshot')
    output.mkdir(parents=True)
    (output/'raw').mkdir(); (output/'metadata').mkdir()
    manifest={'status':'exporting','database':'mimiciv','parent_snapshot_manifest_sha256':sha256(raw_root/'metadata/manifest.json'),
              'parent_cohort_sha256':sha256(source_file),'sources':{},'cohort_source':COHORT,
              'method':'read-only repeatable-read snapshot; exact parent membership; restricted ICU item whitelist',
              'chartevents_state_items':STATE_ITEMS,'chartevents_numeric_items':NUMERIC_ITEMS,'procedure_items':PROCEDURE_ITEMS}
    try:
        current=query(f'SELECT hadm_id,subject_id,admittime,dischtime,discharge_landmark_eligible FROM {COHORT}',snapshot)
        if len(current)!=len(frozen) or {r['hadm_id'] for r in current}!=set(frozen): raise ValueError('live cohort membership differs')
        for r in current:
            old=frozen[r['hadm_id']]
            if any(r[k]!=old[k] for k in ('subject_id','admittime','dischtime','discharge_landmark_eligible')):
                raise ValueError('live cohort context differs')
        manifest['cohort_membership_exactly_matches_parent']=True
        meta=inventory()
        (output/'metadata/schema_inventory.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
        all_items=STATE_ITEMS+NUMERIC_ITEMS+PROCEDURE_ITEMS
        specifications={
            'microbiologyevents': f'SELECT m.* FROM mimiciv_hosp.microbiologyevents m JOIN {COHORT} c USING(subject_id,hadm_id)',
            'icustays':f'SELECT s.* FROM mimiciv_icu.icustays s JOIN {COHORT} c USING(subject_id,hadm_id)',
            'respiratory_procedureevents':f'SELECT p.* FROM mimiciv_icu.procedureevents p JOIN {COHORT} c USING(subject_id,hadm_id) WHERE p.itemid IN ({",".join(map(str,PROCEDURE_ITEMS))})',
            'd_items':f'SELECT * FROM mimiciv_icu.d_items WHERE itemid IN ({",".join(map(str,all_items))})',
        }
        states=','.join(map(str,STATE_ITEMS)); numbers=','.join(map(str,NUMERIC_ITEMS))
        # 设备/模式保留白名单状态记录，数值仅导出每ICU stay的首/末/最小/最大锚点。
        # 不返回全量RR/SpO2/FiO2序列；原源表不修改，选中数值的过程计数只作审计。
        specifications['respiratory_chartevents']=f"""WITH ce AS MATERIALIZED (
          SELECT x.* FROM mimiciv_icu.chartevents x JOIN {COHORT} c USING(subject_id,hadm_id)
          WHERE x.itemid IN ({states},{numbers}) AND c.admittime<=x.charttime AND x.charttime<=x.storetime AND x.storetime<=c.dischtime
        ), ranked AS (
          SELECT *, ROW_NUMBER() OVER w_first AS r_first,ROW_NUMBER() OVER w_last AS r_last,
             ROW_NUMBER() OVER w_min AS r_min,ROW_NUMBER() OVER w_max AS r_max,COUNT(*) OVER(PARTITION BY hadm_id,stay_id,itemid) AS available_numeric_rows
          FROM ce WHERE itemid IN ({numbers}) AND valuenum IS NOT NULL AND valuenum NOT IN ('NaN'::double precision,'Infinity'::double precision,'-Infinity'::double precision)
          WINDOW w_first AS (PARTITION BY hadm_id,stay_id,itemid ORDER BY charttime,storetime,valuenum),
          w_last AS (PARTITION BY hadm_id,stay_id,itemid ORDER BY charttime DESC,storetime DESC,valuenum DESC),
          w_min AS (PARTITION BY hadm_id,stay_id,itemid ORDER BY valuenum,charttime,storetime),
          w_max AS (PARTITION BY hadm_id,stay_id,itemid ORDER BY valuenum DESC,charttime,storetime)
        ) SELECT subject_id,hadm_id,stay_id,charttime,storetime,itemid,value,valuenum,valueuom,warning,
                 'state_record'::text AS selection_role,NULL::bigint AS available_numeric_rows
          FROM ce WHERE itemid IN ({states})
          UNION ALL
          SELECT subject_id,hadm_id,stay_id,charttime,storetime,itemid,value,valuenum,valueuom,warning,
             CONCAT_WS('|',CASE WHEN r_first=1 THEN 'first' END,CASE WHEN r_last=1 THEN 'latest' END,
                       CASE WHEN r_min=1 THEN 'minimum' END,CASE WHEN r_max=1 THEN 'maximum' END),available_numeric_rows
          FROM ranked WHERE r_first=1 OR r_last=1 OR r_min=1 OR r_max=1"""
        for name,sql in specifications.items():
            (output/'metadata'/f'{name}.sql').write_text(sql,encoding='utf-8')
            print(json.dumps({'stage':'exporting','table':name}),flush=True)
            manifest['sources'][name]=stream_export(sql,output/'raw'/f'{name}.csv.gz',snapshot)
            print(json.dumps({'stage':'exported','table':name,'rows':manifest['sources'][name]['rows']}),flush=True)
        eligible={h for h,r in frozen.items() if r['discharge_landmark_eligible']=='t'}
        audit={}
        for name in ('microbiologyevents','icustays','respiratory_procedureevents','respiratory_chartevents'):
            seen=set(); stays=set()
            for r in rows(output/'raw'/f'{name}.csv.gz'):
                if r['hadm_id'] not in frozen or r['subject_id']!=frozen[r['hadm_id']]['subject_id']: raise ValueError('supplement outside parent cohort')
                if r['hadm_id'] in eligible: seen.add(r['hadm_id'])
                if name=='icustays':
                    if r['stay_id'] in stays: raise ValueError('duplicate ICU stay')
                    stays.add(r['stay_id'])
            audit[name]={'eligible_admissions_with_source_rows':len(seen)}
        manifest.update(status='validated_restricted_source_export_not_model_validated',eligible_admissions=len(eligible),
                        retained_admissions=len(frozen),coverage_audit=audit,elapsed_seconds=round(time.perf_counter()-start,2),
                        limitations=['Numeric ICU anchors are sparse query-selected observations, not complete trajectories',
                                     'Respiratory chartevents restricted to hospital-time and stored by discharge',
                                     'Procedure/microbiology raw rows still require time/result interpretation checks',
                                     'No ICU row does not prove no oxygen/ventilation on non-ICU wards'])
        (output/'metadata/manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        return manifest
    except Exception:
        (output/'metadata/status.json').write_text(json.dumps({'status':'failed_preserved_do_not_use'}),encoding='utf-8')
        raise
    finally:
        keeper.communicate('ROLLBACK;\n',timeout=20)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory',action='store_true')
    parser.add_argument('--raw-root',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.inventory:
        print(json.dumps(inventory(),ensure_ascii=False,indent=2))
    elif args.raw_root and args.output:
        result=export(args.raw_root,args.output)
        print(json.dumps({'status':result['status'],'elapsed_seconds':result['elapsed_seconds']}))
    else:
        parser.error('supply --inventory or --raw-root and --output')
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/text_time_coverage.py · lines 1–202

<!-- REVIEW_FRAGMENT {"path":"patient_data/text_time_coverage.py","part":1,"start_line":1,"end_line":202,"source_sha256":"0f92135bc572bdf31f2844b0e612a3937b213bff31a545caf766e35f74b57f58","fragment_sha256":"0f92135bc572bdf31f2844b0e612a3937b213bff31a545caf766e35f74b57f58","added_terminal_newline":false} -->
````python
"""训练折原句→原B/既有候选覆盖；不训练、不按结局筛选、不输出原文到日志。"""
from collections import Counter,defaultdict
from datetime import datetime
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from patient_data.export_mimiciv import rows,sha256
from patient_data.clinical_facts.sentence_rules import COMPILED,WARNING,FUTURE_OUTCOME
from training.v6.encode_screen_candidates import FUTURE
from training.v6.train import write_json

ROOT=Path('patient_data/processed/v6_text_time_coverage_2026-10-13')
SNAP=Path('patient_data/mimiciv_copd/snapshot_2026-09-27')
STATE=Path('patient_data/processed/v6_clinical_state_2026-10-09_r2')
ASSETS=Path('outputs/01a08b96-2988-7001-af1d-b40d901f3400/private/v5_section_aligned')
B=ASSETS/'field_text_baseline_2026_09_27'
SCREEN=Path('patient_data/processed/v6_rule_sentence_screen_2026-10-10_r4')
ENCODED=Path('patient_data/processed/v6_screen_candidate_bert_2026-10-10')
PROTOCOL=Path('docs/V6_TEXT_TIME_COVERAGE_AUDIT_PROTOCOL_2026-10-13.md')
ROLES={'hospital_course','discharge_condition','discharge_physical_exam','discharge_instructions','transitional_issues','followup_instructions'}
DISCHARGE=re.compile(r'\b(?:(?:at|on|upon|by|before|prior to|time of)\s+(?:the\s+)?discharge|discharged|discharging)\b',re.I)
CHANGE=re.compile(r'\b(?:improv\w*|wean\w*|resolv\w*|persist\w*|still|no longer|(?:back|return\w*)\s+to\s+(?:her|his|their|the)?\s*baseline)\b',re.I)
PLAN=re.compile(r'\b(?:will|plan\w*|recommend\w*|should|advis\w*|continue|to complete|follow.?up)\b',re.I)
HISTORY=re.compile(r'\b(?:history of|baseline|prior|previous\w*|at home)\b',re.I)
POST_DISCHARGE=re.compile(r'\b(?:after\s+(?:the\s+)?discharge|post[- ]discharge)\b',re.I)


def digest(text):return hashlib.sha256(text.encode()).hexdigest()


def time_flags(text,role):
    respiratory=bool(COMPILED['respiratory_status'].search(text) or COMPILED['oxygen_ventilation'].search(text))
    flags=dict(respiratory=respiratory,approved_role=role in ROLES,explicit_discharge=bool(DISCHARGE.search(text)),
        change_or_residual=bool(CHANGE.search(text)),plan=bool(PLAN.search(text)),historical=bool(HISTORY.search(text)),
        warning=bool(WARNING.search(text)),future_outcome=bool(FUTURE_OUTCOME.search(text)),post_discharge_words=bool(POST_DISCHARGE.search(text)))
    flags['priority_candidate']=respiratory and flags['approved_role'] and (flags['explicit_discharge'] or flags['change_or_residual']) and not flags['warning'] and not flags['future_outcome'] and not flags['post_discharge_words']
    return flags


def span(original,start,end,quote):
    if not 0<=start<end<=len(original) or original[start:end]!=quote:raise ValueError('original quote offset differs')


def coverage(start,end,intervals):
    if not 0<=start<end:raise ValueError('nonempty forward span required')
    pieces=sorted((max(start,a),min(end,b)) for a,b in intervals if a<end and b>start)
    total=0;right=start
    for a,b in pieces:
        if not 0<=a<b:raise ValueError('invalid intersecting interval')
        total+=max(0,b-max(a,right));right=max(right,b)
    return total/(end-start),any(a<=start and end<=b for a,b in intervals)


def checked_sources():
    sm=json.loads((SNAP/'metadata/manifest.json').read_text());state=json.loads((STATE/'manifest.json').read_text())
    bm=json.loads((B/'summary.json').read_text());bs=json.loads((B/'bert/state.json').read_text())
    qa=json.loads((ASSETS/'dataset_v4/qa_validation.json').read_text());screen=json.loads((SCREEN/'manifest.json').read_text())
    audit=json.loads((SCREEN/'source_audit.json').read_text());em=json.loads((ENCODED/'manifest.json').read_text())
    proof=json.loads((ENCODED/'independent_encoding_verification.json').read_text())
    checks={SNAP/'raw/discharge.csv.gz':sm['sources']['raw/discharge']['sha256'],
        SNAP/'cohort/admissions.csv.gz':sm['sources']['cohort/admissions']['sha256'],STATE/'clinical_state.npz':state['output_sha256']['clinical_state.npz'],
        ASSETS/'dataset_v4/note_sections.csv.gz':qa['output_sha256']['note_sections.csv.gz'],B/'shared_text.csv.gz':bm['manifest_sha256']}
    checks.update({SCREEN/k:v for k,v in screen['output_sha256'].items()});checks.update({ENCODED/k:v for k,v in em['output_sha256'].items()})
    if sm['status']!='validated' or bs['status']!='complete' or bs['manifest_sha256']!=bm['manifest_sha256']:raise ValueError('original source/encoding incomplete')
    if (audit['status']!='all_original_quotes_contexts_and_section_coverage_checked' or audit['manifest_sha256']!=sha256(SCREEN/'manifest.json') or
        em['source_manifest_sha256']!=audit['manifest_sha256'] or proof['manifest_sha256']!=sha256(ENCODED/'manifest.json')):raise ValueError('candidate source proof differs')
    if sha256(ASSETS/'dataset_v4/note_sections.csv.gz')!=bm['cohort_source_sections_sha256']:raise ValueError('B section lineage changed')
    for p,h in checks.items():
        if sha256(p)!=h:raise ValueError('source changed: '+str(p))
    # 额外保留原完整V6依赖的已记录来源指纹，无训练或结局分析。
    for parent in ('v6_module_surgery_2026-10-12_r3','v6_regularization_search_2026-10-12'):
        m=json.loads((Path('training_runs')/parent/'contract.json').read_text())
        for p,h in m['source_sha256'].items():
            if sha256(Path(p))!=h:raise ValueError('protected old source changed: '+p)
        if sha256(B/'bert/bioclinicalbert.float16.npy')!=m['data_provenance']['text']['vectors_sha256']:raise ValueError('original V6 B vectors changed')
    sources={str(p):h for p,h in checks.items()}
    for p in (PROTOCOL,Path(__file__),Path('training/tests/test_v6_text_time_coverage.py'),B/'bert/state.json',B/'bert/bioclinicalbert.float16.npy',
              SCREEN/'manifest.json',SCREEN/'source_audit.json',ENCODED/'manifest.json',ENCODED/'independent_encoding_verification.json'):
        sources[str(p)]=sha256(p)
    return sources,bs


def derive():
    sources,bs=checked_sources()
    with np.load(STATE/'clinical_state.npz') as z:
        hadms=z['hadm_ids'];people=z['subject_ids'];splits=z['patient_splits']
    training={str(h):str(s) for h,s,f in zip(hadms,people,splits) if f=='train'}
    if len(training)!=13592:raise ValueError('training split differs')
    raw={r['note_id']:r for r in rows(SNAP/'raw/discharge.csv.gz') if r['hadm_id'] in training}
    raw_hashes={note:digest(r['text']) for note,r in raw.items()}
    cohort={r['hadm_id']:r for r in rows(SNAP/'cohort/admissions.csv.gz') if r['hadm_id'] in training}
    sections={}
    for r in rows(ASSETS/'dataset_v4/note_sections.csv.gz'):
        if r['hadm_id'] not in training:continue
        original=raw[r['note_id']];a,b=int(r['start_char']),int(r['end_char'])
        if not 0<=a<=b<=len(original['text']) or original['text'][a:b]!=r['text']:raise ValueError('original section offset differs')
        if r['source_sha256']!=raw_hashes[r['note_id']]:raise ValueError('B original document changed')
        sections[r['section_id']]=r
    intervals=defaultdict(list);bchunks={};bcounts=Counter();bnotes=set();max_tokens=0
    for j,r in enumerate(rows(B/'shared_text.csv.gz')):
        if int(r['embedding_row'])!=j:raise ValueError('B rows reordered')
        if r['hadm_id'] not in training:continue
        if r['subject_id']!=training[r['hadm_id']]:raise ValueError('B subject/fold mismatch')
        sec=sections[r['section_id']];note=sec['note_id'];a,b=int(r['source_start_char']),int(r['source_end_char'])
        original=raw[note]['text'];quote=original[a:b]
        span(original,a,b,quote)
        if digest(quote)!=r['source_chunk_sha256'] or r['chunk_text'].partition('\n')[2]!=quote or digest(r['chunk_text'])!=r['text_sha256']:
            raise ValueError('B prefix/source content mismatch')
        if r['source_section_sha256']!=raw_hashes[note]:raise ValueError('B document fingerprint differs')
        intervals[note].append((a,b,j));bchunks[j]=r;bnotes.add(note);bcounts[r['field']]+=1;max_tokens=max(max_tokens,int(r['bert_token_count']))
    # 候选编码索引source_units属于整个组，不把它冒充每个块的内容。
    candidate_units=set();encoded_notes=set();encoded_groups={};encoded_train_chunks=0
    with gzip.open(ENCODED/'chunks.jsonl.gz','rt',encoding='utf-8') as f:
        for r in map(json.loads,f):
            if r['patient_split']!='train':continue
            if r['hadm_id'] not in training or r['subject_id']!=training[r['hadm_id']]:raise ValueError('candidate encoding fold differs')
            encoded_train_chunks+=1;encoded_notes.add(r['note_id']);key=(r['note_id'],r['field'])
            encoded_groups.setdefault(key,[]).append((r['token_start'],r['token_end']))
            candidate_units.update((r['note_id'],int(a),int(b)) for a,b in r['source_units'])
    for ranges in encoded_groups.values():
        right=0
        for a,b in sorted(ranges):
            if a!=right or not a<b:raise ValueError('candidate token chunks gap/overlap')
            right=b
    counters=Counter();roles=defaultdict(Counter);targets=[];banned=set();time_patterns=Counter();note_set=set()
    print(json.dumps(dict(stage='original_and_B_checked',training_B_chunks=len(bchunks),training_B_notes=len(bnotes))),flush=True)
    with gzip.open(SCREEN/'all_units.jsonl.gz','rt',encoding='utf-8') as f:
        for r in map(json.loads,f):
            if r['kind']!='discharge' or r['patient_split']!='train':continue
            if r['hadm_id'] not in training or r['subject_id']!=training[r['hadm_id']]:raise ValueError('candidate fold/subject mismatch')
            original=raw[r['note_id']]
            if r['original_sha256']!=raw_hashes[r['note_id']]:raise ValueError('candidate original document changed')
            a,b=int(r['start_char']),int(r['end_char']);span(original['text'],a,b,r['quote'])
            counters['all_training_discharge_units']+=1;note_set.add(r['note_id'])
            if r['reason']=='observed_future_outcome' or FUTURE.search(r['quote']):banned.add((r['note_id'],r['section_ordinal']))
            flags=time_flags(r['quote'],r['section'])
            if not (flags['approved_role'] and flags['respiratory']):continue
            counters['respiratory_units_in_candidate_roles']+=1
            for k,v in flags.items():
                if v:counters['lexical:'+k]+=1
            roles[r['section']]['respiratory_units']+=1
            if not flags['priority_candidate']:continue
            span_pairs=[(x,y) for x,y,_ in intervals[r['note_id']]];fraction,single=coverage(a,b,span_pairs)
            containing=[j for x,y,j in intervals[r['note_id']] if x<=a and b<=y]
            hit=(r['note_id'],a,b) in candidate_units
            pattern='+'.join(k for k in ('explicit_discharge','change_or_residual','plan','historical') if flags[k]);time_patterns[pattern]+=1
            record={k:r[k] for k in ('note_id','hadm_id','subject_id','section','section_ordinal','start_char','end_char','unit_key','reason')}
            record.update(original_sha256=r['original_sha256'],quote_sha256=digest(r['quote']),flags=flags,selected=r['selected'],
                B_union_character_coverage=fraction,B_single_chunk_full=single,B_containing_rows=containing,
                B_same_document_present=r['note_id'] in bnotes,encoded_candidate_source_unit=hit,
                quote_chars=len(r['quote']),min_containing_B_chars=min((len(bchunks[j]['chunk_text'].partition('\n')[2]) for j in containing),default=None),
                source_stored_after_discharge=datetime.fromisoformat(original['storetime'])>datetime.fromisoformat(cohort[r['hadm_id']]['dischtime']))
            targets.append(record)
    # 完整章节隔离在全文扫描完成后确定，不能先看到安全句就当章节安全。
    for r in targets:
        r['candidate_section_quarantined']=(r['note_id'],r['section_ordinal']) in banned
        if r['encoded_candidate_source_unit'] and r['candidate_section_quarantined']:raise ValueError('quarantined target enters encoded source')
        c=roles[r['section']];c['priority_candidates']+=1
        for name,value in [('B_single_chunk_full',r['B_single_chunk_full']),('B_union_full',r['B_union_character_coverage']==1),
            ('B_no_character_coverage',r['B_union_character_coverage']==0),('B_other_document_or_absent',not r['B_same_document_present']),
            ('candidate_selected',r['selected']),('encoded_candidate_source_unit',r['encoded_candidate_source_unit']),
            ('candidate_section_quarantined',r['candidate_section_quarantined']),('source_stored_after_discharge',r['source_stored_after_discharge'])]:
            if value:counters[name]+=1;c[name]+=1
    fractions=[r['quote_chars']/r['min_containing_B_chars'] for r in targets if r['min_containing_B_chars']]
    result=dict(status='training_only_original_time_coverage_audit_not_semantic_gold',source_sha256=sources,training_admissions=13592,
        training_summary_documents=len(note_set),training_B_chunks=len(bchunks),training_B_notes=len(bnotes),B_chunks_by_field=dict(bcounts),
        B_training_max_recorded_token_count=max_tokens,B_encoding_state=bs,training_encoded_candidate_chunks=encoded_train_chunks,
        counters=dict(counters),by_role={k:dict(v) for k,v in roles.items()},lexical_time_pattern_counts=dict(time_patterns),
        priority_candidate_units=len(targets),priority_candidate_admissions=len({r['hadm_id'] for r in targets}),
        priority_candidate_chars_as_fraction_of_containing_B_chunk_quantiles=np.quantile(fractions,[0,.1,.5,.9,1]).tolist() if fractions else None,
        unencoded_priority_candidate_reasons=dict(Counter(r['reason'] for r in targets if not r['encoded_candidate_source_unit'])),
        limits=['No outcomes used or new fitting','Lexical candidate not verified discharge fact or semantic precision/recall',
            'already encoded does not prove learned semantic importance','small content fraction does not prove pooling is performance root cause',
            'candidate source_units identifies whole group; no individual block containment claim','FINAL retrospective including later storage not real-time availability'])
    return result,targets,bchunks


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('build','verify'),required=True);p.add_argument('--tag',default='a',choices=('a','b'));a=p.parse_args()
    report,records,_=derive()
    if a.mode=='build':
        if ROOT.exists():raise FileExistsError('audit directory exists; inspect')
        ROOT.mkdir(parents=True)
        with gzip.open(ROOT/'priority_candidate_index.jsonl.gz','xt',encoding='utf-8') as f:
            for r in records:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        write_json(ROOT/'audit.json',report)
    else:
        saved=json.loads((ROOT/'audit.json').read_text())
        if saved!=report:raise ValueError('source reconstruction/report differs')
        with gzip.open(ROOT/'priority_candidate_index.jsonl.gz','rt',encoding='utf-8') as f:
            if list(map(json.loads,f))!=records:raise ValueError('source audit index differs')
        write_json(ROOT/f'verification_{a.tag}.json',dict(status='independent_source_rebuild_verified',report_sha256=sha256(ROOT/'audit.json'),
            index_sha256=sha256(ROOT/'priority_candidate_index.jsonl.gz'),maximum_count_or_index_error=0,training_only=True,no_model_fit=True))
    print(json.dumps({k:report[k] for k in ('status','counters','by_role','priority_candidate_units','priority_candidate_admissions',
        'priority_candidate_chars_as_fraction_of_containing_B_chunk_quantiles','unencoded_priority_candidate_reasons')}),flush=True)


if __name__=='__main__':main()
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/unassigned_lab_audit.py · lines 1–128

<!-- REVIEW_FRAGMENT {"path":"patient_data/unassigned_lab_audit.py","part":1,"start_line":1,"end_line":128,"source_sha256":"1476e72e970aa11b5075588b9161efa9f4bd85052d1d0c55f34926cf663780eb","fragment_sha256":"1476e72e970aa11b5075588b9161efa9f4bd85052d1d0c55f34926cf663780eb","added_terminal_newline":false} -->
````python
"""无结局训练折同标本住院锚点审计；不自动填住院或构建模型输入。"""
from collections import Counter,defaultdict
from datetime import datetime
import gzip
import json
from pathlib import Path
import time
import numpy as np
from patient_data.export_mimiciv import rows,sha256

SNAPSHOT=Path('patient_data/mimiciv_copd/snapshot_2026-09-27')
ROOT=Path('training_runs/v6_unassigned_lab_audit_2026-10-14')
PARENT=Path('training_runs/v6_admission_context_2026-10-14')
PROTOCOL=Path('docs/V6_UNASSIGNED_LAB_PROTOCOL_2026-10-14.md')
NULLS=('',r'\N',None)

def key(row):
    return None if row.get('subject_id') in NULLS or row.get('specimen_id') in NULLS else (row['subject_id'],row['specimen_id'])

def visible(row,boundary):
    try:
        chart,stored=(datetime.fromisoformat(row[k]) for k in ('charttime','storetime'))
        return boundary[0]<=chart<=stored<=boundary[1]
    except (ValueError,TypeError,KeyError):return False

def attribution(row,anchor,admissions,eligible,duplicate=False):
    if duplicate:return None,'duplicate_labevent_id'
    if key(row) is None:return None,'missing_patient_or_specimen'
    if not anchor:return None,'no_assigned_specimen_anchor'
    if len(anchor)!=1:return None,'multiple_hospital_anchors'
    h,info=next(iter(anchor.items()))
    if h not in eligible:return None,'anchor_not_eligible_training_admission'
    if not visible(row,admissions[h]):return None,'candidate_not_hospital_visible'
    if row['charttime'] not in info['visible_times']:return None,'no_visible_same_time_anchor'
    return h,'same_specimen_hospital_evidence'

def derive():
    started=time.perf_counter();sm_path=SNAPSHOT/'metadata/manifest.json';sm=json.loads(sm_path.read_text())
    if sm['status']!='validated':raise ValueError('snapshot not validated')
    manifest=json.loads((PARENT/'interface_manifest.json').read_text())
    proof=json.loads((PARENT/'verification.json').read_text())
    if (manifest['sha256']!=sha256(PARENT/'interface.npz') or proof['maximum_error']!=0
            or proof['screening_sha256']!=sha256(PARENT/'screening.json')):raise ValueError('parent proof changed')
    sources={str(sm_path):sha256(sm_path),str(PARENT/'interface.npz'):sha256(PARENT/'interface.npz')}
    for name in ('cohort/admissions','raw/labevents','raw/labevents_unassigned','raw/d_labitems'):
        p=SNAPSHOT/(name+'.csv.gz');h=sha256(p)
        if h!=sm['sources'][name]['sha256']:raise ValueError('source changed: '+name)
        sources[str(p)]=h
    for p in (Path(__file__),PROTOCOL,Path('training/tests/test_v6_unassigned_lab_audit.py')):sources[str(p)]=sha256(p)
    with np.load(PARENT/'interface.npz') as z:
        eligible={str(h):str(s) for h,s,f in zip(z['hadm_ids'],z['subject_ids'],z['patient_splits']) if f=='train'}
    if len(eligible)!=13592:raise ValueError('training membership differs')
    people=set(eligible.values());admissions={};owners={}
    for row in rows(SNAPSHOT/'cohort/admissions.csv.gz'):
        if row['subject_id'] not in people:continue
        h=row['hadm_id']
        if h in admissions:raise ValueError('duplicate cohort')
        admissions[h]=tuple(datetime.fromisoformat(row[k]) for k in ('admittime','dischtime'));owners[h]=row['subject_id']
        if h in eligible and (eligible[h]!=row['subject_id'] or row['discharge_landmark_eligible']!='t'):
            raise ValueError('training qualification differs')
    if not set(eligible)<=set(admissions):raise ValueError('training source missing')
    candidate=[];total=0;ids=Counter();items=Counter();specimens=set();hadm_present=0
    for row in rows(SNAPSHOT/'raw/labevents_unassigned.csv.gz'):
        total+=1
        if row['hadm_id'] not in NULLS:hadm_present+=1
        if row['subject_id'] not in people:continue
        candidate.append(row);ids[row['labevent_id']]+=1;items[row['itemid']]+=1
        if key(row) is not None:specimens.add(key(row))
    if total!=sm['sources']['raw/labevents_unassigned']['rows'] or hadm_present:
        raise ValueError('unassigned source scope differs')
    anchors=defaultdict(lambda:defaultdict(lambda:dict(visible_times=set(),ids=set())))
    raw_total=0;relevant=0;cross_ids=set();seen_assigned=Counter()
    for row in rows(SNAPSHOT/'raw/labevents.csv.gz'):
        raw_total+=1
        if row['labevent_id'] in ids:cross_ids.add(row['labevent_id'])
        k=key(row)
        if k not in specimens:continue
        relevant+=1;seen_assigned[row['labevent_id']]+=1;h=row['hadm_id']
        if h not in admissions or owners[h]!=row['subject_id']:raise ValueError('assigned anchor member conflict')
        slot=anchors[k][h];slot['ids'].add(row['labevent_id'])
        if visible(row,admissions[h]):slot['visible_times'].add(row['charttime'])
    if raw_total!=sm['sources']['raw/labevents']['rows']:raise ValueError('assigned source count differs')
    duplicate_anchor_ids={k for k,v in seen_assigned.items() if v>1}
    records=[];reasons=Counter();bound_items=Counter();bound_hadms=set()
    for row in candidate:
        anchor=anchors.get(key(row),{})
        bad_anchor=any(slot['ids']&duplicate_anchor_ids for slot in anchor.values())
        h,reason=attribution(row,anchor,admissions,eligible,
            duplicate=ids[row['labevent_id']]>1 or row['labevent_id'] in cross_ids or bad_anchor)
        reasons[reason]+=1
        if h is not None:bound_items[row['itemid']]+=1;bound_hadms.add(h)
        records.append(dict(labevent_id=row['labevent_id'],subject_id=row['subject_id'],specimen_id=row['specimen_id'],
            itemid=row['itemid'],attributed_hadm_id=h,reason=reason,
            anchor_hadm_ids=sorted(anchor),anchor_count=sum(len(s['ids']) for s in anchor.values())))
    labels={r['itemid']:r['label'] for r in rows(SNAPSHOT/'raw/d_labitems.csv.gz')}
    result=dict(status='train_only_specimen_anchor_source_audit',source_sha256=sources,
        training_admissions=len(eligible),training_patients=len(people),
        total_unassigned_rows=total,training_patient_candidate_rows=len(candidate),
        training_candidate_specimens=len(specimens),training_candidate_items=len(items),
        assigned_source_rows_scanned=raw_total,assigned_rows_with_candidate_specimens=relevant,
        candidate_specimens_with_anchor=len(anchors),multiple_hospital_anchor_specimens=sum(len(v)>1 for v in anchors.values()),
        duplicate_candidate_ids=sum(v>1 for v in ids.values()),duplicate_relevant_anchor_ids=len(duplicate_anchor_ids),
        overlapping_labevent_ids=len(cross_ids),attribution_reasons=dict(reasons),
        bound_rows=sum(bound_items.values()),bound_training_admissions=len(bound_hadms),
        bound_items={k:dict(label=labels.get(k,'unknown'),rows=v) for k,v in bound_items.items()},
        top_training_candidate_items=[dict(itemid=k,label=labels.get(k,'unknown'),rows=v) for k,v in items.most_common(15)],
        actual_seconds=time.perf_counter()-started,new_fits=0,new_vectors=0,no_outcome_data_read=True,
        next_stage='bound_numeric_specimen_unit_feasibility_needed' if bound_items else 'stop_current_same_specimen_recipe_no_anchors',
        limits=['only preserved COPD hospital source available, not complete allcause hospital inventory',
                'same specimen attribution evidence not clinical gold; time-only assignment not accepted',
                'no clinical validity or incremental performance measured'])
    return result,records

def run():
    if ROOT.exists():raise FileExistsError('audit exists; do not repeat without reason')
    ROOT.mkdir();(ROOT/'state.json').write_text('{"status":"source_audit_running","new_fits":0}\n')
    try:
        result,records=derive()
        with gzip.open(ROOT/'local_evidence.jsonl.gz','wt',encoding='utf-8') as f:
            for row in records:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        result['local_evidence_sha256']=sha256(ROOT/'local_evidence.jsonl.gz')
        (ROOT/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        (ROOT/'state.json').write_text(json.dumps(dict(status='source_audit_complete',next_stage=result['next_stage'],new_fits=0,goal_achieved=False))+'\n')
        print(json.dumps({k:result[k] for k in ('status','training_patient_candidate_rows','assigned_rows_with_candidate_specimens','bound_rows','attribution_reasons','next_stage','actual_seconds')}),flush=True)
    except Exception as error:
        (ROOT/'failure.json').write_text(json.dumps(dict(error=repr(error)))+'\n');raise

if __name__=='__main__':run()
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/unassigned_lab_verify.py · lines 1–51

<!-- REVIEW_FRAGMENT {"path":"patient_data/unassigned_lab_verify.py","part":1,"start_line":1,"end_line":51,"source_sha256":"33d890ce69880ede7678f0ca901b1f3bc350e60d338c66ebd0ed748fccbad957","fragment_sha256":"33d890ce69880ede7678f0ca901b1f3bc350e60d338c66ebd0ed748fccbad957","added_terminal_newline":false} -->
````python
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
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/unified_lab_state.py · lines 1–91

<!-- REVIEW_FRAGMENT {"path":"patient_data/unified_lab_state.py","part":1,"start_line":1,"end_line":91,"source_sha256":"1a61905f575a4671ca4ea0410d29db82afed84b78e102ae9d4847a0c5945c489","fragment_sha256":"1a61905f575a4671ca4ea0410d29db82afed84b78e102ae9d4847a0c5945c489","added_terminal_newline":false} -->
````python
"""额外检验与核心异常使用共同九通道；原派生和源资料不覆盖。"""
from collections import Counter, defaultdict
from datetime import datetime
import argparse
import json
from pathlib import Path
import numpy as np
from patient_data.export_mimiciv import rows, sha256
from patient_data.v6_lab_state import summarize, scale_on_train, AXES
from patient_data.v6_detail_extension import LABS
from patient_data.eos_med_context import EOS_ITEMS, canonical_eos, finite

ROOT = Path('patient_data/processed/v6_unified_lab_state_2026-10-14')
SNAPSHOT = Path('patient_data/mimiciv_copd/snapshot_2026-09-27')
STATE = Path('patient_data/processed/v6_clinical_state_2026-10-09_r2')
EXTENSION = Path('patient_data/processed/v6_detail_extension_2026-10-11_r2')
NAMES = tuple(LABS) + ('eos_absolute', 'eos_percent')
EXTRA_NAMES = tuple(f'{name}_{direction}' for name in NAMES for direction in ('low', 'high'))


def abnormal(value, lower, upper, direction):
    if (direction not in ('low', 'high') or not np.isfinite([value, lower, upper]).all()
            or lower >= upper):
        raise ValueError('invalid laboratory reference/value')
    return max(0., ((lower-value) if direction == 'low' else (value-upper))/(upper-lower))


def build():
    if ROOT.exists(): raise FileExistsError('unified interface exists; do not overwrite')
    sm = json.loads((SNAPSHOT/'metadata/manifest.json').read_text())
    st = json.loads((STATE/'manifest.json').read_text())
    em = json.loads((EXTENSION/'manifest.json').read_text())
    sources = {}
    for name in ('raw/labevents', 'cohort/admissions'):
        p = SNAPSHOT/(name+'.csv.gz'); sources[str(p)] = sha256(p)
        if sources[str(p)] != sm['sources'][name]['sha256']: raise ValueError('raw source changed')
    for p, h in ((STATE/'clinical_state.npz', st['output_sha256']['clinical_state.npz']),
                 (EXTENSION/'extension.npz', em['output_sha256'])):
        sources[str(p)] = sha256(p)
        if sources[str(p)] != h: raise ValueError('derived source changed')
    with np.load(STATE/'clinical_state.npz') as z:
        a = {k:z[k] for k in ('hadm_ids', 'subject_ids', 'patient_splits', 'lab_axis_raw', 'lab_axis_inputs')}
    with np.load(EXTENSION/'extension.npz') as z:
        eos = z['eos']; np.testing.assert_array_equal(z['hadm_ids'], a['hadm_ids'])
    ids = {str(h):i for i,h in enumerate(a['hadm_ids'])}
    dates = {r['hadm_id']:(datetime.fromisoformat(r['admittime']), datetime.fromisoformat(r['dischtime']))
             for r in rows(SNAPSHOT/'cohort/admissions.csv.gz') if r['hadm_id'] in ids}
    lookup = {item:(j, unit, None) for j,(item, unit) in enumerate(LABS.values())}
    lookup.update({item:(5 if kind=='absolute' else 6, None, kind) for item,kind in EOS_ITEMS.items()})
    samples = defaultdict(list); audit = Counter()
    for r in rows(SNAPSHOT/'raw/labevents.csv.gz'):
        if r['hadm_id'] not in ids or r['itemid'] not in lookup: continue
        j, unit, kind = lookup[r['itemid']]; audit[NAMES[j]+':candidate'] += 1
        try: t, s = datetime.fromisoformat(r['charttime']), datetime.fromisoformat(r['storetime'])
        except ValueError: audit['invalid_time'] += 1; continue
        start, end = dates[r['hadm_id']]
        if not start <= t <= s <= end: audit['outside_visible_interval'] += 1; continue
        v, lo, hi = (finite(r[k]) for k in ('valuenum', 'ref_range_lower', 'ref_range_upper'))
        if kind:
            v, lo, hi = (canonical_eos(x, r['valueuom'], kind) for x in (v,lo,hi))
        elif r['valueuom'] != unit:
            audit['invalid_unit'] += 1; continue
        if v is None or lo is None or hi is None or lo >= hi or v < 0 or ((kind=='percent' or unit=='%') and v > 100):
            audit[NAMES[j]+':invalid_value_or_reference'] += 1; continue
        day = (t-start).total_seconds()/86400
        for direction, delta in (('low',0), ('high',1)):
            samples[ids[r['hadm_id']], 2*j+delta].append((day, abnormal(v, lo, hi, direction)))
        audit[NAMES[j]+':accepted'] += 1
    n = len(ids); raw = np.zeros((n, len(AXES)+len(EXTRA_NAMES), 9))
    raw[:, :len(AXES)] = a['lab_axis_raw']
    for (i,j), values in samples.items():
        start,end = dates[str(a['hadm_ids'][i])]
        raw[i, len(AXES)+j], _ = summarize(values, (end-start).total_seconds()/86400)
    values, valid, scales = scale_on_train(raw, a['patient_splits'])
    np.testing.assert_allclose(values[:,:22], a['lab_axis_inputs'], atol=1e-6, rtol=0)
    # 八个激素时间类别仍在检验分支，低/高方向共用，不冒充实际药效。
    context = eos[:,:,12:20].reshape(n,16).astype(np.float32)
    ROOT.mkdir(parents=True)
    np.savez_compressed(ROOT/'interface.npz', **{k:a[k] for k in ('hadm_ids','subject_ids','patient_splits')},
                        raw=raw, values=values, valid=valid, eos_context=context,
                        names=np.asarray((*AXES, *EXTRA_NAMES)))
    manifest = dict(status='unified_abnormality_interface_source_checked', source_sha256=sources,
        output_sha256=sha256(ROOT/'interface.npz'), names=list((*AXES,*EXTRA_NAMES)), channels=9,
        audit=dict(audit), train_coverage={name:int(raw[a['patient_splits']=='train',22+j,6].sum()) for j,name in enumerate(EXTRA_NAMES)},
        normalization=scales, no_outcome_used=True, no_high_frequency=True,
        limits='Extra/EOS missing valid reference is unobserved; original pH/pCO2 clinical definition retained.')
    (ROOT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(status=manifest['status'], train_coverage=manifest['train_coverage'], audit=dict(audit))), flush=True)


if __name__ == '__main__': build()
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/v6_audit_report.py · lines 1–98

<!-- REVIEW_FRAGMENT {"path":"patient_data/v6_audit_report.py","part":1,"start_line":1,"end_line":98,"source_sha256":"5d6bc00054638cc689c170a304389f17b5232bf15fffc71fabb3360ba35962b9","fragment_sha256":"5d6bc00054638cc689c170a304389f17b5232bf15fffc71fabb3360ba35962b9","added_terminal_newline":false} -->
````python
"""从审计产物生成中文全字段说明及未用来源清单；不修改训练合同。"""
from pathlib import Path
import json
import csv
import numpy as np
from patient_data.export_mimiciv import sha256
from patient_data.v6_input_audit_compact import ROOT, LAB_NAMES, LAB_FAMILY, FAMILIES


def run():
    audit=json.loads((ROOT/'audit.json').read_text());rows=audit['rows']
    source_dir=Path('patient_data/processed/v6_resp_micro_context_2026-10-09_r2')
    source=json.loads((source_dir/'manifest.json').read_text())
    with np.load(source_dir/'resp_micro_context.npz') as z:context={k:z[k] for k in z.files}
    # 此时读取的源必须与已保存清单一致；不以历史文档中的零/未导出描述作当前事实。
    for name,digest in source['output_sha256'].items():
        if sha256(source_dir/name)!=digest:raise ValueError('source context changed')
    extra=dict(status='existing_source_inventory_not_added_to_model',respiratory_numeric=source['numeric_coverage'],
      respiratory_support_and_duration=source['support_coverage'],array_shapes={k:list(v.shape) for k,v in context.items()},
      source_manifest_sha256=sha256(source_dir/'manifest.json'),live_postgresql_checked=False,
      live_check_limitation='psql read-only query lacked process credential; no credential written or saved; existing exports independently hash checked')
    (ROOT/'other_sources.json').write_text(json.dumps(extra,ensure_ascii=False,indent=2),encoding='utf-8')
    reasons={
      'acid_base':'通气/酸碱代偿与严重度；需标本确认，不把静脉血气当动脉',
      'inflammation':'感染/炎症及免疫细胞组成；EOS用于ICS反应背景，非单向异常或确证感染',
      'hematology_nutrition':'贫血/营养/炎症背景；临床相关不保证再入院独立增量',
      'renal':'共病/治疗安全及全身严重度，保留用于整体组比较',
      'electrolytes':'呼吸肌与治疗安全/全身病情，非COPD特异',
      'metabolic_perfusion':'糖代谢/激素影响/灌注背景，保留用于整体组比较',
      'cardiac':'心肺重叠及心肌损伤鉴别，不能混合BNP与NT-proBNP或不同肌钙蛋白测定'}
    lines=['# 全输入审计：来源、稀疏、重复与候选取舍','',
      '19,401次预测出院、5,351患者；分母、严格491.21/J44.1任意诊断位结局及患者划分未变。',
      '本表审计的是强参照及全部近期候选的并集，不声称所有39轴/38药物已被采用为默认。原强参照只有22轴+7补充来源、7类治疗等；新接口/新实验分开登记。',
      '细叶阈值仅看训练患者≥20人，不按结局P值或测试分数选字段。少于20人药物仍上卷药效父层；罕见医学核心检验不自动删除。',
      '“有记录”与“异常/阳性”分开：正常记录、未检验、明确阴性、未提及不互换。',
      '', '## 全部检验轴/来源','',
      '|项目|英文键|覆盖住院/患者|覆盖%|异常峰值住院（仅原22轴）|取舍与理由|','|---|---|---:|---:|---:|---|']
    for r in rows:
        if r['group']!='laboratory':continue
        name=r['name'];j=LAB_NAMES.index(name);reason=reasons[FAMILIES[LAB_FAMILY[j]]]
        if name in audit['excluded_lab_candidates']:
            reason='简化包中暂屏蔽，不认定无用。'+reason
            if name=='hematocrit':reason+='；高红细胞压积的慢性缺氧背景并未被低Hb轴完全替代，不能永久删除'
            if name=='protein_total':reason+='；覆盖2.70%，和白蛋白主题部分重叠，不是完全重复'
        if name in ('bicarbonate_low','bicarbonate_high','wbc_low','wbc_high','sodium_low','sodium_high','potassium_low','potassium_high','glucose_low','glucose_high'):
            reason='合并同分析物概念，两个方向的内容保留。'+reason
        abnormal=str(r['positive_admissions']) if j<22 else '未设异常标签'
        lines.append(f"|{r['chinese']}|{name}|{r['observed_admissions']:,}/{r['observed_patients']:,}|{r['observed_percent']:.2f}|{abnormal}|{reason}|")
    lines+=['','高低轴合并5对后为34概念；简化包暂屏蔽4概念后30。高碳酸酸中毒是pH/PCO2组合，保留为知识特征，不当独立测量。',
      '中性/淋巴百分比的训练折最新值Spearman约−0.90，但不是完全重复，不能据此删除其中一个。其它当前审计的额外检验未发现绝对相关≥0.80的成对关系。',
      '','## 全部药物身份与真实可用剂量','',
      '给药、药房及关联处方互补。网络只看到统一治疗记录、单次剂量/单位及频率，不看到执行/医嘱来源类别。来源留作审计，避免声称计划均被执行。',
      '不把复方成分剂量拆成两个单药重复相加；不用累计量、药物日数、首末日期或跨度。mg/mcg/g规范化；puff/inh/neb/ml不互换。',
      '实际最大单次剂量优先，缺失时使用可用医嘱最大单次剂量；实际每记录日频率优先，缺失时使用计划频率。该接口仍不能解释完整药物疗效或住院外依从性。',
      '万古霉素旧实际汇总途径不清，新接口仅采用已核验IV医嘱；不可把口服万古霉素当全身抗呼吸感染。',
      '', '|药物身份|覆盖住院/训练患者|原完整实际剂量住院|医嘱补单次剂量住院|细叶取舍|','|---|---:|---:|---:|---|']
    for r in rows:
        if r['group']=='drug_identity':
            decision='保留细叶' if r['proposal']=='fine_leaf_keep' else '仅保留父药效，无独立细叶'
            lines.append(f"|{r['name']}|{r['observed_admissions']:,}/{r['train_patients']:,}|{r['actual_single_dose']:,}|{r['new_order_dose_only']:,}|{decision}|")
    lines+=['','上表万古霉素“原实际剂量”只作审计，新接口不使用该列。吸入制剂多种单位/组合导致剂量完整度不均，不能仅凭有药名就假装已有可靠mg累计。',
      '药效父层为SABA短效β₂激动剂、SAMA短效抗胆碱能、LABA长效β₂激动剂、LAMA长效抗胆碱能、ICS吸入激素、全身激素、抗菌、黏液相关、甲基黄嘌呤及PDE4抑制剂。',
      'GOLD主要治疗类别提供候选理由；茶碱/愈创甘油醚/广谱抗菌药不等于推荐用于全部COPD患者，也不证明出院后独立预测价值。',
      '', '## 微生物、支持、静态和文本：全部现有字段','',
      '|组|字段|有记录住院/患者|阳性住院|处理|','|---|---|---:|---:|---|']
    for r in rows:
        if r['group'] in ('laboratory','drug_identity'):continue
        reason={'micro':'共享培养mask去重；内容全部保留，阳性不等于活动感染/定植',
          'support':'保留3类；仅ICU记录证据，缺失不是未支持',
          'frozen_history':'不动冻结H34基础风险', 'frozen_current':'不动冻结H34诊断/去向参照',
          'text':'保留B四字段现有冻结向量；缺失不补造正常',
          'drug_class':'上卷共享药效；不重复累计复方剂量'}[r['group']]
        lines.append(f"|{r['group']}|{r['name']}|{r['observed_admissions']:,}/{r['observed_patients']:,}|{r['positive_admissions']:,}|{reason}|")
    lines+=['','静态/文本表中“阳性”仅表示非零编码/存在文本，不是疾病结局阳性。当前诊断编码缺乏不等于临床无病；去向未填不补为回家。',
      '','## 已保存但当前强参照未利用的信息','',
      '|信息|本地核验|是否已入本轮风险网络|注意事项|','|---|---|---|---|']
    for name,r in source['numeric_coverage'].items():
        lines.append(f"|{name}|{r['admissions']:,}住院/{r['patients']:,}患者|否|已保存低频锚点，最后ICU记录不是出院状态|")
    lines+=['|通气时长、拔管/重插管|已保存resp_micro_context.npz及3,133条原始操作记录|否，当前仅3类支持|需确认区间、ICU覆盖与适应证；不是所有病人都可测|',
      '|胸片结果|12,346份报告，4,888次住院；有CheXbert、RadBERT/RadGraph派生|否，已有独立历史实验|重复报告可能用于设备检查；专科抽取不是专家金标准|',
      '|胸部CT形态|2,118份报告，1,872次住院；专科派生已保存|否，已有独立历史实验|覆盖9.65%；不能把缺CT当没有肺气肿/支扩|',
      '|FEV1/FVC、家庭氧疗、吸入技术、依从性、功能/照护|已有小结原句筛选，肺功能主题5,012次句段命中|无独立标准化事实；四字段向量可能包含部分文本|命中句段不是5,012患者或有效肺功能测量；要核日期/否定/单位|',
      '|BMI/体重及其变化|官方OMR存在；本地仅核到旧队列09_validation_omr.csv的历史统计|否|旧队列覆盖不应用作现19,401分母；本轮未成功实时查询，未假装完成覆盖|',
      '|急诊分诊、门诊治疗、服务/转科、操作代码|官方MIMIC对应模块/表；本快照来源没有全部导出|否|ED是否本机导入待确认；服务/收费不自动等同病情，不以ID预测|',
      '|PaO2、血气氧饱和度、碱剩余|此前本快照候选审计有数值，4,261/1,343/4,262次住院（混合标本）|否|不是已确认动脉PaO2；需氧疗上下文，不能随意算P/F|',
      '|PCT|本快照d_labitems的procalcitonin名称检索未见item|否|不据此断言全库或文本没有PCT|',
      '','## 来源与可复核文件','',
      '- all_inputs.csv / audit.json：全部字段覆盖、患者数、异常/阳性、候选及相关性。',
      '- interface.npz / normalization.json：独立新接口，不含结局，训练折剂量尺度。',
      '- other_sources.json：已保存来源、数组形状与本次实时核验限制。',
      '- 原始快照：patient_data/mimiciv_copd/snapshot_2026-09-27；补充来源：同级supplement_resp_micro_2026-10-09。',
      '- 初版无r2目录为开发审计，万古霉素途径修订后只使用r2训练。',
      '','临床/方法依据见docs/V6_AUDIT_LITERATURE_2026-10-12.md；正式结果见training_runs/v6_audited_coarse_fine_2026-10-12。没有删除原始数据或编码。']
    (ROOT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='comprehensive_readme_and_source_inventory_generated',root=str(ROOT))))


if __name__=='__main__':run()
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/v6_detail_extension.py · lines 1–246

<!-- REVIEW_FRAGMENT {"path":"patient_data/v6_detail_extension.py","part":1,"start_line":1,"end_line":246,"source_sha256":"b11fa069b1c98f145f2b6b459204f1afa7ef021d8ee211183b2b7bd18356dbd3","fragment_sha256":"b11fa069b1c98f145f2b6b459204f1afa7ef021d8ee211183b2b7bd18356dbd3","added_terminal_newline":false} -->
````python
"""V6 增量候选：分子级给药计数/剂量和额外检验，原始文件只读。"""
from collections import Counter, defaultdict
from datetime import datetime
import argparse
import json
import math
from pathlib import Path
import re
import time
import numpy as np
from patient_data.export_mimiciv import rows, sha256
from patient_data.preprocess_events import ADMINISTERED
from patient_data.eos_med_context import SYSTEMIC_ROUTES, finite
from patient_data.compact_multisource import INHALED_ROUTES, order_available
from training.v5.numeric_preprocessing import robust_parameters

LABS = {
    'neutrophils_percent': ('51256', '%'),
    'lymphocytes_percent': ('51244', '%'),
    'rdw': ('51277', '%'),
    'anion_gap': ('50868', 'mEq/L'),
    'chloride': ('50902', 'mEq/L'),
}
# 原料/复方层次固定，不把不同分子的 mg 相加或当作等效药效。
MOLECULES = ('albuterol', 'ipratropium', 'prednisone', 'methylprednisolone',
    'hydrocortisone', 'dexamethasone', 'azithromycin', 'ceftriaxone',
    'levofloxacin', 'doxycycline', 'piperacillin_tazobactam', 'vancomycin')
LAB_CHANNELS = ('first', 'minimum', 'maximum', 'latest', 'latest_age_log_days')
MED_CHANNELS = ('recorded_administrations', 'recorded_calendar_days',
    'maximum_administrations_per_calendar_day', 'administrations_per_recorded_day',
    'complete_recorded_dose_mg', 'maximum_recorded_single_dose_mg', 'ordered_doses_per_24h')


def molecule_matches(name):
    name = name.lower()
    result = []
    for molecule in MOLECULES:
        if molecule == 'piperacillin_tazobactam':
            hit = ('piperacillin' in name and 'tazobactam' in name) or 'zosyn' in name
        elif molecule == 'albuterol':
            hit = bool(re.search(r'\balbuterol\b|\bsalbutamol\b', name)) or 'duoneb' in name
        elif molecule == 'ipratropium':
            hit = 'ipratropium' in name or 'duoneb' in name
        else:
            hit = bool(re.search(r'\b'+molecule+r'\b', name))
        if hit: result.append(molecule)
    return tuple(result)


def dose_mg(detail, molecule):
    """只累加扫描子行。相同 ordinal 重复去重、冲突/缺单位/不同分子均拒绝。"""
    components = {}
    for r in detail:
        ordinal = r['parent_field_ordinal']
        if ordinal in ('', r'\N'): continue
        product = r['product_description']
        if product not in ('', r'\N') and molecule_matches(product) != (molecule,): return None, 'ambiguous_product'
        value = finite(r['dose_given'])
        factor = {'mg': 1., 'mcg': .001, 'ug': .001, 'g': 1000., 'gm': 1000.}.get(r['dose_given_unit'].strip().lower())
        if value is None or value < 0 or factor is None: return None, 'unknown_dose_or_unit'
        dose = value * factor
        if ordinal in components and components[ordinal] != dose: return None, 'conflicting_ordinal'
        components[ordinal] = dose
    if not components: return None, 'no_formulary_components'
    return sum(components.values()), 'verified_component_sum'


def scale(values, masks, train):
    transformed = np.log1p(values)
    result = np.zeros_like(values, dtype=np.float32)
    normalization = {}
    for j in range(values.shape[1]):
        for k in range(values.shape[2]):
            valid = train & masks[:, j, k]
            if not valid.any():
                masks[:, j, k] = False
                normalization[f'{j}/{k}'] = {'method': 'no_training_observation'}
                continue
            p = robust_parameters(transformed[valid, j, k])
            result[:, j, k] = np.where(masks[:, j, k], np.clip(
                (transformed[:, j, k]-p['median']) / p['scale'], -8, 8), 0)
            normalization[f'{j}/{k}'] = p
    return result, normalization


def build(snapshot, state_dir, eos_dir, output):
    if output.exists(): raise FileExistsError(output)
    started = time.perf_counter()
    sm = json.loads((snapshot/'metadata/manifest.json').read_text())
    st = json.loads((state_dir/'manifest.json').read_text())
    em = json.loads((eos_dir/'manifest.json').read_text())
    if sm['status'] != 'validated' or em['status'] != 'eos_med_context_validated_not_trained':
        raise ValueError('unvalidated parents')
    if sha256(state_dir/'clinical_state.npz') != st['output_sha256']['clinical_state.npz']:
        raise ValueError('state hash mismatch')
    if em['parent_clinical_state_manifest_sha256'] != sha256(state_dir/'manifest.json'):
        raise ValueError('EOS state lineage differs')
    if sha256(eos_dir/'eos_med_context.npz') != em['output_sha256']['eos_med_context.npz']:
        raise ValueError('EOS output changed')
    with np.load(state_dir/'clinical_state.npz') as z: state = {k:z[k] for k in z.files}
    with np.load(eos_dir/'eos_med_context.npz') as z: eos = {k:z[k] for k in z.files}
    for k in ('hadm_ids', 'subject_ids', 'patient_splits'):
        if not np.array_equal(state[k], eos[k]): raise ValueError('EOS members/splits differ')
    ids = [str(x) for x in state['hadm_ids']]; index = {h:i for i,h in enumerate(ids)}
    if len(index) != 19401: raise ValueError('member drift')
    admissions = {r['hadm_id']: (datetime.fromisoformat(r['admittime']), datetime.fromisoformat(r['dischtime']))
        for r in rows(snapshot/'cohort/admissions.csv.gz') if r['hadm_id'] in index}
    sources = ('raw/labevents', 'raw/emar', 'raw/emar_detail', 'raw/pharmacy', 'raw/d_labitems', 'cohort/admissions')
    hashes = {}
    for name in sources:
        hashes[name] = sha256(snapshot/(name+'.csv.gz'))
        if hashes[name] != sm['sources'][name]['sha256']: raise ValueError('raw source changed '+name)
    dictionary = {r['itemid']:r for r in rows(snapshot/'raw/d_labitems.csv.gz')}
    audit = Counter(); observations = defaultdict(list)
    lab_map = {item:(j,unit) for j,(item,unit) in enumerate(LABS.values())}
    for r in rows(snapshot/'raw/labevents.csv.gz'):
        if r['hadm_id'] not in index or r['itemid'] not in lab_map: continue
        j,unit = lab_map[r['itemid']]; key = list(LABS)[j]
        audit['lab:'+key+':candidate'] += 1
        try:
            t,s = datetime.fromisoformat(r['charttime']),datetime.fromisoformat(r['storetime'])
        except ValueError:
            audit['lab:'+key+':invalid_time'] += 1; continue
        start,end = admissions[r['hadm_id']]; value = finite(r['valuenum'])
        if not start <= t <= s <= end:
            audit['lab:'+key+':outside_visible_interval'] += 1; continue
        if value is None or value < 0 or (unit == '%' and value > 100) or r['valueuom'] != unit:
            audit['lab:'+key+':invalid_value_or_unit'] += 1; continue
        observations[r['hadm_id'],j].append((t,value))
    print('lab sources audited', flush=True)
    # eMAR 父记录只数一次。保存候选键，子行从 detail 补给药途径和可核验剂量。
    records = {}; execution_source = set()
    pharmacy_routes = {}; orders = defaultdict(list); order_source = set()
    for r in rows(snapshot/'raw/pharmacy.csv.gz'):
        h = r['hadm_id']
        if h not in index or not order_available(r,admissions[h])[0]: continue
        order_source.add(h)
        pharmacy_routes[h,r['pharmacy_id']] = (r['route'].strip().upper(),r['starttime'],r['stoptime'])
        f = finite(r['doses_per_24_hrs'])
        if f is None or not 0 < f <= 48: continue
        for molecule in molecule_matches(r['medication']):
            allowed = INHALED_ROUTES if molecule in ('albuterol','ipratropium') else SYSTEMIC_ROUTES
            if r['route'].strip().upper() in allowed: orders[h,molecule].append(f)
    for r in rows(snapshot/'raw/emar.csv.gz'):
        h = r['hadm_id']
        if h not in index: continue
        try: t,s = datetime.fromisoformat(r['charttime']),datetime.fromisoformat(r['storetime'])
        except ValueError: continue
        if not admissions[h][0] <= t <= s <= admissions[h][1]: continue
        execution_source.add(h)
        molecules = molecule_matches(r['medication'])
        if not molecules or r['event_txt'] not in ADMINISTERED: continue
        key = (r['emar_id'], r['emar_seq'])
        if key in records: raise ValueError('duplicate eMAR parent')
        records[key] = (h,t,molecules,r['medication'],r['pharmacy_id'])
    details = defaultdict(list)
    for r in rows(snapshot/'raw/emar_detail.csv.gz'):
        key = (r['emar_id'], r['emar_seq'])
        if key in records: details[key].append(r)
    administered = defaultdict(list)
    for key,(h,t,molecules,name,pid) in records.items():
        detail = details[key]
        routes = {r['route'].strip().upper() for r in detail if r['route'] not in ('', r'\N')}
        if not routes and (h,pid) in pharmacy_routes:
            route,first,last = pharmacy_routes[h,pid]
            if datetime.fromisoformat(first) <= t and (last in ('',r'\N') or t <= datetime.fromisoformat(last)):
                routes = {route}
                audit['route_from_linked_visible_order'] += 1
        for molecule in molecules:
            allowed = INHALED_ROUTES if molecule in ('albuterol','ipratropium') else SYSTEMIC_ROUTES
            if not routes or not routes.issubset(allowed):
                audit['med:'+molecule+':route_not_verified'] += 1; continue
            if any(r['complete_dose_not_given'].strip().upper() in ('Y','YES','TRUE','1') for r in detail):
                audit['med:'+molecule+':incomplete_dose'] += 1; continue
            dose,reason = dose_mg(detail,molecule) if len(molecules)==1 else (None,'combination_dose_not_allocated')
            audit['med:'+molecule+':'+reason] += 1
            # 明确零剂量不计作执行；无法解析剂量仍保留执行记录，但剂量mask关闭。
            if dose == 0: audit['med:'+molecule+':zero_dose'] += 1; continue
            administered[h,molecule].append((t,dose))
    print('execution and formulary components audited', flush=True)
    lv = np.zeros((len(ids),len(LABS),5)); lm = np.zeros_like(lv,dtype=bool)
    mv = np.zeros((len(ids),len(MOLECULES),7)); mm = np.zeros_like(mv,dtype=bool)
    med_source = np.zeros((len(ids),len(MOLECULES),2),dtype=bool)
    coverage = {'labs':{}, 'medications':{}}
    for h,i in index.items():
        for j in range(len(LABS)):
            samples = observations.get((h,j),[])
            if not samples: continue
            by_time = defaultdict(list)
            for t,v in samples: by_time[t].append(v)
            samples = [(t,float(np.median(v))) for t,v in sorted(by_time.items())]
            lv[i,j] = [samples[0][1],min(v for _,v in samples),max(v for _,v in samples),samples[-1][1],
                        math.log1p((admissions[h][1]-samples[-1][0]).total_seconds()/86400)]
            lm[i,j] = True
        for j,molecule in enumerate(MOLECULES):
            med_source[i,j] = h in execution_source,h in order_source
            if h in execution_source: mm[i,j,:4] = True
            samples = administered.get((h,molecule),[])
            if samples:
                days = Counter(t.date() for t,_ in samples)
                mv[i,j,:4] = len(samples),len(days),max(days.values()),len(samples)/len(days)
                doses = [d for _,d in samples]
                if all(d is not None for d in doses):
                    mv[i,j,4:6] = sum(doses),max(doses); mm[i,j,4:6] = True
            if orders.get((h,molecule)):
                mv[i,j,6] = float(np.median(orders[h,molecule])); mm[i,j,6] = True
    train = state['patient_splits']=='train'
    ls,ln = scale(lv,lm,train); ms,mn = scale(mv,mm,train)
    for j,name in enumerate(LABS):
        coverage['labs'][name] = dict(itemid=list(LABS.values())[j][0],label=dictionary[list(LABS.values())[j][0]]['label'],
            unit=list(LABS.values())[j][1],admissions=int(lm[:,j,0].sum()),
            patients=len(np.unique(state['subject_ids'][lm[:,j,0]])))
    for j,name in enumerate(MOLECULES):
        coverage['medications'][name] = dict(recorded_execution=int((mv[:,j,0]>0).sum()),
            full_recorded_mg_dose=int(mm[:,j,4].sum()),ordered_numeric_frequency=int(mm[:,j,6].sum()))
    labs = np.concatenate((ls,lm),2).astype(np.float32)
    meds = np.concatenate((ms,mm,med_source),2).astype(np.float32)
    for a in (labs,meds,eos['eos_features']):
        if not np.isfinite(a).all(): raise ValueError('nonfinite extension')
    output.mkdir(parents=True)
    np.savez_compressed(output/'extension.npz',**{k:state[k] for k in ('hadm_ids','subject_ids','patient_splits')},
        new_labs=labs,eos=eos['eos_features'],med_details=meds,
        lab_values_raw=lv,lab_masks=lm,med_values_raw=mv,med_masks=mm)
    report = dict(status='source_checked_extension_not_trained',admissions=len(ids),
        parent_state_manifest_sha256=sha256(state_dir/'manifest.json'),eos_manifest_sha256=sha256(eos_dir/'manifest.json'),
        raw_source_sha256=hashes,lab_names=list(LABS),med_names=list(MOLECULES),
        lab_channels=list(LAB_CHANNELS),med_channels=list(MED_CHANNELS),coverage=coverage,audit=dict(audit),
        shapes=dict(new_labs=list(labs.shape),eos=list(eos['eos_features'].shape),med_details=list(meds.shape)),
        normalization=dict(labs=ln,medications=mn),output_sha256=sha256(output/'extension.npz'),
        seconds=time.perf_counter()-started,
        limits=['Counts are recorded eMAR events, not complete exposure or prescribed frequency.',
            'Doses only component-verified mg by molecule; no mL/puff/NEB or potency conversion; missing is masked.',
            'Missing execution route may use its linked visible active pharmacy order route; this is an order route, not observed route.',
            'Ordered frequency is median valid pharmacy order frequency, not executed frequency.',
            'No complete drug-lab response sequence or causal treatment effect is inferred.'])
    (output/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(status=report['status'],coverage=coverage,seconds=report['seconds'])),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--snapshot',type=Path,default=Path('patient_data/mimiciv_copd/snapshot_2026-09-27'))
    p.add_argument('--state',type=Path,default=Path('patient_data/processed/v6_clinical_state_2026-10-09_r2'))
    p.add_argument('--eos',type=Path,default=Path('patient_data/processed/v6_eos_med_context_2026-10-09_r3'))
    p.add_argument('--output',type=Path,default=Path('patient_data/processed/v6_detail_extension_2026-10-11_r2'))
    a=p.parse_args(); build(a.snapshot,a.state,a.eos,a.output)
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/v6_expanded_candidates.py · lines 1–143

<!-- REVIEW_FRAGMENT {"path":"patient_data/v6_expanded_candidates.py","part":1,"start_line":1,"end_line":143,"source_sha256":"0808d5f412509cbdd45ad309e4a58ef777da7eb3cb5d3ec970546c0a67c83ee2","fragment_sha256":"0808d5f412509cbdd45ad309e4a58ef777da7eb3cb5d3ec970546c0a67c83ee2","added_terminal_newline":false} -->
````python
"""独立扩展字典和派生数据；旧已哈希来源/生成器不修改。"""
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import argparse
import json
import re
import numpy as np
from patient_data import medication_identity as old
from patient_data.export_mimiciv import rows, sha256
from patient_data.eos_med_context import finite
from training.v5.numeric_preprocessing import robust_parameters

NEW_DRUGS=('levalbuterol','arformoterol','umeclidinium','beclomethasone','mometasone',
 'fluticasone_vilanterol','tiotropium_olodaterol','glycopyrrolate_formoterol',
 'roflumilast','theophylline','acetylcysteine','guaifenesin',
 'amoxicillin_clavulanate','ampicillin_sulbactam','cefepime','moxifloxacin','clarithromycin')
NAMES=old.NAMES+NEW_DRUGS
ALIASES={'levalbuterol':('xopenex',),'arformoterol':('brovana',),'umeclidinium':('incruse',),
 'beclomethasone':('qvar',),'mometasone':('asmanex',),'fluticasone_vilanterol':('breo',),
 'tiotropium_olodaterol':('stiolto',),'glycopyrrolate_formoterol':('bevespi',),
 'roflumilast':('daliresp',),'theophylline':('theo-24','theodur','theo-dur'),
 'acetylcysteine':('mucomyst',),'guaifenesin':('mucinex',),
 'amoxicillin_clavulanate':('augmentin',),'ampicillin_sulbactam':('unasyn',)}
ORIGINAL_IDENTITY=old.identity
INHALED=set(old.INHALED)|set(NEW_DRUGS[:8])|{'acetylcysteine'}
# 候选理由先于结局；同一项目的跨itemid来源只在单位相同时合并。
LABS={
 'neutrophils_absolute':(('52075','53159'),'K/uL','炎症/感染细胞组成'),
 'lymphocytes_absolute':(('51133','52769','53157'),'K/uL','免疫细胞组成'),
 'monocytes_percent':(('51254',),'%','炎症细胞组成'),
 'hematocrit':(('51221',),'%','贫血/红细胞增多与缺氧背景'),
 'mcv':(('51250',),'fL','贫血/营养表型'),
 'calcium_total':(('50893',),'mg/dL','电解质和全身病情背景'),
 'phosphate':(('50970',),'mg/dL','呼吸肌能量/电解质背景'),
 'protein_total':(('50976',),'g/dL','营养/炎症背景'),
 'fibrinogen':(('51214',),'mg/dL','有COPD加重/死亡关联证据的炎症凝血候选'),
 'troponin_t':(('51003',),'ng/mL','心肺重叠及心肌损伤背景，不是COPD特异指标')}


def identity(name):
    s=str(name).lower()
    # 祛痰镇咳复方不当作纯guaifenesin累计剂量；本轮未建独立复方槽位。
    if ('guaifenesin' in s or 'mucinex' in s) and any(x in s for x in ('codeine','dextromethorphan',' dm')):return None
    combos={'fluticasone_vilanterol':('fluticasone','vilanterol'),
      'tiotropium_olodaterol':('tiotropium','olodaterol'),
      'glycopyrrolate_formoterol':('glycopyrrolate','formoterol'),
      'amoxicillin_clavulanate':('amoxicillin','clav'),
      'ampicillin_sulbactam':('ampicillin','sulbactam')}
    # 排除本轮仍未建独立接口的复方，不能把它们误认成单药。
    if ('trelegy' in s or all(x in s for x in ('fluticasone','umeclidinium','vilanterol'))
        or 'dulera' in s or all(x in s for x in ('mometasone','formoterol'))):return None
    for n,parts in combos.items():
        if all(x in s for x in parts) or any(x in s for x in ALIASES.get(n,())):return n
    hits=[n for n in NEW_DRUGS if re.search(r'\b'+n+r'\b',s) or any(x in s for x in ALIASES.get(n,()))]
    prior=ORIGINAL_IDENTITY(s)
    if prior:hits.append(prior)
    return hits[0] if len(set(hits))==1 else None


@contextmanager
def medication_catalog():
    """显式单进程字典适配：仅派生期间生效；终于恢复旧模块，不热改旧文件。"""
    saved=old.NAMES,old.INHALED,old.identity
    old.NAMES,old.INHALED,old.identity=NAMES,INHALED,identity
    try:yield
    finally:old.NAMES,old.INHALED,old.identity=saved


def summarize(observations,end):
    if not observations:return np.zeros(5,np.float32),np.zeros(5,bool)
    # 同项目同时间冲突不求任意平均；完全相同记录去重。
    buckets=defaultdict(set)
    for t,v in observations:buckets[t].add(v)
    items=sorted((t,next(iter(v))) for t,v in buckets.items() if len(v)==1)
    if not items:return np.zeros(5,np.float32),np.zeros(5,bool)
    values=[v for _,v in items]
    return np.asarray((values[0],min(values),max(values),values[-1],
           np.log1p((end-items[-1][0]).total_seconds()/86400)),np.float32),np.ones(5,bool)


def build_labs(snapshot,state_dir,output):
    if output.exists():raise FileExistsError(output)
    sm=json.loads((snapshot/'metadata/manifest.json').read_text());st=json.loads((state_dir/'manifest.json').read_text())
    if sm['status']!='validated' or sha256(state_dir/'clinical_state.npz')!=st['output_sha256']['clinical_state.npz']:raise ValueError('invalid parent')
    with np.load(state_dir/'clinical_state.npz') as z:state={k:z[k] for k in ('hadm_ids','subject_ids','patient_splits')}
    ix={str(h):i for i,h in enumerate(state['hadm_ids'])};sources={}
    for name in ('raw/labevents','raw/d_labitems','cohort/admissions'):
        p=snapshot/(name+'.csv.gz');sources[name]=sha256(p)
        if sources[name]!=sm['sources'][name]['sha256']:raise ValueError('raw changed')
    admissions={r['hadm_id']:(datetime.fromisoformat(r['admittime']),datetime.fromisoformat(r['dischtime']))
      for r in rows(snapshot/'cohort/admissions.csv.gz') if r['hadm_id'] in ix}
    dictionary={r['itemid']:r for r in rows(snapshot/'raw/d_labitems.csv.gz')}
    mapping={item:(j,unit) for j,(items,unit,_) in enumerate(LABS.values()) for item in items}
    obs=defaultdict(list);audit=Counter();units=defaultdict(Counter)
    for r in rows(snapshot/'raw/labevents.csv.gz'):
        h=r['hadm_id']
        if h not in ix or r['itemid'] not in mapping:continue
        j,unit=mapping[r['itemid']];name=list(LABS)[j];units[name][r['valueuom']]+=1
        try:t,s=datetime.fromisoformat(r['charttime']),datetime.fromisoformat(r['storetime'])
        except ValueError:audit[name+'/invalid_time']+=1;continue
        v=finite(r['valuenum']);start,end=admissions[h]
        if not start<=t<=s<=end:audit[name+'/not_visible']+=1;continue
        if v is None or v<0 or (unit=='%' and v>100) or r['valueuom']!=unit:
            audit[name+'/invalid_value_or_unit']+=1;continue
        obs[h,j].append((t,v));audit[name+'/accepted']+=1
    raw=np.zeros((len(ix),len(LABS),5),np.float32);mask=np.zeros_like(raw,bool)
    for (h,j),items in obs.items():raw[ix[h],j],mask[ix[h],j]=summarize(items,admissions[h][1])
    train=state['patient_splits']=='train';scaled=np.zeros_like(raw);normalization={};coverage={}
    for j,name in enumerate(LABS):
        for k in range(5):
            fit=train&mask[:,j,k]
            if not fit.any():mask[:,j,k]=False;continue
            # raw age already log-days; concentration channels transform log1p.
            x=np.log1p(raw[:,j,k]) if k<4 else raw[:,j,k]
            p=robust_parameters(x[fit]);normalization[f'{name}/{k}']=p
            valid=mask[:,j,k];scaled[valid,j,k]=np.clip((x[valid]-p['median'])/p['scale'],-8,8)
        observed=mask[:,j,0];coverage[name]=dict(admissions=int(observed.sum()),
          patients=int(len(np.unique(state['subject_ids'][observed]))),train_patients=int(len(np.unique(state['subject_ids'][observed&train]))),
          missing_admissions=int((~observed).sum()),raw_units=dict(units[name]))
    features=np.concatenate((scaled,mask.astype(np.float32)),2);output.mkdir(parents=True)
    np.savez_compressed(output/'expanded_labs.npz',**state,features=features,values_raw=raw,value_masks=mask,names=np.asarray(list(LABS)))
    (output/'normalization.json').write_text(json.dumps(normalization,indent=2),encoding='utf-8')
    manifest=dict(status='source_checked_expanded_labs',parent_state_manifest_sha256=sha256(state_dir/'manifest.json'),
      output_sha256=sha256(output/'expanded_labs.npz'),sources_sha256=sources,catalog=LABS,
      dictionary={i:dictionary[i] for ids,_,_ in LABS.values() for i in ids},coverage=coverage,audit=dict(audit),
      builder_sha256=sha256(Path(__file__)),rules='charttime<=storetime<=discharge; exact units; conflicting simultaneous values excluded; train-only scaling')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(dict(status=manifest['status'],coverage=coverage)),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,default=Path('patient_data/mimiciv_copd/snapshot_2026-09-27'))
    p.add_argument('--state',type=Path,default=Path('patient_data/processed/v6_clinical_state_2026-10-09_r2'))
    p.add_argument('--output',type=Path,default=Path('patient_data/processed/v6_expanded_candidates_2026-10-12_r2'));a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    with medication_catalog():old.build(a.snapshot,a.state,a.output/'medications')
    build_labs(a.snapshot,a.state,a.output/'labs')


if __name__=='__main__':main()
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/v6_expanded_medication_fix.py · lines 1–55

<!-- REVIEW_FRAGMENT {"path":"patient_data/v6_expanded_medication_fix.py","part":1,"start_line":1,"end_line":55,"source_sha256":"ee971824acc5540b2b2cc91ce3dcb148f4dee857b25a7a1cb2fc0ee0600bb5ab","fragment_sha256":"ee971824acc5540b2b2cc91ce3dcb148f4dee857b25a7a1cb2fc0ee0600bb5ab","added_terminal_newline":false} -->
````python
"""只修正新单药遮蔽已有固定复方的问题；原已哈希字典保留。"""
import argparse
import json
from pathlib import Path
from patient_data import v6_expanded_candidates as catalog
from patient_data import medication_identity as original
from patient_data.export_mimiciv import sha256

PREVIOUS=catalog.identity


def identity(name):
    text=str(name).lower()
    if 'trelegy' in text or all(x in text for x in ('fluticasone','umeclidinium','vilanterol')):return None
    prior=catalog.ORIGINAL_IDENTITY(name)
    if prior in ('albuterol_ipratropium','fluticasone_salmeterol','budesonide_formoterol','umeclidinium_vilanterol'):
        return prior
    return PREVIOUS(name)


def build(snapshot,state,output):
    if output.exists():raise FileExistsError(output)
    saved=catalog.identity;catalog.identity=identity
    try:
        with catalog.medication_catalog():original.build(snapshot,state,output)
    finally:catalog.identity=saved
    path=output/'manifest.json';m=json.loads(path.read_text())
    m['identity_fix']=dict(builder_sha256=sha256(Path(__file__)),previous_catalog_sha256=sha256(Path(catalog.__file__)),
        scope='Recognized existing fixed combinations take precedence over newly introduced constituent single drug.')
    path.write_text(json.dumps(m,indent=2),encoding='utf-8')


def audit_existing(snapshot,output):
    """新增三联排除边界；核对所有来源无该反例，证明已派生输出无需变更。"""
    from patient_data.export_mimiciv import rows
    hits=0
    for file,column in (('pharmacy','medication'),('emar','medication'),('emar_detail','product_description')):
        for row in rows(snapshot/f'raw/{file}.csv.gz'):
            s=row.get(column,'').lower()
            hits+=int(all(x in s for x in ('fluticasone','umeclidinium','vilanterol')))
    if hits:raise ValueError('generic triple found: must rederive with corrected boundary')
    path=output/'manifest.json';m=json.loads(path.read_text());prior=m['identity_fix']['builder_sha256']
    m['identity_fix'].update(builder_sha256=sha256(Path(__file__)),initial_extraction_builder_sha256=prior,
         boundary_equivalence_audit='all raw emar/pharmacy/product_description generic triple hits0; later guard does not change this dataset')
    path.write_text(json.dumps(m,indent=2),encoding='utf-8');print(json.dumps(dict(boundary_audit='passed',generic_triple_hits=hits)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,default=Path('patient_data/mimiciv_copd/snapshot_2026-09-27'))
    p.add_argument('--state',type=Path,default=Path('patient_data/processed/v6_clinical_state_2026-10-09_r2'))
    p.add_argument('--output',type=Path,default=Path('patient_data/processed/v6_expanded_candidates_2026-10-12_r3/medications'))
    p.add_argument('--audit-existing',action='store_true')
    a=p.parse_args()
    if a.audit_existing:audit_existing(a.snapshot,a.output)
    else:build(a.snapshot,a.state,a.output)
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/v6_input_audit_compact.py · lines 1–234

<!-- REVIEW_FRAGMENT {"path":"patient_data/v6_input_audit_compact.py","part":1,"start_line":1,"end_line":234,"source_sha256":"d56ffe3869a866c018b4389444515527e7babb1a1125747bce7ae1d77ccf9a52","fragment_sha256":"d56ffe3869a866c018b4389444515527e7babb1a1125747bce7ae1d77ccf9a52","added_terminal_newline":false} -->
````python
"""全输入审计及独立粗细接口；只读旧来源，训练折拟合新尺度。"""
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import argparse
import csv
import json
import numpy as np
from scipy.stats import spearmanr
from patient_data.export_mimiciv import rows, sha256
from patient_data.compact_multisource import order_available, INHALED_ROUTES, ORDER_SYSTEMIC_ROUTES
from patient_data.v6_expanded_medication_fix import identity
from patient_data.v6_expanded_candidates import INHALED
from patient_data.eos_med_context import finite
from patient_data.medication_identity import UNITS
from training.v5.numeric_preprocessing import robust_parameters
from training.v6.expanded_original_experiment import checked, parser as parent_parser
from training.v6.data import HISTORY_NAMES, CURRENT_NAMES

ROOT = Path('patient_data/processed/v6_audited_coarse_fine_2026-10-12_r2')
LAB_ZH = ('动脉低pH','动脉高二氧化碳分压','低碳酸氢根','高碳酸氢根','高白细胞','低白蛋白','低血红蛋白','高肌酐',
 '高碳酸酸中毒','低白细胞','低钠','高钠','低钾','高钾','高尿素氮','低血糖','高血糖','低血小板','低镁','高乳酸','高CRP','高NT-proBNP',
 '中性粒细胞百分比','淋巴细胞百分比','红细胞分布宽度','阴离子间隙','氯','嗜酸粒细胞绝对值','嗜酸粒细胞百分比',
 '中性粒细胞绝对值','淋巴细胞绝对值','单核细胞百分比','红细胞压积','平均红细胞体积','总钙','磷','总蛋白','纤维蛋白原','肌钙蛋白T')
LAB_NAMES = ('arterial_low_ph','arterial_high_pco2','bicarbonate_low','bicarbonate_high','wbc_high','albumin_low','hemoglobin_low',
 'creatinine_high','arterial_hypercapnic_acidosis','wbc_low','sodium_low','sodium_high','potassium_low','potassium_high',
 'urea_nitrogen_high','glucose_low','glucose_high','platelet_low','magnesium_low','lactate_high','crp_high','ntprobnp_high',
 'neutrophils_percent','lymphocytes_percent','rdw','anion_gap','chloride','eosinophils_absolute','eosinophils_percent',
 'neutrophils_absolute','lymphocytes_absolute','monocytes_percent','hematocrit','mcv','calcium_total','phosphate','protein_total','fibrinogen','troponin_t')
LAB_FAMILY = (0,0,0,0,1,2,2,3,0,1,4,4,4,4,3,5,5,2,4,5,1,6,1,1,2,0,4,1,1,1,1,1,2,2,4,4,2,1,6)
FAMILIES = ('acid_base','inflammation','hematology_nutrition','renal','electrolytes','metabolic_perfusion','cardiac')
MERGE = ((2,3),(4,9),(10,11),(12,13),(15,16))
COARSE_DRUGS = ('SABA','SAMA','LABA','LAMA','ICS','systemic_steroid','antibacterial','mucus_targeted','methylxanthine','PDE4_inhibitor')
DRUG_PARENTS = {
 'albuterol':(0,), 'levalbuterol':(0,), 'ipratropium':(1,), 'salmeterol':(2,), 'formoterol':(2,), 'arformoterol':(2,),
 'tiotropium':(3,), 'umeclidinium':(3,), 'fluticasone':(4,), 'budesonide':(4,), 'beclomethasone':(4,), 'mometasone':(4,),
 'albuterol_ipratropium':(0,1),'fluticasone_salmeterol':(4,2),'budesonide_formoterol':(4,2),
 'umeclidinium_vilanterol':(3,2),'fluticasone_vilanterol':(4,2),'tiotropium_olodaterol':(3,2),'glycopyrrolate_formoterol':(3,2),
 'prednisone':(5,), 'methylprednisolone':(5,), 'hydrocortisone':(5,), 'dexamethasone':(5,),
 'azithromycin':(6,), 'ceftriaxone':(6,), 'levofloxacin':(6,), 'doxycycline':(6,), 'piperacillin_tazobactam':(6,),
 'vancomycin':(6,), 'amoxicillin_clavulanate':(6,), 'ampicillin_sulbactam':(6,), 'cefepime':(6,), 'moxifloxacin':(6,), 'clarithromycin':(6,),
 'acetylcysteine':(7,), 'guaifenesin':(7,), 'theophylline':(8,), 'roflumilast':(9,)}


def scalar_dose(value, unit):
    """不猜范围、浓度或复方有效成分比例；质量单位统一mg，其余保留剂型单位。"""
    v=finite(value); u=str(unit).strip().lower()
    factors={'mg':1.,'mcg':.001,'ug':.001,'g':1000.,'gm':1000.}
    aliases={'puff':'puff','puffs':'puff','neb':'neb','nebs':'neb','inh':'inh','ml':'ml'}
    if v is None or v<=0:return None
    if u in factors:return v*factors[u],UNITS.index('mg')
    if u in aliases:return v,UNITS.index(aliases[u])
    return None


def order_doses(snapshot, ids, drug_names):
    """按pharmacy_id联接可用时间与状态，处方只作缺失单次剂量的补充。"""
    ix={str(h):i for i,h in enumerate(ids)}; di={n:j for j,n in enumerate(drug_names)}
    admissions={r['hadm_id']:(datetime.fromisoformat(r['admittime']),datetime.fromisoformat(r['dischtime']))
                for r in rows(snapshot/'cohort/admissions.csv.gz') if r['hadm_id'] in ix}
    pharmacy={}; observed=np.zeros(len(ids),bool)
    for r in rows(snapshot/'raw/pharmacy.csv.gz'):
        h=r['hadm_id']
        if h in ix and order_available(r,admissions[h])[0]:
            pharmacy[h,r['pharmacy_id']]=r; observed[ix[h]]=True
    obs=defaultdict(list); seen=set(); reasons=Counter(); iv_vanco_frequency=defaultdict(list)
    for r in rows(snapshot/'raw/prescriptions.csv.gz'):
        h=r['hadm_id']; p=pharmacy.get((h,r['pharmacy_id']))
        if h not in ix or p is None:continue
        drug=identity(r['drug'])
        if drug not in di:continue
        route=r.get('route','').strip().upper()
        if route not in (INHALED_ROUTES if drug in INHALED else ORDER_SYSTEMIC_ROUTES):continue
        if drug=='vancomycin' and route not in {'IV','IV DRIP','IV PUSH','IV BOLUS','INTRAVENOUS'}:
            reasons['oral_or_noniv_vancomycin_order_not_systemic']+=1;continue
        if not order_available(r,admissions[h],entered=datetime.fromisoformat(p['entertime']),status=p['status'])[0]:continue
        value=scalar_dose(r.get('dose_val_rx'),r.get('dose_unit_rx'))
        if value is None:reasons['unparseable_scalar_or_unit']+=1;continue
        key=(h,r['pharmacy_id'],drug,r['starttime'],r['stoptime'],value)
        if key in seen:reasons['duplicate_order_dose']+=1;continue
        seen.add(key);obs[ix[h],di[drug]].append(value)
        if drug=='vancomycin':
            frequency=finite(p.get('doses_per_24_hrs'))
            if frequency is not None and 0<frequency<=48:iv_vanco_frequency[ix[h]].append(frequency)
    raw=np.zeros((len(ids),len(drug_names)),np.float32);mask=np.zeros_like(raw,bool);unit=np.zeros_like(raw,np.int64)
    for (i,j),items in obs.items():
        units={u for _,u in items}
        if len(units)!=1:reasons['mixed_order_units_not_combined']+=1;continue
        unit[i,j]=next(iter(units)); raw[i,j]=max(v for v,_ in items);mask[i,j]=True
    vf=np.zeros(len(ids),np.float32);vfm=np.zeros(len(ids),bool)
    for i,values in iv_vanco_frequency.items():vf[i]=np.median(values);vfm[i]=True
    return raw,mask,unit,observed,dict(reasons),vf,vfm


def scale_per_unit(raw, masks, units, train, names):
    x=np.zeros_like(raw);parameters={}
    for j,name in enumerate(names):
        for k in range(raw.shape[2]):
            # 通道0是单次剂量，其余为记录或计划频率；不能跨剂型拟合剂量尺度。
            for unit in (range(len(UNITS)) if k==0 else (-1,)):
                valid=masks[:,j,k] & (units[:,j]==unit if unit>=0 else True)
                fitted=valid&train
                if not fitted.any():masks[valid,j,k]=False;continue
                p=robust_parameters(np.log1p(raw[fitted,j,k]));parameters[f'{name}/{k}/{unit}']=p
                x[valid,j,k]=np.clip((np.log1p(raw[valid,j,k])-p['median'])/p['scale'],-8,8)
    return x,parameters


def run(output):
    if output.exists():raise FileExistsError(output)
    args=parent_parser().parse_args([]);args.expanded=Path('patient_data/processed/v6_expanded_candidates_2026-10-12_r3')
    d,c,h,ext,added,med,prov,active,norm_checks=checked(args)
    train=d.arrays['patient_splits']=='train'; subjects=d.arrays['subject_ids']; n=len(train)
    # checked为了旧实验屏蔽稀有新药；这里重新只读加载，供完整审计与上卷类别。
    with np.load(args.expanded/'medications/medication_identity.npz') as z:med={k:z[k] for k in z.files}
    output.mkdir(parents=True)
    catalog=[]
    def coverage(group,name,observed,positive=None,**extra):
        observed=np.asarray(observed,bool);positive=observed if positive is None else np.asarray(positive,bool)
        item=dict(group=group,name=name,observed_admissions=int(observed.sum()),observed_percent=round(100*observed.mean(),3),
          missing_admissions=int((~observed).sum()),observed_patients=int(len(np.unique(subjects[observed]))),
          train_patients=int(len(np.unique(subjects[observed&train]))),positive_admissions=int(positive.sum()),
          positive_train_patients=int(len(np.unique(subjects[positive&train]))),**extra)
        catalog.append(item);return item
    lab=np.zeros((n,39,24),np.float32); lp=np.zeros((n,39),bool)
    lab[:,:22,:9]=d.arrays['lab_axis_inputs'];lp[:,:22]=lab[:,:22,6]>0
    lab[:,22:27,:10]=ext['new_labs'];lp[:,22:27]=ext['lab_masks'].any(2)
    lab[:,27:29,:20]=ext['eos'];lp[:,27:29]=ext['eos'][:,:,6:12].any(2)
    lab[:,29:,:10]=added['features'];lp[:,29:]=added['value_masks'].any(2)
    for j,name in enumerate(LAB_NAMES):
        coverage('laboratory',name,lp[:,j],lp[:,j]&(d.arrays['lab_axis_raw'][:,j,1]>0) if j<22 else None,
          chinese=LAB_ZH[j],source='labevents + d_labitems; specimen link for arterial gas',
          post_peak_rechecked=int((lab[:,j,7]>0).sum()) if j<22 else None,
          proposal='clinical_core_keep' if j<29 else 'candidate_keep',
          positive_definition='abnormal peak' if j<22 else 'observed, not abnormality classification')
    # 高低双轴合到一个概念，完整保留各9通道，不求平均。
    merged=set(v for pair in MERGE for v in pair);spec=[]
    for j in range(39):
        pair=next((p for p in MERGE if p[0]==j),None)
        if pair:spec.append((pair,LAB_NAMES[j].removesuffix('_low').removesuffix('_high'),LAB_FAMILY[j]))
        elif j not in merged:spec.append(((j,),LAB_NAMES[j],LAB_FAMILY[j]))
    merged_lab=[];merged_present=[];lnames=[];families=[]
    for indices,name,family in spec:
        x=lab[:,indices[0]].copy();p=lp[:,indices[0]].copy()
        if len(indices)==2:x[:,9:18]=lab[:,indices[1],:9];p|=lp[:,indices[1]]
        merged_lab.append(x);merged_present.append(p);lnames.append(name);families.append(family)
    merged_lab=np.stack(merged_lab,1);merged_present=np.stack(merged_present,1)
    # 先验低特异/重叠候选暂离主配方；不删除原数据，需后续同架构整组比较。
    excluded_lab={'hematocrit','mcv','monocytes_percent','protein_total'}
    lab_keep=np.array([name not in excluded_lab for name in lnames])
    duplicates=[]
    for j in range(39):
        for k in range(j+1,39):
            if np.array_equal(lp[:,j],lp[:,k]):duplicates.append(dict(kind='availability',left=LAB_NAMES[j],right=LAB_NAMES[k]))
    correlations=[]
    rawextras=np.concatenate((ext['lab_values_raw'],added['values_raw']),1)
    masks=np.concatenate((ext['lab_masks'],added['value_masks']),1);names=list(LAB_NAMES[22:27])+list(LAB_NAMES[29:])
    for j in range(len(names)):
        for k in range(j+1,len(names)):
            valid=train&masks[:,j,3]&masks[:,k,3]
            if valid.sum()<200:continue
            a,b=rawextras[valid,j,3],rawextras[valid,k,3]
            if np.std(a)==0 or np.std(b)==0:continue
            rho=float(spearmanr(a,b).statistic)
            if abs(rho)>=.8:correlations.append(dict(left=names[j],right=names[k],n=int(valid.sum()),rho=rho))
    ordered,om,ou,order_source,oa,vf,vfm=order_doses(args.snapshot,d.arrays['hadm_ids'],list(med['drug_names']))
    raw=med['values_raw'];mask=med['value_masks'];units=med['dose_unit_id']
    # 不把全期累计量当单次剂量；实际最大单次优先，医嘱中位单次补缺。
    dose=np.where(mask[:,:,5],raw[:,:,5],ordered);dose_mask=mask[:,:,5]|om
    dose_unit=np.where(mask[:,:,5],units,ou)
    freq=np.where(raw[:,:,0]>0,raw[:,:,3],raw[:,:,6]);freqmask=(raw[:,:,0]>0)|mask[:,:,6]
    # 旧万古霉素汇总混合途径，不能声称是全身抗感染。新接口只用已核对IV医嘱。
    van=list(med['drug_names']).index('vancomycin')
    dose[:,van]=ordered[:,van];dose_mask[:,van]=om[:,van];dose_unit[:,van]=ou[:,van]
    freq[:,van]=vf;freqmask[:,van]=vfm
    vals=np.stack((dose,freq),2);vm=np.stack((dose_mask,freqmask),2)
    scaled,norm=scale_per_unit(vals,vm,dose_unit,train,list(med['drug_names']))
    present=med['present']|om
    present[:,van]=om[:,van]|vfm
    med_features=np.concatenate((scaled,vm.astype(np.float32),np.eye(len(UNITS),dtype=np.float32)[dose_unit]),2)*present[:,:,None]
    # 来源及记录次数不直接入网；记录频率与医嘱频率互补，含义为documented schedule。
    coarse=np.zeros((n,10,2),np.float32);coarse[:,:7,0]=c.arrays['med_features'][:,:,:2].any(2)
    coarse[:,:,1]=c.arrays['med_features'][:,:,2:].any((1,2))[:,None]|order_source[:,None]
    drug_parent=np.zeros((len(med['drug_names']),10),np.float32);keep=[]
    for j,name in enumerate(med['drug_names']):
        drug_parent[j,list(DRUG_PARENTS[name])]=1
        for g in DRUG_PARENTS[name]:coarse[:,g,0]=np.maximum(coarse[:,g,0],present[:,j])
        row=coverage('drug_identity',str(name),present[:,j],source='emar + pharmacy + linked prescriptions',
          actual_single_dose=int(mask[:,j,5].sum()),ordered_single_dose=int(om[:,j].sum()),
          dose_after_complement=int(vm[:,j,0].sum()),frequency_after_complement=int(vm[:,j,1].sum()),
          new_order_dose_only=int((om[:,j]&~mask[:,j,5]).sum()),old_enabled=bool(active['drug'][str(name)]['in_model']))
        keep.append(row['train_patients']>=20)
        row['proposal']='fine_leaf_keep' if keep[-1] else 'parent_only_no_fine_embedding'
    for j,name in enumerate(COARSE_DRUGS):coverage('drug_class',name,coarse[:,j,1]>0,coarse[:,j,0]>0,source='union treatment evidence')
    for prefix in ('micro','support'):
        for j,name in enumerate(c.arrays[prefix+'_names']):
            coverage(prefix,str(name),c.arrays[prefix+'_masks'][:,j],c.arrays[prefix+'_values'][:,j]>0,
              source='microbiologyevents' if prefix=='micro' else 'restricted ICU chartevents',proposal='keep_compact_no_species_expansion')
    for j,name in enumerate(HISTORY_NAMES):coverage('frozen_history',name,np.ones(n,bool),d.history_raw[:,j]!=0,proposal='unchanged frozen H34')
    for j,name in enumerate(CURRENT_NAMES):coverage('frozen_current',name,d.current_masks[:,j],d.current[:,j]>0,proposal='unchanged frozen H34')
    text=np.zeros((n,4),int)
    for i,refs in enumerate(d.text_refs):
        for field,_ in refs:text[i,field]+=1
    for j,name in enumerate(('history_of_present_illness','past_medical_history','hospital_course','discharge_condition')):
        coverage('text',name,text[:,j]>0,chunks=int(text[:,j].sum()),source='saved B four-field Bio_ClinicalBERT',proposal='retain frozen vectors')
    micro=np.concatenate((c.arrays['micro_values'],c.arrays['micro_masks'][:,[0,3,4]]),1).astype(np.float32)
    if not all(np.array_equal(c.arrays['micro_masks'][:,0],c.arrays['micro_masks'][:,j]) for j in (1,2)):raise ValueError('culture masks differ')
    # 同一培养可用标记只传一次；细菌类别/病毒/耐药内容不删除。
    support=np.concatenate((c.arrays['support_values'],c.arrays['support_masks']),1).astype(np.float32)
    np.savez_compressed(output/'interface.npz',hadm_ids=d.arrays['hadm_ids'],subject_ids=subjects,patient_splits=d.arrays['patient_splits'],
      labs=merged_lab,lab_present=merged_present,lab_names=np.asarray(lnames),lab_family=np.asarray(families),lab_keep=lab_keep,
      medications=med_features,med_present=present,med_names=med['drug_names'],med_keep=np.asarray(keep),drug_parent=drug_parent,
      coarse=coarse,micro=micro,micro_present=c.arrays['micro_masks'].any(1),support=support,support_present=c.arrays['support_masks'].any(1))
    (output/'normalization.json').write_text(json.dumps(norm,indent=2),encoding='utf-8')
    keys=list(dict.fromkeys(k for row in catalog for k in row))
    with (output/'all_inputs.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,keys);writer.writeheader();writer.writerows(catalog)
    audit=dict(status='audited_interface_not_trained',admissions=n,patients=int(len(np.unique(subjects))),rows=catalog,
      duplicate_availability=duplicates,train_observed_latest_correlations=correlations,order_dose_audit=oa,
      old_axes=39,merged_concepts=len(lnames),compact_concepts=int(lab_keep.sum()),all_drug_leaves=len(keep),compact_drug_leaves=int(sum(keep)),
      excluded_lab_candidates=sorted(excluded_lab),micro_repeated_masks_removed=2,
      medication_contract='union documented treatment; actual maximum single dose when complete, otherwise maximum scalar ordered single dose; actual or planned frequency; no origin channels or count/span; vancomycin IV orders only',
      member_sha256=sha256(args.snapshot/'cohort/admissions.csv.gz'),interface_sha256=sha256(output/'interface.npz'),
      sources=prov,source_code_sha256=sha256(Path(__file__)),normalization_errors=norm_checks,
      limits=['No raw deletion; threshold is train patients >=20 for fine leaf, parent retained.','Candidate clinical simplification not yet adopted.',
              'Prescribed dose is not certified administered; cumulative dose not synthesized from prescription.',
              'Existing vancomycin executed summaries not route-separated and excluded from new interface; only verified IV order retained, indication unknown.',
              'Latest-value correlations are feature-level, not necessarily same-specimen paired measurements.'])
    (output/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in audit.items() if k in ('status','admissions','patients','old_axes','merged_concepts','compact_concepts','compact_drug_leaves')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT);run(p.parse_args().output)
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/v6_lab_state.py · lines 1–280

<!-- REVIEW_FRAGMENT {"path":"patient_data/v6_lab_state.py","part":1,"start_line":1,"end_line":280,"source_sha256":"15c343dbe9f1d92221e972738d75a039f0192ad578e099d920f98683583a4c2d","fragment_sha256":"15c343dbe9f1d92221e972738d75a039f0192ad578e099d920f98683583a4c2d","added_terminal_newline":false} -->
````python
"""V6 全住院检验状态：只读源数据，独立派生，不读取结局内容。"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
import time

import numpy as np

from patient_data.export_mimiciv import rows, sha256
from patient_data.expanded_abnormality_features import AXES, ADDED_AXES, directional_severity
from patient_data.lab_recovery_features import AXIS_CONCEPT, abnormal_severity, PAIRED_AXIS
from patient_data.preprocess_events import LABS
from training.v5.numeric_preprocessing import robust_parameters

NUMERIC = ("first_observed_severity", "observed_peak_severity", "latest_observed_severity",
           "recovery_delta", "latest_measurement_age", "peak_to_latest_elapsed")
FLAGS = ("axis_observed", "post_peak_recheck_valid", "latest_at_abnormal_peak")
# 与已经临床核验的类别一致；不接入 other_antibiotic，不加入利尿剂。
MED_GROUPS = {
    "short_beta_agonist": ("saba",), "short_muscarinic_antagonist": ("sama",),
    "long_beta_agonist": ("laba",), "long_muscarinic_antagonist": ("lama",),
    "inhaled_corticosteroid": ("inhaled_steroid",),
    "systemic_corticosteroid": ("systemic_steroid",),
    "common_systemic_antibacterial": ("macrolide", "beta_lactam", "fluoroquinolone", "tetracycline"),
}


def summarize(samples: list[tuple[float, float]], discharge_day: float) -> tuple[np.ndarray, dict]:
    """同刻取中位数；在无截顶异常程度上定位最早/最后峰值。"""
    if not math.isfinite(discharge_day) or discharge_day <= 0:
        raise ValueError("invalid discharge duration")
    grouped = defaultdict(list)
    for day, severity in samples:
        if not (math.isfinite(day) and math.isfinite(severity)
                and 0 <= day <= discharge_day and severity >= 0):
            raise ValueError("invalid observation")
        grouped[day].append(severity)
    if not grouped:
        return np.zeros(9), {"measurements": 0}
    ordered = [(t, float(np.median(v))) for t, v in sorted(grouped.items())]
    peak = max(v for _, v in ordered)
    # 定位峰值用精确最大值，末次峰标记另用固定浮点容差。
    peak_times = [t for t, v in ordered if v == peak]
    first_peak = peak_times[0]
    latest_time, latest = ordered[-1]
    valid = latest_time > first_peak
    result = np.array([ordered[0][1], peak, latest, peak - latest if valid else 0.,
                       math.log1p(discharge_day - latest_time), math.log1p(latest_time - first_peak),
                       1., float(valid), float(peak > 0 and math.isclose(latest, peak, rel_tol=1e-7, abs_tol=1e-8))])
    return result, {"measurements": len(ordered), "first_peak_day": first_peak,
                    "last_peak_day": peak_times[-1], "latest_day": latest_time}


def scale_on_train(raw: np.ndarray, splits: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """只有有测/有效恢复的训练住院参与尺度拟合；标记保持二元。"""
    if raw.ndim != 3 or raw.shape[2] != 9 or len(splits) != len(raw):
        raise ValueError("invalid state shape")
    scaled = raw.copy()
    valid = np.broadcast_to(raw[:, :, 6:7] > 0, raw[:, :, :6].shape).copy()
    valid[:, :, 3] &= raw[:, :, 7] > 0
    valid[:, :, 5] &= raw[:, :, 7] > 0
    transformed = raw[:, :, :6].copy()
    transformed[:, :, :4] = np.log1p(transformed[:, :, :4])
    parameters = {}
    for a, axis in enumerate(AXES):
        for f, name in enumerate(NUMERIC):
            selected = valid[:, a, f] & (splits == "train")
            key = f"{axis}__{name}"
            if not selected.any():
                parameters[key] = {"method": "no_training_observation", "training_observations": 0}
                valid[:, a, f] = False
                scaled[:, a, f] = 0
                continue
            p = robust_parameters(transformed[selected, a, f])
            values = (transformed[:, a, f] - p["median"]) / p["scale"]
            scaled[:, a, f] = np.where(valid[:, a, f], np.clip(values, -8, 8), 0)
            parameters[key] = p
    return scaled.astype(np.float32), valid, parameters


def strict_interval_exposure(start: float, end: float, times: list[float]) -> bool:
    """只计严格位于两次采样之间的执行记录；边界同刻不推断先后。"""
    if not start < end:
        raise ValueError("interval must increase")
    return bisect_right(times, start) < bisect_left(times, end)


def build(snapshot: Path, processed: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    source_manifest = snapshot / "metadata/manifest.json"
    sm = json.loads(source_manifest.read_text(encoding="utf-8"))
    pm_path = processed / "manifest.json"
    pm = json.loads(pm_path.read_text(encoding="utf-8"))
    if sm["status"] != "validated" or pm["status"] != "preprocessed_and_checked":
        raise ValueError("unvalidated source")
    if pm.get("medication_mapping") != "clinical_reviewed_2026_09":
        raise ValueError("expected clinically reviewed medication mapping")
    inputs = {}
    for name in ("cohort/admissions", "cohort/patient_splits", "raw/labevents"):
        digest = sha256(snapshot / (name + ".csv.gz"))
        if digest != sm["sources"][name]["sha256"] or digest != pm["input_sha256"][name]:
            raise ValueError(f"raw lineage mismatch: {name}")
        inputs[name] = digest
    events_path = processed / "events.csv.gz"
    if sha256(events_path) != pm["output_sha256"]["events.csv.gz"]:
        raise ValueError("processed events changed")
    inputs["processed_events"] = pm["output_sha256"]["events.csv.gz"]
    admissions = {}
    subjects = {}
    for row in rows(snapshot / "cohort/admissions.csv.gz"):
        if row["discharge_landmark_eligible"] not in ("t", "true", "1"):
            continue
        hadm = row["hadm_id"]
        if hadm in admissions:
            raise ValueError("duplicate admission")
        admissions[hadm] = (datetime.fromisoformat(row["admittime"]), datetime.fromisoformat(row["dischtime"]))
        subjects[hadm] = row["subject_id"]
    if len(admissions) != 19401 or len(set(subjects.values())) != 5351:
        raise ValueError("cohort drift")
    split_rows = list(rows(snapshot / "cohort/patient_splits.csv.gz"))
    splits = {r["subject_id"]: r["patient_split"] for r in split_rows}
    if len(splits) != len(split_rows) or not set(splits.values()) <= {"train", "validation", "test"}:
        raise ValueError("split drift")
    hadms = sorted(admissions, key=lambda h: (int(subjects[h]), admissions[h][0], int(h)))
    by_class = {c: g for g, cs in MED_GROUPS.items() for c in cs}
    medication = defaultdict(dict)
    accepted_gas = set()
    for row in rows(events_path):
        h = row["hadm_id"]
        if h not in admissions or row["patient_split"] != splits[subjects[h]]:
            raise ValueError("processed member/split mismatch")
        if row["kind"] == "lab" and row["concept"] in ("ph:arterial", "pco2:arterial"):
            accepted_gas.add(row["source_id"])
        if row["kind"] != "medication" or row["concept"] not in by_class:
            continue
        day = float(row["elapsed_days"])
        length = (admissions[h][1] - admissions[h][0]).total_seconds() / 86400
        if not math.isfinite(day) or not 0 <= day <= length:
            raise ValueError("medication outside admission")
        slot = medication[h, by_class[row["concept"]]]
        previous = slot.setdefault(row["source_id"], day)
        if previous != day:
            raise ValueError("conflicting administration time")
    samples = defaultdict(list)
    paired = defaultdict(dict)
    axis_audit = defaultdict(Counter)
    item_axes = defaultdict(list)
    for axis, concept in AXIS_CONCEPT.items():
        item, unit = next((i, u) for i, (c, u) in LABS.items() if c == concept.split(":")[0])
        item_axes[item].append((axis, "", unit))
    for axis, (item, direction, unit) in ADDED_AXES.items():
        item_axes[str(item)].append((axis, direction, unit))
    total = 0
    for row in rows(snapshot / "raw/labevents.csv.gz"):
        total += 1
        h, item = row["hadm_id"], row["itemid"]
        if h not in admissions or item not in item_axes:
            continue
        start, end = admissions[h]
        try:
            chart, stored = datetime.fromisoformat(row["charttime"]), datetime.fromisoformat(row["storetime"])
            values = tuple(float(row[k]) for k in ("valuenum", "ref_range_lower", "ref_range_upper"))
            value, lower, upper = values
            reason = ("time_unavailable" if not start <= chart <= stored <= end else
                      "invalid_numeric_or_reference" if not all(map(math.isfinite, values)) or value <= 0 or lower >= upper else None)
        except (ValueError, TypeError):
            reason = "invalid_time_numeric_or_reference"
        for axis, direction, unit in item_axes[item]:
            counts = axis_audit[axis]
            counts["raw_rows"] += 1
            reject = reason
            if row["valueuom"].strip() != unit:
                reject = "unit_mismatch_or_missing"
            if axis.startswith("arterial_") and row["labevent_id"] not in accepted_gas:
                reject = "not_in_validated_arterial_events"
            if reject:
                counts[reject] += 1
                continue
            day = (chart - start).total_seconds() / 86400
            severity = (directional_severity(value, lower, upper, direction, severity_cap=None) if direction
                        else abnormal_severity(axis, value, lower, upper, severity_cap=None))
            samples[h, axis].append((day, severity))
            counts["accepted_rows"] += 1
            if axis.startswith("arterial_") and row["specimen_id"] not in ("", "\\N"):
                slot = paired[h, row["specimen_id"]]
                if axis in slot:
                    raise ValueError("duplicate gas component within specimen")
                slot[axis] = (day, severity)
    if total != sm["sources"]["raw/labevents"]["rows"]:
        raise ValueError("raw laboratory count changed")
    for (h, _), slot in paired.items():
        if len(slot) != 2:
            continue
        (t1, s1), (t2, s2) = slot["arterial_low_ph"], slot["arterial_high_pco2"]
        if abs(t1 - t2) <= 1 / 24:
            samples[h, PAIRED_AXIS].append((max(t1, t2), min(s1, s2)))
    print(json.dumps({"stage": "raw_labs_scanned", "rows": total}), flush=True)
    raw = np.zeros((len(hadms), len(AXES), 9), dtype=np.float64)
    med = np.zeros((len(hadms), len(MED_GROUPS)), dtype=np.float32)
    peak_times = np.full((len(hadms), len(AXES), 3), np.nan)
    cycle_stats = defaultdict(Counter)
    for i, h in enumerate(hadms):
        duration = (admissions[h][1] - admissions[h][0]).total_seconds() / 86400
        drug_times = {g: sorted(set(medication[h, g].values())) for g in MED_GROUPS}
        for g, group in enumerate(MED_GROUPS):
            med[i, g] = bool(drug_times[group])
        for a, axis in enumerate(AXES):
            obs = samples.get((h, axis), [])
            raw[i, a], metadata = summarize(obs, duration)
            if metadata["measurements"]:
                peak_times[i, a] = [metadata["first_peak_day"], metadata["last_peak_day"], metadata["latest_day"]]
            for name, flag in zip(FLAGS, raw[i, a, 6:], strict=True):
                axis_audit[axis][name + "_admissions"] += int(flag)
            axis_audit[axis]["observed_recovery_admissions"] += int(raw[i, a, 3] > 0)
            times = sorted({t for t, _ in obs})
            if len(times) < 2:
                continue
            cycle_stats[axis]["admissions_with_pairs"] += 1
            cycle_stats[axis]["pairs"] += len(times) - 1
            for left, right in zip(times[:-1], times[1:], strict=True):
                for group in MED_GROUPS:
                    cycle_stats[axis][group + "_exposed_intervals"] += strict_interval_exposure(left, right, drug_times[group])
    split_array = np.asarray([splits[subjects[h]] for h in hadms])
    scaled, value_masks, parameters = scale_on_train(raw, split_array)
    # 在写文件前再次核对源文件，防止运行期间源被替换。
    for name in ("cohort/admissions", "cohort/patient_splits", "raw/labevents"):
        if sha256(snapshot / (name + ".csv.gz")) != inputs[name]:
            raise ValueError("source changed during build")
    if sha256(events_path) != inputs["processed_events"]:
        raise ValueError("events changed during build")
    output.mkdir(parents=True)
    file = output / "clinical_state.npz"
    np.savez_compressed(file, hadm_ids=np.asarray(hadms, dtype=np.int64),
                        subject_ids=np.asarray([subjects[h] for h in hadms], dtype=np.int64),
                        patient_splits=split_array, lab_axis_raw=raw, lab_axis_inputs=scaled,
                        lab_value_masks=value_masks, axis_names=np.asarray(AXES),
                        feature_names=np.asarray(NUMERIC + FLAGS), peak_audit_days=peak_times,
                        med_exposure=med, med_group_names=np.asarray(tuple(MED_GROUPS)))
    with np.load(file, allow_pickle=False) as check:
        loaded = {k: check[k] for k in check.files}
    if not np.array_equal(loaded["lab_axis_inputs"], scaled) or not np.isfinite(scaled).all():
        raise ValueError("derived roundtrip failed")
    (output / "normalization.json").write_text(json.dumps(parameters, indent=2), encoding="utf-8")
    report = {"status": "derived_and_roundtrip_checked_not_prediction_validated",
              "admissions": len(hadms), "patients": len(set(subjects.values())),
              "lab_shape": list(raw.shape), "med_shape": list(med.shape),
              "source_hashes": inputs, "source_manifest_sha256": sha256(source_manifest),
              "processed_manifest_sha256": sha256(pm_path), "builder_sha256": sha256(Path(__file__)),
              "output_sha256": {"clinical_state.npz": sha256(file),
                                "normalization.json": sha256(output / "normalization.json")},
              "axis_audit": {a: dict(axis_audit[a]) for a in AXES},
              "lab_intervention_interval_coverage": {a: dict(cycle_stats[a]) for a in AXES},
              "medication_coverage": {g: int(med[:, i].sum()) for i, g in enumerate(MED_GROUPS)},
              "limitations": ["Frozen eMAR mapping is inherited, not a new dose or indication adjudication",
                              "No eMAR evidence is not confirmed absence of treatment",
                              "Cycle coverage is not a treatment effect; boundary-time drug events excluded",
                              "Per-axis timestamp aggregation is not a dense state vector",
                              "No new predictors, labels, text encoding or model training in this builder",
                              "Lab measurements require positive numeric values and valid reference interval"],
              "elapsed_seconds": round(time.perf_counter() - started, 2)}
    (output / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.snapshot, args.processed, args.output)
    print(json.dumps({k: result[k] for k in ("status", "admissions", "patients", "elapsed_seconds")}))
````
<!-- END REVIEW_FRAGMENT -->

## patient_data/verify_medication_identity.py · lines 1–50

<!-- REVIEW_FRAGMENT {"path":"patient_data/verify_medication_identity.py","part":1,"start_line":1,"end_line":50,"source_sha256":"fb96c7d8d5bc403289abdba6701469f1239973a3376cd2669c2cea1b34c7004a","fragment_sha256":"fb96c7d8d5bc403289abdba6701469f1239973a3376cd2669c2cea1b34c7004a","added_terminal_newline":false} -->
````python
"""独立验证药物频率、剂量单位、成员/训练折尺度与落盘接口。"""
import argparse
import json
from pathlib import Path
import numpy as np
from patient_data.export_mimiciv import sha256
from patient_data.medication_identity import NAMES,UNITS,CHANNELS
from training.v5.numeric_preprocessing import robust_parameters


def verify(directory,state_dir):
    m=json.loads((directory/'manifest.json').read_text());norm=json.loads((directory/'normalization.json').read_text())
    if m['status']!='validated_medication_identity' or m['parent_state_manifest_sha256']!=sha256(state_dir/'manifest.json') or m['output_sha256']!=sha256(directory/'medication_identity.npz'):
        raise ValueError('source proof changed')
    with np.load(directory/'medication_identity.npz') as z:a={k:z[k] for k in z.files}
    with np.load(state_dir/'clinical_state.npz') as z:
        for key in ('hadm_ids','subject_ids','patient_splits'):
            if not np.array_equal(a[key],z[key]):raise ValueError('member/split differs')
    v=a['values_raw'];mask=a['value_masks'];f=a['features'];unit=a['dose_unit_id'];present=a['present'];train=a['patient_splits']=='train'
    if f.shape!=(19401,21,23) or not np.isfinite(f).all() or not np.isfinite(v).all() or np.any(v<0):raise ValueError('invalid medication tensor')
    if np.any(f[~present]!=0):raise ValueError('empty identity token not masked')
    executed=v[:,:,0]>0
    if np.any(v[:,:,1][executed]>v[:,:,0][executed]) or np.any(v[:,:,2][executed]>v[:,:,0][executed]):raise ValueError('count/day arithmetic impossible')
    np.testing.assert_allclose(v[:,:,3][executed],v[:,:,0][executed]/v[:,:,1][executed],rtol=1e-6)
    complete=mask[:,:,4]
    if np.any(unit[complete]==0) or np.any(unit[complete]==UNITS.index('mixed')) or np.any(v[:,:,4][complete]<v[:,:,5][complete]):
        raise ValueError('unknown/mixed/corrupt cumulative dose admitted')
    max_error=0.
    for j,drug in enumerate(NAMES):
        for k,name in enumerate(CHANNELS):
            for u in (range(len(UNITS)) if k in (4,5) else (-1,)):
                valid=mask[:,j,k] & (unit[:,j]==u if u>=0 else True)
                key=f'{drug}/{name}/{UNITS[u] if u>=0 else "count"}'
                if not valid.any():continue
                p=robust_parameters(np.log1p(v[train & valid,j,k]))
                if any(abs(p[n]-norm[key][n])>1e-7 for n in ('median','scale')):raise ValueError('normalization not train-only')
                expected=np.clip((np.log1p(v[valid,j,k])-p['median'])/p['scale'],-8,8)*present[valid,j]
                max_error=max(max_error,float(np.max(np.abs(expected-f[valid,j,k]))))
    if max_error>2e-6:raise ValueError('scaled values differ')
    proof=dict(status='independently_verified_medication_interface',manifest_sha256=sha256(directory/'manifest.json'),
        npz_sha256=sha256(directory/'medication_identity.npz'),raw_drug_name_dictionary_sha256=sha256(directory/'raw_drug_names.json'),
        normalization_sha256=sha256(directory/'normalization.json'),max_scaled_value_error=max_error,
        train_only_normalization_reproduced=True,frequency_arithmetic_checked=True,mixed_unit_doses_excluded=True,
        limits=['This verifies numerical/source rules, not physician-adjudicated drug mapping sensitivity.'])
    (directory/'independent_verification.json').write_text(json.dumps(proof,indent=2),encoding='utf-8');print(json.dumps(proof))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,default=Path('patient_data/processed/v6_medication_identity_2026-10-12'))
    p.add_argument('--state-dir',type=Path,default=Path('patient_data/processed/v6_clinical_state_2026-10-09_r2'));a=p.parse_args();verify(a.directory,a.state_dir)
````
<!-- END REVIEW_FRAGMENT -->
