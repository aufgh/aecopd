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
