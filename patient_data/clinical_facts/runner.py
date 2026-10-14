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
