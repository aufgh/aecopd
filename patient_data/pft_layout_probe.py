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
