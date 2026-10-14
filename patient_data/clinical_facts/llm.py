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
