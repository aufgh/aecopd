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
