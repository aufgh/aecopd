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
