\set ON_ERROR_STOP on
SET client_encoding = 'UTF8';
BEGIN;

DROP TABLE IF EXISTS project_aecopd.cohort_hadm_ids;
CREATE TABLE project_aecopd.cohort_hadm_ids AS
SELECT subject_id,hadm_id,admittime,dischtime
FROM project_aecopd.aecopd_admissions_strict;
ALTER TABLE project_aecopd.cohort_hadm_ids ADD CONSTRAINT cohort_hadm_ids_pk PRIMARY KEY(hadm_id);
CREATE INDEX cohort_hadm_ids_subject_idx ON project_aecopd.cohort_hadm_ids(subject_id);
ANALYZE project_aecopd.cohort_hadm_ids;
SELECT COUNT(*) AS cohort_n FROM project_aecopd.cohort_hadm_ids \gset

DROP TABLE IF EXISTS project_aecopd.core_field_coverage;
CREATE TABLE project_aecopd.core_field_coverage AS
WITH e AS (
  SELECT s.*,p.gender,p.anchor_age,p.anchor_year,p.anchor_year_group,p.dod
  FROM project_aecopd.aecopd_admissions_strict s
  JOIN mimiciv_hosp.patients p USING(subject_id)
), audit AS (
  SELECT 'mimiciv_hosp.patients' source_table,'gender' field_name,'character' data_type,COUNT(*) FILTER(WHERE gender IS NOT NULL) non_null_admissions,COUNT(DISTINCT gender)::bigint distinct_values,'patient-level; repeated across admissions' audit_note FROM e
  UNION ALL SELECT 'mimiciv_hosp.patients','anchor_age','smallint',COUNT(*) FILTER(WHERE anchor_age IS NOT NULL),COUNT(DISTINCT anchor_age),'deidentified anchor age; derive admission age only in later approved feature work' FROM e
  UNION ALL SELECT 'mimiciv_hosp.patients','anchor_year','smallint',COUNT(*) FILTER(WHERE anchor_year IS NOT NULL),COUNT(DISTINCT anchor_year),'deidentified anchor year' FROM e
  UNION ALL SELECT 'mimiciv_hosp.patients','anchor_year_group','varchar',COUNT(*) FILTER(WHERE anchor_year_group IS NOT NULL),COUNT(DISTINCT anchor_year_group),'deidentified year group' FROM e
  UNION ALL SELECT 'mimiciv_hosp.patients','dod','date',COUNT(*) FILTER(WHERE dod IS NOT NULL),COUNT(DISTINCT dod),'death support field; not a finalized competing-risk definition' FROM e
  UNION ALL SELECT 'mimiciv_hosp.admissions','admittime','timestamp',COUNT(*) FILTER(WHERE admittime IS NOT NULL),COUNT(DISTINCT admittime),'prediction history boundary' FROM e
  UNION ALL SELECT 'mimiciv_hosp.admissions','dischtime','timestamp',COUNT(*) FILTER(WHERE dischtime IS NOT NULL),COUNT(DISTINCT dischtime),'candidate prediction time' FROM e
  UNION ALL SELECT 'mimiciv_hosp.admissions','deathtime','timestamp',COUNT(*) FILTER(WHERE deathtime IS NOT NULL),COUNT(DISTINCT deathtime),'in-hospital death support field' FROM e
  UNION ALL SELECT 'mimiciv_hosp.admissions','admission_type','varchar',COUNT(*) FILTER(WHERE admission_type IS NOT NULL),COUNT(DISTINCT admission_type),'administrative field; audit before modeling' FROM e
  UNION ALL SELECT 'mimiciv_hosp.admissions','admission_location','varchar',COUNT(*) FILTER(WHERE admission_location IS NOT NULL),COUNT(DISTINCT admission_location),'administrative field; transfer QC source' FROM e
  UNION ALL SELECT 'mimiciv_hosp.admissions','discharge_location','varchar',COUNT(*) FILTER(WHERE discharge_location IS NOT NULL),COUNT(DISTINCT discharge_location),'may be known at discharge; administrative/clinical interpretation required' FROM e
  UNION ALL SELECT 'mimiciv_hosp.admissions','insurance','varchar',COUNT(*) FILTER(WHERE insurance IS NOT NULL),COUNT(DISTINCT insurance),'administrative field; fairness and temporal stability review required' FROM e
  UNION ALL SELECT 'mimiciv_hosp.admissions','language','varchar',COUNT(*) FILTER(WHERE language IS NOT NULL),COUNT(DISTINCT language),'administrative field; missingness audit only' FROM e
  UNION ALL SELECT 'mimiciv_hosp.admissions','marital_status','varchar',COUNT(*) FILTER(WHERE marital_status IS NOT NULL),COUNT(DISTINCT marital_status),'administrative field; missingness audit only' FROM e
  UNION ALL SELECT 'mimiciv_hosp.admissions','race','varchar',COUNT(*) FILTER(WHERE race IS NOT NULL),COUNT(DISTINCT race),'fairness-sensitive administrative field; not automatically included' FROM e
  UNION ALL SELECT 'mimiciv_hosp.admissions','edregtime','timestamp',COUNT(*) FILTER(WHERE edregtime IS NOT NULL),COUNT(DISTINCT edregtime),'ED timing support field' FROM e
  UNION ALL SELECT 'mimiciv_hosp.admissions','edouttime','timestamp',COUNT(*) FILTER(WHERE edouttime IS NOT NULL),COUNT(DISTINCT edouttime),'ED timing support field' FROM e
  UNION ALL SELECT 'mimiciv_hosp.admissions','hospital_expire_flag','smallint',COUNT(*) FILTER(WHERE hospital_expire_flag IS NOT NULL),COUNT(DISTINCT hospital_expire_flag),'death support field; not a finalized outcome definition' FROM e
)
SELECT *,:cohort_n::bigint AS cohort_admissions,ROUND(100.0*non_null_admissions/:cohort_n,2) AS coverage_pct
FROM audit;

DROP TABLE IF EXISTS project_aecopd.core_value_distribution;
CREATE TABLE project_aecopd.core_value_distribution AS
WITH e AS (
  SELECT s.*,p.gender,p.anchor_age,p.anchor_year_group,p.dod
  FROM project_aecopd.aecopd_admissions_strict s JOIN mimiciv_hosp.patients p USING(subject_id)
), u AS (
  SELECT 'gender' field_name,COALESCE(gender,'<NULL>')::text value FROM e
  UNION ALL SELECT 'anchor_year_group',COALESCE(anchor_year_group,'<NULL>') FROM e
  UNION ALL SELECT 'admission_type',COALESCE(admission_type,'<NULL>') FROM e
  UNION ALL SELECT 'admission_location',COALESCE(admission_location,'<NULL>') FROM e
  UNION ALL SELECT 'discharge_location',COALESCE(discharge_location,'<NULL>') FROM e
  UNION ALL SELECT 'insurance',COALESCE(insurance,'<NULL>') FROM e
  UNION ALL SELECT 'language',COALESCE(language,'<NULL>') FROM e
  UNION ALL SELECT 'marital_status',COALESCE(marital_status,'<NULL>') FROM e
  UNION ALL SELECT 'race',COALESCE(race,'<NULL>') FROM e
  UNION ALL SELECT 'hospital_expire_flag',COALESCE(hospital_expire_flag::text,'<NULL>') FROM e
)
SELECT field_name,value,COUNT(*) AS admissions,ROUND(100.0*COUNT(*)/:cohort_n,2) AS pct
FROM u GROUP BY field_name,value;

DROP TABLE IF EXISTS project_aecopd.lab_item_coverage_top100;
CREATE TABLE project_aecopd.lab_item_coverage_top100 AS
WITH cohort_labs AS MATERIALIZED (
  SELECT l.subject_id,l.hadm_id,l.itemid,l.charttime,l.valueuom,c.admittime,c.dischtime
  FROM mimiciv_hosp.labevents l
  JOIN project_aecopd.cohort_hadm_ids c ON c.hadm_id=l.hadm_id AND c.subject_id=l.subject_id
), per_admission AS (
  SELECT itemid,hadm_id,MIN(subject_id) subject_id,COUNT(*) measurements_per_admission
  FROM cohort_labs GROUP BY itemid,hadm_id
), aggregate_items AS (
  SELECT
    l.itemid,
    COUNT(*) AS measurements,
    COUNT(DISTINCT l.hadm_id) AS admissions_with_item,
    COUNT(DISTINCT l.subject_id) AS patients_with_item,
    COUNT(DISTINCT l.valueuom) AS unit_count,
    STRING_AGG(DISTINCT COALESCE(l.valueuom,'<NULL>'),' | ' ORDER BY COALESCE(l.valueuom,'<NULL>')) AS observed_units,
    COUNT(*) FILTER(WHERE l.charttime<l.admittime) AS measurements_before_admission,
    COUNT(*) FILTER(WHERE l.charttime>l.dischtime) AS measurements_after_discharge
  FROM cohort_labs l GROUP BY l.itemid
), medians AS (
  SELECT itemid,percentile_cont(0.5) WITHIN GROUP(ORDER BY measurements_per_admission) AS median_measurements_per_admission
  FROM per_admission GROUP BY itemid
)
SELECT
  a.itemid,d.label,d.fluid,d.category,a.measurements,a.admissions_with_item,a.patients_with_item,
  ROUND(100.0*a.admissions_with_item/:cohort_n,2) AS admission_coverage_pct,
  m.median_measurements_per_admission,a.unit_count,a.observed_units,
  a.measurements_before_admission,a.measurements_after_discharge,
  (lower(COALESCE(d.label,'')) ~ '(blood gas|ph$|pco2|po2|oxygen|bicarbonate|base excess|hemoglobin|hematocrit|white blood|wbc|platelet|neutrophil|lymphocyte|creatinine|urea nitrogen|sodium|potassium|chloride|calcium|magnesium|phosphate|lactate|c-reactive|procalcitonin|albumin)') AS copd_clinical_review_candidate
FROM aggregate_items a
JOIN medians m USING(itemid)
LEFT JOIN mimiciv_hosp.d_labitems d USING(itemid)
ORDER BY a.admissions_with_item DESC,a.measurements DESC,a.itemid
LIMIT 100;

DROP TABLE IF EXISTS project_aecopd.medication_coverage_top100;
CREATE TABLE project_aecopd.medication_coverage_top100 AS
SELECT
  p.drug,
  COUNT(*) AS prescription_rows,
  COUNT(DISTINCT p.hadm_id) AS admissions_with_drug,
  COUNT(DISTINCT p.subject_id) AS patients_with_drug,
  ROUND(100.0*COUNT(DISTINCT p.hadm_id)/:cohort_n,2) AS admission_coverage_pct,
  COUNT(DISTINCT p.route) AS route_count,
  STRING_AGG(DISTINCT COALESCE(p.route,'<NULL>'),' | ' ORDER BY COALESCE(p.route,'<NULL>')) AS observed_routes,
  COUNT(*) FILTER(WHERE p.starttime>c.dischtime) AS starts_after_discharge,
  COUNT(*) FILTER(WHERE p.stoptime<c.admittime) AS stops_before_admission
FROM mimiciv_hosp.prescriptions p
JOIN project_aecopd.cohort_hadm_ids c ON c.hadm_id=p.hadm_id AND c.subject_id=p.subject_id
GROUP BY p.drug
ORDER BY admissions_with_drug DESC,prescription_rows DESC,p.drug
LIMIT 100;

DROP TABLE IF EXISTS project_aecopd.pharmacy_coverage_top100;
CREATE TABLE project_aecopd.pharmacy_coverage_top100 AS
SELECT
  p.medication,
  COUNT(*) AS pharmacy_rows,
  COUNT(DISTINCT p.hadm_id) AS admissions_with_medication,
  COUNT(DISTINCT p.subject_id) AS patients_with_medication,
  ROUND(100.0*COUNT(DISTINCT p.hadm_id)/:cohort_n,2) AS admission_coverage_pct,
  COUNT(DISTINCT p.route) AS route_count,
  STRING_AGG(DISTINCT COALESCE(p.route,'<NULL>'),' | ' ORDER BY COALESCE(p.route,'<NULL>')) AS observed_routes,
  COUNT(*) FILTER(WHERE p.starttime>c.dischtime) AS starts_after_discharge,
  COUNT(*) FILTER(WHERE p.verifiedtime>c.dischtime) AS verified_after_discharge
FROM mimiciv_hosp.pharmacy p
JOIN project_aecopd.cohort_hadm_ids c ON c.hadm_id=p.hadm_id AND c.subject_id=p.subject_id
GROUP BY p.medication
ORDER BY admissions_with_medication DESC,pharmacy_rows DESC,p.medication
LIMIT 100;

DROP TABLE IF EXISTS project_aecopd.procedure_coverage_top100;
CREATE TABLE project_aecopd.procedure_coverage_top100 AS
SELECT
  p.icd_version,p.icd_code,d.long_title,
  COUNT(*) AS procedure_rows,
  COUNT(DISTINCT p.hadm_id) AS admissions_with_procedure,
  COUNT(DISTINCT p.subject_id) AS patients_with_procedure,
  ROUND(100.0*COUNT(DISTINCT p.hadm_id)/:cohort_n,2) AS admission_coverage_pct,
  COUNT(*) FILTER(WHERE p.chartdate<c.admittime::date) AS charted_before_admission,
  COUNT(*) FILTER(WHERE p.chartdate>c.dischtime::date) AS charted_after_discharge,
  (lower(COALESCE(d.long_title,'')) ~ '(ventilat|intubat|respirat|oxygen|trache|airway|thorac)') AS respiratory_support_review_candidate
FROM mimiciv_hosp.procedures_icd p
JOIN project_aecopd.cohort_hadm_ids c ON c.hadm_id=p.hadm_id AND c.subject_id=p.subject_id
LEFT JOIN mimiciv_hosp.d_icd_procedures d USING(icd_version,icd_code)
GROUP BY p.icd_version,p.icd_code,d.long_title
ORDER BY admissions_with_procedure DESC,procedure_rows DESC,p.icd_version,p.icd_code
LIMIT 100;

DROP TABLE IF EXISTS project_aecopd.icu_admission_audit;
CREATE TABLE project_aecopd.icu_admission_audit AS
SELECT
  c.subject_id,c.hadm_id,
  COUNT(i.stay_id) AS icu_stay_count,
  MIN(i.intime) AS first_icu_intime,
  MAX(i.outtime) AS last_icu_outtime,
  SUM(i.los) AS total_icu_los_days,
  (COUNT(i.stay_id)>0) AS has_icu_stay,
  COUNT(i.stay_id) FILTER(WHERE i.intime<c.admittime OR i.outtime>c.dischtime) AS icu_stays_outside_hospital_time
FROM project_aecopd.cohort_hadm_ids c
LEFT JOIN mimiciv_icu.icustays i ON i.hadm_id=c.hadm_id AND i.subject_id=c.subject_id
GROUP BY c.subject_id,c.hadm_id;
ALTER TABLE project_aecopd.icu_admission_audit ADD CONSTRAINT icu_admission_audit_pk PRIMARY KEY(hadm_id);

DROP TABLE IF EXISTS project_aecopd.icu_coverage_summary;
CREATE TABLE project_aecopd.icu_coverage_summary AS
SELECT
  COUNT(*) AS cohort_admissions,
  COUNT(*) FILTER(WHERE has_icu_stay) AS admissions_with_icu,
  ROUND(100.0*COUNT(*) FILTER(WHERE has_icu_stay)/COUNT(*),2) AS icu_coverage_pct,
  SUM(icu_stay_count) AS icu_stays,
  COUNT(*) FILTER(WHERE icu_stay_count>1) AS admissions_with_multiple_icu_stays,
  SUM(icu_stays_outside_hospital_time) AS icu_stays_outside_hospital_time
FROM project_aecopd.icu_admission_audit;

DROP TABLE IF EXISTS project_aecopd.icu_selection_audit;
CREATE TABLE project_aecopd.icu_selection_audit AS
SELECT
  i.has_icu_stay,
  COUNT(*) AS admissions,
  percentile_cont(0.5) WITHIN GROUP(ORDER BY s.hospital_los_days) AS median_hospital_los_days,
  percentile_cont(0.5) WITHIN GROUP(ORDER BY p.anchor_age) AS median_anchor_age,
  COUNT(*) FILTER(WHERE s.hospital_expire_flag=1) AS in_hospital_deaths,
  ROUND(100.0*COUNT(*) FILTER(WHERE s.hospital_expire_flag=1)/COUNT(*),2) AS in_hospital_death_pct,
  COUNT(*) FILTER(WHERE n.has_discharge_note) AS with_discharge_note,
  ROUND(100.0*COUNT(*) FILTER(WHERE n.has_discharge_note)/COUNT(*),2) AS discharge_note_coverage_pct
FROM project_aecopd.icu_admission_audit i
JOIN project_aecopd.aecopd_admissions_strict s USING(subject_id,hadm_id)
JOIN mimiciv_hosp.patients p USING(subject_id)
JOIN project_aecopd.discharge_note_admission_audit n USING(subject_id,hadm_id)
GROUP BY i.has_icu_stay;

DROP TABLE IF EXISTS project_aecopd.structured_source_coverage;
CREATE TABLE project_aecopd.structured_source_coverage AS
WITH source_hadm AS (
  SELECT 'mimiciv_hosp.labevents'::text AS data_source,l.hadm_id
  FROM mimiciv_hosp.labevents l JOIN project_aecopd.cohort_hadm_ids c ON c.hadm_id=l.hadm_id AND c.subject_id=l.subject_id
  GROUP BY l.hadm_id
  UNION ALL
  SELECT 'mimiciv_hosp.prescriptions',p.hadm_id
  FROM mimiciv_hosp.prescriptions p JOIN project_aecopd.cohort_hadm_ids c ON c.hadm_id=p.hadm_id AND c.subject_id=p.subject_id
  GROUP BY p.hadm_id
  UNION ALL
  SELECT 'mimiciv_hosp.pharmacy',p.hadm_id
  FROM mimiciv_hosp.pharmacy p JOIN project_aecopd.cohort_hadm_ids c ON c.hadm_id=p.hadm_id AND c.subject_id=p.subject_id
  GROUP BY p.hadm_id
  UNION ALL
  SELECT 'mimiciv_hosp.procedures_icd',p.hadm_id
  FROM mimiciv_hosp.procedures_icd p JOIN project_aecopd.cohort_hadm_ids c ON c.hadm_id=p.hadm_id AND c.subject_id=p.subject_id
  GROUP BY p.hadm_id
)
SELECT data_source,COUNT(*) AS admissions_with_data,ROUND(100.0*COUNT(*)/:cohort_n,2) AS admission_coverage_pct
FROM source_hadm GROUP BY data_source;

DROP TABLE IF EXISTS project_aecopd.derived_inventory;
CREATE TABLE project_aecopd.derived_inventory AS
SELECT n.nspname AS table_schema,c.relname AS object_name,
       CASE c.relkind WHEN 'r' THEN 'table' WHEN 'p' THEN 'partitioned table' WHEN 'v' THEN 'view' WHEN 'm' THEN 'materialized view' WHEN 'f' THEN 'foreign table' ELSE c.relkind::text END AS object_type
FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
WHERE n.nspname='mimiciv_derived' AND c.relkind IN('r','p','v','m','f');

DROP TABLE IF EXISTS project_aecopd.hosp_field_inventory;
CREATE TABLE project_aecopd.hosp_field_inventory AS
SELECT source_table AS data_source,field_name AS field_or_item,NULL::text AS item_id,data_type,
       cohort_admissions,non_null_admissions AS admissions_with_data,NULL::bigint AS patients_with_data,
       coverage_pct,NULL::bigint AS total_records,NULL::text AS observed_units_or_routes,
       (source_table='mimiciv_hosp.admissions') AS has_hadm_id,
       CASE WHEN data_type IN('timestamp','date') THEN field_name ELSE NULL END AS time_fields,
       audit_note,
       'researcher_review_required'::text AS later_model_consideration
FROM project_aecopd.core_field_coverage
UNION ALL
SELECT 'mimiciv_hosp.labevents',COALESCE(label,'<missing dictionary label>'),itemid::text,'lab item',
       :cohort_n,admissions_with_item,patients_with_item,admission_coverage_pct,measurements,observed_units,
       TRUE,'charttime/storetime','aggregated within cohort; no imputation or unit harmonization; before/after admission flags retained',
       'candidate_after_window_and_unit_rules'
FROM project_aecopd.lab_item_coverage_top100
UNION ALL
SELECT 'mimiciv_hosp.prescriptions',COALESCE(drug,'<NULL>'),NULL,'raw drug name',
       :cohort_n,admissions_with_drug,patients_with_drug,admission_coverage_pct,prescription_rows,observed_routes,TRUE,'starttime/stoptime',
       'raw names/routes not standardized; timing anomalies counted','candidate_after_drug_normalization_and_time_rules'
FROM project_aecopd.medication_coverage_top100
UNION ALL
SELECT 'mimiciv_hosp.pharmacy',COALESCE(medication,'<NULL>'),NULL,'raw pharmacy medication',
       :cohort_n,admissions_with_medication,patients_with_medication,admission_coverage_pct,pharmacy_rows,observed_routes,TRUE,'starttime/stoptime/entertime/verifiedtime',
       'raw medication/routes/status not standardized; timing anomalies counted','candidate_after_medication_normalization_and_time_rules'
FROM project_aecopd.pharmacy_coverage_top100
UNION ALL
SELECT 'mimiciv_hosp.procedures_icd',COALESCE(long_title,'<missing dictionary title>'),icd_version::text||':'||icd_code,'ICD procedure',
       :cohort_n,admissions_with_procedure,patients_with_procedure,admission_coverage_pct,procedure_rows,NULL,TRUE,'chartdate',
       'candidate only; procedure and respiratory-support interpretation requires clinical review','candidate_after_clinical_code_review'
FROM project_aecopd.procedure_coverage_top100;

COMMIT;

\copy (SELECT * FROM project_aecopd.core_field_coverage ORDER BY source_table,field_name) TO 'outputs/05_core_field_coverage.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.core_value_distribution ORDER BY field_name,admissions DESC,value) TO 'outputs/05_core_value_distribution.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.lab_item_coverage_top100) TO 'outputs/05_lab_item_coverage_top100.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.medication_coverage_top100) TO 'outputs/05_medication_coverage_top100.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.pharmacy_coverage_top100) TO 'outputs/05_pharmacy_coverage_top100.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.procedure_coverage_top100) TO 'outputs/05_procedure_coverage_top100.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.icu_admission_audit ORDER BY subject_id,hadm_id) TO 'outputs/05_icu_admission_audit.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.icu_coverage_summary) TO 'outputs/05_icu_coverage.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.icu_selection_audit ORDER BY has_icu_stay) TO 'outputs/05_icu_selection_audit.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.structured_source_coverage ORDER BY data_source) TO 'outputs/05_structured_source_coverage.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.derived_inventory) TO 'outputs/05_derived_inventory.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.hosp_field_inventory ORDER BY data_source,coverage_pct DESC,field_or_item) TO 'outputs/05_hosp_field_inventory.csv' CSV HEADER



