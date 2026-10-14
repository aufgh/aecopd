\set ON_ERROR_STOP on
SET client_encoding = 'UTF8';
BEGIN;
CREATE SCHEMA IF NOT EXISTS project_aecopd;

DROP TABLE IF EXISTS project_aecopd.aecopd_admissions_strict;
CREATE TABLE project_aecopd.aecopd_admissions_strict AS
WITH dx AS (
  SELECT
    subject_id,
    hadm_id,
    MIN(seq_num) AS best_seq_num,
    STRING_AGG(DISTINCT icd_code, ',' ORDER BY icd_code) AS aecopd_codes,
    STRING_AGG(DISTINCT icd_version::text, ',' ORDER BY icd_version::text) AS aecopd_icd_versions
  FROM mimiciv_hosp.diagnoses_icd
  WHERE (icd_version=10 AND icd_code='J441')
     OR (icd_version=9 AND icd_code='49121')
  GROUP BY subject_id, hadm_id
), joined AS (
  SELECT
    a.subject_id,
    a.hadm_id,
    a.admittime,
    a.dischtime,
    a.deathtime,
    a.admission_type,
    a.admission_location,
    a.discharge_location,
    a.insurance,
    a.language,
    a.marital_status,
    a.race,
    a.edregtime,
    a.edouttime,
    a.hospital_expire_flag,
    dx.best_seq_num,
    dx.aecopd_codes,
    dx.aecopd_icd_versions,
    EXTRACT(EPOCH FROM (a.dischtime-a.admittime))/86400.0 AS hospital_los_days,
    LAG(a.dischtime) OVER (PARTITION BY a.subject_id ORDER BY a.admittime,a.hadm_id) AS previous_strict_dischtime
  FROM dx
  JOIN mimiciv_hosp.admissions a USING (subject_id,hadm_id)
)
SELECT
  *,
  (admittime IS NULL OR dischtime IS NULL) AS qc_missing_admission_time,
  (dischtime <= admittime) AS qc_nonpositive_los,
  (previous_strict_dischtime IS NOT NULL AND admittime < previous_strict_dischtime) AS qc_overlap_previous_strict,
  (admission_location='TRANSFER FROM HOSPITAL') AS qc_transfer_from_hospital
FROM joined;

ALTER TABLE project_aecopd.aecopd_admissions_strict ADD CONSTRAINT aecopd_admissions_strict_pk PRIMARY KEY (hadm_id);
CREATE INDEX aecopd_admissions_strict_subject_time_idx ON project_aecopd.aecopd_admissions_strict(subject_id,admittime,hadm_id);

DROP TABLE IF EXISTS project_aecopd.aecopd_cohort_qc;
CREATE TABLE project_aecopd.aecopd_cohort_qc AS
SELECT 'rows'::text metric, COUNT(*)::text value FROM project_aecopd.aecopd_admissions_strict
UNION ALL SELECT 'distinct_hadm_id', COUNT(DISTINCT hadm_id)::text FROM project_aecopd.aecopd_admissions_strict
UNION ALL SELECT 'distinct_subject_id', COUNT(DISTINCT subject_id)::text FROM project_aecopd.aecopd_admissions_strict
UNION ALL SELECT 'duplicate_hadm_id_groups', COUNT(*)::text FROM (SELECT hadm_id FROM project_aecopd.aecopd_admissions_strict GROUP BY hadm_id HAVING COUNT(*)>1) q
UNION ALL SELECT 'missing_admission_time', COUNT(*) FILTER (WHERE qc_missing_admission_time)::text FROM project_aecopd.aecopd_admissions_strict
UNION ALL SELECT 'nonpositive_los', COUNT(*) FILTER (WHERE qc_nonpositive_los)::text FROM project_aecopd.aecopd_admissions_strict
UNION ALL SELECT 'overlap_previous_strict', COUNT(*) FILTER (WHERE qc_overlap_previous_strict)::text FROM project_aecopd.aecopd_admissions_strict
UNION ALL SELECT 'transfer_from_hospital', COUNT(*) FILTER (WHERE qc_transfer_from_hospital)::text FROM project_aecopd.aecopd_admissions_strict
UNION ALL SELECT 'los_min_days', MIN(hospital_los_days)::text FROM project_aecopd.aecopd_admissions_strict
UNION ALL SELECT 'los_p25_days', percentile_cont(0.25) WITHIN GROUP (ORDER BY hospital_los_days)::text FROM project_aecopd.aecopd_admissions_strict
UNION ALL SELECT 'los_median_days', percentile_cont(0.50) WITHIN GROUP (ORDER BY hospital_los_days)::text FROM project_aecopd.aecopd_admissions_strict
UNION ALL SELECT 'los_p75_days', percentile_cont(0.75) WITHIN GROUP (ORDER BY hospital_los_days)::text FROM project_aecopd.aecopd_admissions_strict
UNION ALL SELECT 'los_max_days', MAX(hospital_los_days)::text FROM project_aecopd.aecopd_admissions_strict;

COMMIT;

\copy (SELECT * FROM project_aecopd.aecopd_admissions_strict ORDER BY subject_id,admittime,hadm_id) TO 'outputs/02_aecopd_admissions_strict.csv' CSV HEADER
\copy (SELECT * FROM project_aecopd.aecopd_cohort_qc) TO 'outputs/02_aecopd_cohort_qc.csv' CSV HEADER



