-- 在独立 mimiciii 数据库执行；源表为 public.admissions / public.diagnoses_icd。
-- 与 MIMIC-IV 版本使用相同的 ICD-9 患者入组与住院层语义；不跨库合并患者。
\set ON_ERROR_STOP on
BEGIN;

CREATE SCHEMA IF NOT EXISTS project_aecopd;
CREATE TABLE project_aecopd.copd_longitudinal_v1 AS
WITH codebook(icd9_code, code_group) AS (
  VALUES
    ('49120', 'copd_chronic'),
    ('49121', 'aecopd_strict'),
    ('49122', 'copd_acute_bronchitis'),
    ('4928',  'emphysema'),
    ('496',   'copd_chronic')
), diagnosis_flags AS (
  SELECT d.subject_id, d.hadm_id,
         bool_or(d.seq_num <= 3) AS has_top3_qualifying_code,
         bool_or(c.code_group = 'aecopd_strict') AS is_strict_aecopd,
         bool_or(c.code_group = 'copd_acute_bronchitis') AS has_49122,
         bool_or(c.code_group = 'emphysema') AS has_emphysema_code,
         bool_or(c.code_group = 'copd_chronic') AS has_chronic_copd_code,
         array_agg(DISTINCT d.icd9_code ORDER BY d.icd9_code) AS matching_codes
  FROM public.diagnoses_icd d
  JOIN codebook c USING (icd9_code)
  GROUP BY d.subject_id, d.hadm_id
), first_qualification AS (
  SELECT DISTINCT ON (a.subject_id)
         a.subject_id, a.hadm_id AS first_qualifying_hadm_id,
         a.admittime AS first_qualifying_admittime
  FROM public.admissions a
  JOIN diagnosis_flags d USING (subject_id, hadm_id)
  WHERE d.has_top3_qualifying_code
  ORDER BY a.subject_id, a.admittime, a.hadm_id
), selected_admissions AS (
  SELECT a.subject_id, a.hadm_id, a.admittime, a.dischtime,
         a.admission_type, a.admission_location, a.discharge_location,
         a.hospital_expire_flag,
         q.first_qualifying_hadm_id, q.first_qualifying_admittime,
         COALESCE(d.has_top3_qualifying_code, false) AS has_top3_qualifying_code,
         COALESCE(d.is_strict_aecopd, false) AS is_strict_aecopd,
         COALESCE(d.has_49122, false) AS has_49122,
         false AS has_j440,
         COALESCE(d.has_emphysema_code, false) AS has_emphysema_code,
         COALESCE(d.has_chronic_copd_code, false) AS has_chronic_copd_code,
         d.matching_codes,
         (a.admittime, a.hadm_id) >=
           (q.first_qualifying_admittime, q.first_qualifying_hadm_id)
           AS at_or_after_first_qualification
  FROM public.admissions a
  JOIN first_qualification q USING (subject_id)
  LEFT JOIN diagnosis_flags d USING (subject_id, hadm_id)
)
SELECT s.*,
       row_number() OVER (PARTITION BY subject_id ORDER BY admittime, hadm_id) AS all_admission_order,
       row_number() OVER (PARTITION BY subject_id ORDER BY admittime, hadm_id) - 1 AS prior_all_admission_count,
       coalesce(sum(is_strict_aecopd::integer) OVER (
         PARTITION BY subject_id ORDER BY admittime, hadm_id
         ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0)::integer
         AS prior_strict_aecopd_count
FROM selected_admissions s;

ALTER TABLE project_aecopd.copd_longitudinal_v1 ADD PRIMARY KEY (hadm_id);
CREATE INDEX copd_longitudinal_v1_subject_time_idx
  ON project_aecopd.copd_longitudinal_v1(subject_id, admittime, hadm_id);

COMMIT;
