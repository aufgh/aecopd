-- 在 mimiciv 数据库执行。创建独立纵向住院表，不重建既有 master/model 表。
-- 患者层：任一次住院前三位出现 COPD/常见肺气肿编码即可入组。
-- 住院层：保留该患者在本数据库的全部住院，包括首次入组编码之前的住院。
-- 事件层：严格 AECOPD 与 COPD+急性支气管炎、COPD+感染分列，不在此处定义预测标签。
\set ON_ERROR_STOP on
BEGIN;

CREATE TABLE project_aecopd.copd_longitudinal_v1 AS
WITH codebook(icd_version, icd_code, code_group) AS (
  VALUES
    (9, '49120', 'copd_chronic'),
    (9, '49121', 'aecopd_strict'),
    (9, '49122', 'copd_acute_bronchitis'),
    (9, '4928',  'emphysema'),
    (9, '496',   'copd_chronic'),
    (10, 'J431', 'emphysema'),
    (10, 'J432', 'emphysema'),
    (10, 'J438', 'emphysema'),
    (10, 'J439', 'emphysema'),
    (10, 'J440', 'copd_acute_infection'),
    (10, 'J441', 'aecopd_strict'),
    (10, 'J449', 'copd_chronic')
), diagnosis_flags AS (
  SELECT d.subject_id, d.hadm_id,
         bool_or(d.seq_num <= 3) AS has_top3_qualifying_code,
         bool_or(c.code_group = 'aecopd_strict') AS is_strict_aecopd,
         bool_or(c.code_group = 'copd_acute_bronchitis') AS has_49122,
         bool_or(c.code_group = 'copd_acute_infection') AS has_j440,
         bool_or(c.code_group = 'emphysema') AS has_emphysema_code,
         bool_or(c.code_group = 'copd_chronic') AS has_chronic_copd_code,
         array_agg(DISTINCT d.icd_code ORDER BY d.icd_code) AS matching_codes
  FROM mimiciv_hosp.diagnoses_icd d
  JOIN codebook c ON c.icd_version=d.icd_version AND c.icd_code=d.icd_code
  GROUP BY d.subject_id, d.hadm_id
), first_qualification AS (
  SELECT DISTINCT ON (a.subject_id)
         a.subject_id, a.hadm_id AS first_qualifying_hadm_id,
         a.admittime AS first_qualifying_admittime
  FROM mimiciv_hosp.admissions a
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
         COALESCE(d.has_j440, false) AS has_j440,
         COALESCE(d.has_emphysema_code, false) AS has_emphysema_code,
         COALESCE(d.has_chronic_copd_code, false) AS has_chronic_copd_code,
         d.matching_codes,
         (a.admittime, a.hadm_id) >=
           (q.first_qualifying_admittime, q.first_qualifying_hadm_id)
           AS at_or_after_first_qualification
  FROM mimiciv_hosp.admissions a
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
