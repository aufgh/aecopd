-- 在 mimiciv 数据库执行；只创建 v2，不覆盖 v1 或任何模型表。
-- 修订：49120、496、J449 仅主诊断(seq_num=1)可触发患者入组；
-- 其余急性/肺气肿编码仍允许诊断前 3 位。入组后保留患者全部住院。
-- v1 是上述条件的超集，因此用其全住院资料作稳定底座；不复用 v1 的入组标记。
\set ON_ERROR_STOP on
BEGIN;

CREATE TABLE project_aecopd.copd_longitudinal_v2 AS
WITH qualifying_admissions AS (
  SELECT DISTINCT d.subject_id, d.hadm_id
  FROM mimiciv_hosp.diagnoses_icd d
  WHERE (d.seq_num <= 3 AND (
           (d.icd_version = 9 AND d.icd_code IN ('49121','49122','4928'))
        OR (d.icd_version = 10 AND d.icd_code IN ('J431','J432','J438','J439','J440','J441'))
        ))
     OR (d.seq_num = 1 AND (
           (d.icd_version = 9 AND d.icd_code IN ('49120','496'))
        OR (d.icd_version = 10 AND d.icd_code = 'J449')
        ))
), first_qualification AS (
  SELECT DISTINCT ON (v.subject_id)
         v.subject_id, v.hadm_id AS first_qualifying_hadm_id,
         v.admittime AS first_qualifying_admittime
  FROM project_aecopd.copd_longitudinal_v1 v
  JOIN qualifying_admissions q USING (subject_id, hadm_id)
  ORDER BY v.subject_id, v.admittime, v.hadm_id
)
SELECT v.subject_id, v.hadm_id, v.admittime, v.dischtime,
       v.admission_type, v.admission_location, v.discharge_location,
       v.hospital_expire_flag,
       f.first_qualifying_hadm_id, f.first_qualifying_admittime,
       (q.hadm_id IS NOT NULL) AS has_qualifying_code,
       v.is_strict_aecopd, v.has_49122, v.has_j440,
       v.has_emphysema_code, v.has_chronic_copd_code, v.matching_codes,
       (v.admittime, v.hadm_id) >=
         (f.first_qualifying_admittime, f.first_qualifying_hadm_id)
         AS at_or_after_first_qualification,
       v.all_admission_order, v.prior_all_admission_count,
       v.prior_strict_aecopd_count
FROM project_aecopd.copd_longitudinal_v1 v
JOIN first_qualification f USING (subject_id)
LEFT JOIN qualifying_admissions q USING (subject_id, hadm_id);

ALTER TABLE project_aecopd.copd_longitudinal_v2 ADD PRIMARY KEY (hadm_id);
CREATE INDEX copd_longitudinal_v2_subject_time_idx
  ON project_aecopd.copd_longitudinal_v2(subject_id, admittime, hadm_id);

COMMIT;
