-- MIMIC-III：沿用 v1 患者入组规则，仅保留本次住院诊断含 v1 编码的住院。
-- 在独立 mimiciii 数据库执行；不修改 v1/v2 或原始表。
\set ON_ERROR_STOP on
BEGIN;

CREATE TABLE project_aecopd.copd_coded_admissions_v1_gate AS
WITH codebook(icd9_code) AS (
  VALUES ('49120'),('49121'),('49122'),('4928'),('496')
), coded AS (
  SELECT v.subject_id, v.hadm_id, v.admittime, v.dischtime,
         v.admission_type, v.admission_location, v.discharge_location,
         v.hospital_expire_flag,
         v.first_qualifying_hadm_id, v.first_qualifying_admittime,
         v.at_or_after_first_qualification,
         v.has_top3_qualifying_code,
         v.is_strict_aecopd, v.has_49122, v.has_j440,
         v.has_emphysema_code, v.has_chronic_copd_code,
         v.matching_codes,
         min(d.seq_num) AS best_copd_seq_num
  FROM project_aecopd.copd_longitudinal_v1 v
  JOIN public.diagnoses_icd d USING (subject_id, hadm_id)
  JOIN codebook b ON b.icd9_code=d.icd9_code
  WHERE v.matching_codes IS NOT NULL
  GROUP BY v.subject_id, v.hadm_id, v.admittime, v.dischtime,
           v.admission_type, v.admission_location, v.discharge_location,
           v.hospital_expire_flag,
           v.first_qualifying_hadm_id, v.first_qualifying_admittime,
           v.at_or_after_first_qualification,
           v.has_top3_qualifying_code,
           v.is_strict_aecopd, v.has_49122, v.has_j440,
           v.has_emphysema_code, v.has_chronic_copd_code,
           v.matching_codes
)
SELECT 'MIMIC-III'::text AS source_dataset, c.*,
       row_number() OVER (PARTITION BY subject_id ORDER BY admittime, hadm_id)
         AS coded_admission_order,
       row_number() OVER (PARTITION BY subject_id ORDER BY admittime, hadm_id)-1
         AS prior_coded_admission_count,
       (hospital_expire_flag=0 AND dischtime>admittime)
         AS discharge_landmark_eligible
FROM coded c;

ALTER TABLE project_aecopd.copd_coded_admissions_v1_gate ADD PRIMARY KEY (hadm_id);
CREATE INDEX copd_coded_admissions_v1_gate_subject_time_idx
  ON project_aecopd.copd_coded_admissions_v1_gate(subject_id,admittime,hadm_id);

DO $$ BEGIN
  IF (SELECT count(DISTINCT subject_id)
      FROM project_aecopd.copd_coded_admissions_v1_gate)
     <> (SELECT count(DISTINCT subject_id)
         FROM project_aecopd.copd_longitudinal_v1)
     OR (SELECT count(*) FROM project_aecopd.copd_coded_admissions_v1_gate)
        <> (SELECT count(*) FROM project_aecopd.copd_longitudinal_v1
            WHERE matching_codes IS NOT NULL)
     OR EXISTS (SELECT 1 FROM project_aecopd.copd_coded_admissions_v1_gate
                WHERE best_copd_seq_num IS NULL) THEN
    RAISE EXCEPTION 'COPD coded-admission cohort consistency check failed';
  END IF;
END $$;

COMMIT;
