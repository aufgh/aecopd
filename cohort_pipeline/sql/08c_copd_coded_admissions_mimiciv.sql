-- MIMIC-IV：沿用 v1 患者入组规则，仅保留本次住院诊断含 v1 编码的住院。
-- 创建独立版本；不修改 v1/v2、master_cohort 或任何模型标签。
\set ON_ERROR_STOP on
BEGIN;

CREATE TABLE project_aecopd.copd_coded_admissions_v1_gate AS
WITH codebook(icd_version, icd_code) AS (
  VALUES (9,'49120'),(9,'49121'),(9,'49122'),(9,'4928'),(9,'496'),
         (10,'J431'),(10,'J432'),(10,'J438'),(10,'J439'),
         (10,'J440'),(10,'J441'),(10,'J449')
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
  JOIN mimiciv_hosp.diagnoses_icd d USING (subject_id, hadm_id)
  JOIN codebook b ON b.icd_version=d.icd_version AND b.icd_code=d.icd_code
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
SELECT 'MIMIC-IV'::text AS source_dataset, c.*,
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
