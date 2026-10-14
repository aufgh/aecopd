-- MIMIC-IV：新队列的逐住院来源/时间可用性清单，不等于模型特征张量。
-- 处方行只证明医嘱存在，不证明实际给药。
\set ON_ERROR_STOP on
BEGIN;

CREATE TABLE project_aecopd.copd_coded_dataset_manifest AS
WITH cohort AS MATERIALIZED (
  SELECT * FROM project_aecopd.copd_coded_admissions_v1_gate
), labs AS MATERIALIZED (
  SELECT c.hadm_id,
         count(*) AS lab_rows_charted_before_discharge,
         count(*) FILTER (WHERE x.storetime<=c.dischtime) AS lab_rows_result_time_verified
  FROM cohort c JOIN mimiciv_hosp.labevents x ON x.hadm_id=c.hadm_id
  WHERE x.charttime BETWEEN c.admittime AND c.dischtime
  GROUP BY c.hadm_id
), rx AS MATERIALIZED (
  SELECT c.hadm_id,count(*) AS rx_rows_started_before_discharge
  FROM cohort c JOIN mimiciv_hosp.prescriptions x ON x.hadm_id=c.hadm_id
  WHERE x.starttime BETWEEN c.admittime AND c.dischtime
  GROUP BY c.hadm_id
), micro AS MATERIALIZED (
  SELECT c.hadm_id,
         count(*) AS micro_rows_charted_before_discharge,
         count(*) FILTER (WHERE x.storetime<=c.dischtime) AS micro_rows_result_time_verified
  FROM cohort c JOIN mimiciv_hosp.microbiologyevents x ON x.hadm_id=c.hadm_id
  WHERE x.charttime BETWEEN c.admittime AND c.dischtime
  GROUP BY c.hadm_id
), icu AS MATERIALIZED (
  SELECT c.hadm_id,count(*) AS icu_stay_count
  FROM cohort c JOIN mimiciv_icu.icustays x ON x.hadm_id=c.hadm_id
  GROUP BY c.hadm_id
)
SELECT c.subject_id,c.hadm_id,c.source_dataset,c.admittime,c.dischtime,
       c.best_copd_seq_num,c.is_strict_aecopd,
       c.at_or_after_first_qualification,c.discharge_landmark_eligible,
       c.coded_admission_order,c.prior_coded_admission_count,
       n.discharge_summary_rows>0 AS has_discharge_summary,
       n.timed_discharge_summary_rows>0 AS has_timed_discharge_summary,
       n.timed_radiology_rows>0 AS has_timed_other_note,
       COALESCE(l.lab_rows_charted_before_discharge,0) AS lab_rows_charted_before_discharge,
       COALESCE(l.lab_rows_result_time_verified,0) AS lab_rows_result_time_verified,
       COALESCE(r.rx_rows_started_before_discharge,0) AS rx_rows_started_before_discharge,
       COALESCE(m.micro_rows_charted_before_discharge,0) AS micro_rows_charted_before_discharge,
       COALESCE(m.micro_rows_result_time_verified,0) AS micro_rows_result_time_verified,
       COALESCE(i.icu_stay_count,0) AS icu_stay_count
FROM cohort c
JOIN project_aecopd.copd_coded_note_availability n USING(hadm_id)
LEFT JOIN labs l USING(hadm_id)
LEFT JOIN rx r USING(hadm_id)
LEFT JOIN micro m USING(hadm_id)
LEFT JOIN icu i USING(hadm_id);

ALTER TABLE project_aecopd.copd_coded_dataset_manifest ADD PRIMARY KEY (hadm_id);
CREATE INDEX copd_coded_dataset_manifest_subject_time_idx
  ON project_aecopd.copd_coded_dataset_manifest(subject_id,admittime,hadm_id);

DO $$ BEGIN
  IF (SELECT count(*) FROM project_aecopd.copd_coded_dataset_manifest)
     <> (SELECT count(*) FROM project_aecopd.copd_coded_admissions_v1_gate) THEN
    RAISE EXCEPTION 'MIMIC-IV dataset manifest row count mismatch';
  END IF;
END $$;

COMMIT;
