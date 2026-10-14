-- MIMIC-III：新队列的逐住院来源/时间可用性清单，不等于模型特征张量。
-- LABEVENTS 与 MICROBIOLOGYEVENTS 无结果存储时刻；结果当时可见性不可验证。
\set ON_ERROR_STOP on
BEGIN;

CREATE TABLE project_aecopd.copd_coded_dataset_manifest AS
SELECT c.subject_id,c.hadm_id,c.source_dataset,c.admittime,c.dischtime,
       c.best_copd_seq_num,c.is_strict_aecopd,
       c.at_or_after_first_qualification,c.discharge_landmark_eligible,
       c.coded_admission_order,c.prior_coded_admission_count,
       n.discharge_summary_rows>0 AS has_discharge_summary,
       n.timed_discharge_summary_rows>0 AS has_timed_discharge_summary,
       n.timed_other_note_rows>0 AS has_timed_other_note,
       l.lab_rows_charted_before_discharge,
       NULL::bigint AS lab_rows_result_time_verified,
       r.rx_rows_started_before_discharge,
       m.micro_rows_charted_before_discharge,
       NULL::bigint AS micro_rows_result_time_verified,
       i.icu_stay_count
FROM project_aecopd.copd_coded_admissions_v1_gate c
JOIN project_aecopd.copd_coded_note_availability n USING(hadm_id)
LEFT JOIN LATERAL (
  SELECT count(*) AS lab_rows_charted_before_discharge
  FROM public.labevents x
  WHERE x.hadm_id=c.hadm_id AND x.charttime BETWEEN c.admittime AND c.dischtime
) l ON true
LEFT JOIN LATERAL (
  SELECT count(*) AS rx_rows_started_before_discharge
  FROM public.prescriptions x
  WHERE x.hadm_id=c.hadm_id AND x.startdate BETWEEN c.admittime::date AND c.dischtime::date
) r ON true
LEFT JOIN LATERAL (
  SELECT count(*) AS micro_rows_charted_before_discharge
  FROM public.microbiologyevents x
  WHERE x.hadm_id=c.hadm_id AND x.charttime BETWEEN c.admittime AND c.dischtime
) m ON true
LEFT JOIN LATERAL (
  SELECT count(*) AS icu_stay_count
  FROM public.icustays x WHERE x.hadm_id=c.hadm_id
) i ON true;

ALTER TABLE project_aecopd.copd_coded_dataset_manifest ADD PRIMARY KEY (hadm_id);
CREATE INDEX copd_coded_dataset_manifest_subject_time_idx
  ON project_aecopd.copd_coded_dataset_manifest(subject_id,admittime,hadm_id);

DO $$ BEGIN
  IF (SELECT count(*) FROM project_aecopd.copd_coded_dataset_manifest)
     <> (SELECT count(*) FROM project_aecopd.copd_coded_admissions_v1_gate) THEN
    RAISE EXCEPTION 'MIMIC-III dataset manifest row count mismatch';
  END IF;
END $$;

COMMIT;
