-- MIMIC-IV：v1 患者门槛不变；每次存活且正时长的 COPD 编码住院出院均可作为预测起点。
-- 包含首次前三位编码资格出现之前的 COPD 编码住院；显式保留该标记供选择偏倚审计。
-- 结局仍为下一次非转院续接的严格 AECOPD 再入院，死亡为竞争事件。
\set ON_ERROR_STOP on
BEGIN;

CREATE TABLE project_aecopd.copd_coded_all_index_labels_v2 AS
WITH strict_ordered AS MATERIALIZED (
  SELECT c.subject_id,c.hadm_id,c.admittime,c.dischtime,c.admission_location,
         lag(c.dischtime) OVER (
           PARTITION BY c.subject_id ORDER BY c.admittime,c.hadm_id
         ) AS previous_strict_dischtime
  FROM project_aecopd.copd_coded_admissions_v1_gate c
  WHERE c.is_strict_aecopd
), strict_events AS MATERIALIZED (
  SELECT *,
         COALESCE((admission_location='TRANSFER FROM HOSPITAL'
          AND admittime>=previous_strict_dischtime
          AND admittime<=previous_strict_dischtime+interval '2 days'),false)
           AS is_transfer_continuation
  FROM strict_ordered
), event_times AS (
  SELECT a.subject_id,a.hadm_id,a.source_dataset,a.admittime,a.dischtime,
         a.best_copd_seq_num,a.coded_admission_order,
         a.is_strict_aecopd AS index_is_strict_aecopd,
         a.at_or_after_first_qualification,
         n.hadm_id AS next_observed_aecopd_hadm_id,
         n.admittime AS next_observed_aecopd_admittime,
         extract(epoch FROM (n.admittime-a.dischtime))/86400.0
           AS next_observed_aecopd_days,
         CASE WHEN p.dod>=a.dischtime::date
              THEN (p.dod-a.dischtime::date)::double precision END
           AS observed_death_days
  FROM project_aecopd.copd_coded_admissions_v1_gate a
  JOIN mimiciv_hosp.patients p USING(subject_id)
  LEFT JOIN LATERAL (
    SELECT b.hadm_id,b.admittime
    FROM strict_events b
    WHERE b.subject_id=a.subject_id
      AND (b.admittime,b.hadm_id)>(a.admittime,a.hadm_id)
      AND b.admittime>=a.dischtime
      AND NOT b.is_transfer_continuation
    ORDER BY b.admittime,b.hadm_id LIMIT 1
  ) n ON true
  WHERE a.discharge_landmark_eligible
), labels AS (
  SELECT *, CASE
    WHEN next_observed_aecopd_days<=365
      AND (observed_death_days IS NULL
           OR next_observed_aecopd_days<=observed_death_days) THEN 1
    WHEN observed_death_days<=365
      AND (next_observed_aecopd_days IS NULL
           OR observed_death_days<next_observed_aecopd_days) THEN 2
    ELSE 0 END AS event_type
  FROM event_times
)
SELECT subject_id,hadm_id,source_dataset,admittime,dischtime,
       best_copd_seq_num,coded_admission_order,index_is_strict_aecopd,
       at_or_after_first_qualification,
       next_observed_aecopd_hadm_id,next_observed_aecopd_admittime,
       next_observed_aecopd_days,observed_death_days,event_type,
       CASE event_type WHEN 1 THEN 'AECOPD_READMISSION'
         WHEN 2 THEN 'ALL_CAUSE_DEATH' ELSE 'CENSORED_365D' END AS event_name,
       CASE event_type WHEN 1 THEN next_observed_aecopd_days
         WHEN 2 THEN observed_death_days ELSE 365.0 END AS time_days,
       'World A: database-observed readmission; 365d administrative censoring; individual follow-up completeness unverified'::text
         AS followup_assumption
FROM labels;

ALTER TABLE project_aecopd.copd_coded_all_index_labels_v2 ADD PRIMARY KEY(hadm_id);
CREATE INDEX copd_coded_all_index_labels_v2_subject_idx
  ON project_aecopd.copd_coded_all_index_labels_v2(subject_id,admittime,hadm_id);

DO $$ BEGIN
  IF (SELECT count(*) FROM project_aecopd.copd_coded_all_index_labels_v2)
     <> (SELECT count(*) FROM project_aecopd.copd_coded_admissions_v1_gate
         WHERE discharge_landmark_eligible)
     OR EXISTS (
       SELECT 1 FROM project_aecopd.copd_coded_all_index_labels_v2 l
       LEFT JOIN project_aecopd.copd_coded_admissions_v1_gate c USING(hadm_id)
       WHERE c.hadm_id IS NULL OR NOT c.discharge_landmark_eligible
         OR l.coded_admission_order<>c.coded_admission_order
         OR l.event_type NOT IN (0,1,2)
         OR l.time_days<0 OR l.time_days>365
     ) THEN
    RAISE EXCEPTION 'MIMIC-IV all-coded index label consistency check failed';
  END IF;
END $$;
COMMIT;
