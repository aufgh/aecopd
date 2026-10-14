-- 每次合格出院后首次再次发生任意清单 COPD／肺气肿编码住院；保留旧严格标签表不变。
-- 数据库只新增独立表；若目标已存在则失败，不自动覆盖。
\set ON_ERROR_STOP on
BEGIN;

CREATE TABLE project_aecopd.copd_broad_readmission_labels_v1 AS
WITH ordered AS MATERIALIZED (
  SELECT c.subject_id,c.hadm_id,c.admittime,c.dischtime,c.admission_location,
         c.best_copd_seq_num,c.is_strict_aecopd,c.has_49122,c.has_j440,
         lag(c.dischtime) OVER (
           PARTITION BY c.subject_id ORDER BY c.admittime,c.hadm_id
         ) AS previous_copd_dischtime
  FROM project_aecopd.copd_coded_admissions_v1_gate c
), candidates AS MATERIALIZED (
  SELECT *,
         COALESCE((admission_location='TRANSFER FROM HOSPITAL'
           AND admittime>=previous_copd_dischtime
           AND admittime<=previous_copd_dischtime+interval '2 days'),false)
           AS is_transfer_continuation
  FROM ordered
), event_times AS (
  SELECT a.subject_id,a.hadm_id,a.source_dataset,a.admittime,a.dischtime,
         a.best_copd_seq_num,a.coded_admission_order,
         a.is_strict_aecopd AS index_is_strict_aecopd,
         a.at_or_after_first_qualification,
         n.hadm_id AS next_copd_hadm_id,
         n.admittime AS next_copd_admittime,
         n.best_copd_seq_num AS next_copd_best_code_seq_num,
         n.is_strict_aecopd AS next_is_strict_aecopd,
         n.has_49122 AS next_has_49122,
         n.has_j440 AS next_has_j440,
         extract(epoch FROM (n.admittime-a.dischtime))/86400.0 AS next_copd_days,
         CASE WHEN p.dod>=a.dischtime::date
              THEN (p.dod-a.dischtime::date)::double precision END
           AS observed_death_days
  FROM project_aecopd.copd_coded_admissions_v1_gate a
  JOIN mimiciv_hosp.patients p USING(subject_id)
  LEFT JOIN LATERAL (
    SELECT b.hadm_id,b.admittime,b.best_copd_seq_num,b.is_strict_aecopd,
           b.has_49122,b.has_j440
    FROM candidates b
    WHERE b.subject_id=a.subject_id
      AND (b.admittime,b.hadm_id)>(a.admittime,a.hadm_id)
      AND b.admittime>=a.dischtime
      AND NOT b.is_transfer_continuation
    ORDER BY b.admittime,b.hadm_id LIMIT 1
  ) n ON true
  WHERE a.discharge_landmark_eligible
), labels AS (
  SELECT *,CASE
    WHEN next_copd_days<=365
      AND (observed_death_days IS NULL OR next_copd_days<=observed_death_days) THEN 1
    WHEN observed_death_days<=365
      AND (next_copd_days IS NULL OR observed_death_days<next_copd_days) THEN 2
    ELSE 0 END AS event_type
  FROM event_times
)
SELECT subject_id,hadm_id,source_dataset,admittime,dischtime,
       best_copd_seq_num,coded_admission_order,index_is_strict_aecopd,
       at_or_after_first_qualification,
       next_copd_hadm_id,next_copd_admittime,next_copd_best_code_seq_num,
       next_is_strict_aecopd,next_has_49122,next_has_j440,
       next_copd_days,observed_death_days,event_type,
       CASE event_type WHEN 1 THEN 'COPD_CODED_READMISSION'
         WHEN 2 THEN 'ALL_CAUSE_DEATH' ELSE 'CENSORED_365D' END AS event_name,
       CASE event_type WHEN 1 THEN next_copd_days
         WHEN 2 THEN observed_death_days ELSE 365.0 END AS time_days,
       'World A: database-observed readmission; 365d administrative censoring; individual follow-up completeness unverified'::text
         AS followup_assumption
FROM labels;

ALTER TABLE project_aecopd.copd_broad_readmission_labels_v1 ADD PRIMARY KEY(hadm_id);
CREATE INDEX copd_broad_readmission_labels_v1_subject_idx
  ON project_aecopd.copd_broad_readmission_labels_v1(subject_id,admittime,hadm_id);

DO $$ BEGIN
  IF (SELECT count(*) FROM project_aecopd.copd_broad_readmission_labels_v1)
       <> (SELECT count(*) FROM project_aecopd.copd_coded_admissions_v1_gate
           WHERE discharge_landmark_eligible)
     OR EXISTS (
       SELECT 1 FROM project_aecopd.copd_broad_readmission_labels_v1 l
       LEFT JOIN project_aecopd.copd_coded_admissions_v1_gate c USING(hadm_id)
       WHERE c.hadm_id IS NULL OR NOT c.discharge_landmark_eligible
         OR l.event_type NOT IN (0,1,2)
         OR l.time_days<0 OR l.time_days>365
         OR (l.event_type=1 AND l.next_copd_hadm_id IS NULL)
     ) THEN
    RAISE EXCEPTION 'Broad COPD-coded readmission label validation failed';
  END IF;
END $$;
COMMIT;
