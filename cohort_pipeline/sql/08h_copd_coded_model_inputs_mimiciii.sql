-- MIMIC-III：与 IV 相同的低维结构化输入；LABEVENTS 无 storetime，
-- 只能按 charttime 限制，不能证明每个检验结果已在出院前写入。
\set ON_ERROR_STOP on
BEGIN;

CREATE TABLE project_aecopd.copd_coded_model_base_v1 AS
SELECT l.subject_id,l.hadm_id,l.source_dataset,l.event_type,l.time_days,
       LEAST(90,GREATEST(0,extract(epoch FROM (l.admittime-p.dob))/31557600.0))::double precision
         AS age_at_admission,
       CASE p.gender WHEN 'M' THEN 1 WHEN 'F' THEN 0 END AS male,
       c.prior_coded_admission_count,
       (SELECT count(*) FROM project_aecopd.copd_coded_admissions_v1_gate h
        WHERE h.subject_id=l.subject_id AND h.is_strict_aecopd
          AND h.dischtime<=l.admittime) AS prior_strict_aecopd_count,
       m.has_timed_discharge_summary,m.has_timed_other_note
FROM project_aecopd.copd_coded_strict_index_labels_v1 l
JOIN project_aecopd.copd_coded_admissions_v1_gate c USING(hadm_id)
JOIN public.patients p ON p.subject_id=l.subject_id
JOIN project_aecopd.copd_coded_dataset_manifest m USING(hadm_id);

ALTER TABLE project_aecopd.copd_coded_model_base_v1 ADD PRIMARY KEY(hadm_id);

CREATE TABLE project_aecopd.copd_coded_model_labs_v1 AS
WITH selected AS MATERIALIZED (
  SELECT l.hadm_id,x.itemid,x.charttime,x.row_id,x.valuenum
  FROM project_aecopd.copd_coded_strict_index_labels_v1 l
  JOIN public.labevents x ON x.hadm_id=l.hadm_id
  WHERE x.itemid IN (50818,50821,51301,51222,50912)
    AND x.charttime BETWEEN l.admittime AND l.dischtime
    AND x.valuenum IS NOT NULL
    AND x.valuenum>=0
), ranked AS (
  SELECT *,
         row_number() OVER (PARTITION BY hadm_id,itemid ORDER BY charttime,row_id)
           AS first_rank,
         row_number() OVER (PARTITION BY hadm_id,itemid ORDER BY charttime DESC,row_id DESC)
           AS last_rank
  FROM selected
)
SELECT hadm_id,itemid,count(*) AS measurement_count,
       max(valuenum) FILTER (WHERE first_rank=1) AS first_value,
       max(valuenum) FILTER (WHERE last_rank=1) AS last_value,
       min(valuenum) AS min_value,max(valuenum) AS max_value
FROM ranked GROUP BY hadm_id,itemid;

ALTER TABLE project_aecopd.copd_coded_model_labs_v1 ADD PRIMARY KEY(hadm_id,itemid);

DO $$ BEGIN
  IF (SELECT count(*) FROM project_aecopd.copd_coded_model_base_v1)
     <> (SELECT count(*) FROM project_aecopd.copd_coded_strict_index_labels_v1)
     OR EXISTS (SELECT 1 FROM project_aecopd.copd_coded_model_base_v1
                WHERE age_at_admission IS NULL OR male IS NULL
                   OR prior_coded_admission_count<0 OR prior_strict_aecopd_count<0)
     OR EXISTS (SELECT 1 FROM project_aecopd.copd_coded_model_labs_v1
                WHERE measurement_count<1 OR first_value IS NULL OR last_value IS NULL)
  THEN RAISE EXCEPTION 'MIMIC-III model input consistency check failed'; END IF;
END $$;
COMMIT;
